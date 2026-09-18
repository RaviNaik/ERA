"""Tiny, inspectable 32-rank data-parallel and state-sharding demonstration."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


class DemoMLP(nn.Module):
    """A deliberately small model that makes CPU experiments inexpensive."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass(frozen=True)
class VirtualStepResult:
    world_size: int
    samples_per_rank: int
    reference_loss: float
    mean_rank_loss: float
    max_gradient_difference: float
    max_parameter_difference_after_step: float


def make_regression_batch(
    world_size: int = 32,
    samples_per_rank: int = 2,
    input_dim: int = 8,
    seed: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic synthetic data divisible across virtual ranks."""

    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(world_size * samples_per_rank, input_dim, generator=generator)
    true_weights = torch.linspace(-0.8, 0.9, input_dim).unsqueeze(1)
    targets = inputs @ true_weights + 0.15
    return inputs, targets


def _flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().reshape(-1) for tensor in tensors])


def partition_tensor(tensor: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    """Split a flat tensor into balanced contiguous rank-owned shards."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    flat = tensor.detach().reshape(-1)
    base, remainder = divmod(flat.numel(), world_size)
    shards: list[torch.Tensor] = []
    start = 0
    for rank in range(world_size):
        width = base + int(rank < remainder)
        shards.append(flat[start : start + width].clone())
        start += width
    return shards


def reconstruct_tensor(shards: list[torch.Tensor]) -> torch.Tensor:
    """Reverse :func:`partition_tensor` in rank order."""

    return torch.cat(shards) if shards else torch.empty(0)


def virtual_data_parallel_step(
    world_size: int = 32,
    samples_per_rank: int = 2,
    learning_rate: float = 0.05,
    seed: int = 7,
    verbose: bool = True,
) -> VirtualStepResult:
    """Compare one virtual data-parallel SGD step with a global-batch step."""

    torch.manual_seed(seed)
    initial_model = DemoMLP()
    inputs, targets = make_regression_batch(world_size, samples_per_rank)
    loss_function = nn.MSELoss(reduction="mean")

    reference_model = copy.deepcopy(initial_model)
    reference_loss = loss_function(reference_model(inputs), targets)
    reference_loss.backward()
    reference_gradients = [
        parameter.grad.detach().clone() for parameter in reference_model.parameters()
    ]
    with torch.no_grad():
        for parameter, gradient in zip(
            reference_model.parameters(), reference_gradients, strict=True
        ):
            parameter.add_(gradient, alpha=-learning_rate)

    rank_gradients: list[list[torch.Tensor]] = []
    rank_losses: list[float] = []
    for rank in range(world_size):
        rank_model = copy.deepcopy(initial_model)
        start = rank * samples_per_rank
        stop = start + samples_per_rank
        rank_loss = loss_function(rank_model(inputs[start:stop]), targets[start:stop])
        rank_loss.backward()
        rank_losses.append(rank_loss.item())
        rank_gradients.append(
            [parameter.grad.detach().clone() for parameter in rank_model.parameters()]
        )
        if verbose:
            gradient_norm = torch.linalg.vector_norm(
                _flatten(rank_gradients[-1])
            ).item()
            print(
                f"rank {rank:02d} | samples [{start:02d}:{stop:02d}] | "
                f"loss={rank_loss.item():9.6f} | grad_norm={gradient_norm:9.6f}"
            )

    averaged_gradients = [
        torch.stack([rank[layer] for rank in rank_gradients]).mean(dim=0)
        for layer in range(len(rank_gradients[0]))
    ]
    virtual_model = copy.deepcopy(initial_model)
    with torch.no_grad():
        for parameter, gradient in zip(
            virtual_model.parameters(), averaged_gradients, strict=True
        ):
            parameter.add_(gradient, alpha=-learning_rate)

    gradient_difference = max(
        (reference - virtual).abs().max().item()
        for reference, virtual in zip(
            reference_gradients, averaged_gradients, strict=True
        )
    )
    parameter_difference = max(
        (reference - virtual).abs().max().item()
        for reference, virtual in zip(
            reference_model.parameters(), virtual_model.parameters(), strict=True
        )
    )
    return VirtualStepResult(
        world_size=world_size,
        samples_per_rank=samples_per_rank,
        reference_loss=reference_loss.item(),
        mean_rank_loss=sum(rank_losses) / len(rank_losses),
        max_gradient_difference=gradient_difference,
        max_parameter_difference_after_step=parameter_difference,
    )


def demo_state_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Create illustrative mixed-precision Adam state for ownership checks."""

    parameters = _flatten([parameter for parameter in model.parameters()]).to(
        torch.float16
    )
    gradients = torch.linspace(-1.0, 1.0, parameters.numel(), dtype=torch.float16)
    master_parameters = parameters.float()
    first_moment = torch.zeros_like(master_parameters)
    second_moment = torch.ones_like(master_parameters)
    return {
        "parameters_fp16": parameters,
        "gradients_fp16": gradients,
        "master_parameters_fp32": master_parameters,
        "adam_first_moment_fp32": first_moment,
        "adam_second_moment_fp32": second_moment,
    }


def sharded_state_names(stage: int) -> set[str]:
    """Return the state categories partitioned at a given ZeRO stage."""

    if stage not in range(4):
        raise ValueError("stage must be 0, 1, 2, or 3")
    names: set[str] = set()
    if stage >= 1:
        names.update(
            {
                "master_parameters_fp32",
                "adam_first_moment_fp32",
                "adam_second_moment_fp32",
            }
        )
    if stage >= 2:
        names.add("gradients_fp16")
    if stage >= 3:
        names.add("parameters_fp16")
    return names


def verify_shard_reconstruction(
    model: nn.Module, stage: int, world_size: int = 32
) -> dict[str, bool]:
    """Partition stage-owned states and verify exact all-gather reconstruction."""

    states = demo_state_tensors(model)
    sharded = sharded_state_names(stage)
    checks: dict[str, bool] = {}
    for name, tensor in states.items():
        if name in sharded:
            reconstructed = reconstruct_tensor(partition_tensor(tensor, world_size))
            checks[name] = torch.equal(reconstructed, tensor)
        else:
            checks[name] = True
    return checks
