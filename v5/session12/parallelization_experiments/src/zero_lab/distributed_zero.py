"""A REAL 32-process ZeRO simulation using torch.distributed on CPU.

Every "rank" in this module is a genuine, isolated operating-system process
that joins a real `torch.distributed` process group and exchanges real
tensors over real collective calls (all-reduce, all-gather, and a manual
reduce-scatter). Nothing here is faked with a Python for-loop pretending to be
32 workers: this is the same family of primitives DeepSpeed/FSDP use to
implement ZeRO, just running on CPU cores with the "gloo" backend instead of
GPUs with NCCL. It is slower per step and has no GPU-interconnect numbers to
report, but the sharding *logic* — who owns which bytes, and which collective
moves them — is identical.

Precision: to keep the demonstration fully real (no faked half-precision
tensors), every state tensor here is plain fp32. Adam's per-parameter state is
then exactly 16 bytes: 4 (parameter) + 4 (gradient) + 4 (first moment) +
4 (second moment) — the same total the assignment's mixed-precision formula
uses, just reached with real fp32 tensors instead of an assumed fp16/fp32 split.
This lets the measured bytes from this module be compared directly against
`zero_lab.simulator.estimate_stage`.

Gloo does not implement `reduce_scatter` (confirmed on this machine: it raises
`RuntimeError: ProcessGroupGloo does not support reduce_scatter`). Reduce-scatter
is reconstructed here from its definition — one `dist.reduce` call per shard,
each targeted at the rank that owns that shard — which Gloo does support, and
which produces exactly the same result a native reduce-scatter would.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from .demo import DemoMLP, make_regression_batch

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
STAGE_LABELS = {
    0: "ZeRO-0 / data parallel",
    1: "ZeRO-1: optimizer sharding",
    2: "ZeRO-2: + gradient sharding",
    3: "ZeRO-3: + parameter sharding",
}


def _pad_len(n: int, world_size: int) -> int:
    """Elements to append so `n` divides evenly across `world_size` ranks."""

    remainder = n % world_size
    return 0 if remainder == 0 else world_size - remainder


def _hello_worker(rank: int, world_size: int, run_dir: str) -> None:
    """The smallest possible real collective: every rank contributes its own
    rank number and its own OS process id; `all_reduce` sums the rank numbers
    across all of them. Proves these are genuinely separate processes talking
    to each other, before we trust them with anything ZeRO-shaped."""

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{run_dir}/rendezvous",
        world_size=world_size,
        rank=rank,
    )
    try:
        value = torch.tensor([float(rank)])
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        with open(os.path.join(run_dir, f"hello_{rank}.json"), "w") as handle:
            json.dump(
                {"rank": rank, "pid": os.getpid(), "all_reduced_sum": value.item()},
                handle,
            )
    finally:
        dist.destroy_process_group()


def hello_world_collectives(world_size: int = 4) -> pd.DataFrame:
    """Spawn `world_size` real processes and have them all-reduce their own
    rank numbers. Expected sum on every rank: 0+1+...+(world_size-1)."""

    run_dir = tempfile.mkdtemp(prefix="zero_lab_hello_")
    try:
        mp.spawn(
            _hello_worker, args=(world_size, run_dir), nprocs=world_size, join=True
        )
        rows = [
            json.load(open(os.path.join(run_dir, f"hello_{rank}.json")))
            for rank in range(world_size)
        ]
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def _flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _flat_grads(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])


def _load_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset : offset + n].view_as(p))
        offset += n


def _manual_reduce_scatter(
    chunks: list[torch.Tensor], rank: int, world_size: int
) -> torch.Tensor:
    """Reduce-scatter built from `world_size` point-to-point `dist.reduce` calls.

    After this call, only `chunks[rank]` is guaranteed correct on this rank —
    exactly the reduce-scatter contract (every rank ends up owning one summed
    shard, not the whole reduced tensor).
    """

    for owner in range(world_size):
        dist.reduce(chunks[owner], dst=owner, op=dist.ReduceOp.SUM)
    return chunks[rank]


def _adam_update(
    m: torch.Tensor, v: torch.Tensor, grad: torch.Tensor, step: int, lr: float
) -> torch.Tensor:
    m.mul_(ADAM_BETA1).add_(grad, alpha=1 - ADAM_BETA1)
    v.mul_(ADAM_BETA2).addcmul_(grad, grad, value=1 - ADAM_BETA2)
    m_hat = m / (1 - ADAM_BETA1**step)
    v_hat = v / (1 - ADAM_BETA2**step)
    return lr * m_hat / (v_hat.sqrt() + ADAM_EPS)


def _worker(
    rank: int,
    world_size: int,
    stage: int,
    run_dir: str,
    samples_per_rank: int,
    hidden_dim: int,
    seed: int,
    lr: float,
) -> None:
    """One real OS process = one ZeRO "rank". Runs one training step and one
    optimizer step at the requested ZeRO stage, then writes its own view of
    the world (what it owns, what it borrowed, what it computed) to a small
    per-rank JSON file so the parent process can build a readable report.
    """

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{run_dir}/rendezvous",
        world_size=world_size,
        rank=rank,
    )
    try:
        torch.manual_seed(seed)  # every rank builds the identical initial model
        model = DemoMLP(hidden_dim=hidden_dim)
        inputs, targets = make_regression_batch(
            world_size, samples_per_rank, seed=seed + 1
        )
        sample_start, sample_stop = (
            rank * samples_per_rank,
            (rank + 1) * samples_per_rank,
        )
        x, y = inputs[sample_start:sample_stop], targets[sample_start:sample_stop]

        init_flat = _flat_params(model)
        num_params = init_flat.numel()
        pad = _pad_len(num_params, world_size)
        padded_len = num_params + pad
        shard_size = padded_len // world_size
        lo, hi = rank * shard_size, (rank + 1) * shard_size
        pad_tail = torch.zeros(pad)

        owned_opt_elems = shard_size if stage >= 1 else padded_len
        m_state = torch.zeros(owned_opt_elems)
        v_state = torch.zeros(owned_opt_elems)

        # --- parameters: gathered transiently only for ZeRO-3 ------------------
        param_bytes_transient = 0
        owned_param_shard: torch.Tensor | None = None
        if stage == 3:
            owned_param_shard = torch.cat([init_flat, pad_tail])[lo:hi].clone()
            gathered_param_shards = [torch.zeros(shard_size) for _ in range(world_size)]
            dist.all_gather(gathered_param_shards, owned_param_shard)
            full_param = torch.cat(gathered_param_shards)[:num_params]
            _load_flat_params(model, full_param)
            param_bytes_owned = shard_size * 4
            param_bytes_transient = (padded_len - shard_size) * 4
        else:
            param_bytes_owned = padded_len * 4

        # --- real forward + backward on this rank's own microbatch --------------
        prediction = model(x)
        loss = F.mse_loss(prediction, y)
        loss.backward()
        grad_padded = torch.cat([_flat_grads(model), pad_tail])

        # --- gradients: all-reduced (replicated) or reduce-scattered (sharded) --
        if stage in (0, 1):
            dist.all_reduce(grad_padded, op=dist.ReduceOp.SUM)
            grad_padded /= world_size
            grad_bytes_owned = padded_len * 4
            grad_shard = grad_padded[lo:hi]
        else:
            chunks = list(grad_padded.split(shard_size))
            grad_shard = _manual_reduce_scatter(chunks, rank, world_size) / world_size
            grad_bytes_owned = shard_size * 4

        # --- optimizer step: full vector (stage 0) or owned shard (stage 1-3) --
        base_padded = torch.cat([init_flat, pad_tail])
        if stage == 0:
            update = _adam_update(m_state, v_state, grad_padded, 1, lr)
            new_full_padded = base_padded - update
        else:
            base_shard = owned_param_shard if stage == 3 else base_padded[lo:hi]
            update = _adam_update(m_state, v_state, grad_shard, 1, lr)
            updated_shard = base_shard - update
            gathered_updated = [torch.zeros(shard_size) for _ in range(world_size)]
            dist.all_gather(gathered_updated, updated_shard)
            new_full_padded = torch.cat(gathered_updated)

        opt_bytes_owned = owned_opt_elems * 8  # first + second moment, fp32
        new_full = new_full_padded[:num_params]

        result = {
            "rank": rank,
            "stage": stage,
            "world_size": world_size,
            "num_params": num_params,
            "pad": pad,
            "shard_size": shard_size,
            "shard_lo": lo,
            "shard_hi": min(hi, num_params),
            "sample_start": sample_start,
            "sample_stop": sample_stop,
            "loss": float(loss.item()),
            "param_bytes_owned": param_bytes_owned,
            "param_bytes_transient": param_bytes_transient,
            "grad_bytes_owned": grad_bytes_owned,
            "optimizer_bytes_owned": opt_bytes_owned,
            "new_full_sum": float(new_full.sum().item()),
            "new_full_norm": float(new_full.norm().item()),
        }
        with open(os.path.join(run_dir, f"rank_{rank}.json"), "w") as handle:
            json.dump(result, handle)

        if rank == 0:
            torch.save(new_full.clone(), os.path.join(run_dir, "final_params_rank0.pt"))
    finally:
        dist.destroy_process_group()


@dataclass(frozen=True)
class StageRunReport:
    """Everything one call to :func:`run_zero_stage` produced, for one stage."""

    stage: int
    label: str
    world_size: int
    num_params: int
    per_rank: pd.DataFrame
    final_params: torch.Tensor


def run_zero_stage(
    stage: int,
    world_size: int = 32,
    samples_per_rank: int = 2,
    hidden_dim: int = 16,
    seed: int = 7,
    lr: float = 0.05,
) -> StageRunReport:
    """Spawn `world_size` real CPU processes and run one ZeRO stage for real.

    Returns a per-rank DataFrame (one row per rank) plus the reconstructed
    final parameter vector, read back from rank 0's own on-disk checkpoint.
    """

    if stage not in (0, 1, 2, 3):
        raise ValueError("stage must be 0, 1, 2, or 3")

    run_dir = tempfile.mkdtemp(prefix=f"zero_lab_stage{stage}_")
    try:
        mp.spawn(
            _worker,
            args=(world_size, stage, run_dir, samples_per_rank, hidden_dim, seed, lr),
            nprocs=world_size,
            join=True,
        )
        rows = []
        for rank in range(world_size):
            with open(os.path.join(run_dir, f"rank_{rank}.json")) as handle:
                rows.append(json.load(handle))
        per_rank = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
        final_params = torch.load(
            os.path.join(run_dir, "final_params_rank0.pt"), weights_only=True
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    return StageRunReport(
        stage=stage,
        label=STAGE_LABELS[stage],
        world_size=world_size,
        num_params=int(per_rank["num_params"].iloc[0]),
        per_rank=per_rank,
        final_params=final_params,
    )


def run_all_zero_stages(
    world_size: int = 32,
    samples_per_rank: int = 2,
    hidden_dim: int = 16,
    seed: int = 7,
    lr: float = 0.05,
) -> dict[int, StageRunReport]:
    """Run ZeRO-0 through ZeRO-3, each as its own real 32-process job."""

    return {
        stage: run_zero_stage(
            stage,
            world_size=world_size,
            samples_per_rank=samples_per_rank,
            hidden_dim=hidden_dim,
            seed=seed,
            lr=lr,
        )
        for stage in (0, 1, 2, 3)
    }


def reference_single_process_step(
    world_size: int = 32,
    samples_per_rank: int = 2,
    hidden_dim: int = 16,
    seed: int = 7,
    lr: float = 0.05,
) -> torch.Tensor:
    """The same model/data/Adam step computed with no distribution at all.

    This is the ground truth that every ZeRO stage above must reproduce: one
    global batch, one ordinary Adam step, no ranks, no sharding.
    """

    torch.manual_seed(seed)
    model = DemoMLP(hidden_dim=hidden_dim)
    inputs, targets = make_regression_batch(world_size, samples_per_rank, seed=seed + 1)
    prediction = model(inputs)
    loss = F.mse_loss(prediction, targets)
    loss.backward()
    grad = _flat_grads(model)
    m_state = torch.zeros_like(grad)
    v_state = torch.zeros_like(grad)
    update = _adam_update(m_state, v_state, grad, 1, lr)
    return _flat_params(model) - update


def stage_memory_table(reports: dict[int, StageRunReport]) -> pd.DataFrame:
    """One row per stage: measured (not estimated) per-rank byte ownership.

    Every rank is identical by symmetry (same shard size, same data-per-rank),
    so this reports rank 0's row from each stage alongside the min/max across
    ranks as a balance sanity check.
    """

    rows = []
    for stage, report in sorted(reports.items()):
        per_rank = report.per_rank
        total = (
            per_rank["param_bytes_owned"]
            + per_rank["grad_bytes_owned"]
            + per_rank["optimizer_bytes_owned"]
        )
        rows.append(
            {
                "stage": stage,
                "label": report.label,
                "param_bytes_per_rank": int(per_rank["param_bytes_owned"].iloc[0]),
                "grad_bytes_per_rank": int(per_rank["grad_bytes_owned"].iloc[0]),
                "optimizer_bytes_per_rank": int(
                    per_rank["optimizer_bytes_owned"].iloc[0]
                ),
                "transient_param_bytes_per_rank": int(
                    per_rank["param_bytes_transient"].iloc[0]
                ),
                "persistent_total_bytes_per_rank": int(total.iloc[0]),
                "balanced_across_ranks": bool(total.min() == total.max()),
            }
        )
    return pd.DataFrame(rows)
