"""Analytical model-state accounting for data parallelism and ZeRO.

The simulator models persistent mixed-precision Adam state and communication.
It deliberately reports activations separately because activation memory is not
sharded by ZeRO and depends on batch size, sequence length, and checkpointing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum

import pandas as pd


class Stage(IntEnum):
    """ZeRO stage, with stage 0 representing ordinary data parallelism."""

    ZERO_0 = 0
    ZERO_1 = 1
    ZERO_2 = 2
    ZERO_3 = 3


@dataclass(frozen=True)
class ModelProfile:
    """Inputs needed for an interpretable per-rank resource estimate."""

    parameters: int
    largest_layer_parameters: int
    world_size: int = 32
    parameter_bytes: int = 2
    gradient_bytes: int = 2
    master_parameter_bytes: int = 4
    adam_moment_bytes: int = 8
    activation_bytes_per_rank: int = 0

    def __post_init__(self) -> None:
        if self.parameters <= 0:
            raise ValueError("parameters must be positive")
        if not 0 < self.largest_layer_parameters <= self.parameters:
            raise ValueError("largest_layer_parameters must be in [1, parameters]")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")

    @property
    def optimizer_bytes(self) -> int:
        return self.master_parameter_bytes + self.adam_moment_bytes


@dataclass(frozen=True)
class StageEstimate:
    stage: int
    label: str
    parameter_gib: float
    gradient_gib: float
    optimizer_gib: float
    persistent_model_state_gib: float
    activation_gib: float
    transient_gather_gib: float
    estimated_peak_gib: float
    communication_gib_per_rank_step: float
    training_flops_relative: float
    main_collectives: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


GIB = 1024**3


def ring_phase_bytes(payload_bytes: float, world_size: int) -> float:
    """Per-rank traffic for one ring reduce-scatter or all-gather phase."""

    return payload_bytes * (world_size - 1) / world_size


def estimate_stage(profile: ModelProfile, stage: Stage | int) -> StageEstimate:
    """Estimate persistent state, peak memory, and communication for one rank.

    Communication is an algorithmic volume estimate for ring collectives, not a
    wall-clock prediction. ZeRO-3 assumes parameters are gathered once for the
    forward and once for backward, then released layer by layer.
    """

    stage = Stage(stage)
    world_size = profile.world_size
    parameters = profile.parameters
    parameter_bytes = parameters * profile.parameter_bytes
    gradient_bytes = parameters * profile.gradient_bytes
    optimizer_bytes = parameters * profile.optimizer_bytes

    parameter_divisor = world_size if stage >= Stage.ZERO_3 else 1
    gradient_divisor = world_size if stage >= Stage.ZERO_2 else 1
    optimizer_divisor = world_size if stage >= Stage.ZERO_1 else 1

    local_parameters = parameter_bytes / parameter_divisor
    local_gradients = gradient_bytes / gradient_divisor
    local_optimizer = optimizer_bytes / optimizer_divisor
    persistent = local_parameters + local_gradients + local_optimizer

    transient_gather = 0.0
    if stage == Stage.ZERO_3:
        transient_gather = (
            profile.largest_layer_parameters
            * profile.parameter_bytes
            * (world_size - 1)
            / world_size
        )

    gradient_phase = ring_phase_bytes(gradient_bytes, world_size)
    parameter_phase = ring_phase_bytes(parameter_bytes, world_size)
    if stage == Stage.ZERO_0:
        # Gradients replicated -> full all-reduce (2 phases). Every rank applies
        # the identical update to its identical full optimizer state, so no
        # parameter traffic is needed afterwards.
        communication = 2.0 * gradient_phase
        collectives = "gradient all-reduce"
    elif stage == Stage.ZERO_1:
        # Gradients are still replicated (full all-reduce, 2 phases), but the
        # optimizer state is sharded: each rank can only step the shard it owns,
        # so the freshly updated shards must be all-gathered back out (+1 phase).
        communication = 2.0 * gradient_phase + parameter_phase
        collectives = "gradient all-reduce + updated-parameter all-gather"
    elif stage == Stage.ZERO_2:
        # Gradients are now sharded too: reduce-scatter replaces all-reduce
        # (1 phase instead of 2), which exactly pays for the extra all-gather
        # ZeRO-1 introduced -> total volume returns to the ZeRO-0 baseline.
        communication = gradient_phase + parameter_phase
        collectives = "gradient reduce-scatter + updated-parameter all-gather"
    else:
        # Parameters are sharded at rest, so the full vector must be
        # reconstructed twice per step: once for the forward pass, and again
        # for backward (freed after forward to keep the 1/N memory footprint,
        # so it is not still resident when backward needs it) -- on top of
        # the ZeRO-2 gradient traffic.
        communication = gradient_phase + 2.0 * parameter_phase
        collectives = (
            "parameter all-gather (forward) + parameter all-gather (backward) "
            "+ gradient reduce-scatter"
        )

    labels = {
        Stage.ZERO_0: "ZeRO-0 / data parallel",
        Stage.ZERO_1: "ZeRO-1: optimizer sharding",
        Stage.ZERO_2: "ZeRO-2: + gradient sharding",
        Stage.ZERO_3: "ZeRO-3: + parameter sharding",
    }
    activation_gib = profile.activation_bytes_per_rank / GIB
    persistent_gib = persistent / GIB
    transient_gib = transient_gather / GIB
    return StageEstimate(
        stage=int(stage),
        label=labels[stage],
        parameter_gib=local_parameters / GIB,
        gradient_gib=local_gradients / GIB,
        optimizer_gib=local_optimizer / GIB,
        persistent_model_state_gib=persistent_gib,
        activation_gib=activation_gib,
        transient_gather_gib=transient_gib,
        estimated_peak_gib=persistent_gib + activation_gib + transient_gib,
        communication_gib_per_rank_step=communication / GIB,
        training_flops_relative=1.0,
        main_collectives=collectives,
    )


def estimate_all_stages(profile: ModelProfile) -> pd.DataFrame:
    """Return one readable row per ZeRO stage."""

    return pd.DataFrame(estimate_stage(profile, stage).as_dict() for stage in Stage)
