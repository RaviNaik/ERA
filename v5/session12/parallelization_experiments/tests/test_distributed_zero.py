"""Tests for the REAL torch.distributed (gloo, CPU-process) ZeRO implementation.

These spawn genuine OS processes, so a small world size keeps the suite fast
while still exercising every collective (all-reduce, all-gather, and the
manual reduce-scatter) exactly as the 32-rank notebook run does.
"""

from __future__ import annotations

import pytest
import torch

from zero_lab.distributed_zero import (
    reference_single_process_step,
    run_all_zero_stages,
    stage_memory_table,
)
from zero_lab.simulator import GIB, ModelProfile, estimate_stage

WORLD_SIZE = 4


@pytest.fixture(scope="module")
def reports():
    return run_all_zero_stages(
        world_size=WORLD_SIZE, samples_per_rank=2, hidden_dim=8, seed=3
    )


def test_every_stage_matches_the_single_process_reference(reports) -> None:
    reference = reference_single_process_step(
        world_size=WORLD_SIZE, samples_per_rank=2, hidden_dim=8, seed=3
    )
    for stage, report in reports.items():
        diff = (report.final_params - reference).abs().max().item()
        assert diff < 1e-5, f"stage {stage} diverged from the reference by {diff:.2e}"


def test_every_rank_ends_with_identical_replicated_parameters(reports) -> None:
    for report in reports.values():
        sums = report.per_rank["new_full_sum"]
        norms = report.per_rank["new_full_norm"]
        assert sums.max() - sums.min() == pytest.approx(0.0, abs=1e-4)
        assert norms.max() - norms.min() == pytest.approx(0.0, abs=1e-4)


def test_data_is_partitioned_without_overlap_or_gaps(reports) -> None:
    per_rank = reports[0].per_rank.sort_values("rank")
    starts = per_rank["sample_start"].tolist()
    stops = per_rank["sample_stop"].tolist()
    assert starts[0] == 0
    assert stops[-1] == WORLD_SIZE * 2
    assert all(stops[i] == starts[i + 1] for i in range(len(starts) - 1))


def test_measured_bytes_match_the_analytical_formula(reports) -> None:
    """The real per-rank byte counts should equal `estimate_stage` exactly,
    once the profile uses the same real fp32 byte widths and the same padded
    length the real run actually operates on (parameter counts are padded up
    to a multiple of `world_size` so collectives split evenly; this is
    standard practice, not an approximation error)."""

    num_params = reports[0].num_params
    pad = (-num_params) % WORLD_SIZE
    padded_len = num_params + pad
    profile = ModelProfile(
        parameters=padded_len,
        largest_layer_parameters=padded_len,  # single flat shard for this tiny model
        world_size=WORLD_SIZE,
        parameter_bytes=4,
        gradient_bytes=4,
        master_parameter_bytes=0,
        adam_moment_bytes=8,
    )
    for stage, report in reports.items():
        estimate = estimate_stage(profile, stage)
        measured_persistent = (
            report.per_rank["param_bytes_owned"].iloc[0]
            + report.per_rank["grad_bytes_owned"].iloc[0]
            + report.per_rank["optimizer_bytes_owned"].iloc[0]
        )
        assert measured_persistent == pytest.approx(
            estimate.persistent_model_state_gib * GIB, abs=1
        )


def test_sharding_progressively_shrinks_owned_state(reports) -> None:
    table = stage_memory_table(reports)
    totals = table.set_index("stage")["persistent_total_bytes_per_rank"]
    assert totals[0] > totals[1] > totals[2] > totals[3]
    assert table["balanced_across_ranks"].all()


def test_zero3_is_the_only_stage_with_a_transient_parameter_gather(reports) -> None:
    table = stage_memory_table(reports)
    transient = table.set_index("stage")["transient_param_bytes_per_rank"]
    assert transient[0] == transient[1] == transient[2] == 0
    assert transient[3] > 0
