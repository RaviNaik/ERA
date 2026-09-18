from __future__ import annotations

import pytest
import torch

from zero_lab.demo import DemoMLP, partition_tensor, reconstruct_tensor
from zero_lab.demo import verify_shard_reconstruction, virtual_data_parallel_step
from zero_lab.simulator import GIB, ModelProfile, Stage, estimate_stage


def test_expected_mixed_precision_adam_bytes_per_parameter_at_32_ranks() -> None:
    profile = ModelProfile(
        parameters=1_000_000, largest_layer_parameters=100_000, world_size=32
    )
    expected = {
        Stage.ZERO_0: 16.0,
        Stage.ZERO_1: 4.375,
        Stage.ZERO_2: 2.4375,
        Stage.ZERO_3: 0.5,
    }
    for stage, expected_bytes in expected.items():
        estimate = estimate_stage(profile, stage)
        actual_bytes = estimate.persistent_model_state_gib * GIB / profile.parameters
        assert actual_bytes == pytest.approx(expected_bytes)


def test_zero3_peak_includes_largest_layer_gather_and_activations() -> None:
    profile = ModelProfile(1_000, 100, world_size=10, activation_bytes_per_rank=1234)
    estimate = estimate_stage(profile, Stage.ZERO_3)
    assert estimate.estimated_peak_gib * GIB == pytest.approx(1600 + 180 + 1234)


def test_zero2_matches_baseline_but_zero1_and_zero3_cost_more() -> None:
    """With equal gradient/parameter byte widths: ZeRO-2 == baseline, while
    ZeRO-1 (extra parameter all-gather) and ZeRO-3 (extra parameter all-gather
    before compute) both cost 1.5x baseline, for different reasons."""

    profile = ModelProfile(
        1_000_000, 100_000, world_size=32, parameter_bytes=4, gradient_bytes=4
    )
    baseline = estimate_stage(profile, Stage.ZERO_0).communication_gib_per_rank_step
    assert estimate_stage(
        profile, Stage.ZERO_1
    ).communication_gib_per_rank_step == pytest.approx(1.5 * baseline)
    assert estimate_stage(
        profile, Stage.ZERO_2
    ).communication_gib_per_rank_step == pytest.approx(baseline)
    assert estimate_stage(
        profile, Stage.ZERO_3
    ).communication_gib_per_rank_step == pytest.approx(1.5 * baseline)


def test_partition_is_balanced_and_reconstructs_exactly() -> None:
    tensor = torch.arange(101)
    shards = partition_tensor(tensor, 32)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    assert torch.equal(reconstruct_tensor(shards), tensor)


@pytest.mark.parametrize("stage", range(4))
def test_every_stage_reconstructs_demo_state(stage: int) -> None:
    assert all(verify_shard_reconstruction(DemoMLP(), stage, 32).values())


def test_virtual_rank_update_matches_global_batch() -> None:
    result = virtual_data_parallel_step(verbose=False)
    assert result.reference_loss == pytest.approx(result.mean_rank_loss, abs=2e-6)
    assert result.max_gradient_difference < 2e-6
    assert result.max_parameter_difference_after_step < 2e-7
