import torch
from collections import namedtuple

from dtphi import devcopy


def test_via_cpu_returns_non_gpu_inputs_unchanged():
    x = torch.arange(3)
    assert devcopy.via_cpu(x, torch.device("cpu")) is x


def test_accelerate_send_to_device_keeps_model_output_types_with_the_patch():
    """The bug that cost a 30-minute calibration: the model output turned into a plain dict (no .loss)."""
    import accelerate.hooks as hooks
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast
    restore = devcopy.install_global()
    try:
        out = MoeCausalLMOutputWithPast(loss=torch.tensor(1.0), logits=torch.zeros(1, 2, 3), past_key_values=None)
        moved = hooks.send_to_device(out, "cpu", skip_keys=["past_key_values"])
        assert type(moved) is MoeCausalLMOutputWithPast and float(moved.loss) == 1.0
    finally:
        restore()


def test_global_patch_leaves_ordinary_copies_alone_and_is_idempotent():
    restore = devcopy.install_global()
    try:
        assert devcopy.install_global()() is None
        x = torch.arange(4)
        assert x.to(torch.float32).dtype == torch.float32 and x.to("cpu").equal(x) and x.to(x).equal(x)
        assert x.to(device="cpu", dtype=torch.float64).dtype == torch.float64
        y = torch.zeros(4, dtype=torch.long); y.copy_(x)
        assert y.equal(x)
        assert devcopy._target_device((torch.float32,), {}) is None
        assert devcopy._target_device(("cuda:1",), {}) == torch.device("cuda:1")
        assert devcopy._target_device((), {"device": torch.device("cpu")}) == torch.device("cpu")
        assert not devcopy._cross_gpu(x, torch.device("cuda:1"))   # a CPU tensor is never a GPU-to-GPU copy
    finally:
        restore()
    assert not devcopy._ORIG
