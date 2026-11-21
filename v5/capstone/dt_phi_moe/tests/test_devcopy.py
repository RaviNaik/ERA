import torch
from collections import namedtuple

from dtphi import devcopy


def test_via_cpu_keeps_structure_and_values_on_the_same_device():
    P = namedtuple("P", "a b")
    obj = {"x": torch.arange(3), "y": [torch.ones(2), (torch.zeros(1), 5)], "z": P(torch.ones(1), "s"), "n": None}
    out = devcopy.via_cpu(obj, torch.device("cpu"))
    assert out["x"].tolist() == [0, 1, 2] and out["y"][1][1] == 5 and isinstance(out["z"], P) and out["n"] is None


def test_wrapped_send_to_device_skips_keys_and_delegates():
    import accelerate.hooks as hooks
    restore = devcopy.install()
    try:
        assert hooks.send_to_device._dtphi_via_cpu
        out = hooks.send_to_device({"a": torch.ones(2), "b": torch.zeros(2)}, "cpu", skip_keys=["b"])
        assert out["a"].sum() == 2 and out["b"].sum() == 0
        assert devcopy.install()() is None   # second install is a no-op
    finally:
        restore()
    assert not getattr(hooks.send_to_device, "_dtphi_via_cpu", False)
