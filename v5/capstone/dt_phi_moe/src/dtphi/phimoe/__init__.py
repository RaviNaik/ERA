"""Phi-mini-MoE model code (Microsoft; see the licence headers in the files), copied from the checkpoint folder with three small patches.

`modeling_phimoe.py` is `modeling_slimmoe.py` of microsoft/Phi-mini-MoE-instruct with: (1) the two unused imports of einops and flash_attn removed, (2) `GenerationMixin` added as an explicit base of
PhiMoEForCausalLM, (3) a module switch `DETERMINISTIC_MOE` (default False = original behaviour) that makes the MoE blocks route as in eval mode while training. `configuration_slimmoe.py` is unchanged. Loading from here needs no trust_remote_code and
no flash_attn."""
from .configuration_slimmoe import PhiMoEConfig
from .modeling_phimoe import PhiMoEForCausalLM

__all__ = ["PhiMoEConfig", "PhiMoEForCausalLM"]
