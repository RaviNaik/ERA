import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODEL_DIR = Path(os.environ.get("DT_MODEL_DIR", "/mnt/Dev/dt_transfer/phi-mini-moe-instruct"))
DATA_DIR = Path(os.environ.get("DT_DATA_DIR", "/mnt/Dev/dt_transfer/data/gsm8k"))
TEAM10_DIR = Path(os.environ.get("DT_TEAM10_DIR", "/mnt/Dev/dt_transfer/data/team10_sft_tests"))
TOK_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
             "generation_config.json"]


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    """A random 2-layer, 4-expert Phi-MoE with the real tokenizer, saved in fp32 (no jitter noise)."""
    if not (MODEL_DIR / "config.json").exists():
        pytest.skip("real model folder not found (set DT_MODEL_DIR)")
    from dtphi.phimoe import PhiMoEConfig, PhiMoEForCausalLM
    d = tmp_path_factory.mktemp("tiny")
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    cfg.update(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
               intermediate_size=32, num_local_experts=4, input_jitter_noise=0.0, router_jitter_noise=0.0,
               torch_dtype="float32")
    cfg.pop("auto_map", None)
    (d / "config.json").write_text(json.dumps(cfg))
    for f in TOK_FILES:
        shutil.copy(MODEL_DIR / f, d / f)
    torch.manual_seed(0)
    conf = PhiMoEConfig.from_pretrained(d)
    conf._attn_implementation = "sdpa"
    m = PhiMoEForCausalLM(conf).float()
    for n, p in m.named_parameters():  # larger than the default init so Wanda scores and logits are not degenerate
        if p.dim() == 2:
            torch.nn.init.normal_(p, 0.0, 0.08)
    m.save_pretrained(d, safe_serialization=True)
    return d


@pytest.fixture(scope="session")
def data_dir():
    if not (DATA_DIR / "main").exists():
        pytest.skip("GSM8K folder not found (set DT_DATA_DIR)")
    return DATA_DIR
