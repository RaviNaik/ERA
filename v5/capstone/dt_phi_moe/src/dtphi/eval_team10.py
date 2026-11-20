"""Team 10 Indic and safety row sets: generate with the Phi chat template, score with the team scorers.

The tool-call set is not scored: its replies are chat-format tool calls with Team 10 special tokens, which the Phi
tokenizer does not have. The scorers come from teams/10-post-training/sft/tests/scorers.py (copied unchanged) with a
plain-text stand-in for chat-format/render.py (see team10/chat-format/render.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from .eval_gsm8k import stop_ids
from .modeling import input_device
from . import data

_SCORERS = Path(__file__).resolve().parent / "team10" / "sft" / "tests"


def load_scorers():
    sys.path.insert(0, str(_SCORERS))
    import scorers  # noqa: E402
    return scorers


@torch.inference_mode()
def generate_replies(model, tok, rows: list[dict], batch_size: int, max_new_tokens: int) -> dict[str, str]:
    model.eval()
    stops = set(stop_ids(model, tok))
    dev = input_device(model)
    prompts = [list(tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=True)) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -len(prompts[i]))
    out: dict[str, str] = {}
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids = data.pad_left([prompts[i] for i in idx], tok.pad_token_id).to(dev)
        attn = data.pad_left([[1] * len(prompts[i]) for i in idx], 0).to(dev)
        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
        for k, i in enumerate(idx):
            resp = gen[k, ids.shape[1]:].tolist()
            cut = next((j for j, t in enumerate(resp) if t in stops), None)
            resp = resp if cut is None else resp[:cut]
            out[rows[i]["id"]] = tok.decode(resp, skip_special_tokens=True)
    return out


def run_team10(model, tok, sets_dir: str, batch_size: int, max_new_tokens: int) -> dict:
    sc = load_scorers()
    result = {}
    for name, fn in (("indic_v1", sc.score_indic_set), ("safety_v1", sc.score_safety_set)):
        rows = sc.load_jsonl(Path(sets_dir) / f"{name}.jsonl")
        gens = generate_replies(model, tok, rows, batch_size, max_new_tokens)
        rep = fn(rows, gens)
        rep["generations"] = gens
        result[name] = rep
    return result
