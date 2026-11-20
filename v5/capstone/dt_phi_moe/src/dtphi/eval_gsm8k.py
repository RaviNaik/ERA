"""GSM8K test accuracy (greedy generation) and the forward KL of the official retention script."""
from __future__ import annotations

import re
import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from . import data, official  # noqa: F401
from .modeling import input_device

# "#### 72" followed by a character that cannot continue the number: the answer is complete
ANSWER_DONE = re.compile(r"####\s*-?\d[\d,]*(\.\d+)?[^\d,.]")


def stop_ids(model, tok) -> list[int]:
    e = model.generation_config.eos_token_id
    if e is None:
        e = tok.eos_token_id
    return list(e) if isinstance(e, (list, tuple)) else [e]


class AnswerDone(StoppingCriteria):
    """Per sequence: stop once the first '#### <number>' of the reply is complete."""

    def __init__(self, tok, prompt_len: int):
        self.tok, self.prompt_len = tok, prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tok.batch_decode(input_ids[:, self.prompt_len:], skip_special_tokens=True)
        return torch.tensor([ANSWER_DONE.search(t) is not None for t in texts], device=input_ids.device)


def trim_to_answer(tok, ids: list[int]) -> list[int]:
    """The shortest prefix of the reply whose text already contains the complete first answer; the whole reply if
    there is none."""
    if not ANSWER_DONE.search(tok.decode(ids, skip_special_tokens=True)):
        return ids
    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if ANSWER_DONE.search(tok.decode(ids[:mid], skip_special_tokens=True)):
            hi = mid
        else:
            lo = mid + 1
    return ids[:lo]


@torch.inference_mode()
def generate_gsm8k(model, tok, rows: list[dict], prompt_fn, batch_size: int, max_new_tokens: int) -> list[dict]:
    """Greedy replies. Record: idx, prompt_ids, response_ids (before the first stop token, cut after the first
    complete '#### n'), text, correct. Prompts are processed longest first in left-padded batches."""
    model.eval()
    stops = set(stop_ids(model, tok))
    prompts = [prompt_fn(tok, r["question"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -len(prompts[i]))
    dev = input_device(model)
    out: list[dict | None] = [None] * len(rows)
    t0 = time.time()
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids = data.pad_left([prompts[i] for i in idx], tok.pad_token_id).to(dev)
        attn = data.pad_left([[1] * len(prompts[i]) for i in idx], 0).to(dev)
        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.pad_token_id,
                             stopping_criteria=StoppingCriteriaList([AnswerDone(tok, ids.shape[1])]))
        for k, i in enumerate(idx):
            resp = gen[k, ids.shape[1]:].tolist()
            cut = next((j for j, t in enumerate(resp) if t in stops), None)
            resp = trim_to_answer(tok, resp if cut is None else resp[:cut])
            text = tok.decode(resp, skip_special_tokens=True)
            out[i] = {"idx": i, "prompt_ids": prompts[i], "response_ids": resp, "text": text,
                      "correct": data.is_correct(text, rows[i]["answer"])}
        if (s // batch_size) % 5 == 0:
            print(f"gsm8k generation {min(s + batch_size, len(order))}/{len(order)} ({time.time() - t0:.0f} s)",
                  flush=True)
    return out  # type: ignore[return-value]


def gsm8k_metrics(records: list[dict]) -> dict:
    n = len(records)
    c = sum(r["correct"] for r in records)
    f = sum("####" in r["text"] for r in records)
    return {"n": n, "correct": c, "accuracy": c / n if n else 0.0, "share_with_####": f / n if n else 0.0}


def forward_kl(base_lp: list[torch.Tensor], masked_lp: list[torch.Tensor]) -> dict:
    """The official retention_forward_kl.py estimate: per prompt the mean over response tokens of
    log pi_0(y_t) - log pi(y_t), averaged over prompts that have a response (that script's main() does this inline,
    so it is repeated here; the log-probabilities come from its response_logprobs)."""
    per = [(b - m).mean().item() for b, m in zip(base_lp, masked_lp) if b.numel()]
    return {"forward_kl_per_token_mean": sum(per) / len(per) if per else float("nan"), "n_prompts": len(per)}
