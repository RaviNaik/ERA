"""Prior-task accuracy with lm-evaluation-harness: the six benchmarks, metrics and settings of the official
retention_prior_tasks.py (zero-shot, chat template, bf16). The official script loads the masked model from a saved
directory through lm-eval's `hf` model; here the model object that already has the mask applied is wrapped instead
(Phi-mini-MoE needs its own model code, which that path cannot load)."""
from __future__ import annotations

import os

# Opt-in required by the `code_eval` metric of HumanEval (official retention_prior_tasks.py does the same).
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

from . import official  # noqa: F401
from retention_prior_tasks import TASK_TO_METRIC  # official: lm-eval task of each benchmark and its primary metric


def run_prior_tasks(model, tok, tasks: list[str], batch_size: int, limit: int) -> dict:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    unknown = [t for t in tasks if t not in TASK_TO_METRIC]
    if unknown:
        raise ValueError(f"unknown prior tasks {unknown}; choose from {sorted(TASK_TO_METRIC)}")
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, max_length=4096)
    res = evaluator.simple_evaluate(model=lm, tasks=tasks, apply_chat_template=True, limit=limit or None,
                                    confirm_run_unsafe_code=True, log_samples=False, bootstrap_iters=0)
    scores = {}
    for t in tasks:
        r = res["results"][t]
        if TASK_TO_METRIC[t] not in r:
            raise KeyError(f"{t}: metric {TASK_TO_METRIC[t]} not in {sorted(r)}")
        scores[t] = r[TASK_TO_METRIC[t]]
    return {"scores": scores, "prior_task_mean": sum(scores.values()) / len(scores), "limit": limit,
            "metrics": {t: TASK_TO_METRIC[t] for t in tasks}}


def decode_keep_leading_space(tok, ids, skip_special_tokens: bool = True) -> str:
    """Decode generated token ids without losing the first space. A SentencePiece tokenizer drops the leading space of
    a text it decodes on its own, so the continuation '    for x' (tokens '▁▁▁', '▁for', ...) comes back with 3 spaces
    and every HumanEval body then has an indentation error. Decoding after one known token and cutting that token's
    text off keeps the space."""
    ids = [ids] if isinstance(ids, int) else list(ids)
    anchor = tok.encode("\n", add_special_tokens=False)[-1:]
    head = tok.decode(anchor, skip_special_tokens=False)
    full = tok.decode(anchor + ids, skip_special_tokens=skip_special_tokens)
    return full[len(head):] if full.startswith(head) else tok.decode(ids, skip_special_tokens=skip_special_tokens)


def make_lm(model, tok, batch_size: int, keep_leading_space: bool = False):
    """The lm-eval wrapper of the prior-task evaluation. keep_leading_space=True is for code generation (HumanEval); the other
    tasks stay on the harness decode so earlier numbers remain comparable."""
    from lm_eval.models.huggingface import HFLM

    class _HFLMKeepSpace(HFLM):
        def tok_decode(self, tokens, skip_special_tokens=True):
            return decode_keep_leading_space(self.tokenizer, tokens, skip_special_tokens)

    return (_HFLMKeepSpace if keep_leading_space else HFLM)(pretrained=model, tokenizer=tok, batch_size=batch_size,
                                                            max_length=4096)
