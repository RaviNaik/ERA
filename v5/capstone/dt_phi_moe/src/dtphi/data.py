"""GSM8K for the official SFT code: the same `{question}\\n{answer}` samples as official MMLU-Pro SFT, loss on the
answer tokens only (official sft_data.encode), left-padded batches (official sft_data.collate_train)."""
from __future__ import annotations

import random
import re
from pathlib import Path

import pyarrow.parquet as pq
import torch

from . import official  # noqa: F401  (puts official/src on sys.path)
import sft_data  # official

INSTRUCTION = "Solve the problem step by step, and end your answer with '#### <number>'."
NEWLINE_ID = 13  # the "\n" byte token of the Llama-style Phi tokenizer; official sft_data.tokenize needs a BOS token
_CALC = re.compile(r"<<[^>]*>>")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def load_gsm8k(data_dir: str | Path, split: str) -> list[dict]:
    files = sorted(Path(data_dir, "main").glob(f"{split}-*.parquet"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one {split} parquet in {data_dir}/main, found {len(files)}")
    return [{"question": r["question"], "answer": r["answer"]} for r in pq.read_table(files[0]).to_pylist()]


def clean_answer(answer: str) -> str:
    """The reference solution without the <<calculator>> annotations; the final '#### n' line is kept."""
    return _CALC.sub("", answer).strip()


def encode_text(tok, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def plain_prompt_ids(tok, question: str) -> list[int]:
    """The SFT prompt: question tokens + newline. Generation after this continues with the answer."""
    return encode_text(tok, question.strip()) + [NEWLINE_ID]


def chat_prompt_ids(tok, question: str) -> list[int]:
    """The chat prompt of the instruct model, with the answer-format instruction."""
    msgs = [{"role": "user", "content": f"{question.strip()}\n\n{INSTRUCTION}"}]
    return list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True))


class Gsm8kSplits:
    """What the official `train_all_tasks` needs of a splits object: `.tasks`, `.tokenizer` and `.train_set(task)`.
    One task, "gsm8k". The training pool is `pool` rows of the seeded-shuffled GSM8K train split (0 = all 7,473);
    official Wanda calibration runs over the whole pool, so pool=2048 gives the paper's 2,048 calibration samples."""

    def __init__(self, tok, data_dir: str | Path, seed: int, pool: int, max_len: int):
        self.tokenizer = tok
        self.tasks = ["gsm8k"]
        rows = load_gsm8k(data_dir, "train")
        random.Random(seed).shuffle(rows)
        if pool:
            rows = rows[:pool]
        self.dropped = 0
        self._samples: list[dict] = []
        for r in rows:
            prompt = torch.tensor(plain_prompt_ids(tok, r["question"]), dtype=torch.long)
            answer = torch.tensor(encode_text(tok, clean_answer(r["answer"])), dtype=torch.long)
            s = sft_data.encode(prompt_ids=prompt, answer_ids=answer)
            if len(s["input_ids"]) > max_len:
                self.dropped += 1
            else:
                self._samples.append(s)

    def train_set(self, task: str) -> list[dict]:
        assert task == "gsm8k"
        return self._samples


def gold_number(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def _norm_number(s: str) -> str | None:
    s = s.replace(",", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def extract_answer(text: str) -> str | None:
    """The number after the FIRST '####' (a model that was not taught to stop may go on), else a \\boxed{...}
    number, else the last number of the reply."""
    i = text.find("####")
    if i >= 0:
        m = _NUM.search(text[i + 4:])
        if m:
            return _norm_number(m.group(0))
    b = re.findall(r"\\boxed\{([^}]*)\}", text)
    if b:
        m = _NUM.search(b[-1])
        if m:
            return _norm_number(m.group(0))
    nums = _NUM.findall(text)
    return _norm_number(nums[-1]) if nums else None


def is_correct(text: str, gold_answer: str) -> bool:
    got = extract_answer(text)
    return got is not None and got == _norm_number(gold_number(gold_answer))


def pad_left(seqs: list[list[int]], value: int) -> torch.Tensor:
    n = max(len(s) for s in seqs)
    return torch.tensor([[value] * (n - len(s)) + list(s) for s in seqs], dtype=torch.long)
