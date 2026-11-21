"""Deletion Tuning SFT on GSM8K with the OFFICIAL training code (official/src/train_sft.py, unmodified).

This file only builds the inputs the official `train_all_tasks` expects and calls it:
  - the model (Phi-mini-MoE, bf16, gradient checkpointing as in official main()), split into two blocks of layers over
    the two GPUs;
  - the splits object (GSM8K in the official `{question}\\n{answer}` SFT format, data.Gsm8kSplits);
  - deterministic MoE routing during training (see below);
  - the target module list (official TARGET_MODULES = ["gate_proj"] -> the Phi expert gate projection ".w1").
Everything else (Wanda calibration and prior, SparseLinear, sharpness schedule, regulariser with linear decay, AdamW,
clipping, mask saving) is official code. Runtime wrappers around official functions (the files stay unmodified):
  - convert_to_sparse_model: afterwards every mask logit tensor is moved to the device of its layer's weight. The
    official function puts all logits on `model.device`, and the official Wanda prior then adds a tensor from the
    weight's device to the logits (`gate.data += ...`), which only works when both are on one GPU.
  - sum_sparse_weight_norms: the same sum as the official function, accumulated on the device of the loss. The
    official one starts from a tensor on the first parameter's device, so it fails when layers sit on two GPUs.
  - init_masks_from_wanda: calls the official function, then logs the deleted share at step 0.
  - the logging handler collects the official loss lines into train_log.jsonl."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch

from . import official  # noqa: F401
from .data import Gsm8kSplits
from .modeling import load_model, load_tokenizer

GATE_TARGET = ".w1"  # the gate projection of a Phi-MoE expert: act(w1(x)) * w3(x) -> w2


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data_dir", required=True, help="folder holding main/train-*.parquet")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_gpus", type=int, default=2)
    # the official flags and their values in scripts/train_sft.sh (paper): train_steps 400, batch 4, lr 5e-4,
    # init_sparsity 1/2048, sharpness 100 -> 1000, weight_reg 1e-5
    p.add_argument("--train_steps", type=int, default=400)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--init_sparsity", type=float, default=0.00048828125)
    p.add_argument("--prior_strength", type=float, default=3.0)
    p.add_argument("--sharpness_start", type=float, default=100.0)
    p.add_argument("--sharpness_end", type=float, default=1000.0)
    p.add_argument("--weight_reg", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    # ours: the GSM8K pool the official code trains and calibrates on
    p.add_argument("--train_pool", type=int, default=2048)
    p.add_argument("--max_len", type=int, default=512)
    a = p.parse_args(argv)
    a.fullparam = False  # official flag: this driver only trains masks
    return a


class _LossCapture(logging.Handler):
    """Collects the official '[task] iter i/N | loss=x' log lines."""
    PAT = re.compile(r"iter (\d+)/(\d+) \| loss=([-\d.eE+naninf]+)")

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records: list[dict] = []
        self.t_train_start: float | None = None
        self.t0 = time.time()

    def emit(self, record):
        msg = record.getMessage()
        if "training:" in msg and "iters" in msg and self.t_train_start is None:
            self.t_train_start = time.time()  # the official line printed right before the training loop starts
        m = self.PAT.search(msg)
        if m:
            self.records.append({"iter": int(m.group(1)), "of": int(m.group(2)), "loss": float(m.group(3)),
                                 "elapsed_s": round(time.time() - self.t0, 1)})


def run(args: argparse.Namespace) -> dict:
    official.verify_vendored()
    import train_sft  # official
    import wanda_calibration  # official
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from learnable_sparsity.sparse_linear import SparseLinear

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    use_cuda = args.n_gpus > 0
    accelerator = Accelerator(mixed_precision="bf16" if use_cuda else "no", cpu=not use_cuda)
    set_seed(args.seed)
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    capture = _LossCapture()
    logging.getLogger().addHandler(capture)

    t_all = time.time()
    tok = load_tokenizer(args.model_dir)
    splits = Gsm8kSplits(tok, args.data_dir, args.seed, args.train_pool, args.max_len)
    n_train = len(splits.train_set("gsm8k"))
    print(f"GSM8K training pool: {n_train} samples ({splits.dropped} longer than {args.max_len} tokens dropped)",
          flush=True)
    if n_train < args.train_batch_size:
        raise ValueError("fewer training samples than the batch size")

    from .phimoe import modeling_phimoe
    # official main() calls model.train(); in this model's code that switches on random expert sampling (Gumbel), a random
    # router-gradient estimator and 1% input jitter. The mask is evaluated with deterministic top-2 routing, so train with it too.
    modeling_phimoe.DETERMINISTIC_MOE = True
    model = load_model(args.model_dir, args.n_gpus)
    # as official main(): gradient checkpointing, parameters frozen (the official model is also passed through
    # accelerator.prepare there; a model spread over two GPUs is used as it is)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad = False
    train_sft.TARGET_MODULES[:] = [GATE_TARGET]  # official module-level list, used by convert and by Wanda

    # accelerate's dispatch hook on a model spread over several GPUs returns the outputs, hence the loss, on the input
    # device (cuda:0), not on the LM head's device. The regulariser term must be added on that same device.
    loss_device = accelerator.device
    original_convert = train_sft.convert_to_sparse_model
    original_norms = train_sft.sum_sparse_weight_norms

    def convert_and_colocate(model, target_modules, sharpness, total_steps):
        m = original_convert(model=model, target_modules=target_modules, sharpness=sharpness, total_steps=total_steps)
        for mod in m.modules():
            if isinstance(mod, SparseLinear) and mod.diff_mask.gate.device != mod.weight.device:
                mod.diff_mask.gate.data = mod.diff_mask.gate.data.to(mod.weight.device)
        return m

    def sum_norms_on_loss_device(model):
        total = torch.zeros((), device=loss_device)
        for module in model.modules():
            if isinstance(module, SparseLinear) and hasattr(module, "sparse_weight_norm"):
                total = total + module.sparse_weight_norm.float().to(loss_device)
        return total

    train_sft.convert_to_sparse_model = convert_and_colocate
    train_sft.sum_sparse_weight_norms = sum_norms_on_loss_device
    info: dict = {}
    original_init = wanda_calibration.init_masks_from_wanda

    def init_and_log(model, wanda_scores, init_sparsity, prior_strength):
        original_init(model=model, wanda_scores=wanda_scores, init_sparsity=init_sparsity,
                      prior_strength=prior_strength)
        layers = [(n, m) for n, m in model.named_modules() if isinstance(m, SparseLinear)]
        deleted = sum(int((m.diff_mask.gate <= 0).sum()) for _, m in layers)
        total = sum(m.diff_mask.gate.numel() for _, m in layers)
        info.update({"sparse_layers": len(layers), "layers_with_wanda_scores": len(wanda_scores),
                     "layers_without_scores": len(layers) - len(wanda_scores),
                     "deleted_step0": deleted, "targeted": total, "deleted_frac_step0": deleted / total})
        print("after Wanda prior:", info, flush=True)

    wanda_calibration.init_masks_from_wanda = init_and_log
    t_train = time.time()
    try:
        train_sft.train_all_tasks(args=args, model=model, splits=splits, accelerator=accelerator)
    finally:
        wanda_calibration.init_masks_from_wanda = original_init
        train_sft.convert_to_sparse_model = original_convert
        train_sft.sum_sparse_weight_norms = original_norms
        logging.getLogger().removeHandler(capture)
    t_train = time.time() - t_train

    mask_dir = out / "masks" / "gsm8k"
    stats = json.loads((mask_dir / "config.json").read_text())
    peak = {f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 2**30, 1)
            for i in range(torch.cuda.device_count())} if use_cuda else {}
    (out / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in capture.records))
    summary = {"args": vars(args), "official_commit": official.COMMIT, "train_samples": n_train,
               "dropped_too_long": splits.dropped, "wanda_prior": info, "mask": stats, "train_seconds": round(t_train, 1),
               "total_seconds": round(time.time() - t_all, 1),
               # training loop only (no model load, no Wanda calibration); includes the mask save
               "seconds_per_step": (round((time.time() - capture.t_train_start) / args.train_steps, 2)
                                    if capture.t_train_start else None),
               "peak_gpu_gib": peak,
               "command": " ".join(sys.argv)}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done: {stats['zero_params']:,d} of {stats['total_params']:,d} gate weights deleted "
          f"({stats['sparsity_ratio']:.4%}), mask {stats['compressed_size_mb']:.1f} MB, {t_train:.0f} s", flush=True)
    return summary


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
