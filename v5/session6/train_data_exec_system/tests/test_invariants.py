"""
test_invariants.py - Automated tests for the important invariants of the TDES.
Run with: uv run pytest tests/ -v
"""
from __future__ import annotations
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.corpus.corpus_generator import build_corpus
from src.tokenizer.frozen_tokenizer import FrozenTokenizer
from src.shards.shard_creator import create_shard, verify_shard_integrity
from src.manifests.manifest_builder import (build_manifest, get_admitted_shards,
                                             get_eval_shards)
from src.firewall.eval_firewall import EvalFirewall, EvalFirewallViolation
from src.packing.packer import Packer, DocToken, compute_packing_stats, extract_agentic_masked_spans
from src.mixture.mixture_scheduler import MixtureScheduler, PROTECTED_FLOORS
from src.mixture.opus_selector import OPUSSelector, summarize_decisions
from src.dataloader.deterministic_dataloader import DeterministicDataLoader, batch_hash
from src.model.tiny_gpt import TinyGPT
from src.ledgers.consumption_ledger import ConsumptionLedger
from src.ledgers.learning_ledger import LearningLedger
from src.checkpoints.checkpoint_manager import CheckpointManager
from src.training.trainer import Trainer, SimulatedCrashError


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def tokenizer():
    return FrozenTokenizer()


@pytest.fixture(scope="session")
def corpus():
    return build_corpus()


@pytest.fixture(scope="session")
def tmp_base(tmp_path_factory):
    return tmp_path_factory.mktemp("tdes_test")


@pytest.fixture(scope="session")
def shards_and_manifests(corpus, tokenizer, tmp_base):
    shard_dir = tmp_base / "shards"
    manifest_dir = tmp_base / "manifests"
    shard_infos = [create_shard(doc, tokenizer, shard_dir) for doc in corpus]
    manifests = [build_manifest(info, manifest_dir) for info in shard_infos]
    return shard_infos, manifests


@pytest.fixture(scope="session")
def firewall(shards_and_manifests):
    _, manifests = shards_and_manifests
    eval_shards = get_eval_shards(manifests)
    fw = EvalFirewall(eval_shards)
    fw.verify_eval_shards_blocked(manifests)
    return fw


@pytest.fixture(scope="session")
def doc_tokens(shards_and_manifests, tokenizer):
    _, manifests = shards_and_manifests
    admitted = get_admitted_shards(manifests)
    from src.shards.shard_creator import load_shard_tokens
    tokens = []
    for m in admitted:
        arr = load_shard_tokens(m["shard_path"])
        is_agentic = m["capability_lane"] == "agentic"
        masked = extract_agentic_masked_spans(arr.tolist(), tokenizer) if is_agentic else []
        tokens.append(DocToken(
            doc_id=m["document_ids"][0],
            shard_id=m["shard_id"],
            lane=m["capability_lane"],
            token_ids=arr.tolist(),
            is_agentic=is_agentic,
            masked_spans=masked,
        ))
    return tokens


# ── Test 1: Tokenizer SHA Stability ──────────────────────────────────────────


def test_tokenizer_hash_stable(tokenizer):
    """Same tokenizer always produces same SHA."""
    t2 = FrozenTokenizer()
    assert tokenizer.tokenizer_sha == t2.tokenizer_sha, \
        "Tokenizer SHA must be deterministic across instances"
    assert tokenizer.tokenizer_sha.startswith("tok_"), \
        "SHA must have 'tok_' prefix"
    assert len(tokenizer.tokenizer_sha) > 10, "SHA must be non-trivial"


# ── Test 2: Shard Immutability ────────────────────────────────────────────────


def test_shard_immutability(shards_and_manifests):
    """Modifying shard content changes hash (proving immutability)."""
    shard_infos, _ = shards_and_manifests
    info = shard_infos[0]
    # Verify original is OK
    assert verify_shard_integrity(info["shard_path"], info["content_hash"])
    # Wrong hash should FAIL
    bad_hash = "sha256_" + "0" * 64
    assert not verify_shard_integrity(info["shard_path"], bad_hash), \
        "Modified hash should not verify"


# ── Test 3: Manifest Admission Gate ───────────────────────────────────────────


def test_manifest_admission_gate(shards_and_manifests):
    """Eval shards must be blocked; safe shards must be admitted."""
    _, manifests = shards_and_manifests
    eval_m = [m for m in manifests if m.get("is_eval")]
    safe_m = [m for m in manifests if not m.get("is_eval")]
    assert len(eval_m) == 3, f"Expected 3 eval shards, got {len(eval_m)}"
    for m in eval_m:
        assert m["admission"] == "blocked", f"Eval shard {m['shard_id']} must be blocked"
    for m in safe_m:
        assert m["admission"] == "admitted", f"Safe shard {m['shard_id']} should be admitted"


# ── Test 4: Eval Firewall Blocks ──────────────────────────────────────────────


def test_eval_firewall_blocks(shards_and_manifests, firewall):
    """Eval shards must NEVER enter a training batch."""
    _, manifests = shards_and_manifests
    eval_shards = get_eval_shards(manifests)
    assert len(eval_shards) > 0, "Must have at least one eval shard"
    for m in eval_shards:
        with pytest.raises(EvalFirewallViolation):
            firewall.check_batch([m["shard_id"]])

    # Admitted shards should NOT raise
    admitted = get_admitted_shards(manifests)
    firewall.check_batch([m["shard_id"] for m in admitted[:2]])  # should not raise


# ── Test 5: Packing Utilization ────────────────────────────────────────────────


def test_packing_utilization(doc_tokens):
    """Structure-preserving packing must achieve >= 70% utilization."""
    packer = Packer(context_len=128, policy="structure_preserving")
    batches = packer.pack(list(doc_tokens))
    stats = compute_packing_stats(batches, 128)
    assert stats["utilization_pct"] >= 70.0, \
        f"Packing utilization {stats['utilization_pct']:.1f}% < 70%"
    assert stats["num_sequences"] > 0, "Must produce at least one sequence"


def test_pad_each_doc_lower_utilization(doc_tokens):
    """Pad-each-doc must have lower utilization than structure-preserving."""
    packer_pad = Packer(context_len=128, policy="pad_each_doc")
    packer_sp = Packer(context_len=128, policy="structure_preserving")
    batches_pad = packer_pad.pack(list(doc_tokens))
    batches_sp = packer_sp.pack(list(doc_tokens))
    stats_pad = compute_packing_stats(batches_pad, 128)
    stats_sp = compute_packing_stats(batches_sp, 128)
    assert stats_pad["utilization_pct"] < stats_sp["utilization_pct"], \
        "Pad-each-doc should have lower utilization than structure-preserving"


# ── Test 6: Loss Mask — Agentic Tokens ────────────────────────────────────────


def test_loss_mask_agentic(doc_tokens, tokenizer):
    """Agentic tool observation tokens must have loss_mask=0."""
    agentic_docs = [dt for dt in doc_tokens if dt.is_agentic]
    if not agentic_docs:
        pytest.skip("No agentic documents in corpus")

    packer = Packer(context_len=256, policy="structure_preserving")
    batches = packer.pack(agentic_docs[:2])

    checked_a_masked_doc = False
    for batch in batches:
        if batch.lane == "agentic" or any(dt.is_agentic for dt in agentic_docs
                                           if dt.shard_id in batch.shard_ids):
            # For agentic, loss_mask should NOT be all-ones (some masked)
            # At minimum, verify loss_mask is valid [0,1]
            assert batch.loss_mask.min() >= 0.0
            assert batch.loss_mask.max() <= 1.0
            # Agentic docs with masked spans must have some 0 positions —
            # this is the actual property that matters: tool observations
            # must not be loss-bearing, or the model learns to hallucinate
            # tool results instead of calling tools.
            for dt in agentic_docs:
                if dt.shard_id in batch.shard_ids and dt.masked_spans:
                    total_loss = batch.loss_mask.sum()
                    total_positions = len(batch.loss_mask)
                    assert total_loss < total_positions, (
                        f"doc {dt.doc_id} has masked_spans={dt.masked_spans} but "
                        f"loss_mask sum ({total_loss}) == total positions "
                        f"({total_positions}) — no tokens were actually masked"
                    )
                    checked_a_masked_doc = True
                    break

    assert checked_a_masked_doc, (
        "No agentic doc with masked_spans was found in any packed batch — "
        "the test didn't actually verify the masking property"
    )


# ── Test 7: Ledger Monotonicity ────────────────────────────────────────────────


def test_ledger_monotonic(tmp_path):
    """Consumption ledger steps must be monotonically increasing."""
    ledger = ConsumptionLedger(tmp_path / "test_consumption.jsonl")

    # Mock a batch for recording
    class MockBatch:
        shard_ids = ["shard_001"]
        doc_spans = [("doc_001", 0, 10)]
        lane = "web"
        policy = "structure_preserving"
        utilization = 0.8
        useful_tokens = 80
        pad_count = 20
        loss_mask = np.ones(100)

    for step in range(5):
        ledger.record(step, f"batch_{step:04d}", MockBatch(), {"web": 80})

    entries = ledger.get_all()
    steps = [e["step"] for e in entries]
    assert steps == sorted(steps), "Ledger steps must be monotonically increasing"
    assert steps == list(range(5)), "Ledger must have contiguous steps"


# ── Test 8: Deterministic Batch Order ─────────────────────────────────────────


def test_deterministic_batch_order(shards_and_manifests, tokenizer, firewall):
    """Same seed must produce identical batch sequences."""
    _, manifests = shards_and_manifests
    loader1 = DeterministicDataLoader(
        manifests=manifests, tokenizer=tokenizer, firewall=firewall,
        context_len=128, policy="structure_preserving", seed=42, run_id="test")
    loader2 = DeterministicDataLoader(
        manifests=manifests, tokenizer=tokenizer, firewall=firewall,
        context_len=128, policy="structure_preserving", seed=42, run_id="test")

    ids1 = [bid for bid, _ in loader1._packed_batches]
    ids2 = [bid for bid, _ in loader2._packed_batches]
    assert ids1 == ids2, "Same seed must produce identical batch order"

    hashes1 = [loader1.get_batch_hash(i) for i in range(len(loader1))]
    hashes2 = [loader2.get_batch_hash(i) for i in range(len(loader2))]
    assert hashes1 == hashes2, "Same seed must produce identical batch hashes"


# ── Test 9: Resume — No Skip, No Repeat ────────────────────────────────────────


def test_resume_no_skip_no_repeat(shards_and_manifests, tokenizer, firewall, tmp_path):
    """After crash+resume, the next batch must be exactly the expected batch."""
    _, manifests = shards_and_manifests
    loader = DeterministicDataLoader(
        manifests=manifests, tokenizer=tokenizer, firewall=firewall,
        context_len=128, policy="structure_preserving", seed=42, run_id="test")

    original_sequence = [bid for bid, _ in loader._packed_batches]
    crash_at = 3
    resume_from = 2  # last checkpoint before crash

    # Simulate: steps 0,1,2 completed, checkpoint at 2, crash at 3
    # Resume should give step 3 exactly
    loader.seek(resume_from)
    resumed_batch_id, _ = loader.get_batch_at(resume_from)
    expected = original_sequence[resume_from]
    assert resumed_batch_id == expected, \
        f"After resume, got {resumed_batch_id}, expected {expected}"

    # Ensure seek(0) followed by next(crash_at) gives same as original[crash_at]
    loader.seek(crash_at)
    crash_batch_id, _ = loader.get_batch_at(crash_at)
    assert crash_batch_id == original_sequence[crash_at], \
        "Batch at crash point must match original"


# ── Test 10: Replay Hash Match ────────────────────────────────────────────────


def test_replay_hash_match(shards_and_manifests, tokenizer, firewall):
    """Replayed batches must produce identical hashes to the original run."""
    _, manifests = shards_and_manifests
    loader1 = DeterministicDataLoader(
        manifests=manifests, tokenizer=tokenizer, firewall=firewall,
        context_len=128, policy="structure_preserving", seed=42, run_id="run_A")
    loader2 = DeterministicDataLoader(
        manifests=manifests, tokenizer=tokenizer, firewall=firewall,
        context_len=128, policy="structure_preserving", seed=42, run_id="run_A")

    for i in range(min(5, len(loader1))):
        h1 = loader1.get_batch_hash(i)
        h2 = loader2.get_batch_hash(i)
        assert h1 == h2, f"Replay hash mismatch at step {i}: {h1} != {h2}"


# ── Test 11: Protected Floor Enforcement ──────────────────────────────────────


def test_floor_protection():
    """Indic must never fall below 12% in enforced weights."""
    scheduler = MixtureScheduler()
    # Test all stages
    for step in [0, 10, 18]:
        weights = scheduler.get_lane_weights(step, 20)
        indic_pct = weights.get("indic", 0.0)
        agentic_pct = weights.get("agentic", 0.0)
        assert indic_pct >= PROTECTED_FLOORS["indic"] - 0.001, \
            f"step={step}: Indic {indic_pct:.3f} < floor {PROTECTED_FLOORS['indic']}"
        assert agentic_pct >= PROTECTED_FLOORS["agentic"] - 0.001, \
            f"step={step}: Agentic {agentic_pct:.3f} < floor {PROTECTED_FLOORS['agentic']}"


# ── Test 12: OPUS Audit Trail ─────────────────────────────────────────────────


def test_opus_audit_trail():
    """Every batch must have an OPUS decision record."""
    opus = OPUSSelector(proxy_mix="balanced", keep_fraction=0.5)
    batch_infos = [
        {"batch_id": f"batch_{i:03d}", "lane": lane, "token_count": 100}
        for i, lane in enumerate(["web", "code", "indic", "agentic", "reasoning"])
    ]
    all_scores = [opus._proxy_score(bi) for bi in batch_infos]
    decisions = []
    for bi in batch_infos:
        dec, score, reason = opus.select_batch(bi, all_scores)
        decisions.append(dec)

    recorded = opus.get_decisions()
    assert len(recorded) == len(batch_infos), \
        "Every batch must have a recorded OPUS decision"
    for record in recorded:
        assert "decision" in record and record["decision"] in ("accept", "reject", "defer")
        assert "score" in record
        assert "batch_id" in record


# ── Test 13: Checkpoint Save/Load ─────────────────────────────────────────────


def test_checkpoint_save_load(tmp_path):
    """Checkpoint must restore model weights exactly."""
    model = TinyGPT(vocab_size=100, n_embd=16, n_head=2, n_layer=1, context_len=32)
    original_hash = model.get_weights_hash()

    ckpt_manager = CheckpointManager(tmp_path / "checkpoints")
    ckpt_id = ckpt_manager.save(
        step=5, model=model, optimizer_lr=1e-3,
        ledger_offset=5, run_id="test", branch_id="main"
    )

    # Modify model
    model.step_update(1.0, lr=0.1)
    assert model.get_weights_hash() != original_hash, "Model should change after update"

    # Load and verify
    loaded = ckpt_manager.load(ckpt_id)
    model.load_state_dict(loaded["state"])
    assert model.get_weights_hash() == original_hash, \
        "Loaded checkpoint must restore original weights"
    assert loaded["meta"]["ledger_offset"] == 5


# ── Test 14: Tokenizer SHA in Manifests ───────────────────────────────────────


def test_tokenizer_sha_in_manifests(shards_and_manifests, tokenizer):
    """Every manifest must carry the correct tokenizer SHA."""
    _, manifests = shards_and_manifests
    for m in manifests:
        assert m["tokenizer_hash"] == tokenizer.tokenizer_sha, \
            f"Manifest {m['shard_id']} has wrong tokenizer SHA"


# ── Test 15: No Eval Shard in Admitted ────────────────────────────────────────


def test_no_eval_in_admitted(shards_and_manifests):
    """Admitted shards must contain zero eval shards."""
    _, manifests = shards_and_manifests
    admitted = get_admitted_shards(manifests)
    eval_in_admitted = [m for m in admitted if m.get("is_eval")]
    assert len(eval_in_admitted) == 0, \
        f"Found {len(eval_in_admitted)} eval shards in admitted list!"


# ── Test 16: Curriculum Stage Weights Actually Affect OPUS Scoring ────────────


def test_curriculum_stage_weights_affect_opus_score():
    """
    MixtureScheduler's per-stage lane weights must have a real, measurable
    effect on OPUS's proxy score — not just be computed for display. Indic is
    weighted much higher in the anneal stage (28%) than the seed stage (16%),
    so the same candidate batch must score higher when evaluated under the
    anneal stage's weights.
    """
    opus = OPUSSelector(proxy_mix="balanced", keep_fraction=0.4)
    batch_info = {"batch_id": "batch_stage_test", "lane": "indic", "token_count": 100}

    scheduler = MixtureScheduler()
    seed_weights = scheduler.get_lane_weights(0, 100)     # stage 1 (seed): indic=16%
    anneal_weights = scheduler.get_lane_weights(99, 100)  # stage 3 (anneal): indic=28%
    assert anneal_weights["indic"] > seed_weights["indic"], \
        "test assumption: anneal stage should weight indic higher than seed stage"

    # Same batch_id -> same noise draw, so only the stage weighting differs.
    score_seed = opus._proxy_score(batch_info, stage_weights=seed_weights)
    score_anneal = opus._proxy_score(batch_info, stage_weights=anneal_weights)
    assert score_anneal > score_seed, (
        f"indic candidate should score higher under the anneal stage's weights "
        f"than the seed stage's: seed={score_seed:.4f} anneal={score_anneal:.4f}"
    )

    # And the un-adjusted score sits between them (sanity check that
    # stage_weights=None is a neutral baseline, not accidentally a no-op alias
    # for one of the two stages).
    score_unadjusted = opus._proxy_score(batch_info)
    assert score_seed <= score_unadjusted <= score_anneal


# ── Test 17: Defer and Reject Both Skip Training ───────────────────────────────


def test_defer_and_reject_skip_training(shards_and_manifests, tokenizer, firewall, tmp_path):
    """
    Only accepted batches may enter the consumption/learning ledgers. With
    keep_fraction=0.0, almost every candidate is rejected or deferred — none
    of those batch_ids should ever appear as a trained (ledger-recorded) step.
    """
    _, manifests = shards_and_manifests
    loader = DeterministicDataLoader(
        manifests=manifests, tokenizer=tokenizer, firewall=firewall,
        context_len=128, policy="structure_preserving", seed=42, run_id="test_defer")
    model = TinyGPT(vocab_size=tokenizer.vocab_size, n_embd=16, n_head=2, n_layer=1, context_len=128)
    consumption = ConsumptionLedger(tmp_path / "consumption_defer.jsonl")
    learning = LearningLedger(tmp_path / "learning_defer.jsonl")
    ckpt_manager = CheckpointManager(tmp_path / "checkpoints_defer")
    scheduler = MixtureScheduler()
    opus = OPUSSelector(proxy_mix="balanced", keep_fraction=0.0)
    trainer = Trainer(
        model=model, loader=loader, consumption_ledger=consumption,
        learning_ledger=learning, checkpoint_manager=ckpt_manager,
        scheduler=scheduler, opus=opus, run_id="test_defer", branch_id="main",
        checkpoint_every=1000, crash_at_step=None, lr=1e-3, total_steps=len(loader),
    )
    trainer.run(start_step=0)

    decisions = opus.get_decisions()
    non_accepted = [d for d in decisions if d["decision"] != "accept"]
    assert non_accepted, "Expected at least one reject/defer decision with keep_fraction=0.0"

    consumed_batch_ids = {e["batch_id"] for e in consumption.get_all()}
    learned_batch_ids = {e["batch_id"] for e in learning.get_all()}
    for d in non_accepted:
        assert d["batch_id"] not in consumed_batch_ids, (
            f"batch {d['batch_id']} was {d['decision']} but appears in the "
            f"consumption ledger — defer/reject must both skip training"
        )
        assert d["batch_id"] not in learned_batch_ids, (
            f"batch {d['batch_id']} was {d['decision']} but appears in the "
            f"learning ledger — defer/reject must both skip training"
        )


# ── Test 18: Trainer Respects the total_steps Budget ───────────────────────────


def test_trainer_respects_total_steps_budget(shards_and_manifests, tokenizer, firewall, tmp_path):
    """
    Trainer.run() must stop at its configured total_steps budget even when
    the dataloader has more batches available (previously total_steps was
    accepted but never actually enforced).
    """
    _, manifests = shards_and_manifests
    loader = DeterministicDataLoader(
        manifests=manifests, tokenizer=tokenizer, firewall=firewall,
        context_len=128, policy="structure_preserving", seed=42, run_id="test_budget")
    assert len(loader) > 3, "fixture corpus must yield more than 3 batches for this test"

    model = TinyGPT(vocab_size=tokenizer.vocab_size, n_embd=16, n_head=2, n_layer=1, context_len=128)
    consumption = ConsumptionLedger(tmp_path / "consumption_budget.jsonl")
    learning = LearningLedger(tmp_path / "learning_budget.jsonl")
    ckpt_manager = CheckpointManager(tmp_path / "checkpoints_budget")
    scheduler = MixtureScheduler()
    opus = OPUSSelector(proxy_mix="balanced", keep_fraction=1.0)  # accept everything
    trainer = Trainer(
        model=model, loader=loader, consumption_ledger=consumption,
        learning_ledger=learning, checkpoint_manager=ckpt_manager,
        scheduler=scheduler, opus=opus, run_id="test_budget", branch_id="main",
        checkpoint_every=1000, crash_at_step=None, lr=1e-3, total_steps=3,
    )
    result = trainer.run(start_step=0)

    assert trainer._step == 3, f"trainer advanced to step {trainer._step}, past its total_steps=3 budget"
    assert result["steps_run"] == 3, f"expected exactly 3 trained steps, got {result['steps_run']}"


# ── Test 19: OPUS Decisions Merge Correctly Across a Resume Boundary ──────────


def test_opus_decision_merge_across_resume():
    """
    A fresh OPUSSelector is used for the resumed run, so the persisted audit
    trail is built by merging decisions from two instances. Merged records
    must be step-numbered continuously (not restart from 0), and
    summarize_decisions() must summarize the merged list correctly.
    """
    pre = OPUSSelector(proxy_mix="balanced", keep_fraction=0.5)
    post = OPUSSelector(proxy_mix="balanced", keep_fraction=0.5)

    pre_infos = [{"batch_id": f"batch_{i:04d}", "lane": "web", "token_count": 10} for i in range(4)]
    pre_scores = [pre._proxy_score(bi) for bi in pre_infos]
    for i, bi in enumerate(pre_infos):
        pre.select_batch(bi, pre_scores, step=i)

    post_infos = [{"batch_id": f"batch_{i:04d}", "lane": "code", "token_count": 10} for i in range(4, 8)]
    post_scores = [post._proxy_score(bi) for bi in post_infos]
    for i, bi in enumerate(post_infos, start=4):
        post.select_batch(bi, post_scores, step=i)

    merged = pre.get_decisions() + post.get_decisions()
    assert [d["step"] for d in merged] == list(range(8)), \
        "merged decisions must be step-numbered continuously across the resume boundary"

    summary = summarize_decisions(merged, keep_fraction_target=0.5)
    assert summary["total_candidates"] == 8
    assert summary["accepted"] + summary["rejected"] + summary["deferred"] == 8
