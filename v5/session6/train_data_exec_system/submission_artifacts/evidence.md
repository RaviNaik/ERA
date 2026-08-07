# Training Data Execution System — Evidence Bundle

Generated: 2026-08-07T08:44:31Z

## Requirement Results

| Requirement | Result | Evidence |
|-------------|--------|----------|
| Tokenizer integrity | ✅ PASS | submission_artifacts/tokenizer_spec.json |
| Evaluation firewall | ✅ PASS | submission_artifacts/manifests/manifest_summary.json |
| Packing correctness | ✅ PASS | submission_artifacts/ledgers/packing_report.json |
| Mixture compliance | ✅ PASS | submission_artifacts/ledgers/mixture_actual.json |
| OPUS audit trail | ✅ PASS | submission_artifacts/ledgers/opus_decisions.jsonl |
| Crash recovery | ✅ PASS | submission_artifacts/checkpoints/ckpt_run_A_main_step00010/meta.json |
| Replay hash match | ✅ PASS | submission_artifacts/ledgers/replay_hashes.json |
| Learning trace | ✅ PASS | submission_artifacts/ledgers/doc_loss_summary.json |
| Throughput & utilization | ✅ PASS | submission_artifacts/performance.json |

## Log Excerpt

```
[14:14:13] 
============================================================
[14:14:13] ============================================================
[14:14:13] [PASS] tokenizer_integrity: tokenizer_sha=tok_27b6a09dd8613291
[14:14:13] [PASS] tokenizer_hash_verified
[14:14:13] 
============================================================
[14:14:13] ============================================================
[14:14:13]   [PASS] All 27 shards passed integrity verification
[14:14:13] 
============================================================
[14:14:13] ============================================================
[14:14:13]   [PASS] eval_shard_blocked: shard=shard_ca51f610fcf6
[14:14:13]   [PASS] eval_shard_blocked: shard=shard_f8dea85b952b
[14:14:13]   [PASS] eval_shard_blocked: shard=shard_8f703fea119c
[14:14:13] [PASS] eval_firewall: 3 eval shards blocked, 0 violations
[14:14:13] [PASS] eval_shard_blocked (all eval shards intercepted)
[14:14:13] 
============================================================
[14:14:13] ============================================================
[14:14:13] [PASS] packing_correctness: structure_preserving: 82.7% util, agentic masks applied
[14:14:13] [PASS] packing_correctness: util=82.7%
[14:14:13] 
============================================================
[14:14:13] ============================================================
[14:14:13] 
============================================================
[14:14:13] ============================================================
[14:14:16]   [CHECKPOINT] ckpt_run_A_main_step00005 at ledger_offset=5
[14:14:20]   [CHECKPOINT] ckpt_run_A_main_step00010 at ledger_offset=10
[14:14:20] [CRASH] Simulated crash at step 12
[14:14:20] 
[CRASH] Deliberate crash at step 12
[14:14:20] [PASS] crash_simulated
[14:14:20] [PASS] checkpoint_saved: ckpt_run_A_main_step00010 at ledger_offset=10
[14:14:20] 
============================================================
[14:14:20] ============================================================
[14:14:20] [PASS] crash_recovery: next_batch=run_A_batch_0010 matched, hash verified
[14:14:20] [PASS] resume_next_batch_matched: expected=run_A_batch_0010, got=run_A_batch_0010
[14:14:24]   [CHECKPOINT] ckpt_run_A_main_step00015 at ledger_offset=15
[14:14:27]   [CHECKPOINT] ckpt_run_A_main_step00020 at ledger_offset=20
[14:14:28] [PASS] run_resumed successfully
[14:14:28] 
============================================================
[14:14:28] ============================================================
[14:14:28] [PASS] replay: All 5 replayed batches hash-matched original
[14:14:28] [PASS] replay_hash_matched: 5 batches verified
[14:14:28] 
============================================================
[14:14:28] ============================================================
[14:14:31] [PASS] branch_forked
[14:14:31] 
============================================================
[14:14:31] ============================================================
[14:14:31] [PASS] opus_audit_trail: 25 decisions logged (pre-crash + post-resume), 4 floor overrides
[14:14:31] [PASS] opus_decisions_recorded
[14:14:31] [PASS] mixture_compliance: Curriculum stages executed, floors enforced
[14:14:31] [PASS] mixture_compiled
[14:14:31] [PASS] learning_trace: Loss linked to 18 source documents
[14:14:31] [PASS] learning_trace: per-document loss recorded
[14:14:31] 
============================================================
[14:14:31] ============================================================
[14:14:31] [PASS] throughput: packing=82.7%, avg_tps=628
[14:14:31] [PASS] performance_measured
[14:14:31] 
============================================================
[14:14:31] ============================================================
```