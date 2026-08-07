# Training Data Execution System — Evidence Bundle

Generated: 2026-08-07T05:55:57Z

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
[11:25:39] 
============================================================
[11:25:39] ============================================================
[11:25:39] [PASS] tokenizer_integrity: tokenizer_sha=tok_27b6a09dd8613291
[11:25:39] [PASS] tokenizer_hash_verified
[11:25:39] 
============================================================
[11:25:39] ============================================================
[11:25:40]   [PASS] All 27 shards passed integrity verification
[11:25:40] 
============================================================
[11:25:40] ============================================================
[11:25:40]   [PASS] eval_shard_blocked: shard=shard_ca51f610fcf6
[11:25:40]   [PASS] eval_shard_blocked: shard=shard_f8dea85b952b
[11:25:40]   [PASS] eval_shard_blocked: shard=shard_8f703fea119c
[11:25:40] [PASS] eval_firewall: 3 eval shards blocked, 0 violations
[11:25:40] [PASS] eval_shard_blocked (all eval shards intercepted)
[11:25:40] 
============================================================
[11:25:40] ============================================================
[11:25:40] [PASS] packing_correctness: structure_preserving: 82.7% util, agentic masks applied
[11:25:40] [PASS] packing_correctness: util=82.7%
[11:25:40] 
============================================================
[11:25:40] ============================================================
[11:25:40] 
============================================================
[11:25:40] ============================================================
[11:25:43]   [CHECKPOINT] ckpt_run_A_main_step00005 at ledger_offset=5
[11:25:46]   [CHECKPOINT] ckpt_run_A_main_step00010 at ledger_offset=10
[11:25:47] [CRASH] Simulated crash at step 12
[11:25:47] 
[CRASH] Deliberate crash at step 12
[11:25:47] [PASS] crash_simulated
[11:25:47] [PASS] checkpoint_saved: ckpt_run_A_main_step00010 at ledger_offset=10
[11:25:47] 
============================================================
[11:25:47] ============================================================
[11:25:47] [PASS] crash_recovery: next_batch=run_A_batch_0010 matched, hash verified
[11:25:47] [PASS] resume_next_batch_matched: expected=run_A_batch_0010, got=run_A_batch_0010
[11:25:50]   [CHECKPOINT] ckpt_run_A_main_step00015 at ledger_offset=15
[11:25:53]   [CHECKPOINT] ckpt_run_A_main_step00020 at ledger_offset=20
[11:25:54] [PASS] run_resumed successfully
[11:25:54] 
============================================================
[11:25:54] ============================================================
[11:25:54] [PASS] replay: All 5 replayed batches hash-matched original
[11:25:54] [PASS] replay_hash_matched: 5 batches verified
[11:25:54] 
============================================================
[11:25:54] ============================================================
[11:25:57] [PASS] branch_forked
[11:25:57] 
============================================================
[11:25:57] ============================================================
[11:25:57] [PASS] opus_audit_trail: 12 decisions logged, 2 floor overrides
[11:25:57] [PASS] opus_decisions_recorded
[11:25:57] [PASS] mixture_compliance: Curriculum stages executed, floors enforced
[11:25:57] [PASS] mixture_compiled
[11:25:57] [PASS] learning_trace: Loss linked to 16 source documents
[11:25:57] [PASS] learning_trace: per-document loss recorded
[11:25:57] 
============================================================
[11:25:57] ============================================================
[11:25:57] [PASS] throughput: packing=82.7%, avg_tps=696
[11:25:57] [PASS] performance_measured
[11:25:57] 
============================================================
[11:25:57] ============================================================
```