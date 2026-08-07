# Training Data Execution System — Evidence Bundle

Generated: 2026-08-07T06:40:38Z

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
[12:10:17] 
============================================================
[12:10:17] ============================================================
[12:10:17] [PASS] tokenizer_integrity: tokenizer_sha=tok_27b6a09dd8613291
[12:10:17] [PASS] tokenizer_hash_verified
[12:10:17] 
============================================================
[12:10:17] ============================================================
[12:10:17]   [PASS] All 27 shards passed integrity verification
[12:10:17] 
============================================================
[12:10:17] ============================================================
[12:10:17]   [PASS] eval_shard_blocked: shard=shard_ca51f610fcf6
[12:10:17]   [PASS] eval_shard_blocked: shard=shard_f8dea85b952b
[12:10:17]   [PASS] eval_shard_blocked: shard=shard_8f703fea119c
[12:10:17] [PASS] eval_firewall: 3 eval shards blocked, 0 violations
[12:10:17] [PASS] eval_shard_blocked (all eval shards intercepted)
[12:10:17] 
============================================================
[12:10:17] ============================================================
[12:10:17] [PASS] packing_correctness: structure_preserving: 82.7% util, agentic masks applied
[12:10:17] [PASS] packing_correctness: util=82.7%
[12:10:17] 
============================================================
[12:10:17] ============================================================
[12:10:17] 
============================================================
[12:10:17] ============================================================
[12:10:21]   [CHECKPOINT] ckpt_run_A_main_step00005 at ledger_offset=5
[12:10:24]   [CHECKPOINT] ckpt_run_A_main_step00010 at ledger_offset=10
[12:10:25] [CRASH] Simulated crash at step 12
[12:10:25] 
[CRASH] Deliberate crash at step 12
[12:10:25] [PASS] crash_simulated
[12:10:25] [PASS] checkpoint_saved: ckpt_run_A_main_step00010 at ledger_offset=10
[12:10:25] 
============================================================
[12:10:25] ============================================================
[12:10:25] [PASS] crash_recovery: next_batch=run_A_batch_0010 matched, hash verified
[12:10:25] [PASS] resume_next_batch_matched: expected=run_A_batch_0010, got=run_A_batch_0010
[12:10:29]   [CHECKPOINT] ckpt_run_A_main_step00015 at ledger_offset=15
[12:10:33]   [CHECKPOINT] ckpt_run_A_main_step00020 at ledger_offset=20
[12:10:34] [PASS] run_resumed successfully
[12:10:34] 
============================================================
[12:10:34] ============================================================
[12:10:34] [PASS] replay: All 5 replayed batches hash-matched original
[12:10:34] [PASS] replay_hash_matched: 5 batches verified
[12:10:34] 
============================================================
[12:10:34] ============================================================
[12:10:38] [PASS] branch_forked
[12:10:38] 
============================================================
[12:10:38] ============================================================
[12:10:38] [PASS] opus_audit_trail: 12 decisions logged, 2 floor overrides
[12:10:38] [PASS] opus_decisions_recorded
[12:10:38] [PASS] mixture_compliance: Curriculum stages executed, floors enforced
[12:10:38] [PASS] mixture_compiled
[12:10:38] [PASS] learning_trace: Loss linked to 18 source documents
[12:10:38] [PASS] learning_trace: per-document loss recorded
[12:10:38] 
============================================================
[12:10:38] ============================================================
[12:10:38] [PASS] throughput: packing=82.7%, avg_tps=624
[12:10:38] [PASS] performance_measured
[12:10:38] 
============================================================
[12:10:38] ============================================================
```