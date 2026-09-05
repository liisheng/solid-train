# Project status

Human-readable coordination snapshot. The canonical gate definitions remain
[`configs/operations/measurement_v1.yaml`](../configs/operations/measurement_v1.yaml), and
evidence—not this page—determines whether a gate passes.

| Snapshot | Value |
|---|---|
| Last updated | 2026-09-05 20:02 UTC+8 |
| Current milestone | **G1 — Data** |
| Current branch | **`g1-evidence`** |
| Last verified implementation | **G1 pipeline checkpoint at branch tip** (587 tests passed) |

## Milestones

| Gate | Purpose | Working state | Exit condition |
|---|---|---|---|
| G0 | Foundation | REVIEW | Every G0 evidence key is audited PASS, including external teammate checks. |
| G1 | Data | RUNNING | Token/profile thresholds, provenance, reserved margin, isolation, decontamination, shards, and schedules all verify. |
| G2 | Tiny end-to-end | BLOCKED_BY_G1 | Real-shard train/resume/corruption/export checks pass on both machines as required. |
| G3 | Minimum campaign | BLOCKED_BY_G2 | P1–P4/P8 and F1/F2 evidence exists and frozen decision rules are applied. |
| G4 | Final freeze | BLOCKED_BY_G3 | Two-person freeze, real-shard throughput, and takeover rehearsal pass. |
| G5 | Campaign | NOT_RUN | Stable/fallback lineage, confirmations, counters, and frozen artifacts reconcile. |
| G6 | Release | NOT_RUN | Fresh evaluation, exports, documentation, public access, and both approvals pass. |

## Active G1 queue

| ID | State | Owner | Preferred machine | Work / completion evidence |
|---|---|---|---|---|
| G1-01 | COMPLETE | repository | Either | Final 12,288-token tokenizer and verification evidence. |
| G1-02 | COMPLETE | repository | Either | Pinned benchmark quarantine inputs and digest/count evidence (`a7475b3`). |
| G1-03 | COMPLETE | repository | Either | Disk-backed streaming shard builder, atomic publish, and tests (`6aadedc`). |
| G1-04 | IMPLEMENTED / SLICE_NOT_RUN | repository | 3070 preferred to run | Restartable pinned-source acquisition, filtering, global deduplication, indexed decontamination, assignment, atomic publication, and reason-coded evidence are implemented and tested. A real slice must prove them. |
| G1-05 | IMPLEMENTED / FINAL_NOT_RUN | repository | Either | Bounded-memory aggregate verification of source shares, totals, reserved margin, profile selection, shard integrity, and isolation evidence is implemented and tested. Final artifacts do not exist yet. |
| G1-06 | RUNNING — BOTTLENECK FOUND | 4070 machine | Current local workspace | Decontamination is active but measured at only about 1–2 documents/minute. Preserve the checkpoint; optimize and equivalence-test indexed matching before relying on a completion forecast. |
| G1-07 | BLOCKED_BY_06 | Unclaimed | 3070 preprocessing / 4070 mirror | Build the full accepted corpus and pass token, provenance, contamination, and split-isolation gates. |
| G1-08 | BLOCKED_BY_07 | Unclaimed | 4070 canonical mirror | Produce and verify final source-tagged shards and deterministic schedules. |
| G1-09 | BLOCKED_BY_08 | Both | Either | Review the machine-readable evidence bundle and record the required corpus freeze approval. |

The 3070/64GB machine is the planned bounded-preprocessing host. The 4070 machine is the
planned canonical accepted-data/checkpoint mirror and primary trainer. Either teammate may
implement or review repository work; do not duplicate the same long-running job.

## Human decisions and blockers

- The active 1% state and Hugging Face cache are local, ignored artifacts. A replacement
  agent on this machine must check for an existing `scripts/prepare_corpus.py` process and
  monitor it rather than launching a duplicate. A different machine needs an explicit
  state/cache transfer or a fresh run.
- Review the measured disk/runtime forecast before G1-07 starts. No forecast may be replaced
  by an invented estimate.
- The current single-process decontamination path is not viable for the full run. Prefer
  targeted indexed digest probes and batched commits; measure that change before adding
  multiprocessing or proceeding to the 2–5% slice.
- Both teammates must approve the corpus freeze at G1-09. A checked task is not a substitute
  for its hashes, tests, and approval event.

## Update rule

After a meaningful checkpoint, update only the current milestone, active queue, blockers,
and last verified commit. Keep detailed technical history in `.agent/CONTINUITY.md`. Never
mark a gate complete because code exists; mark it complete only when its canonical evidence
keys pass.
