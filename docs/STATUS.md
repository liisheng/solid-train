# Project status

Human-readable coordination snapshot. The canonical gate definitions remain
[`configs/operations/measurement_v1.yaml`](../configs/operations/measurement_v1.yaml), and
evidence—not this page—determines whether a gate passes.

**Accepted scope:** [reduced campaign](REDUCED_CAMPAIGN.md), pinned in
`configs/campaign/submission_scope_v1.yaml`: 1B baseline, conditional 3–5B, one optional
comparison. The original gate table is retained; reduced scope does not imply its passes.

| Snapshot | Value |
|---|---|
| Last updated | 2026-09-07 14:30 UTC+8 |
| Current milestone | **G1 — Reduced-scope evidence complete; canonical G1 remains unpassed** |
| Current branch | **`g1-evidence`** |
| Last verified implementation | **`47dd13c`**: reduced-scope G1 pipeline and aggregate evidence implementation; focused checks passed. |

## Milestones

| Gate | Purpose | Working state | Exit condition |
|---|---|---|---|
| G0 | Foundation | REVIEW | Every G0 evidence key is audited PASS, including external teammate checks. |
| G1 | Data | ENGINEERING_RUNNING | Token/profile thresholds, provenance, reserved margin, isolation, decontamination, shards, schedules, and real-shard engineering checks all verify. |
| G2 | Tiny end-to-end | **BLOCKED_BY_G1 / NOT_RUN** | Canonical G2 still requires independent machine evidence and approvals; no G2 claim is made here. |
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
| G1-04 | IMPLEMENTED / REDUCED EVIDENCE ONLY | repository | Current local workspace | Acquisition, filtering, deduplication, decontamination, assignment, and publication paths are implemented and tested; reduced-scope artifacts are documented separately. |
| G1-05 | IMPLEMENTED / REDUCED CHECK | repository | Current local workspace | Scope-bound aggregate evidence is `runs/reduced_campaign/reduced_5pct_v1/aggregate.json`; canonical full-scale thresholds remain unpassed. |
| G1-06 | TOP-UP PASS / PUBLISHED | 4070 machine | Current local workspace | Corrected ordered bundle and original top-up bundle both verify; 143,934 decisions, 106,551 accepted rows, 114,478,342 assigned tokens, and zero boundary/slice isolation violations. |
| G1-07 | SHARDS/SCHEDULES BUILT; LOCAL RECOVERY PASS | repository | Current local workspace | `data/shards/slice_topup` and stable/validation/recovery schedules are materialized. Fresh recovery evidence passes exact state, update-input hashes, corruption rejection, and export/reload; shard aggregate/profile reconciliation remains deferred. |
| G1-08 | LOCAL PROFILE MEASURED | 4070 machine | Current local workspace | 1,883.02 post-warmup seconds measured on real shards: p10 53,339.55, median 64,831.86, p90 66,601.79 tok/s; weighted 62,785.89; 5.45 GiB peak. Other-machine profile remains NOT_RUN. |
| G1-09 | BLOCKED_BY_08 | Both | Either | Review the machine-readable evidence bundle and record the required corpus freeze approval. |
| G1-10 | REDUCED 5% SCOPE PASS | repository | Current local workspace | Aggregate PASS (69/69) at `runs/reduced_campaign/reduced_5pct_v1/aggregate.json`; first parser-bug failure preserved as `aggregate_failed_parser_bug.json`. This is scope-bound engineering evidence, not canonical full-scale G1. |

The 3070/64GB machine is the planned bounded-preprocessing host. The 4070 machine is the
planned canonical accepted-data/checkpoint mirror and primary trainer. Either teammate may
implement or review repository work; do not duplicate the same long-running job.

## Human decisions and blockers

- Bounded engineering pilot completed 64 final-model updates and limited evaluation:
  59,007 tok/s after warmup, validation loss 9.11 to 6.80. This flat-sampler pilot does
  not pass G2/G4. [Report and budget options](VERIFIED_SLICE_PILOT.md).
  No training/evaluation process remains; full budget/scope remains undecided.
- Evaluation mapping corrected in v2: explicit pinned WikiText-103, live smoke complete.
  Final scoring settings remain provisional; old v1 evidence remains historical.
- The corrected top-up publication is complete under
  `data/pipeline/slice_topup_ordered_output`; its evidence is
  `runs/reduced_campaign/slice_ordered.pipeline.json`. It records 110,098,650 stable-train
  tokens and 114,478,342 assigned tokens; all declared source selections pass. The original
  top-up state and bundle remain preserved.
- Shards and schedules are materialized under `data/shards/slice_topup` and
  `data/schedules/slice_topup`. The shard builder reports 369 PASS and one deferred
  streaming-mixture check; each schedule reports eight PASS and one deferred real-scale
  construction check. These deferred checks prevent a G1 pass claim.
- Fresh local recovery evidence at `runs/reduced_campaign/recovery_verified_inputs_v2/evidence.json`
  passes exact state, all eight per-update input hashes, deliberate corruption rejection, and
  export/reload checks. The completed local real-shard profile at
  `runs/reduced_campaign/profile_verified_inputs_v1/measurement.json` reports `MEASURED` with
  no violations: 1,883.02 post-warmup seconds, 1,901.71 optimizer seconds, 1,930.81 wall seconds,
  p10/median/p90 53,339.55/64,831.86/66,601.79 tok/s, weighted 62,785.89 tok/s, and 5.45 GiB
  peak allocation. Other-machine verification and teammate approvals remain unconfirmed. The
  No main baseline training has started.

- The reduced 5% expansion now has a scope-bound aggregate PASS at
  `runs/reduced_campaign/reduced_5pct_v1/aggregate.json`: 550,094,903 selected stable tokens,
  fresh recovery, and a new real-shard profile with 1,901.95 optimizer seconds. The first
  aggregate failure is retained as `aggregate_failed_parser_bug.json`; a bounded manifest-parser
  compaction defect was fixed and the focused verifier/aggregate tests passed. This does not claim
  the original 11B-token G1 threshold, other-machine verification, baseline training, G2/G4, or
  release approval.

- V3 supersedes v2's short-answer over-quarantine; the original state and migration
  evidence are preserved. Short standalone fields and paraphrases remain limitations.
  Re-measure runtime and accepted-data yield; earlier 5–7-hour / 22–29-day forecasts
  describe the superseded v2 scan and must not be reused as v3 predictions.
- Official submission deadline verified on Devpost: **September 21, 2026, 23:45 UTC+8**.
  The September 5 G4 target is unmet. Full unchanged campaign timing is unsupported;
  sustained real-shard training throughput and a viable accepted-data yield remain required.
- The saved 1% state and Hugging Face cache are local, ignored artifacts. A replacement
  agent on this machine must check for an existing `scripts/prepare_corpus.py` process and
  monitor it rather than launching a duplicate. A different machine needs an explicit
  state/cache transfer or a fresh run.
- Saved states: `data/pipeline/slice_1pct_v3/state.sqlite` (original) and
  `data/pipeline/slice_1pct_v3_topup/state.sqlite` (top-up). The viewer default still shows the
  retained v2 state; pass an explicit state path when inspecting either v3 state. The completed
  top-up evidence and corrected publication are the sources for the current shard run.
- Review the measured disk/runtime forecast before G1-07 starts. No forecast may be replaced
  by an invented estimate.
- Full-scale runtime remains a blocker until v3 slice measurements support a forecast.
  Reassess controlled multiprocessing after the slice evidence, before G1-07.
- Both teammates must approve the corpus freeze at G1-09. A checked task is not a substitute
  for its hashes, tests, and approval event.

## Update rule

After a meaningful checkpoint, update only the current milestone, active queue, blockers,
and last verified commit. Keep detailed technical history in `.agent/CONTINUITY.md`. Never
mark a gate complete because code exists; mark it complete only when its canonical evidence
keys pass.
