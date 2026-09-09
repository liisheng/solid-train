# Accepted submission scope — 2026-09-06

The user accepted a 1B-token baseline, conditional 3–5B target, at most one optional
comparison, and retention of existing sources. The immutable record is
`configs/campaign/submission_scope_v1.yaml` with its digest sidecar. It reduces the
original experiment campaign; correctness checks and honest gate reporting remain.

**Current handoff:** Reduced G2 is verified under `configs/operations/g2_scope_v2.yaml`.
G3 section 6 has passing local integration/recovery evidence and 675 passing tests;
recipe readiness remains blocked by unavailable Docker and Ruff checks. Use
[integration evidence](g3/INTEGRATION_EVIDENCE.md), [G4_HANDOFF.md](g3/G4_HANDOFF.md),
and [STATUS.md](STATUS.md). The current aggregate is
`runs/verification/g3-section6-integration-final-v2/reduced_g3_report.json`.
The slice-topup paths below remain historical preparation evidence.

## Execution

1. **Complete.** Top up math, narrative and textbook for the 1% slice and recompute
   duplicate-safe selections. The corrected ordered publication is
   `data/pipeline/slice_topup_ordered_output`, with evidence in
   `runs/reduced_campaign/slice_ordered.pipeline.json`; it preserves the original top-up
   ledger content and passes all declared source selections and split-isolation checks.
2. **Partly complete.** The corpus/ledger, isolation and quotas pass. Streaming shards and
   deterministic stable/validation/recovery schedules are materialized under
   `data/shards/slice_topup` and `data/schedules/slice_topup`. Aggregate mixture/profile
   reconciliation remains deferred.
3. **Complete locally.** `scripts/verify_real_training.py` passed from fresh evidence at
   `runs/reduced_campaign/recovery_verified_inputs_v2/evidence.json`: uninterrupted versus
   pause/resume state is exact, all eight update-level schedule-reference hashes match,
   a deliberately corrupted checkpoint is rejected, and export/reload passes. The prior
   cross-machine wording is historical frozen-v1 evidence; active G2 uses the any-machine
   amendment and `g2_report_local.json`.
4. **Complete locally.** `scripts/profile_real_training.py` recorded `MEASURED` at
   `runs/reduced_campaign/profile_verified_inputs_v1/measurement.json`: 1,883.02 post-warmup
   seconds (451 samples), p10/median/p90 53,339.55/64,831.86/66,601.79 tok/s, weighted
   62,785.89 tok/s, 1,901.71 optimizer seconds, 1,930.81 wall seconds, and 5.45 GiB peak
   allocation. All 455 runtime update hashes matched the five-pass real-shard schedule. Use p10
   for planning. The source-local profile is efficiency evidence only; machine identity is not
   an active G2 gate.
5. **Complete for reduced preparation.** The isolated expansion contains 550,094,903 distinct
   stable tokens (minimum 500M), with declared shares and scaled reserved/validation targets;
   the scope-bound reconciliation is `runs/reduced_campaign/reduced_5pct_v1/aggregate.json`.
6. **Recipe implemented; readiness blocked.** The fixed baseline recipe and bounded
   recovery are verified locally; G4 must resolve verification-tool blockers and complete
   sustained profiling, applicable recovery/takeover, and teammate review before G5 trains
   3,815 updates = 1,000,079,360 loss tokens. Secure
   an evaluated export before optional expansion/comparison. Conditional 3–5B work needs
   a declared new horizon/run identity and sufficient data/time; never silently extend a
   decayed checkpoint. Original 11B G1 and multi-experiment G3 are not claimed passed.

Current engineering checkpoint: the reduced 5% expansion aggregate passes all 69 reduced-scope
checks at `runs/reduced_campaign/reduced_5pct_v1/aggregate.json`. Fresh local recovery is in
`runs/reduced_campaign/reduced_5pct_v1/g2_recovery_luna_fix3/evidence.json`, source-export fresh
process inference is in `runs/reduced_campaign/g2_source_local.json`, and the combined report
is `runs/reduced_campaign/g2_report_luna_final.json` (`REDUCED_SCOPE_G2_PASS`). This is
`G2_PASS_UNDER_AMENDED_SCOPE`; canonical full-scale G1/G2 and G4/campaign promotion remain
unclaimed. No main baseline training has started.

## Training controls

The baseline uses the section-2 two-component finite exposure, with exact aggregate source
quotas and one absolute cursor, plus complete development validation. Its unchanged
contract is `configs/training/baseline_reduced_v1.yaml`; generated identities and current
checks live in the section-6 external report. The repeated-schedule controls below remain
available for the earlier engineering workflows.

The one-pass schedule stays immutable. `--train-epochs N` explicitly repeats it a finite
number of times under `FINITE_IDENTICAL_SCHEDULE_PASSES_V1`. A composite hash binds the
base schedule and repeat count to run identity. One absolute checkpoint cursor spans all
passes; exhaustion fails closed, and supply is checked before model allocation. Ordering
repeats identically per pass. Report consumed tokens separately from distinct corpus tokens.

`--stop-after-updates N` saves at an accumulation boundary without changing `--steps`.
`--stop-after-training-seconds S` is a profiling control measured within that invocation.
Resume with the original horizon and omit stopping controls to continue. Evaluation uses
absolute update numbers so resume does not insert an extra validation event.

## Evaluation runtime binding

The reduced baseline uses the section-5 runtime binding on top of frozen evaluation v2.
It enforces all five dataset revisions against decontamination v3, verifies installed
task implementation identities, and records the binding in the baseline run identity.
See [evaluation evidence](g3/EVALUATION_EVIDENCE.md) for exact rehearsal and later G6
commands, source custody, checks, and limitations. Smoke mode requires an integer limit
of 1–100 examples per task; full evaluation is a separate explicit action.

### Historical v2 correction

`evaluate.py` defaults to sidecar-pinned `evaluation_provisional_v2.yaml` and all five tasks.
The misleading legacy `wikitext` alias is refused. `wikitext103` explicitly loads
`Salesforce/wikitext`, `wikitext-103-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`.

V2 scores nonempty raw test paragraphs independently, without detokenization, with rolling
1,024-token context. Word denominators use whitespace splitting; byte denominators use
raw UTF-8 length. These are provisional choices, different from the old WikiText-2 task.
Remove smoke limits for full evaluation. The live 100-row v2 smoke completed, and its
bundle passed 12 integrity checks. V1 evidence remains preserved and superseded.

## Recovery

Original: `data/pipeline/slice_1pct_v3/state.sqlite`.
Top-up copy: `data/pipeline/slice_1pct_v3_topup/state.sqlite`.
Top-up helper/logs: `runs/reduced_campaign/topup.py`, `topup.stdout.log`, `topup.stderr.log`.
Top-up evidence: `runs/reduced_campaign/slice_topup.pipeline.json`.
Corrected publication/evidence: `data/pipeline/slice_topup_ordered_output` and
`runs/reduced_campaign/slice_ordered.pipeline.json`.
Engineering launcher/logs: `runs/reduced_campaign/finish_engineering.py`,
`engineering.stdout.log`, `engineering.stderr.log`, and the per-stage logs in the same
directory.

The helper retained original per-document checks in `saved_decontamination` while new global
dedup ran, then restored them before checking new documents. It reset downstream selection in
the copy to account for cluster merges. Original state remains intact. Do not launch a duplicate
engineering chain while `finish_engineering.py` is active.

The active G2 result is valid on any compatible machine when the exact fresh-process and
real-shard commands in `G2_HANDOFF.md` pass. Historical teammate-transfer notes remain
preserved in earlier evidence and do not block the amended active result.
