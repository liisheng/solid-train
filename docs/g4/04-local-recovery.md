# Section 4 — Prove current-source local recovery

## Goal and inputs

Show that the final implementation can interrupt and resume engineering training without
changing its durable state or hiding repeated work. Read the common handoff, section-3
evidence, `docs/g3/REVIEW_FIXES.md`, and the recovery portions of
`scripts/run_g3_integration.py`, `train.py`, `src/tinybench_lm/checkpointing.py`,
`src/tinybench_lm/metric_ledger.py`, and their focused tests.

Historical G3 eight-update versus four-plus-resume evidence is useful background, but it
predates later repairs. Section-1 CPU tests and input reads are not successor CUDA proof.
Reuse evidence only after explicitly comparing source, inputs, environment and scenario.

## Work

1. Write a scenario/coverage table before running anything: valid existing evidence,
   uncovered current-source resume, crash/log rollback, corruption, and timing custody.
   Choose the smallest bounded real-model BF16 runs covering the missing scenarios.
   The existing integration harness defaults to eight versus four-plus-resume; inspect
   and adapt its applicability rather than treating the historical report as current.
2. Use distinct engineering directories and fresh, identical initial conditions for
   uninterrupted and resumed paths. Preserve the full recipe horizon; bound execution
   with absolute completed-update stops. A resumed stop must exceed its checkpoint.
3. Compare actual model and optimizer tensors, RNG/scaler state, best-validation state,
   schedule cursor/order, run identity, frozen hashes and counters. Do not compare only
   abbreviated tensor representations or scalar loss. Keep exact same-environment
   recovery assertions; report a failure rather than weakening them after the fact.
4. Exercise the repaired ledger path with metrics beyond the durable checkpoint, and
   verify that superseded work is preserved, the active prefix is canonical, and retry
   does not duplicate records. Isolate fault injection to task-owned rehearsal processes
   and copies; never corrupt the retained primary checkpoint or another running job.
5. Verify corruption rejection and early-export rejection, reusing valid negative
   fixtures where applicable. Preserve per-invocation timing and outer wall time for
   interrupted/replayed work; keep canonical progress separate from total compute spent.
6. Review any fixes and run relevant code checks. Identify whether a fix invalidates
   section-3 timing; repeat only the affected measurement if necessary, in a later
   continuation of the appropriate section rather than silently relabeling evidence.

## Deliverables and completion

Write `docs/g4/RECOVERY_EVIDENCE.md`: scenario table, exact commands/source/input hashes,
durable comparisons, archive/counter/timing results, artifact custody and remaining gaps.
Name the verified checkpoint available for the target-machine rehearsal. It remains an
engineering checkpoint and cannot initialize the submission baseline.

PASS covers local recovery only. Cross-machine transfer/resume is section 5. No sustained
reprofile, full scoring, optional experiment, main baseline or human approval claim here.

## New-chat prompt

```text
Execute only G4 section 4, docs/g4/04-local-recovery.md. Read the common G4
handoff/index and the targeted recovery evidence/code. Reuse applicable checks,
then run only the bounded current-source real-model recovery scenarios needed.
Verify exact state, corruption rejection, ledger rollback and complete timing.
Do not start takeover, a sustained reprofile, the baseline or full scoring.
Record RECOVERY_EVIDENCE.md and a short handoff for section 5.
```
