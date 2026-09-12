# Section 6 — Confirm the baseline horizon and dated budget

## Goal and inputs

Turn verified measurements into a realistic execution/recovery/evaluation budget for
the existing 1B baseline. Read the common handoff, profile/recovery evidence and
[single-machine amendment](SINGLE_MACHINE_SCOPE.md),
the accepted scope, baseline recipe and operations `horizon` rules. No training or full
benchmark scoring is included in this planning section.

## Work

1. Reconfirm 3,815 updates, 1,000,079,360 loss tokens, fresh seed 1337 and WSD 38/3395/382.
   Record this as the already fixed engineering recipe, not a new LR-search winner.
   The original September 5 freeze target is historical. Confirm the actual submission
   cutoff/year/timezone and machine availability before publishing a dated schedule;
   the operations contract records a September 17 fit constraint, not proof of today's
   organizer rules. Never silently alter the frozen contract to resolve a date conflict.
2. Calculate optimizer time from consumed tokens divided by measured weighted rate.
   Add preparation, all 40 full-dev events, 39 scheduled recovery saves, best-save
   allowance, completion handling, verification/export and other measured overhead.
   Use the actual intervals measured and explain extrapolation. Avoid double-counting
   phases already included in an end-to-end rate.
3. Show central and conservative scenarios with their input provenance. Use throughput
   percentiles in the correct direction; high throughput is not pessimistic runtime.
   Report measured time, extrapolated time and discretionary reserve separately.
   Retain complete costs for failed, interrupted and repeated work outside canonical
   progress counters. A GPU-hours or total-campaign claim needs those costs.
4. Budget checkpoint storage/atomic-write transients, logs, verification copies and
   export alongside training. Schedule training, evaluation and export sequentially
   on this RTX 4070 SUPER. Account for background workloads, local recovery from
   section 4, backup costs and downtime reserve. Section 5 and target-machine timing
   are not prerequisites under g4_scope_v2; no alternate-machine capacity is assumed.
5. Reserve evaluation using the contract's measured p90 evaluation runtime × exact
   candidate count × 1.25 when those measurements exist. The one-example CPU smoke is
   not full CUDA/BF16 timing. Do not export an early engineering checkpoint as a baseline
   to manufacture this input. If full-runtime evidence is absent, explicitly label any
   provisional reserve assumption, leave the measured claim blocked, and describe a
   separate authorized calibration task with an eligible artifact and complete task
   coverage. Do not start scoring from this brief or silently relax the formula.
6. Produce a calendar with start/end windows, timezone, owner, dependencies and recovery/
   evaluation margins. A calendar based on unavailable machine time or unmeasured inputs
   is provisional. Name exactly which condition must change for a justified fit claim.
7. Keep optional 3–5B work and at most one comparison conditional on securing the baseline
   export and full evaluation. Do not lengthen a decayed baseline; any later horizon
   change needs a new declared run identity. No optional work is launched here.

## Deliverables and completion

Write `docs/g4/BASELINE_BUDGET.md` plus a small machine-readable calculation with units,
formulas, source evidence paths/hashes, candidate count and assumptions. Include a table
of missing inputs/owners/next actions. State READY, conditional, or blocked in prose;
for contract requirements use PASS/FAIL/BLOCKED/NOT_RUN as applicable.

Completion means the budget is checkable and its limits explicit. It does not imply a
measured full-campaign forecast or gate pass when a required input remains absent.
Pass the report and open prerequisites to section 7; no blanket claim that everything fits.

## New-chat prompt

```text
Execute only G4 section 6, docs/g4/06-budget.md. Read the common G4 handoff/index
and measured profile/recovery evidence plus configs/operations/g4_scope_v2.yaml.
Skip section 5 under the approved single-machine scope. Build a traceable dated budget
for the fixed 1B baseline, with full overhead, storage, recovery and evaluation
reserve. Distinguish measurements, extrapolations and missing inputs. Do not
train, score benchmarks, extend the horizon or start another section. Record
BASELINE_BUDGET.md, calculation evidence and the final-review handoff.
```
