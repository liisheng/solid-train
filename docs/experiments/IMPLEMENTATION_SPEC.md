# Approved bounded experiment package — implementation specification

Execution amendment: the user subsequently removed hard time limits. The runtime
and budget requirements below are historical where they conflict with
`configs/campaign/experiment_execution_v1.json` and the current OPERATOR_GUIDE.md.
Estimates are advisory, launch requires an affirmative operator response, and
elapsed time/calendar targets never terminate a job. Experiment recipes and
screen/confirmation dependencies remain unchanged.

Recorded 2026-09-09T13:36:44+08:00. User approved the Astra proposal in this task.
This document specifies implementation and later operation; this task does not
authorize production training, scoring, publication, or a G4/G5 pass.

## Scope and exact design

Replace the unapproved seven-run proxy draft with a versioned successor
`configs/campaign/pre_campaign_v2.json`. Preserve v1 as historical, never execute
it through the new entrypoint. Update the uncommitted scope-v2 draft to identify
the approved successor, explicitly superseding its previous draft contents.
Keep existing frozen baseline/data/operation/evaluation contracts unchanged.

All five possible jobs use `configs/final_49m.json`, 49,658,368 parameters, fresh
weights, BF16, sequence length 1024, 262144 loss tokens/update, 382 updates =
100139008 loss tokens. WSD warmup4/stable340/decay38; final LR exactly zero.
Baseline batch8/accumulation32 is the starting choice for both lanes; a failed
capacity check blocks launch, never silently changes the batch. Use deterministic
training and fixed materialized schedules. Within each matched block, only the
declared treatment differs. A new capacity/configuration decision needs a version.

| ID | Lane | Seed | LR | Mixture | Dependency |
|---|---|---|---|---|---|
| S0 | rtx_4070 | 1001 | 0.0006 | base | prepared inputs and preflight |
| SLR | rtx_4070 | 1001 | 0.001 | base | S0 complete |
| SMIX | rtx_4070 | 1001 | 0.0006 | edu | SLR complete |
| C0 | rtx_3070 | 1002 | 0.0006 | base | complete screen selects a candidate |
| C1 | rtx_3070 | 1002 | chosen single change | chosen single change | C0 complete |

Base shares are FineWeb-Edu70/general-FineWeb20/math7/narrative3; edu85/5/7/3.
The internal source ID `dclm` means general FineWeb, not a new dataset. Preserve
math/narrative reference multisets between mixtures. One deterministic schedule
per mixture, independent of initialization seed, is sufficient. No new corpus,
tokenizer, architecture, optimizer, backend search, or later A/B/C campaign.

## Selection contract and coverage

Use the completed endpoint, not the best intermediate checkpoint. Score the full
existing validation_dev schedule once per validation event (753 references /
771072 targets on production inputs). Full-dev validation cadence remains
completed update1, multiples100, and completion. Save/recover cadence100/completion.
Produce actual token-weighted loss sums and counts for each of the four protected
slices. All four must be present, finite, positive-count, and reconcile with global
coverage. Missing/malformed coverage cannot qualify any candidate.

Production dev manifests currently have exactly one protected slice per shard:
dclm=broad_general, fineweb_edu=educational_science,
narrative=narrative_coreference, openwebmath=math_technical. Derive attribution
from each shard's actual declared protected_slices, never guess from source labels.
Reject ambiguous multi-slice shards in this bounded protocol. References stay
within a shard. Preserve RNG, data cursor, wrap state, and model mode in validation.

Candidate qualifies iff (control NLL - candidate NLL)/control NLL >=0.003 and
every slice (candidate NLL-control NLL)/control NLL <=0.01. Among qualifying SLR
and SMIX select the greater relative global reduction; exact ties favor SLR.
Never combine treatments. Repeat selected candidate against C0 at seed1002;
adopt it only if the same criteria pass. Otherwise choose the base/0.0006 control.
No confidence interval/significance/seed-SD claim: these are engineering rules.
No validation_final or submission benchmark input is permitted in this workflow.

Selection must recompute from verified checkpoint/metrics evidence, not accept
an edited selection JSON or latest file labelled PASS. Bind report to contract,
source, inputs, identities, checkpoint/metric hashes, exact counters and complete
coverage. C0/C1 enforce dependencies when launched, not just in documentation.
If an operation is interrupted, no completed report or candidate promotion occurs.
An explicit close-incomplete action may produce a conservative control handoff
with missing/failed jobs and reasons visible; it never marks the experiment gate
fully passed. No later baseline execution exists in these scripts.

## Budget, launch, and custody

Aggregate ceiling12 GPU-hours includes GPU preflight, smoke, full training,
validation/checkpoint overhead and failed/repeated invocations. Implement two
non-transferable6h lane allowances so offline machines cannot each spend12h.
Charge outer child wall time conservatively; distinguish this charged allocation
from optimizer timing and total project compute. CPU analysis/preparation costs
are measured separately. Include previous engineering costs in later disclosure.

One canonical budget ledger per lane/check-out under runs/pre_campaign, outside
individual bundles. Refuse mismatched hardware/lane, concurrent launch and ambiguous
unfinished reservation. Write an immutable reservation BEFORE GPU work, settle it
afterward; an uncertain crashed invocation consumes its reserved amount. Enforce
timeouts with child termination and durable exit evidence. Changing bundle or retry
directory must not reset allowance. Cross-machine transfers copy inputs/results,
not overwrite local lane ledgers. Calendar hard stop2026-09-12T23:59:59+08:00.
No new job may cross it or exceed remaining lane budget. No changing the cap via CLI.
Budget exhaustion -> incomplete/control, never select shorter unequal endpoints.

Plan/check actions do not allocate models or start GPU work. No args should show
help or a plan. Every GPU action requires explicit --execute. Real GPU preflight
is a bounded future action; preparation in this task is CPU/read-only production
verification. Detect real GPU/VRAM/RAM/environment; historical evidence is4070
SUPER12GB,3070Ti8GB. A readable arbitrary BF16 evidence file is not a stability
proof. Implement measured smoke readiness for exact source/config/data and actual
machine, with finite training metrics and >=10% memory headroom. Read-only inspect
may report support/capacity but cannot claim stability/sustained throughput.

Use unique experiment run IDs and directories; explicit resume keeps original
identity/horizon and verifies checkpoint; failed invocations and superseded metrics
remain accounted. No overwrite/delete to resolve a conflict. Paths and identities
must transfer across different checkout locations/CRLF settings. Verify production
data against existing trusted pins and schedules against deterministic rebuilding.
Avoid trusting hashes stored only alongside mutable values. Validate before launch,
not on each batch. Existing baseline trust checks must not be weakened by the new
experiment branch in train.py.

## Deliverables and acceptance

Integer exposure clarification from review: retain the control's largest-remainder
quotas (educational 68,454; general 19,558; math 6,846; narrative 2,934).
For the educational candidate, lock the math/narrative quotas and reference
multisets, then divide the remaining 88,012 references in the ratio 85:5 using
largest remainders: educational 83,122 and general 4,890. Do not round the four
candidate shares independently; that changes a protected reference count.

Luna runner author: contracts/shared job definitions/runner/budget/preflight/tests.
Luna validation author: optional experiment slice reporting in train.py and analyzer
plus focused tests. Coordinate stable APIs before edits; do not edit each other's
files. Root owns operator docs, status/continuity, integrated validation and review.
Sol and Terra independently review the finished combined change at medium reasoning.

Tests: normal and invalid job config, missing/empty/nonfinite slices, wrong relative
threshold denominator, same-block comparisons, deterministic tie, no combination,
conditional confirmation, tampered evidence/identity/config/schedule, wrong GPU,
default no-launch, concurrency/stale budget/failed-attempt costs/bundle reset,
deadline/timeout/resume, plus a tiny CPU real-trainer full-vs-resume fixture with
slice reporting. No production training/scoring or tests requiring ignored corpus.
Root builds Docker image, runs suite/lint/compile and real-input CPU prepare/plan.

Operator docs must give exact prerequisites, interpreter, source/data transfer and
hash checks, per-machine commands, one-job-per-GPU rule, dependencies, read-only vs
GPU actions, resume/retry/failure instructions, output descriptions, ledger custody,
analysis and closure, and self-contained prompts for another LLM. Never claim fit,
runtime or launch readiness before measured evidence. Tests passing mean package
verified; experiments and G4 remain unrun. Preserve earlier G4 tooling/evidence.

## Calendar and next gate

Targets in UTC+8: Sep9–10 implementation/review/preflight; Sep11–12 experiments;
Sep13 remaining G4; Sep14–15 fresh1B main baseline; Sep16–17 full evaluation,
reproduction/demo/docs; Sep18 submission; Sep19–21 contingency. Official deadline
Sep21 23:45 UTC+8, verified at https://gibc-v2.devpost.com/rules on2026-09-09.
These dates are allocations, not measured finish guarantees. Measure before launch;
if budget/calendar cannot fit, keep control and document incomplete optional work.

Experiment completion produces a selected-settings handoff. Integrate a successor
baseline contract/run identity and affected schedules/trust tests before continuing
G4 sections3–7. Reuse only unaffected section1–2 evidence. Experiment weights never
initialize G5. This task prepares that handoff mechanism, not a selected baseline
before results exist. Source plan GIBC-Track01-Final-Plan-v5.md is missing locally;
scope/gate/release contracts and the approved user plan govern this implementation.
