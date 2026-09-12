# Completed selection — 2026-09-12

Final C0/C1 confirmation retains CONTROL: LR 0.0006, base 70/20/7/3.
All five jobs and the failed-smoke history are consolidated in
`runs/pre_campaign/v2-advisory`; the 3070 runtime ledger remains separate.
See [return review](C0C1_RETURN_REVIEW.md) and
[successor baseline integration](../g4/SELECTED_RECIPE.md). The experiment
commands below are historical/operator reference; no additional selection job is pending.

# Bounded pre-campaign experiments

Approved 2026-09-09. This is the active experiment plan; the earlier seven-run
P1–P8/F1–F2 draft is retired. [Package verification passed](VERIFICATION.md):
795 tests, CPU preparation, and independent Sol/Terra review for the original package.
The later [advisory-runtime amendment](ADVISORY_RUNTIME.md) removes hard time limits;
use its updated source and operator guide. No
production experiment result, recipe winner, or G4 pass exists yet.

## What this stage accomplishes

Compare the current recipe with one higher learning rate and one more educational
data mixture. Confirm only the strongest qualifying change with a second random
seed. Carry the settings and evidence into G4, then train the main model from
fresh weights at G5. Experiment weights never initialize the main run.

The primary goal is a strong model with defensible compute, speed, and memory
figures. Each possible job trains the actual49M architecture for382 updates /
100,139,008 loss tokens. Three initial jobs cost300,417,024 tokens; the maximum
five cost500,695,040, excluding engineering probes and failed/repeated work.
These are exact token budgets, not predictions of runtime or benchmark quality.

## Machines and jobs

| Computer | Jobs in order | Purpose |
|---|---|---|
| Your RTX4070 lane | S0, SLR, SMIX | Matched screen at seed1001 |
| Teammate's RTX3070 lane | C0, C1, only if selected | Matched confirmation at seed1002 |

S0 and C0 use LR0.0006 and source shares70/20/7/3. SLR changes only LR to0.001.
SMIX changes only the shares to85/5/7/3 (educational/general/math/narrative).
C1 repeats ONE selected change; it never combines the two candidates. Each
within-machine comparison keeps initialization seed, training length, batch,
schedule policy, model, precision, and validation protocol matched.

Historical device reports show RTX4070SUPER12GB and RTX3070Ti~8GiB. The user's
recollection differs for the latter; actual preflight detection takes precedence.
The source identifiers `rtx_4070` and `rtx_3070` name lanes, not exact marketing
models or assumed VRAM. The internal data label `dclm` is general FineWeb.

Teammate preparation can proceed while the screen runs. C0/C1 cannot start until
the complete screen selects a candidate. Each computer runs one GPU job at a time.
This is independent-job collaboration, not distributed training of one model.

## Decision and gate boundary

Only completed endpoints qualify; an early checkpoint cannot win. Each candidate
must reduce the control's token-weighted development NLL by at least0.3%, with no
more than1% relative regression on any of four protected slices. The largest
qualifying global reduction wins; an exact tie favors SLR. All slices must have
real, finite, complete, reconciled evidence. The second-seed candidate must meet
the same conditions against its own matched control.

The outcome is either a confirmed single change or the unchanged control.
Missing/failed/incomplete work is reported explicitly and cannot support a winner.
These are conservative engineering rules, not statistical-significance claims.
A100M-token result does not prove superiority at the1B-token main horizon.
Submission benchmarks and validation_final stay outside the selection loop.

After results, a new baseline contract/run identity and affected exposure inputs
must bind the decision. Review affected integration evidence, then resume G4:
sustained profile, local recovery, target takeover, measured budget, two-person
freeze. Earlier G4 sections1–2 tools are reusable; evidence is reused only when
its bound source/settings are unaffected. These scripts do not launch G5 or
automatically mark the original full-scale G3/G4 gates passed.

## Estimated time and operator choice

The user removed hard time limits. The current execution policy is
`configs/campaign/experiment_execution_v1.json`, superseding the historical budget
section in pre_campaign_v2. There is no six-hour lane allowance, twelve-hour cap,
five-minute smoke timeout, forecast-based rejection or automatic calendar cutoff.

Before launching, the wrapper prints a same-job smoke-based runtime forecast and
asks `Start this job? [y/N]`. Only yes starts it; --execute alone does not answer
the prompt. Estimates include a conservative margin and are not guarantees.
Before the first smoke there is no measured estimate; this is stated explicitly.
Once started, a job may exceed its estimate and will not be killed for elapsed time.

Actual elapsed time, failures and unknown interrupted durations remain recorded
in each machine's runtime ledger for efficiency reporting. Preserve old ledgers;
unresolved active executions still block concurrent work. September 12 remains
a planning target, and September 18 the intended submission date; calendar targets
do not terminate processes. The operator decides whether each job fits the schedule.
Use the new `runs/pre_campaign/v2-advisory` bundle so old evidence is preserved.

| Dates (UTC+8) | Reserved work |
|---|---|
| Sep9–10 | Implementation, review, transfer, machine checks |
| Sep11–12 | Screen and conditional confirmation |
| Sep13 | Remaining G4 checks and joint freeze |
| Sep14–15 | Fresh1B main baseline and checkpoint verification |
| Sep16–17 | Full evaluation, reproduction, README, video, screenshots |
| Sep18 | Early submission and public-access checks |
| Sep19–21 | Contingency |

Dates are allocations, not measured finish promises. The official deadline is
2026-09-21T23:45:00+08:00, verified on2026-09-09 from the
[official rules](https://gibc-v2.devpost.com/rules). The
[criteria](https://gibc-v2.devpost.com/) reward model quality and compute efficiency,
speed, memory, innovation, and documentation. No numerical weights were published
in the pages reviewed. Include experiment costs in project disclosure and show
main-model training costs separately. The referenced original
GIBC-Track01-Final-Plan-v5.md is missing from this checkout; the approved plan,
versioned scope, and existing gate/release contracts govern this implementation.

## Operator handoff

Use the [operator guide](OPERATOR_GUIDE.md) for exact per-machine commands and
the [LLM handoff](LLM_HANDOFF.md) to brief another assistant. Do not use commands
in the retired draft. Use branch `codex/g3.5-pre-campaign-experiments` for the
reviewed code and handoff. Corpus files, prepared bundles and run artifacts are
ignored by Git and must be transferred separately as the operator guide explains.
Read [implementation specification](IMPLEMENTATION_SPEC.md) for the complete
acceptance criteria. Only explicit operator-authorized GPU execution may launch
the future runs; this preparation task performs CPU tests and read-only real-input
verification only. No automatic commit, push, or publication is included.
