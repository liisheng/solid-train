# Next-run operator runbook

Status: PREPARATION_ONLY. Read [readiness](README.md), [specification](CANDIDATE_SPEC.md)
and [implementation requirements](PREPARATION.md). There is no verified 3B launch command
yet. This runbook defines what must be produced and how to operate the later run.

## Safe inspection commands available now

Run from `D:\SWE\benchmark-50m-lm` in PowerShell. These inspect existing state or CLI
help; they do not launch training or acquisition:

```powershell
git status --short
git branch --show-current
Get-Content .agent/CONTINUITY.md -TotalCount 20
Get-Content docs/next_run/README.md
Get-FileHash runs/reduced_campaign/reduced_baseline_v2/run/baseline_export.pt -Algorithm SHA256
Get-FileHash data/tokenizer_final/tokenizer.json -Algorithm SHA256
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py --help
.\.venv\Scripts\python.exe scripts/prepare_corpus.py --help
.\.venv\Scripts\python.exe scripts/build_schedule.py --help
.\.venv\Scripts\python.exe scripts/verify_shards_streaming.py --help
```

The four help commands were executed successfully while preparing this handoff.
Do not substitute a new output directory into historical launch scripts: their
contracts and defaults still describe the 1B baseline or old 5% corpus.

## Artifacts required to declare TRAINING_READY

Every item must have concrete content and a hash-bound verification receipt:

| Artifact to create | Required contents |
| --- | --- |
| Successor scope and recipe | Exact specification, initialization, adoption policy and old/new lineage boundary |
| Data readiness report | License/revision/filter/index identities, per-source counts, isolation, distinct positions and reuse checks |
| Exposure bundle | Two components, all quotas, order/hash/cursor semantics and 3,000,238,080-token reconciliation |
| Baseline development reference | Complete global and four-slice NLL sums/counts; model/tokenizer/dev hashes |
| Generation diagnostic set | 40 prompts, rubric and exact seeds/decoding settings; no benchmark input |
| Code and runtime verification | Reviewed source hashes, container/test/lint reports and measured CUDA checks |
| Recovery and profile reports | New-input uninterrupted/resume equality, rollback/corruption coverage, measured resources/time |
| Budget and backup plan | Data/preflight/training/evaluation/retry costs, free space, destination, custodian and cadence |
| Freeze/approval receipt | Exact bundle digest and applicable operator approval; preserve old approvals |
| Launch package | Actual command file, controller, observer, expected endpoint and postverification script |

The generated launch command must explicitly specify the successor config, new run
directory and execution action. Initial launch must have no resume checkpoint.
Generate resume/export commands from the same verified identity. Parse and test the
actual commands before use; do not manually bypass guards with raw `train.py` options.

## P3: launch and observation

Refresh actual Python/CUDA versions, GPU/host memory, disk space, competing jobs and
machine availability. Check previous controller receipts and OS process identity before
starting another job. A stale PID number alone is not proof a job is running or dead.
The new namespace must be empty for fresh launch; otherwise inspect it before action.

Disclose the measured estimate, remaining calendar buffer and known uncertainty. The
old baseline projects about 13.1 h training / 16.4 h with a 25% planning margin, excluding
data and verification; these are advisory estimates until remeasured. Confirm the
organizer deadline before making a calendar commitment. Do not enforce the retired
experiment cutoffs or terminate merely because an estimate has elapsed.

Launch one controller with durable active-process and final receipts. Give the user
the observer command immediately. Record child PIDs, argv, start/end timestamps,
stdout/stderr, exit codes and outer durations. Keep nested optimizer/harness times
separate so accounting never double-counts them. Do not ask the observer to rerun a job.

Observe finite loss/gradient norms, monotonic tokens/cursor, disk and memory headroom,
checkpoint cadence, complete dev coverage and expected LR. Do not adjust LR, mixture
or horizon in response to benchmark scores or informal generations mid-run.

## Recovery and backups

On a failure, retain logs and failed outputs. Identify whether setup, data, trainer,
export or postverification failed. Check process liveness before any retry. Resume only
from a verified durable checkpoint with identical recipe/source/input identities and
restored optimizer/RNG/cursor; account for replay and archived metric tails.

A changed recipe, corrupted checkpoint or unknown custody requires diagnosis, not a
forced resume. Never initialize this candidate from the 1B completed checkpoint. Verify
available fallback checkpoints and keep the old baseline immutable throughout.

Set a concrete backup destination and cadence before launch. Proposed cadence: a
verified prelaunch bundle, every 1,000 completed updates at a stable checkpoint boundary,
and completion. If a controller cannot safely snapshot rotating files, implement that
before using this cadence. Verify copied hashes and retain previous generations; test
an isolated restoration. Same-disk copies do not provide protection against disk failure.
Storage estimates must include transient checkpoints, backup generations, datasets,
indices and logs. Do not delete old evidence to make room without explicit direction.

## P4: complete, verify and select

Training exit zero is necessary but insufficient. Require 11,445 completed updates,
3,000,238,080 loss tokens, cursor 2,929,920, final LR zero, complete dev coverage,
finite weights and valid lineage. Independently verify/export/reload/recount the endpoint
and reconcile all invocations, recovery, metrics and hashes. A rehearsal endpoint is
never an eligible candidate. Perform completion backup and record custody.

Apply the exact development rule in CANDIDATE_SPEC.md to unrounded endpoint values.
Record the decision before benchmark execution. If rejected or incomplete, retain the
baseline and report why; no automatic further experiment. Generation diagnostics remain
reported limitations and cannot secretly replace the fixed decision rule.

## P5: evaluate and release

For an accepted candidate, run full evaluation once in a verified fresh environment.
Use the existing five-task protocol and all document-coverage checks; retain full-run
sample records, dataset revisions, raw metrics, standard errors, bundle hashes and
actual costs. Do not rerun the baseline if its comparable verified result already exists.
If official settings change, compare both artifacts under one new immutable protocol.

Report the current exposed benchmark history. Do not choose a different checkpoint
after seeing candidate scores. Complete the [G6 release checklist](../g6/README.md),
including package contents, public accessibility and exact-hash human approvals.
Include all prior experiments, failed runs and the baseline in total campaign costs;
show the candidate's marginal cost separately. Keep the AI-assistance disclosure accurate.

## Final handoff format

Each stage ends with: scope and status; source/config/data hashes; evidence paths;
checks and failures; elapsed and projected time; live-process state; owner; next command.
Use PASS only for completed evidence, NOT_RUN for missing measurements, and BLOCKED
with a concrete owner/action. Update STATUS.md and continuity without copying raw logs.
