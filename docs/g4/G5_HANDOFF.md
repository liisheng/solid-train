# G5 handoff — prepared, launch blocked

Recorded 2026-09-12. This document paraphrases the accepted contracts. Review
`FREEZE_REVIEW.md` and verify `FREEZE_BUNDLE.json.sha256` before using it.
G4 remains BLOCKED; these commands are instructions for a separately authorized G5.

## Fixed identity and prerequisites

Use branch `codex/g4-selected-recipe`, HEAD
`d1080576789794cb35f8839d6789f540223b7653`, plus the exact freeze-manifest overlay.
The measured 172-file source set is unchanged from verified implementation
`6293354814a619c7b318b9d15b694c83d871083c`. Documentation and g4_scope_v2 are
separately bound; a commit alone does not identify the working tree.

Fresh CONTROL v2: seed1337, 49,658,368 parameters, BF16, LR0.0006,
batch8 x accumulation32 x length1024, WSD38/3395/382,
3815 updates, 1,000,079,360 loss tokens, final cursor976640 and LR0.
Runner identity `baseline-f23398e32dc3100e`; training semantics
`run-146ddcb9c11da742`. These identities are shared with engineering runs;
the fresh directory and step-zero provenance distinguish this baseline attempt.
Never initialize it from experiment, pilot or G4 weights.

Before launch, close the blockers in the review, regenerate the bundle if its
inputs/policies change, and obtain two distinct human approvals of that exact digest.
Verify every manifest entry, refreshed environment/loader binding, GPU identity,
no existing training process, and at least30GiB free on D. Reduce unnecessary
background load with the operator: measured minimum host RAM was98.574MiB.
Do not change backend, batch, horizon or precision to work around a failed check.

The directory below was absent during section7. Check again immediately before
launch. If present, inspect provenance, receipts and processes; never overwrite or
silently resume it. The plan command verifies inputs but does not train:

```powershell
Set-Location D:\SWE\benchmark-50m-lm
$env:PYTHONPATH = 'src'
if (Test-Path runs/reduced_campaign/reduced_baseline_v2/run) { throw 'Inspect existing baseline lineage before proceeding' }
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py launch --config configs/training/baseline_reduced_v2.yaml --run-dir runs/reduced_campaign/reduced_baseline_v2/run
```

Section7 executed that plan successfully; the exact child argument array is in
`runs/verification/g4/section-07/20260912-freeze-01/launch-plan-access.json`.
It has no resume or engineering stop. The initial sandbox-access failure and
successful access-enabled retry are retained separately.

## Executable form for the G5 operator

Only after readiness and separate G5 authorization, execute the same command with
`--execute` appended. The runner does not enforce the external G4 human-approval
matrix, so successful CLI preparation is not permission to launch.

```powershell
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py launch --config configs/training/baseline_reduced_v2.yaml --run-dir runs/reduced_campaign/reduced_baseline_v2/run --execute
```

The G5 launcher must retain exact arguments, UTC start/end, process identity,
stdout/stderr, exit code and outer wall receipt, including every failed/repeated
attempt. Run one training process; return its PID and observer to the operator.
Once metrics exist, this read-only observer can be stopped without stopping training:

```powershell
Get-Content D:\SWE\benchmark-50m-lm\runs\reduced_campaign\reduced_baseline_v2\run\metrics.jsonl -Tail 5 -Wait
```

Full-dev covers753 references/771072 targets/95 batches at update1, multiples100,
and3815 (40 events). Recovery is every100 and completion (39 saves), plus best saves.
Keep metrics, phase histories, invocation archives, runner identity, run config,
step-zero provenance and checkpoint sidecars. Reconcile canonical updates/tokens
separately from archived replay. Do not sum nested optimizer/child/controller timers.

## Stop, recovery and backup policy

Stop and preserve evidence on nonfinite values, OOM, identity/hash/cursor drift,
missing checkpoint verification, storage exhaustion or failed correctness checks.
On interruption, confirm process exit before any restart. An emergency stop can
lose work after the last verified checkpoint. Do not erase metric tails or archives;
the tested trainer reconciles them and charges replay. Resume only this lineage,
same source/config/environment, after checksum and identity verification:

```powershell
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py launch --config configs/training/baseline_reduced_v2.yaml --run-dir runs/reduced_campaign/reduced_baseline_v2/run --resume runs/reduced_campaign/reduced_baseline_v2/run/latest.pt
```

Inspect the resume plan, then append `--execute` in authorized G5 recovery. A corrupt
latest checkpoint requires a verified same-lineage backup; do not substitute an
engineering checkpoint. Preserve corrupt bytes for diagnosis. If no usable recovery
exists, escalate to the operator for a new fresh attempt and revised cost ledger.
Hardware failure pauses this campaign until restoration; another machine requires
a scope and compatibility review. Budget one100-update replay plus600s restart and
24h downtime provisionally; exceeding either triggers rebudgeting.

Backup destination and custodian are **UNCONFIRMED**, copy/restore **NOT_RUN**.
Before approval, the operator names storage and a human custodian. Prefer storage
outside this machine; a same-disk copy does not cover machine failure. Proposed
procedure: at a quiescent verified save boundary, copy checkpoint plus sidecar and
the complete lineage metadata/log/archive set to a unique versioned destination;
hash source and destination, record UTC time, destination and custodian, and retain
the previous verified generation until the new one passes. Back up at each planned
pause and at completion; uninterrupted-run cadence must be agreed with the custodian.
Stage restoration into a new isolated directory, compare all hashes, verify durable
checkpoint identity/counters and environment/input availability, and perform a bounded
same-lineage restore rehearsal with separately charged costs before claiming recovery.
Do not copy a changing latest.pt concurrently with an atomic save. No backup was made
in section7. Destination/cadence/restore evidence must become a successor bundle.

## Calendar, endpoint and transfer

The user confirmed continuous machine availability on2026-09-12, committing to keep
it on as long as needed. This removes the availability question, not runtime or
deadline constraints. Retain section6's provisional sequential Sep13 08:00 to
Sep15 16:00 UTC+8 windows (training, export/backup, evaluation, downtime/recovery).
Rebase if readiness misses the start. Training projection is4.406–4.753h;
54.057h conservative total includes provisional reserves. Evaluation p90 is NOT_RUN;
the24h slot is not a measurement. See BASELINE_BUDGET for the eligible-artifact
calibration dependency and separately charged costs. Historical conflicting organizer
dates remain unresolved; no deadline extension or measured-fit claim is made here.

At clean exit0 require3815 updates, 1,000,079,360 tokens, cursor976640, zero final LR,
correct WSD/model/input/runner identity and genuine fresh provenance. Verify and export:

```powershell
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py verify --config configs/training/baseline_reduced_v2.yaml --run-dir runs/reduced_campaign/reduced_baseline_v2/run --checkpoint runs/reduced_campaign/reduced_baseline_v2/run/completed.pt
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py export --config configs/training/baseline_reduced_v2.yaml --run-dir runs/reduced_campaign/reduced_baseline_v2/run --checkpoint runs/reduced_campaign/reduced_baseline_v2/run/completed.pt --destination runs/reduced_campaign/reduced_baseline_v2/run/baseline_export.pt
```

Retain export reload/recount/provenance evidence and hashes. Early best/latest
checkpoints are not eligible endpoints. Full evaluation and release are G6 work,
with fresh-environment coverage and provisional labels until organizer settings resolve.

For transfer, copy each `files_sha256` entry of the freeze bundle preserving its
repository-relative path, and verify raw-byte SHA256 after transfer. Git omits data,
runs and evidence; those must be transferred separately. The bundle includes active
stable/dev payloads and exposure components, not closed final/reserved payloads or
the entire environment. Recover those from their separately verified source before
any later use, without opening holdouts during training. Recreate the documented
environment from Dockerfile/constraints and ENVIRONMENT.md; rerun environment and
loader checks. To regenerate exposure, use SELECTED_RECIPE.md instructions with
the pinned selection report and original components. Regenerated receipts are new
evidence; they must not be presented as byte-identical historical receipts.
