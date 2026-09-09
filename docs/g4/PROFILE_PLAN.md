# G4 section 2: bounded production profiler

**Section 2 PASS**, verified 2026-09-09, branch `codex/g4-verification`, base HEAD
`4e27266e5afafd50ddc3bb7c840e358bde3c74ec`. This is a section-2 preparation record;
changes are uncommitted. No sustained profile, G5 baseline, full scoring or later section
was launched. G4 itself remains unpassed.

## Scope and source custody

Use `scripts/profile_baseline_training.py`, which delegates to the actual reduced
baseline runner. Keep the historical `scripts/profile_real_training.py` and its evidence
unchanged. Fixed recipe, optimizer, data reader, backend and validation policy remain
bound to their existing contracts. Lightweight timer instrumentation measures the existing
training path; it does not select or optimize a backend.

Evidence root: `runs/verification/g4/section-02/20260909-1127/` (ignored local evidence,
requiring transfer or reproduction separately from Git). Initial custody refresh matched
all 148 files in the section-1 manifest
`0b20afde26466900ad23ae78345f16e8873f1e281d149f003269fe89fc9f6da1`, all 133
stable/development shard payloads, four manifest files and two schedules. The closed
holdout/reserved payloads were not opened. Original section-1 input-opening evidence can
be reused because those inputs and readers have not changed. New instrumentation and
profiler code require successor source/test evidence; historical timing is not promoted.

Fresh real runner preparation succeeded with `baseline-9d50a02e9d741daf` and
exposure content hash `6b1b747e47ea907810bf68f9ff941f61319ba7dbc7a034e3e2a746775666b0e3`.
The full identity and exact frozen input digests are in `runner-prepare.json`, SHA-256
`53e3d58085acf2724c80ad537d4a060c9b3c90ddc0fe38c2f760366db62e9acd`.

## Measurement policy selected before observing throughput

Preserve the fresh seed 1337, 49,658,368 parameters, BF16, 8 x 32 x 1,024 batch,
LR 0.0006, 3,815-update horizon and WSD 38/3395/382 schedule. The engineering stop
does not shorten that horizon or create an export-eligible baseline endpoint.

Exclude completed updates 1–38. Require 1,800–3,600 valid post-warmup optimizer
seconds, contiguous update records and ordinary full-development validation and recovery
cadence at multiples of 100. Select a 2,400 optimizer-second stop, 800-update absolute
cap and 3,600-second outer watchdog. Reaching a bound without enough valid evidence is
a failed measurement, never a shorter accepted window. A repeat requires a specific
invalidation and a new directory; there is no automatic retry or baseline launch.

Report nearest-rank p10/median/p90 of per-update tokens/time and weighted total tokens
divided by total optimizer seconds. Keep optimizer time, elapsed training window and
outer command launch-to-exit time distinct. Measure preparation directly. Never label
outer-minus-trainer time as preparation, or add data-wait time a second time when it is
already included in optimizer timing. Elapsed training window is the start of update 39
through the final optimizer finish: intervening validation/saves are included; trailing
saves are covered by phase and outer timings. Data wait means host wall time inside
`get_batch`, including any blocking work there, not isolated asynchronous device time.
Two clock reads per microbatch add no device synchronization. Preserve phase timing and invocation
history alongside outer invocation accounting, including failed attempts.

## Machine and telemetry prerequisites

The read-only environment refresh passed 98 checks: Windows CPython 3.12.6,
torch 2.5.1+cu124, CUDA 12.4, GPU 0 NVIDIA GeForce RTX 4070 SUPER, driver 591.86,
UUID `GPU-ca5933af-db8e-2638-f9b1-5bc4ce8d7cab`. Device-total telemetry reports
12,282 MiB. These are machine facts, not performance or sustained-headroom evidence.

Sample at a declared five-second cadence, recording actual collection times/failures:
RAM, actual pagefile/swap usage, device-total/used VRAM and sampled headroom, GPU
utilization, temperature and throttle reasons. Retain trainer peak allocated VRAM,
data-wait, validation and checkpoint timings. Windows commit limit is not swap usage;
use actual pagefile usage. Unavailable telemetry remains unavailable and prevents a
complete measurement. Sampled device peaks can miss between-sample spikes.

The initial read-only checks found no training process. Other application containers
are running and initial GPU utilization was nonzero. Section 3 must refresh process/load,
free space, source and input custody before launch; do not silently stop unrelated apps.

## Verified commands, evidence and stop handling

The exact read-only command executed from the repository root was:

```powershell
.\.venv\Scripts\python.exe scripts/profile_baseline_training.py --output-dir runs/verification/g4/section-02/20260909-1127/dry-run
```

It exited 0 with `PLAN_ONLY`, retained the same runner ID and created no `train/`
directory. The directory is now evidence and cannot be reused. Preparation took 92.748s;
source/input/telemetry preflight took 2.677s. Externally measured wrapper launch-to-exit
was 102.509s; its own main-entry wall was 95.508s. These are preparation costs, not
training-throughput measurements. The one telemetry sample was collected during the
container build: roughly 0.86 GB RAM available, 7.55 GB pagefile in use and 45% GPU
utilization. Those competing-load observations must be refreshed before section 3.

For a **separately authorized section 3**, after its preflight and confirming the new
directory is absent, the reviewed wrapper command is:

```powershell
.\.venv\Scripts\python.exe scripts/profile_baseline_training.py --output-dir runs/verification/g4/section-03/primary-01 --execute
```

This command was not executed. It rebuilds the verified identity and uses exactly the
fixed recipe plus the declared bounds. `profile_plan.json` retains the complete child
argument array. Output paths must be fresh and below `runs/verification/g4`; the final
baseline directory is rejected. No resume or automatic retry is offered.

Child failure, missing/invalid telemetry, sampler failure, operator interruption or the
watchdog makes the attempt FAIL. The watchdog interrupts the child process group, allows
10s to exit, then kills and waits at most another 10s. An unconfirmed termination remains
explicit; inspect its PID before doing anything else. A forcibly killed wrapper may leave
a RUNNING receipt, which also requires reconciliation. No unrelated process is stopped.
Training's existing finite-value/OOM checks remain effective. Thermal/load observations
are reported for operator assessment; there is no new backend or temperature tuning policy.

Artifacts: `profile_plan.json`, `source_manifest.json`, `input_manifest.json`,
`telemetry_preflight.json`, `wrapper_receipt.json`; on execution, `active_process.json`,
`stdout.log`, `stderr.log`, `telemetry.jsonl`, `outer_invocation_history.jsonl`,
`measurement.json`, and `train/` metrics, phase history, identity and bounded checkpoint.
The report hashes raw telemetry and the phase/metric/identity evidence, reconciles
optimizer totals/history/counters and checks source/input identity again after success.
Each failed/repeated attempt retains its own outer cost and directory; aggregate all
attempt receipts, never only the accepted one. Checkpoint inspection here is manifest,
size and counters; exact durable recovery verification belongs to section 4.

| Final check | Result |
|---|---|
| Clean Docker build; full suite | PASS; **756 passed**, zero skips, 132.96s |
| Windows profiler/telemetry CPU tests | **41 passed**, 5.96s; includes watchdog/process cleanup |
| Ruff, compilation, dependency consistency | PASS |
| Environment refresh | Windows 98 / CPU container 97 checks PASS |
| Source custody | All 151 source files match dry-run; all 149 copied image files and 34 installed modules match |
| Input custody | 133 payloads plus four manifests/two schedules unchanged; all 14 dry-run input hashes match |
| Reviews | Sol and Terra approve final source; root verifies hashes and checks |
| Final process/space observation | No Python processes; about 201 GB free C: and 1.37 TB D: |

The 151-file source scope includes the new profiler, telemetry helper and tests; it
omits the non-code `configs/README.md` from the earlier 148-file scope. Only `train.py`
and its existing CPU ledger test changed among those original files. Both container
build-control files are bound but intentionally not copied into the image. Historical
failed fixture/lint/custody attempts remain beside successor passing receipts.

Final evidence under the root above:

- `report.json`: `2f24bafcb22962a0505062857bd1f66be7e401fcc1c400e2cfa006e3beccc226`
- `dry-run/source_manifest.json`: `63e83aaac746a7752208d27fbd964e8202193192c22eaeca29a6857f917da656`
- `dry-run/input_manifest.json`: `dfefb12cdd17aba1779f9c99d919926c130ae908d6cb1133a0bd4f963deebc44`
- `dry-run/profile_plan.json`: `16f3dc919707c4a9625c3f143eedbf6c1715386eda48bd0feedba51c630fc998`
- `reviews.json`: `bb46e1e75e2bd3fa295efb507d9f2315831fbeed4a0a75ad9fc4d74c1f80977a`

No required telemetry collector remains unresolved for this machine. Section 3 must
produce the actual sustained evidence; historical ~51k/~67k rates remain unexplained.
Agent reviews do not satisfy G4 human teammate approval. No commit/push was made.
