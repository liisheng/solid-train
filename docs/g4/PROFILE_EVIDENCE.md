# G4 section 3: sustained production profile

**Section 3 PASS — source lane**, verified 2026-09-12. The single profile exited
0 at 13:05:04+08:00. This does not pass G4 or start the G5 baseline.

## Scope and custody

Branch `codex/g4-selected-recipe`, implementation HEAD
`6293354814a619c7b318b9d15b694c83d871083c`. No implementation changes in this
section. All 172 source/configuration/test files in the profiler scope match the
selected-recipe verification snapshot: Docker 826 tests, Windows 826 tests,
Ruff, compilation, builds and dependency checks remain applicable.

Evidence root: `runs/verification/g4/section-03/`. The `20260912-preflight`
directory retains environment, process, payload and source checks plus outer
command receipts. `20260912-plan` is a successful, separate PLAN_ONLY invocation;
`20260912-primary-01` is the single execution attempt, now EXITED. No training
process remained at the final process check. No retry was needed.

The preflight rehashed all 133 stable/development payloads; all match their
manifests. Environment: 98 checks PASS, Windows CPython 3.12.6,
torch 2.5.1+cu124, RTX 4070 SUPER, UUID
`GPU-ca5933af-db8e-2638-f9b1-5bc4ce8d7cab`, driver 591.86, total 12,282 MiB.
Disk free at preflight: C 198,146,813,952 bytes; D 1,226,639,503,360 bytes.
No competing training process was found; desktop GPU applications and six
existing AegisVault containers remained running. Their contribution cannot be
separated from machine-level telemetry. This is the measured occupied-desktop
environment, not an isolated-machine capacity claim. No slow interval was removed
or relabeled; all post-warmup updates contribute to the primary result.

Recipe: selected CONTROL v2, fresh seed 1337, 49,658,368 parameters, BF16,
8 × 32 × 1024 batch, LR 0.0006, full 3815-update horizon. Runner identity
`baseline-f23398e32dc3100e`; training identity `run-146ddcb9c11da742`.
Exposure content SHA `c41bc538d0f53ee6bb08ec8b50165afbe7132ada38f3f03a0c0d46c44b792101`.
Source manifest SHA `fbd314e0702c413b0ff5bc5803d1bce0cb5ba004051e20df43357a86e36cb915`.

## Declared measurement window

The reviewed policy was fixed before execution: stop at the first completed
update reaching 2400 optimizer seconds, absolute cap 800 updates, outer watchdog
3600 seconds. Exclude completed updates 1–38; require 1800–3600 post-warmup
optimizer seconds. Keep full-development validation and recovery saves at
multiples of 100, plus initial development validation and best saves.

Weighted throughput is included loss tokens divided by included optimizer time;
percentiles use nearest rank. Elapsed window begins at update 39 and ends at the
last optimizer finish, including intervening validation/saves. Outer child wall,
wrapper wall, preparation, validation and checkpoint costs stay separate.
Data-wait time is included in optimizer time and must not be added twice.
Telemetry cadence is five seconds; sampled device peaks may miss brief spikes.
Headroom uses device capacity and device-level used memory, not allocated tensors.

Exact execution command from the repository root:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/profile_baseline_training.py --output-dir runs/verification/g4/section-03/20260912-primary-01 --execute
```

## Result

| Measurement | Verified result |
|---|---:|
| Completed updates / total loss tokens | 590 / 154,664,960 |
| Included updates / included loss tokens | 39–590 (552 samples) / 144,703,488 |
| Post-warmup optimizer window | 2243.512 s (37.392 min) |
| Weighted optimizer throughput | **64,498.66 tokens/s** |
| Per-update p10 / median / p90 | 60,735.30 / 65,441.63 / 67,322.89 tokens/s |
| Elapsed included training window | 2275.116 s |
| All-update optimizer time | 2401.250 s |
| Development validation / checkpoint time | 20.649 / 17.066 s |
| Data wait, already inside optimizer time | 8.559 s (0.356%) |
| Trainer main / child launch-to-exit | 2466.863 / 2472.413 s |
| Executed wrapper preparation / full outer command | 28.258 / 2504.916 s |
| Separate PLAN_ONLY preparation / full outer command | 54.451 / 57.144 s |
| Total of both wrapper command invocations | 2562.060 s |
| Peak allocated GPU memory | 5.445 GiB |
| Sampled device peak / capacity | 8373 / 12,282 MiB |
| Minimum sampled device headroom | **31.827% — PASS**, minimum required 10% |
| Machine peak RAM used / minimum RAM available | 15.622 GiB / 98.574 MiB |
| Peak actual pagefile use during the profile | 6.489 GiB |
| GPU temperature / utilization range, entire invocation | 52–84°C / 2–99% |
| Telemetry samples / maximum sampling gap | 494 / 5.016 s |

Both command invocations are retained; their sum is not total campaign cost.
Wrapper source/input/telemetry preflight took a further 0.556 s in the executed
invocation. The difference between total wall and named phases is not labeled
preparation. Postverification and operator observation are outside these command
timings. No failed training invocation was omitted.

The 590-update prefix is contiguous; loss and gradient norms are finite throughout.
Full-development events at updates 1, 100, 200, 300, 400 and 500 each cover
753 sequences, 771,072 targets and 95 batches. Recovery writes at 100–500 and
the bounded ending update follow the unchanged fail-closed trainer path; completed
execution and terminal verification establish successful writes. Intermediate
`latest.pt` versions are overwritten, not separately retained. Best saves are
included in checkpoint timing; `best.pt` ends at update 500. The final recovery
checkpoint is 596,076,206 bytes, update 590, cursor 151,040. Its checksum,
payload integrity, run/exposure identity, embedded runner-manifest hash and fixed
controls all reverify. It retains LR 0.0006 and the 3815-update horizon, so it is
not an eligible baseline export. Exact resumed-state equality is not claimed here.

RAM pressure is material: the lowest available RAM was 98.574 MiB at
12:41:16+08:00; actual pagefile use peaked at 6.489 GiB. These are whole-machine
measurements, not training-process allocations or paging traffic. The run completed
without OOM or telemetry loss; these observations do not establish spare host-RAM
capacity for additional work. Five-second sampling cannot rule out brief VRAM spikes.
Observed clock-reason masks were 0, 1 (idle) and 4 (software power cap). Detailed
local `nvidia-smi` snapshots identify power capping and zero software/hardware thermal
slowdown counters; no thermal slowdown was observed. Idle/startup/validation periods
are retained in the utilization range.

Read-only postverification passes 17 checks: rederived metric/phase/telemetry results,
source/input/raw-file custody, all 133 payloads, finite values, both checkpoints and
ending controls/counters. Checkpoint reports retain remote-copy verification as
NOT_RUN; that is outside this source-lane profile. The observer script passes
PowerShell syntax parsing. No project source changed, so the bound 826-test suites,
lint and builds were reused rather than rerun during measurement.

## Historical rate comparison

Historical G2 measured 66,989.59 weighted tokens/s over 1886.16 optimizer
seconds, excluding its first four updates. It used the same model/batch/BF16
and stable shard corpus but a single materialized schedule, 2048-update horizon,
two warmup updates and eight validation batches per event. Current v2 uses the
composite exposure, 3815-update WSD horizon, 38 warmup updates and all 95
development batches. Source and instrumentation also changed.

The G3 approximately 51k observation came from updates 2–8 of an eight-update
correctness rehearsal, not a sustained window. Validation/checkpoint overhead
cannot by itself explain an optimizer-only rate difference because it is outside
the per-update optimizer timer. These runs do not isolate causal effects of
reader changes, instrumentation, clock/load conditions or warmup.

Current weighted throughput is 3.718% below historical G2, while current p90 is
close to its approximately 67k rate. Consecutive blocks of up to 100 included
updates measured 64.7k, 61.1k, 64.4k, 65.1k, 66.3k and 66.6k tokens/s. This
demonstrates variation within the current run and does not reproduce a sustained
51k ceiling. The exact cause of the earlier 51k observation remains **unresolved**;
there is no controlled backend/reader comparison or defensible speedup attribution.
Full-development and checkpoint overhead is now measured explicitly rather than
folded into a throughput explanation. No tuning or backend promotion occurred.

## Evidence hashes and recovery handoff

Paths below are relative to `runs/verification/g4/section-03/`. The manifest binds
43 local evidence files, including both checkpoint binaries, commands, stdout/stderr,
telemetry, phase history, source/input manifests and postverification. Git alone
does not transfer these ignored files.

| Artifact | SHA-256 |
|---|---|
| `evidence-manifest.json` | `883d04403fda6560175294788fc66bedf278ea6fd8de162ef9ea21dd7cdcb0fc` |
| `postverification.json` | `6425fa719355e41c9d10b3cf6e0b4f029d12771c91dd964840972eda0f6a87cc` |
| `20260912-primary-01/measurement.json` | `8801db95051a71658f13eec8c9016f6bb56fce753a293bb5f1ac717ea80e0382` |
| `20260912-primary-01/profile_plan.json` | `9ede1c25d815fb6ec7ed6f0b20c7a4f455eeeb9c57d8821065f6752109975c8b` |
| `20260912-primary-01/input_manifest.json` | `54ebcb99ad59f72b3bb5a74fa52881435a5dd6b41ef4461a47908b19fcccb4c9` |
| `20260912-primary-01/train/latest.pt` | `efcfdd3045d097474a69299e69711e3bd4e2dd776010dada84bad2a41f7562dc` |
| `20260912-primary-01/telemetry.jsonl` | `78158a7083b1faf0f97d62cb83f93c1dd73af0a995edabc60d506f790b218977` |

The section-4 input is the verified engineering `train/latest.pt`, its sidecar,
runner/run identities, provenance, run configuration, metrics and phase history.
Keep its ending cursor 151,040 and 590-update counter; do not confuse it with a
fresh G5 initialization. Safe next command: `Get-Content docs/g4/04-local-recovery.md`.

## Remaining scope

Exact local resume/crash recovery belongs to section 4; target-machine profile/takeover,
dated budget and two-person freeze approval remain later work. No early
engineering checkpoint is an eligible completed-baseline export.
