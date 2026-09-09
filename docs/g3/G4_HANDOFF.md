# G4 handoff from reduced G3

Current disposition: local correctness passed; recipe readiness is
`REDUCED_BASELINE_RECIPE_BLOCKED_FOR_G4` because Docker verification and lint could not run.

Use [INTEGRATION_EVIDENCE.md](INTEGRATION_EVIDENCE.md) and its final machine report
for section-6 disposition and source identity. Work remains on
`codex/g3-baseline-recipe`, based on `f643e9f09c0700686222712eb58d90e4a0fe1de2`;
the G3 changes are uncommitted. The source manifest binds both tracked and untracked
implementation files. Agent review does not replace the teammate review required by G4.
The final [artifact index](../../runs/verification/g3-section6-integration-final-v2/artifact_manifest.json)
includes the tracked working-tree patch and the untracked-file hash manifests.

## Fixed baseline and scope

The baseline remains fresh seed 1337, 49,658,368 parameters, BF16, microbatch 8,
accumulation 32, length 1,024, and 3,815 updates / 1,000,079,360 loss tokens.
WSD is 38 warmup, 3,395 stable, and 382 decay updates to zero LR. The final cursor
is 976,640. There are 40 full-development events (update 1, multiples of 100, and
completion) and 39 scheduled recovery saves, plus improving best-development saves.
Do not initialize from either engineering rehearsal lineage. The eight-update
checkpoints are deliberately ineligible for baseline export.

## G4 work before execution

1. Verify the current source/input manifests and local ignored artifacts, environment,
   free space, and absence of another training job. Reproduce unavailable verification
   checks on a working environment. Preserve the protected historical run directories.
2. Run the one required 30–60 minute sustained profile on the actual composite exposure,
   full-dev validation, and final training controls. Record optimizer and total process
   wall time separately, validation/checkpoint/preparation overhead, RAM/VRAM/headroom,
   data wait, thermal behavior, and sustained rate distribution. Do not reuse an old
   sampled-validation or repeated-one-component profile as changed-input evidence.
3. Resolve the short-run rate difference: approximately 51.1k tokens/s in this rehearsal
   versus historical weighted 66,989.59. These imply about 5.4 and 4.15 optimizer-hours
   respectively *if sustained*, before validation, saves, preparation, export, and
   evaluation. Neither is a current total-campaign forecast or a measured speedup.
4. Reuse the section-6 exact-recovery evidence if its source, inputs, environment and
   required recovery scenario remain applicable; rerun only the G4 recovery/takeover
   check not covered by that evidence. Record the teammate approval and target-machine
   custody required by the applicable G4 contract. The G2 any-machine amendment does
   not silently amend G4. Keep full-evaluation and recovery reserve in the dated budget.

## Exact runner commands

These commands are for the later G4/G5 tasks, from the repository root. The first two
are read-only plans. The `--execute` command starts the full baseline and must only be
used after G4's checks and review are complete.

```powershell
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py prepare --run-dir runs/reduced_campaign/reduced_baseline_v1/run
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py launch --run-dir runs/reduced_campaign/reduced_baseline_v1/run

# G5: fresh baseline, after G4
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py launch --execute --run-dir runs/reduced_campaign/reduced_baseline_v1/run

# G5: recover the same run, retaining its original horizon and identity
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py launch --execute --run-dir runs/reduced_campaign/reduced_baseline_v1/run --resume runs/reduced_campaign/reduced_baseline_v1/run/latest.pt

# Completed endpoint only; an early checkpoint must fail this check
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py verify --run-dir runs/reduced_campaign/reduced_baseline_v1/run --checkpoint runs/reduced_campaign/reduced_baseline_v1/run/completed.pt
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py export --run-dir runs/reduced_campaign/reduced_baseline_v1/run --checkpoint runs/reduced_campaign/reduced_baseline_v1/run/completed.pt --destination runs/reduced_campaign/reduced_baseline_v1/run/baseline_export.pt
```

Absolute `--stop-after-updates` bounds an engineering run without shortening its
3,815-update horizon. Never resume a rehearsal into the submission baseline.

## Evaluation and resource custody

G6 full provisional evaluation, after verifying the eligible export:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/reduced_campaign/reduced_baseline_v1/run/baseline_export.pt --tokenizer runs/reduced_campaign/g2_handoff_v2/artifacts/data/tokenizer_final/tokenizer.json --device cuda --precision bfloat16 --full --output runs/evaluation/baseline-provisional/results.json --bundle runs/evaluation/baseline-provisional/bundle
```

Verify the saved evaluation bundle and measure its own runtime. The one-example CPU
smoke is not a full benchmark or a CUDA/BF16 equivalence result. Organizer settings
and observable harness Git provenance remain unresolved for an official result;
the local runtime content and five loader revisions are pinned. Keep validation-final
closed until its later authorized use; benchmarks do not select training changes.

The real rehearsal used about 5.445 GiB peak allocated VRAM; this is not total-device
headroom certification. Durable payloads were about 596 MB each. Latest, best and
completed payloads therefore need approximately 1.67 GiB steady storage, plus atomic
write transients, clean export, verification copies and logs. Thirty-nine scheduled
recovery writes imply approximately 21.7 GiB cumulative payload writes, excluding best
saves, verification reads and the completion copy. Measure actual costs in G4/G5.

The approximate baseline training estimate `6 × parameters × consumed tokens` is
2.98e17 FLOPs; it is not a hardware measurement and excludes evaluation/engineering
runs. Preserve per-invocation timing and reconcile the complete campaign ledger after
any resume. Data, checkpoints and reports under `data/` and `runs/` are ignored local
artifacts and need separate transfer or reproduction.

The trainer's `phase_timing.json` and append-only `phase_timing_history.jsonl` measure
from entry into `train.main()`, after imports. They are not complete process-launch
timers. Measure each G5 runner invocation externally from launch to exit as well, and
retain the exit status and timing for failed/interrupted attempts. The section-6 harness
does this in each command's `wall_seconds`. Its separately timed initial preparation is
outside those command times; do not label the difference between outer and trainer wall
time as pure preparation or call the raw sum of command times a full campaign total.
