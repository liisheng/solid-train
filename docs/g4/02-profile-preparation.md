# Section 2 — Prepare the production profiler

## Goal

Deliver a tested, bounded measurement path for section 3. Separate instrumentation and
correctness work from the long GPU run. This section permits implementation, CPU tests,
and read-only production preparation, not the sustained profile or baseline training.

## Targeted inputs

Read the common index/handoff and section-1 verification. Inspect
`scripts/profile_real_training.py`, `scripts/run_reduced_baseline.py` (`prepare`,
`launch_command`), `scripts/run_g3_integration.py` (outer timing/postverification),
`train.py` (update timing, validation/saves, stopping), and the relevant telemetry helpers
in `scripts/hardware_inventory.py` / `src/tinybench_lm/operations.py`. Read the hardware
measurement portion of the operations contract and fixed baseline config.

## Work

1. Check the section-1 source identity, ignored inputs, environment, branch and local
   changes. Establish which checks can be reused. Record the runner's fresh identity.
2. Correct or add a narrow profiler wrapper around the actual fixed baseline runner.
   The existing `profile_real_training.py` is historical: it uses a repeated schedule,
   2,048-step horizon, two-step warmup and sampled validation. It cannot be used unchanged
   as the composite/full-dev G4 profile. Preserve old evidence and avoid duplicating
   training logic in the wrapper.
3. Capture exact source/input identities, command, start/end timestamps, exit status and
   outer launch-to-exit time. Retain trainer phase timing and append-only invocation
   history. Measure preparation separately; do not label outer-minus-trainer time as
   pure preparation. Include failed/repeated invocations in the accounting design.
4. Measure sustained update rates with p10/median/p90 and weighted tokens/time. Define
   warmup exclusion before observing results and ensure at least 1,800 valid seconds
   remain after exclusion. Distinguish optimizer-time, elapsed training window and full
   command wall time. The current stop flag counts optimizer seconds, not total wall time.
5. Supply RAM/swap, device-total and peak-used VRAM/headroom, allocated VRAM, data wait,
   validation/checkpoint costs, GPU utilization/temperature and throttle reasons. Record
   sampling cadence, units and collection failures. Do not infer zero from unavailable
   telemetry. Keep instrumentation lightweight; avoid per-microbatch synchronization
   or repeated integrity scans.
6. Produce a reviewed exact command, unique engineering directory, bounded stop and
   abort/watchdog policy. The 3,815-update recipe remains intact; a short command must
   not accidentally run the whole baseline. Rehearsal output is never the G5 run directory.
7. Test command construction, wrong-input rejection, timing math and missing/invalid
   measurement rejection using synthetic/CPU fixtures. Run appropriate lint/build/tests.
   Ensure the declared interval includes ordinary full-dev and save cadence when run.

## Deliverables and completion

Write `docs/g4/PROFILE_PLAN.md`: selected tool/command, source and input hashes, machine,
measurement definitions, warmup/window policy, telemetry coverage, stop/abort behavior,
evidence filenames, and checks. Save dry-run output and tests under the section-2 evidence
directory. A normal `launch` without `--execute` is only a plan, not measurement evidence.

PASS means the wrapper and measurement plan are ready for a separately requested section 3.
Missing required telemetry is a named unresolved prerequisite, not a guessed value.
Update the short handoff. No GPU throughput or G4 PASS claim belongs here.

## New-chat prompt

```text
Execute only G4 section 2, docs/g4/02-profile-preparation.md. Read AGENTS.md,
.agent/CONTINUITY.md, docs/STATUS.md, docs/g4/README.md and CURRENT_HANDOFF.md,
then the targeted code. Prepare and test the correct bounded production profiler
and its exact dry-run command. Reuse valid section-1 evidence. Do not launch the
sustained profile, the baseline, full scoring or another section. Record the plan,
test evidence and a short handoff ready for section 3.
```
