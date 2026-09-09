# Section 3 — Run and analyze one sustained profile

## Goal and prerequisite

Execute the section-2 plan on actual production inputs, then establish reliable measured
throughput and costs. Required: `PROFILE_PLAN.md` and its passing implementation checks.
Read it with the common index/handoff, fixed baseline recipe, measurement contract,
and only the selected wrapper/verification code. Do not reimplement the profiler here.

## Work

1. Refresh machine identity, environment, free disk, active training processes and GPU
   load. Docker may have restarted unrelated application containers in section 1.
   Identify contention; do not stop unrelated user workloads without authorization.
   Verify source/input identities and required telemetry before using the GPU.
2. Start one fresh engineering run in the plan's unique directory, with the fixed final
   model, BF16, composite exposure, full-dev validation and recovery-save controls.
   Keep the full baseline horizon; use the reviewed bounded stopping path. Never
   initialize from a prior pilot or place this run in the future submission directory.
3. Collect 30–60 minutes of valid sustained evidence under the plan's declared clock
   and warmup rules. Retain all samples, invocation timings and overhead. A short, failed,
   interrupted or contended interval cannot silently become the primary profile.
   If a tool session disconnects, inspect process identity/logs before relaunching.
4. Check finite loss/gradients and counter continuity; confirm full-dev coverage and
   expected saves in the measured interval. Verify the ending engineering checkpoint
   and hashes. No early checkpoint is an eligible completed-baseline export.
5. Report weighted throughput and p10/median/p90, optimizer/validation/checkpoint/outer
   wall times, RAM/swap, GPU headroom and thermal/data-wait behavior. Apply the required
   10% headroom rule to device capacity/peak usage, not allocator usage alone.
6. Explain the historical ~51k versus ~67k tokens/s difference using source/config/input,
   window, validation and instrumentation differences plus current measurements. An
   unresolved cause should be labeled unresolved; a throughput regression can be honest
   evidence. No requirement says to beat the old rate or find a faster backend here.
7. If a correctness/instrumentation failure requires changes, preserve the attempt,
   repair and validate the affected code, then explicitly justify successor measurement.
   Do not silently broaden this section into tuning or repeated benchmark searches.

## Deliverables and completion

Write `docs/g4/PROFILE_EVIDENCE.md` with a compact result table, source/input/run identities,
raw evidence paths/hashes, exact window definitions, overhead accounting, machine facts,
headroom result and rate-gap explanation/limits. Preserve the full profile locally.

PASS requires real validated measurements and no bypassed correctness/telemetry failure.
Separate any still-unmeasured target-machine profile from the source-lane result. Do not
claim total-campaign runtime yet; section 6 combines these measurements with reserves.
Carry the verified engineering checkpoint and custody information to section 4.

## New-chat prompt

```text
Execute only G4 section 3, docs/g4/03-sustained-profile.md. Read the common G4
handoff/index and PROFILE_PLAN.md. Verify prerequisites, then run the one bounded
30–60 minute engineering profile on real production inputs with final controls.
Measure and analyze the required telemetry and full timing; preserve all attempts.
Do not start the G5 baseline, full scoring, tuning or another section. Finish with
PROFILE_EVIDENCE.md, evidence hashes, status and the short section-4 handoff.
```
