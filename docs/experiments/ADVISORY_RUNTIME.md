# Advisory runtime amendment

The user removed hard experiment time limits on 2026-09-09. Current policy is
configs/campaign/experiment_execution_v1.json, superseding the historical budget
fields in pre_campaign_v2. Job design and selection thresholds are unchanged.

The runner displays a same-job smoke-based estimate and asks for an affirmative
response before creating an execution record or launching a child. Blank, no and
unavailable stdin decline. Before smoke, the estimate is explicitly unavailable.
Full/resume forecasts preserve startup and validation overhead and add a 50% margin.
They are advisory estimates, not guarantees or enforced limits.

Removed from the active execution path: six-hour lane cap, twelve-hour aggregate
cap, five-minute smoke timeout, forecast-based admission refusal, and automatic
September 12 cutoff. The wrapper waits without a training timeout. It still
terminates a child when handling explicit interruption or an execution error;
no graceful checkpoint-on-interrupt behavior is claimed.

New per-lane .runtime.json files retain actual elapsed duration and unknown crash
durations. They enforce one active job per lane/device and preserve failure records.
Historical .ledger.json files remain intact; unresolved historical executions or
locks need reconciliation before starting new work. No estimate becomes a charge.

Use runs/pre_campaign/v2-advisory with the updated source. Old prepared bundles
bind the previous source and cannot be reused by editing their hashes. The original
bundle remains preserved. Send the new bundle and updated source to the teammate;
the shard/tokenizer inputs have not changed. Runtime-only changes do not implement
concurrent C0: confirmation still follows completed screening.

See OPERATOR_GUIDE.md for exact prepare, plan, smoke/run, usage and recovery commands.
Verification completed on 2026-09-09: the frozen final CPU image passed all 821
tests in 99.06 seconds, including 41 focused runner/runtime tests. Build, Ruff,
compile and dependency checks passed. Sol-medium independently reviewed the final
implementation with no unresolved findings; Luna-medium authored the runtime ledger.
Image source/tests match the checkout, and all 37 installed modules match source.
Real-input bundle preparation and production input validation passed. No GPU job ran.

Evidence: runs/advisory-runtime-tests-frozen.log,
runs/advisory-runtime-verification.json and runs/advisory-runtime-sol-hashes.json.
The earlier mutable-checkout test run detected an edit during source validation;
the frozen-image run supersedes it. Final image ID:
`sha256:02406a1ddf298d93ce1ddb9ef0d005e65155e05d8770ba3527dff8316730c5ff`.
New bundle SHA256:
`15a3e7405f21fca4ee22cacaf4225eed757813de90f5196ad807ec35124df9fe`.
The user authorized publication to `codex/g3.5-pre-campaign-experiments` on
2026-09-09. Transfer the new bundle separately; ignored run artifacts are not in Git.
