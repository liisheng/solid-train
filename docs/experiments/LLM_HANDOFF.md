# Brief for the next LLM

Operate only the approved pre-campaign experiments on the assigned machine.
Read AGENTS.md, .agent/CONTINUITY.md, docs/STATUS.md, docs/ENVIRONMENT.md,
this directory's OPERATOR_GUIDE.md, and the implementation specification with
its execution amendment. Current policy: configs/campaign/experiment_execution_v1.json.

The user removed hard time limits. Present the available same-job smoke-based
estimate and uncertainty, then let the operator decide. Do not answer the runner's
`Start this job? [y/N]` prompt without the user's explicit decision for that job.
`--execute` alone is not an affirmative response. Before the initial smoke, no
measured estimate is available; explain that honestly. No process is killed or
refused because it exceeds six hours, a forecast, or a calendar planning target.

Use the updated source and a new runs/pre_campaign/v2-advisory bundle. Keep old
bundles, identities and ledgers intact; source changes invalidate old identities.
Follow the operator guide's exact setup, transfer, verification and run commands.
Do not copy another machine's virtual environment or runtime ledger. Source and
tokenizer are in Git; shard payloads/manifests, dev schedule and bundle are separate.
Check actual hardware and the CUDA environment before claiming readiness.

The 4070 runs S0, SLR and SMIX in sequence, each with its own two-update smoke.
The 3070 prepares in parallel, then runs C0/C1 only if the complete screen selects
a candidate. C0 concurrency has not been implemented. C1 repeats exactly the
selected single change with a matched control seed. Do not edit recipe settings,
thresholds, schedules, source hashes or identities to bypass a failure.

Wait for each job and verify its endpoint. Preserve checkpoints and sidecars,
metrics, command/hardware/exit/verification/failure receipts and runtime history.
Report observed elapsed time separately from estimates. Unknown crash duration
is not zero and is not replaced with an estimate. Inspect active processes before
recovering an interrupted ledger entry. Resume only the same identity from a
verified, incomplete latest checkpoint. Never run two jobs on the same GPU.

Use completed endpoints and all four protected validation slices for selection;
do not use submission benchmarks or validation_final. Report settings, evidence,
actual hardware, measured/unknown durations, and blockers. If the user chooses to
stop experiments, record explicit incomplete closure rather than a gate pass.
Plan toward September 18 submission, preserving September 19–21 reserve.
After recipe integration, complete G4 sections 3–7 before fresh G5 main training.
No model publication, commit or push is implied by an experiment execution request.
