# Experiment operator guide

The current execution policy is `configs/campaign/experiment_execution_v1.json`.
The user removed automatic time limits. There is no six-hour allowance, aggregate
cap, five-minute smoke timeout, or automatic calendar cutoff. The five-job recipe,
two-update smokes, integrity checks, memory checks and dependencies are unchanged.

The wrapper prints an estimated duration, then asks `Start this job? [y/N]`.
Only `y` or `yes` starts the process. Empty input, no, or unavailable stdin starts
nothing. `--execute` enables this decision step; it does not answer it. A running
job can exceed its estimate without being terminated for elapsed time. Normal
completion remains 382 updates / 100,139,008 loss tokens, or two updates for smoke.

## Setup and transfer

Read AGENTS.md, docs/STATUS.md, .agent/CONTINUITY.md, docs/ENVIRONMENT.md and
IMPLEMENTATION_SPEC.md with its execution amendment. Use the updated reviewed
checkout on both machines and each machine's own verified CUDA environment.
Do not copy `.venv` or install/change host packages without authorization.
The CPU Docker image is a test environment, not the production GPU environment.

Source and tokenizer come from Git. Transfer these ignored inputs at matching
relative paths; see [the teammate ZIP placement table](LLM_HANDOFF.md#teammate-zip-contents-and-placement)
for the six top-level ZIP items and their exact destinations. Required inputs are at
relative paths: `data/shards/reduced_5pct_v1/stable/`, `validation_dev/` under the
same root, `stable_train.manifest.json`, `validation_dev.manifest.json` under that
root, and `data/schedules/reduced_5pct_v1/validation_dev.json`.

The new source invalidates old prepared identities. Prepare a separate bundle at
`runs/pre_campaign/v2-advisory`; keep `runs/pre_campaign/v2` unchanged. Do not edit
old source hashes or resume old-identity runs under the new code. No GPU runs had
started when this amendment was requested. Transfer the new bundle to the teammate.

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiment prepare
.\.venv\Scripts\python.exe -m scripts.run_experiment check
.\.venv\Scripts\python.exe -m scripts.run_experiment plan --job S0
```

These commands default to the new bundle. The full-run estimate becomes available
after a verified smoke for that same job. Before smoke, the tool explicitly reports
that a measured estimate is unavailable. Do not substitute a guessed runtime.

## Your RTX4070 lane

Run each command separately, inspect its result, and decide whether to proceed at
each prompt. The full-run forecast comes from its own smoke, including a conservative
overhead allowance and 50% margin. It is not a sustained profile or a guarantee.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiment smoke --job S0 --lane rtx_4070 --execute
.\.venv\Scripts\python.exe -m scripts.run_experiment run --job S0 --lane rtx_4070 --execute
.\.venv\Scripts\python.exe -m scripts.run_experiment smoke --job SLR --lane rtx_4070 --execute
.\.venv\Scripts\python.exe -m scripts.run_experiment run --job SLR --lane rtx_4070 --execute
.\.venv\Scripts\python.exe -m scripts.run_experiment smoke --job SMIX --lane rtx_4070 --execute
.\.venv\Scripts\python.exe -m scripts.run_experiment run --job SMIX --lane rtx_4070 --execute
.\.venv\Scripts\python.exe -m scripts.analyze_experiments --stage screen --bundle runs/pre_campaign/v2-advisory --output runs/pre_campaign/v2-advisory/selection.screen.json
```

## Teammate's RTX3070 lane

The current implementation still waits for all screening results before C0/C1.
Allowing concurrent C0 was discussed but is not part of this runtime amendment.
If the screen retains control, neither confirmation job is needed. Otherwise
transfer the complete updated bundle including screen checkpoints, sidecars,
metrics and invocation receipts. Re-run `check` on her machine.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiment smoke --job C0 --lane rtx_3070 --execute
.\.venv\Scripts\python.exe -m scripts.run_experiment run --job C0 --lane rtx_3070 --execute
.\.venv\Scripts\python.exe -m scripts.run_experiment smoke --job C1 --lane rtx_3070 --execute
.\.venv\Scripts\python.exe -m scripts.run_experiment run --job C1 --lane rtx_3070 --execute
.\.venv\Scripts\python.exe -m scripts.analyze_experiments --stage final --bundle runs/pre_campaign/v2-advisory --output runs/pre_campaign/v2-advisory/selection.final.json
```

C1's treatment is recomputed from verified screening evidence. Both machines must
detect their actual GPU/VRAM; historical 3070 Ti evidence records about 8GB.

## Accounting and interruption

`usage --lane rtx_4070` (or `rtx_3070`) reports observed elapsed time and unknown
duration attempts. `budget` remains a compatibility alias for this report; it has
no allowance. Each lane keeps its own `runs/pre_campaign/<lane>.runtime.json`.
Do not replace it with the other machine's ledger. Old `.ledger.json` files remain
historical evidence and are not converted into new measured costs.

One job per GPU remains enforced. Failed work still counts in observed elapsed
time. A crash with an unknown duration is explicitly marked unknown, not charged
the estimate and not claimed to cost zero. Preserve all logs and checkpoints.
After independently confirming an interrupted process is dead, use:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiment recover --lane rtx_4070 --reservation-token TOKEN --confirm-process-dead
.\.venv\Scripts\python.exe -m scripts.run_experiment run --job S0 --lane rtx_4070 --resume --execute
```

The token is in the invocation's `launch.json` and runtime ledger. Resume preserves
the identity and horizon; it shows an estimate for remaining updates and asks again.
Already complete checkpoints cannot resume. Unresolved old budget reservations or
operation locks require investigation before any new process; never delete history
to bypass a concurrent execution. Explicit interruption can lose progress since the
last checkpoint; removal of time limits does not add a graceful-stop protocol.

September 12 is an experiment planning target, and September 18 the intended early
submission date. The operator decides whether estimates leave sufficient time for
G4, the main campaign and submission. If stopping the optional experiment stage,
the analyzer's `--close-incomplete budget|deadline|hardware|failed` records a control
fallback with reasons; these are operator decisions, not automatic shutdowns.
Integrate the selected recipe, finish G4 sections 3–7, then train G5 from fresh weights.
