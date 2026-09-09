# Pre-campaign v2 operator guide

This guide operates the approved bounded package in `configs/campaign/pre_campaign_v2.json`. It does not authorize a main baseline, holdout/submission scoring, publication, or G4/G5 completion. The package contains at most three screen jobs and two conditional confirmation jobs, each full run being 382 updates and 100139008 loss tokens, plus one two-update smoke per job. The aggregate ceiling is 12 GPU-hours, split into non-transferable six-hour `rtx_4070` and `rtx_3070` allowances. The hard stop is `2026-09-12T23:59:59+08:00`.

Before operating, read `AGENTS.md`, `docs/STATUS.md`, `.agent/CONTINUITY.md`, `docs/ENVIRONMENT.md`, and `docs/experiments/IMPLEMENTATION_SPEC.md`. Use the reviewed source checkout and the repository `.venv`; do not install host packages or use CPU training as a production shortcut. Confirm that the same reviewed source, tokenizer, manifests, schedules, and configuration hashes are available on every participating machine. Historical hardware records include a 4070 SUPER 12GB and a 3070 Ti around 8GB; detect the actual device and do not infer readiness from those records.

From the repository root, prepare and inspect the package:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiment prepare --bundle runs/pre_campaign/v2
.\.venv\Scripts\python.exe -m scripts.run_experiment check --bundle runs/pre_campaign/v2
.\.venv\Scripts\python.exe -m scripts.run_experiment plan --bundle runs/pre_campaign/v2 --job S0 --lane rtx_4070
```

Preparation verifies trusted production pins and shard integrity, copies the trusted final model configuration, and deterministically materializes both mixtures' schedules. On a receiving checkout, repeat `check`. Transfer the complete bundle, including schedules, identities, metrics, invocation receipts and checkpoints with their sidecars. Also transfer production inputs at the paths listed below. Each machine keeps its own parent `runs/pre_campaign` lane ledger; do not replace it with the other machine's ledger.

On the 4070, run S0 smoke and then S0 full run:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiment smoke --bundle runs/pre_campaign/v2 --job S0 --lane rtx_4070 --execute
.\.venv\Scripts\python.exe -m scripts.run_experiment run --bundle runs/pre_campaign/v2 --job S0 --lane rtx_4070 --execute
```

After a verified S0 endpoint, run SLR and then SMIX, each with its own smoke before its full run. The runner checks predecessor completion, actual GPU family, BF16 support, memory headroom, exact identity, and the persistent lane reservation before starting a child. A smoke is a readiness measurement, not a sustained throughput or fit claim.

Screen selection must be recomputed from verified endpoint evidence:

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze_experiments --stage screen --bundle runs/pre_campaign/v2 --output runs/pre_campaign/v2/selection.screen.json
```

If a candidate is selected, the 3070 operator rechecks custody, transfers the complete screen bundle and evidence, then runs C0 smoke/full. C1 is conditional on C0 and derives its single treatment from the screen analyzer; operators never pass a manual treatment JSON. Run the final analyzer with a unique output path:

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze_experiments --stage final --bundle runs/pre_campaign/v2 --output runs/pre_campaign/v2/selection.final.json
```

An explicit close action is available for budget, deadline, hardware, or failed evidence:

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze_experiments --stage final --bundle runs/pre_campaign/v2 --close-incomplete failed --output runs/pre_campaign/v2/selection.final.incomplete.json
```

That result is `INCOMPLETE_CONTROL`; it is never a fully passed experiment gate. Selection requires the completed endpoint, exact full-dev coverage, four reconciled protected slices, verified checkpoint/metric/input/source hashes, and the frozen 0.003 global improvement and 0.01 per-slice limits. No validation-final or submission benchmark input is permitted.

Inspect the lane ledger before and after each reservation. Never delete or reset a stale lock or ledger. If a process is independently confirmed dead, recover an uncertain reservation only with the explicit dead-process confirmation and expect the full reservation to remain charged. An interrupted job may resume only with the same job identity and latest durable checkpoint:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiment run --bundle runs/pre_campaign/v2 --job S0 --lane rtx_4070 --resume --execute
```

Preserve failed directories, logs, checkpoints, and invocation timing. Do not delete evidence to resolve a conflict. If budget or deadline cannot fit the remaining work, close conservatively to control. Smoke memory and timing observations do not satisfy the later G4 sustained-profile requirement. Integrate the selected settings into a successor baseline recipe and identity, complete G4 sections 3–7, then train the fresh 1B baseline at G5. Experiment weights never initialize it.

Exact follow-on commands:

    .\.venv\Scripts\python.exe -m scripts.run_experiment smoke --bundle runs/pre_campaign/v2 --job SLR --lane rtx_4070 --execute
    .\.venv\Scripts\python.exe -m scripts.run_experiment run --bundle runs/pre_campaign/v2 --job SLR --lane rtx_4070 --execute
    .\.venv\Scripts\python.exe -m scripts.run_experiment smoke --bundle runs/pre_campaign/v2 --job SMIX --lane rtx_4070 --execute
    .\.venv\Scripts\python.exe -m scripts.run_experiment run --bundle runs/pre_campaign/v2 --job SMIX --lane rtx_4070 --execute
    .\.venv\Scripts\python.exe -m scripts.run_experiment smoke --bundle runs/pre_campaign/v2 --job C0 --lane rtx_3070 --execute
    .\.venv\Scripts\python.exe -m scripts.run_experiment run --bundle runs/pre_campaign/v2 --job C0 --lane rtx_3070 --execute
    .\.venv\Scripts\python.exe -m scripts.run_experiment smoke --bundle runs/pre_campaign/v2 --job C1 --lane rtx_3070 --execute
    .\.venv\Scripts\python.exe -m scripts.run_experiment run --bundle runs/pre_campaign/v2 --job C1 --lane rtx_3070 --execute

Ledger inspection and explicit recovery after independent process inspection:

    .\.venv\Scripts\python.exe -m scripts.run_experiment budget --lane rtx_4070
    .\.venv\Scripts\python.exe -m scripts.run_experiment recover --lane rtx_4070 --reservation-token TOKEN --confirm-process-dead

The bundle does not contain all production inputs. Transfer it together with the trusted paths under
`data/shards/reduced_5pct_v1`, `data/schedules/reduced_5pct_v1`, `data/tokenizer_final/tokenizer.json`, and model/config paths recorded by
the trusted baseline configuration. Obtain the reviewed source from branch
`codex/g3.5-pre-campaign-experiments`; Git does not include the ignored inputs or bundles.
Integrate the successor baseline recipe and run identity before G4 sections 3–7;
create the fresh 1B baseline after those G4 sections for G5.
