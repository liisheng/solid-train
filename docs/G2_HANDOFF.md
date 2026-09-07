# G2 handoff — reduced-scope real-shard verification

**Verified:** 2026-09-07T17:03:00+08:00 (Singapore)

The active G2 contract is the narrow, user-authorized amendment in
`configs/operations/g2_scope_v2.yaml` (digest in its sidecar). It applies to the accepted
`reduced_5pct_v1` scope and allows any compatible machine. The machine name and hardware
facts are recorded for reproducibility and efficiency reporting; they are not a gate.
`configs/operations/measurement_v1.yaml` remains unchanged and canonical full-scale G1/G2
claims remain separate.

The frozen G2 engineering checks are real-shard training with decreasing loss, exact
interruption/resume state and schedule identity, closed-fail corruption handling,
export/reload, and evaluation of the original source engineering export in a fresh process.
The amended artifact requirement is `fresh_process_evaluates_source_artifact`; the old
cross-machine wording is retained only in historical evidence.

## Existing reduced evidence

`runs/reduced_campaign/reduced_5pct_v1/aggregate.json` records 69/69 reduced checks and
550,094,903 selected stable tokens. Fresh local recovery is in
`runs/reduced_campaign/reduced_5pct_v1/recovery/evidence.json` and reports exact state,
ordered update-reference hashes, corruption rejection, and export/reload. Its validation
loss decreases from 9.111642837524414 to 7.794910430908203. The associated run config records
BF16 supported/measured stable with PASS, bfloat16, strict deterministic algorithms, and TF32
disabled.

The local real-shard profile is retained as efficiency evidence in
`runs/reduced_campaign/reduced_5pct_v1/profile/measurement.json`: 1,901.947794 optimizer
seconds and 66,989.592558 weighted tokens/s. Keep training, validation, checkpoint,
evaluation, and preparation timings separate; these are observed local reduced-profile facts,
not projections or canonical G4 promotion evidence.

## Any-machine execution

Check the environment into a new report for the machine being used. This records capability
facts but does not replace BF16/determinism assertions from the fresh run config:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py --json `
  --output runs\reduced_campaign\environment_check_<machine>.json
```

Before execution, verify exact source artifact identities with the focused helper. The active
path requires only the reduced shards/schedules already used by recovery, the source export,
its step-zero provenance, the scope amendment, and the environment report. No full
corpus/state transfer or machine distinction is required:

Run recovery in a fresh, nonexistent directory. This is the only bounded training run
required by G2; do not launch the baseline campaign or a long new profile:

```powershell
.\.venv\Scripts\python.exe scripts\verify_real_training.py `
  --shard-root data\shards\reduced_5pct_v1 `
  --train-schedule data\schedules\reduced_5pct_v1\recovery.json `
  --validation-schedule data\schedules\reduced_5pct_v1\validation_dev.json `
  --output-dir runs\reduced_campaign\reduced_5pct_v1\g2_recovery_local
```

Evaluate the original source export in a fresh process. The helper binds SHA256/size for the
export, provenance, scope amendment, and environment report; verifies embedded step-zero
provenance; and calls existing `verify-release`, which performs the deterministic fixed-batch
logits/loss inference check:

```powershell
.\.venv\Scripts\python.exe scripts\g2_handoff.py verify-local `
  --export runs\reduced_campaign\reduced_5pct_v1\recovery\engineering_export.pt `
  --provenance runs\reduced_campaign\reduced_5pct_v1\recovery\resumed\step_zero_provenance.json `
  --scope configs\operations\g2_scope_v2.yaml `
  --environment-report runs\reduced_campaign\environment_check_<machine>.json `
  --output runs\reduced_campaign\g2_source_local.json
```

Map the active amended requirements:

```powershell
.\.venv\Scripts\python.exe scripts\g2_handoff.py report `
  --scope configs\operations\g2_scope_v2.yaml `
  --source-evidence runs\reduced_campaign\g2_source_local.json `
  --recovery-evidence runs\reduced_campaign\reduced_5pct_v1\g2_recovery_local\evidence.json `
  --output runs\reduced_campaign\g2_report_local.json
```

`REDUCED_SCOPE_G2_PASS` means every amended requirement has concrete evidence on the chosen
machine. The report also states `canonical_full_scale_g1: NOT_RUN` and
`canonical_full_scale_g2: NOT_RUN`; reduced success does not promote those gates. Missing,
tampered, substituted, malformed, or unsupported artifacts fail closed.
