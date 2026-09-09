# Retired seven-run draft — do not execute

This file preserves the unapproved, interrupted seven-run proposal. Its job names,
selection logic, commands, and authorization statements below are superseded.
The user approved Astra's three-run screen plus conditional two-run confirmation
on2026-09-09. Use [the active guide](README.md) and
[approved implementation specification](IMPLEMENTATION_SPEC.md). Nothing below
is current launch guidance or evidence of a gate pass.

This is the bounded successor to the historical experiment plan. Its contract is
[`configs/campaign/pre_campaign_v1.json`](../../configs/campaign/pre_campaign_v1.json),
and its authorization is recorded in
[`configs/campaign/submission_scope_v2.yaml`](../../configs/campaign/submission_scope_v2.yaml).
It informs the recipe for the fresh main campaign; it is not the main campaign.

## Jobs and machine assignment

| Job | Stage | Question | Machine lane | Dependency |
|---|---|---|---|---|
| P1 | 31M proxy | control, seed 1001 | RTX 4070 | none |
| P2 | 31M proxy | control, seed 1002 | RTX 3070 | none |
| P3 | 31M proxy | control, seed 1003 | RTX 3070 | none |
| P4 | 31M proxy | learning rate 0.001 | RTX 4070 | none |
| P8 | 31M proxy | educational mixture | RTX 4070 | none |
| F1 | 49M confirmation | full model control | RTX 4070 | all P jobs + selection |
| F2 | 49M confirmation | selected proxy recipe | RTX 4070 | P jobs, selection, verified F1 |

The 4070 is the mainline/control lane. The 3070 is used for the two seed
replications because their purpose is variability estimation. Do not silently
swap lanes: the runner checks the reported GPU model. If measured hardware
capability changes, issue a successor contract first.

## Prepare once

Run these commands from the repository root on a machine with the verified
reduced corpus and CUDA environment. Preparation only reads and hashes inputs;
it does not train.

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/run_experiment.py prepare
python scripts/run_experiment.py check
```

The bundle is written under `runs/pre_campaign/v1`. Copy the bundle and this
checkout to the other machine, then run `check` there. A changed contract,
source file, schedule, tokenizer, or manifest blocks the job.

## Plan, smoke, and run

Every planned command is read-only. `smoke` and `run` refuse to start unless
`--execute` is supplied. A reviewed measured BF16 stability report for the
current machine is also required.

For each proxy job, first print and review its immutable identity and command:

```powershell
python scripts/run_experiment.py plan --job P1 --lane rtx_4070
```

Then run the two-update smoke test on the assigned machine:

```powershell
python scripts/run_experiment.py smoke --job P1 --lane rtx_4070 `
  --bf16-evidence runs/verification/<machine>/bf16.json --execute
```

If smoke verification passes, start the full job using the same command with
`run` in place of `smoke`. Replace `P1` and the lane for each job. Run P1, P4,
and P8 on the 4070; run P2 and P3 on the 3070. They may run in parallel only
when each machine has its own checkout/bundle and no GPU is shared.

An interrupted full job is resumed with `--resume --execute` after checking
the latest checkpoint and invocation log:

```powershell
python scripts/run_experiment.py run --job P1 --lane rtx_4070 `
  --bf16-evidence runs/verification/<machine>/bf16.json --resume --execute
```

Never delete an interrupted run to make it look complete. The runner records
the command, GPU, evidence hash, console log, exit status, and verification
under `runs/pre_campaign/v1/jobs/<JOB>/invocations/`.

## Selection and confirmation order

After all five proxy jobs verify successfully, inspect the selection report:

```powershell
python scripts/analyze_experiments.py runs/pre_campaign/v1 `
  > runs/pre_campaign/v1/selection.json
```

The default is the control (`lr: 0.0006`, `mixture: base`). A candidate must
clear the preregistered relative improvement and slice-regression rules. Missing
slice evidence is inconclusive and retains control. No final holdout or
submission benchmark is used for selection.

Plan and run F1 only after the proxy selection is recorded. F2 is conditional:
it is run only when a proxy candidate survives selection, and only after F1 is
verified. If control is retained, omit F2 and record that it was redundant.

```powershell
python scripts/run_experiment.py plan --job F1 --lane rtx_4070
python scripts/run_experiment.py run --job F1 --lane rtx_4070 `
  --bf16-evidence runs/verification/<machine>/bf16.json --execute
python scripts/run_experiment.py plan --job F2 --lane rtx_4070
python scripts/run_experiment.py run --job F2 --lane rtx_4070 `
  --bf16-evidence runs/verification/<machine>/bf16.json --execute
```

The full-model result is a safety/confirmation check, not permission to launch
the baseline. A selected recipe receives a successor baseline contract and run
identity, followed by the remaining G4 checks.

## Boundary with G4 and G5

Experiment checkpoints never initialize the fresh main baseline. The baseline
starts from new weights under its own identity, even if an experiment is the
selected recipe. After P and applicable F jobs are complete, carry forward the
recipe, evidence, manifests, and selection decision into G4. Re-run affected
recipe/integration checks before sustained profiling, recovery, takeover, and
two-person freeze. Do not use this package to launch G5; there is no command in
this guide that starts the main baseline.
