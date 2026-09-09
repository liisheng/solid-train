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

## Teammate ZIP: contents and placement

Use branch `codex/g3.5-pre-campaign-experiments`, implementation commit
`664c617` or a later compatible documentation-only update. The sender will provide
a ZIP with the following six items at its top level. Folders must include every
file and subfolder. Windows may hide the `.json` extensions; keep the actual names.

| Item at ZIP top level | Destination relative to the teammate's repository root |
| --- | --- |
| `stable/` | `data/shards/reduced_5pct_v1/stable/` |
| `validation_dev/` (folder) | `data/shards/reduced_5pct_v1/validation_dev/` |
| `stable_train.manifest.json` | `data/shards/reduced_5pct_v1/stable_train.manifest.json` |
| `validation_dev.manifest.json` | `data/shards/reduced_5pct_v1/validation_dev.manifest.json` |
| `validation_dev.json` (file) | `data/schedules/reduced_5pct_v1/validation_dev.json` |
| `v2-advisory/` | `runs/pre_campaign/v2-advisory/` |

On the sender's machine, obtain each item from the destination path in this table
under the sender's repository root. Use the actual `v2-advisory` folder, not a
renamed copy of `v2`. Optionally include this handoff in the ZIP for convenience;
it is also available at `docs/experiments/LLM_HANDOFF.md` in the checkout.

Extract the ZIP to a temporary folder, then create the destination parents and
copy the six items to the paths above. The repository root is the directory with
`train.py` and `pyproject.toml`. Avoid extra nesting such as `stable/stable/` or
`v2-advisory/v2-advisory/`. Preserve any existing runs; investigate conflicting
files rather than replacing an active run. The code and tokenizer come from Git,
and the teammate uses her own verified CUDA environment.

From that repository root, verify the transferred bundle:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiment check --bundle runs/pre_campaign/v2-advisory
```

A successful input check is preparation, not permission to begin C0/C1. The initial
ZIP does not yet contain completed screening evidence. If screening selects a
candidate, the sender must later send the updated complete `v2-advisory/` folder,
including selection, checkpoints, sidecars, metrics and receipts. Check it again
before following the teammate commands in OPERATOR_GUIDE.md.

## Experiment execution

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
