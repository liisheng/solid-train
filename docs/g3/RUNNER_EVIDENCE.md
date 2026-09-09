# Reduced G3 section 4 runner evidence

Recorded 2026-09-08 on `codex/g3-baseline-recipe`. This section implements the bounded
runner and export custody; it does not launch the baseline or claim canonical G3/G4/G5/G6.

## Implemented custody

`scripts/run_reduced_baseline.py` provides read-only `prepare` and `launch` planning, plus
`verify` and `export` actions. Preparation loads the persisted two-component exposure plan,
verifies the ordered component identities and quotas, verifies the validation-dev manifest and
schedule, checks the frozen input hashes, and emits the exact command. The plan does not write
a run directory. Executable baseline launch remains blocked while section 5's evaluation
binding is `PENDING_SECTION_5`; bounded fixtures are separate from production launch custody.

The real-input dry-run was reproduced with:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py prepare > runs/verification/g3-section4-prepare.json
```

The final output is [g3-section4-prepare.json](../../../runs/verification/g3-section4-prepare.json),
SHA-256 `d47954310e3ba793e44007368d89f22bb1dd8641c87cebb5bf1416a1d18230`.
It records run identity `baseline-543f61c057466dc0`, the complete 44-argument command, exposure content hash
`6b1b747e47ea907810bf68f9ff941f61319ba7dbc7a034e3e2a746775666b0e3`, exposure plan file
hash `76f16bfc98620227e2070645e5e901d8a4a8811c3970509b92769ebd84f71f8f`, and development
schedule hash `50ddce9f4f12c5f70e6369170332d168c9a1783ee530c61ab4857c1b16ca61fd`.

The bounded command surface for the next sections is:

```powershell
# read-only identity/input plan
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py prepare

# print the exact launch command; --execute remains blocked while evaluation is pending
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py launch --run-dir runs/<baseline-run>

# bounded rehearsal plans preserve the 3815-update horizon; execution still needs section 5
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py launch --run-dir runs/g3-section6-engineering --stop-after-updates 8
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py launch --run-dir runs/g3-section6-engineering --stop-after-training-seconds 60

# resume plan after an interruption; use the identical run directory and identity
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py launch --run-dir runs/g3-section6-engineering --resume runs/g3-section6-engineering/latest.pt --stop-after-updates 16

# eligibility check at completion (an early rehearsal checkpoint must fail)
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py verify --run-dir runs/<baseline-run> --checkpoint runs/<baseline-run>/completed.pt

# export only the independently verified completed endpoint
.\.venv\Scripts\python.exe scripts/run_reduced_baseline.py export --run-dir runs/<baseline-run> --checkpoint runs/<baseline-run>/completed.pt --destination runs/<baseline-run>/baseline_export.pt
```

Currently `evaluation_binding` is the literal `PENDING_SECTION_5`; no resolved-binding CLI
exists yet. Section 5 owns implementing and validating the effective protocol and loader/runtime
binding, including provisional labels and final-holdout custody. A proposed successor object
has `status`, `protocol_id`, `protocol_sha256`, and `loader_lock_sha256`; this is a handoff
proposal, not a currently accepted schema. Execute remains blocked until that integration.
`--stop-after-training-seconds` bounds optimizer time, not total process wall time. The stop
controls do not shorten the declared WSD horizon or make an early checkpoint export-eligible.
The update stop is an absolute completed-update target, so a run stopped at 8 resumes toward
16 for another eight updates; it is not a per-invocation update count.

The training path accepts the exposure plan and both component schedules and opens one
`CompositeTokenStream`. Its single absolute cursor is persisted by the existing durable
checkpoint machinery. Full-dev validation remains the section-3 token-weighted replay. Final
scope requires the section-4 runner identity and fixed 100-update validation and recovery
cadence.

The runner identity binds scope, recipe/model/tokenizer/protocol hashes, the ordered exposure
plan and component hashes, development manifest/schedule identity, validation mode, cadence,
seed, horizon, WSD phases, batch geometry, precision and deterministic policy, environment,
and a pending evaluation binding. Launch writes `runner_identity.json` once. `train.py` records
its byte hash in every durable checkpoint and refuses a resume if the current identity differs.

`verify` and `export` require the durable source checkpoint itself to have a valid manifest,
fresh step-zero provenance, exactly 3,815 completed updates, 1,000,079,360 consumed loss
tokens, the completed composite cursor 976,640, fixed full-dev/100 cadence, matching batch/LR
arguments, WSD final LR exactly zero, and a matching runner identity. Export then reloads and
verifies a clean inference artifact and writes evidence bound to the source checkpoint hash.
Early-stopped, zero-decay, nonzero-final-LR, substituted-input, mismatched-cursor, or identity-
drifted sources fail closed. Best-dev, rolling latest, and the separately named completed
endpoint/fallback roles remain separate; no deletion or
automatic retention mutation was added.

## Verification

Focused regressions passed:

```text
115 passed in 11.00s (Sol); 83 passed in 10.63s (Terra, same set excluding release_evidence)
.\.venv\Scripts\python.exe -m pytest -q tests/test_reduced_baseline_runner.py tests/test_release_evidence.py tests/test_validation.py tests/test_exposure.py tests/test_durable_checkpoints.py tests/test_provenance.py tests/test_training_recipe.py
.\.venv\Scripts\python.exe -m compileall -q train.py scripts/run_reduced_baseline.py src/tinybench_lm/exposure.py tests/test_reduced_baseline_runner.py
```

The tiny durable fixture passed source eligibility, completed-decay checks, clean export, and
independent reload. Negatives cover early stopping, cursor drift, applied optimizer LR,
provenance drift, wrong expected run/frozen recipe, runner-manifest drift, substituted config,
direct train/input mismatch and invalid/overrun stops. Durable regressions cover source checksum
tampering; recipe regressions cover phase/counter arithmetic. The wheel fallback
build passed:

```powershell
.\.venv\Scripts\python.exe -m pip wheel . --no-deps --no-build-isolation --wheel-dir runs/verification/g3-section4-final/wheels
```

Artifact: `runs/verification/g3-section4-final/wheels/tinybench_lm-0.1.0-py3-none-any.whl`,
SHA-256 `be3147ec543bf5e3dcce5559bb32cb93d2f0ad6091319bc9d1eb7c3f3462b230`.
Ruff was unavailable. The required Docker attempt was blocked because Docker Desktop's Linux
engine pipe was unavailable. No package installation, main training, long profile, benchmark,
or baseline export was performed. Full exposure validation is startup work and the composite reader uses
the already-verified component identities without a second component scan.

Checkpoint custody planning remains bounded: fixed 100-update recovery plus completion gives
39 rolling recovery writes (about 21.7 GiB cumulative payload writes at historical size,
excluding verification reads, best-dev saves and completion copy). Existing best/latest retention remains, with a separately named
`completed.pt` endpoint/fallback and no automatic deletion. Historical G2 durable payloads were
596,075,694 bytes each; the steady latest+best+completed custody is approximately 1.67 GiB,
with transient checkpoint/export space additional. Actual campaign phase timings and completed
baseline checkpoint/export hashes remain future G5/G6 evidence.

## Approval state

Sol and Terra approved the identical final source below. Luna authored the initial implementation;
Sol repaired core identity/export and command/stop checks, which Terra independently reviewed.
Terra expanded adversarial tests, which Sol reviewed. Root checked source/artifact hashes,
preserved contract/RULES identities, command documentation and handoff. Agent approval does not
replace G4 teammate approval. No resolved evaluation binding, production launch or canonical
gate result is claimed. No new per-batch scans or synchronization were added; duplicate
startup component validation was removed. Actual throughput and full-dev timing remain unmeasured.

| Final reviewed file | SHA-256 |
|---|---|
| `train.py` | `6201677a862c48f1a9872cd35e74e8fbda42f1c4cfe660c840d5fb6895ccb8e0` |
| `scripts/run_reduced_baseline.py` | `747dc9e1fa8d544e81cf448c76139d0a1e1d98464b597073c853923948496659` |
| `src/tinybench_lm/exposure.py` | `7769ce45cd7ffad9c4ff7a86194dc17515bc62ad3f5d7f19735bee64ee062973` |
| `tests/test_reduced_baseline_runner.py` | `77f493b937297b0bcbb9e5b92b2882eb553f03dc135a293b07babaccf25868d5` |

Section 4 supersedes section 2/3 source hashes only for the changed exposure/trainer files;
their input artifacts and frozen contract remain unchanged. Evidence/artifacts under `runs/`
are ignored local files and require separate transfer or reproduction. Next: section 5 evaluation
binding and bounded evaluation rehearsal, then section 6 integrated checks.
