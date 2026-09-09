# Section 1 — Verification prerequisites (complete)

## Purpose and current result

Make the development and CPU verification environments usable, and verify the actual
production inputs before any expensive GPU work. This section completed on 2026-09-09.
Do not repeat it wholesale when opening section 2.

Read [VERIFICATION.md](VERIFICATION.md), `docs/ENVIRONMENT.md`, and the current handoff.
The authoritative local report is `runs/verification/g4-part1-20260909/report.json`.
Its final source manifest covers 148 files with SHA-256
`0b20afde26466900ad23ae78345f16e8873f1e281d149f003269fe89fc9f6da1`.

## Work already completed

- Docker socket startup failure repaired without resetting/deleting Docker data.
- CPU image corrected: constrained CPU PyTorch, Ruff 0.11.13, required repository
  files, checkout-bound subprocess imports, local corpus/build metadata excluded.
- Ninety-four lint findings and two test portability issues resolved.
- Full Docker suite: 715 passed in 81.59 seconds; Windows: 715 in 133.90 seconds;
  no failed or skipped tests. Lint, compile, dependency consistency and build pass.
- Windows environment: 98 checks; CPU environment: 97 checks; all pass.
- All 34 installed modules match source; 141 real-input checks and actual CPU
  start/component-boundary/end/development batch reads pass.
- No training process detected at the recorded check. Disk and process observations
  are point-in-time facts and must be refreshed before the next GPU run.

## Reopen only if evidence becomes inapplicable

Check source hashes and changed-file scope first. A fresh machine needs environment
verification and actual availability/integrity of ignored artifacts. A changed source
needs the applicable regression checks. Missing logs require transfer/reproduction;
an old PASS label alone is insufficient.

```powershell
docker build -t tinybench-lm:verify .
docker run --rm tinybench-lm:verify
docker run --rm tinybench-lm:verify python -m ruff check src scripts tests train.py generate.py evaluate.py
.\.venv\Scripts\python.exe scripts/check_environment.py
```

Use unique successor evidence paths if rerunning. Preserve the G3 and section-1 history,
the fixed configs, production data, and unrelated local changes. Do not launch a profile
or the baseline as part of this section.

## Completion and next chat

All required verification checks must pass or be explicitly unresolved. Retain source
and input identities, logs/exit codes, environment facts, and process/space observations.
Update [VERIFICATION.md](VERIFICATION.md) with a dated successor if needed. Next is
[section 2](02-profile-preparation.md), preparing a correct measurement harness.

## Prompt if this section must be reopened

```text
Work only on G4 section 1 in docs/g4/01-verification.md. Read AGENTS.md,
.agent/CONTINUITY.md, docs/STATUS.md, docs/g4/README.md and CURRENT_HANDOFF.md.
Determine which existing verification evidence is still applicable and rerun only
the invalidated or missing checks. Fix findings and record successor evidence.
Do not start profiling, production training or another section. Finish with the
required short handoff and status updates.
```
