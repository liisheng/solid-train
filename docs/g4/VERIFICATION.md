# G4 part 1: verification environment and input readiness

Recorded 2026-09-09 on `codex/g4-verification`, based on G3 repair commit
`d5016f00739708687261875d15cbbf8a03b67904`. Verified implementation and the chat guide
were committed as `201bf38f7a1507f9bc139d9ef41c10e9d16c3302` on the user's later request.
Disposition: **G4 part 1 PASS**. This report does not pass G4 or authorize a baseline launch.

## Verification

Evidence directory: `runs/verification/g4-part1-20260909/` (ignored local files;
transfer or reproduce it separately from Git).

| Check | Result | Evidence |
|---|---|---|
| Docker build | PASS | `docker-build-final-v2.log` |
| Full CPU container suite | 715 passed, 81.59 seconds | `docker-pytest-final-v2.log` |
| Full Windows suite, final tree | 715 passed, 133.90 seconds | `windows-pytest-final-v2.log` |
| Ruff 0.11.13, all package/script/test/root Python entrypoints | PASS | `ruff-final-v2.log` |
| Windows CUDA environment | 98 checks PASS | `environment.json` |
| CPU container environment and dependency consistency | 97 checks PASS; dependencies consistent | `container-environment-final-v2.json`; `pip check` exit 0 |
| Compilation | PASS | Container `compileall`, exit 0 |
| Installed package custody | All 34 modules byte-match source | `container-source-final-v2.json` |
| Production artifacts | 141/141 checks PASS | `final/input-check.json` |
| Real input-opening preflight | PASS | `final/runner-prepare.json`, `final/batch-source-preflight.json` |
| Space and processes | No detected training process; about 214 GB free C:, 1.37 TB D: | `system-postcheck.json` |

The final 148-file source/config/tooling manifest is
`final-v2/source-manifest.json`, SHA-256
`0b20afde26466900ad23ae78345f16e8873f1e281d149f003269fe89fc9f6da1`.
The image ID is recorded in `image-id-final-v2.txt`.
The consolidated result and evidence hashes are in `report.json`; 107 copied
Python/project files match the final manifest. Neither full test suite skipped a test.

## Changes and findings resolved

- Restored Docker Desktop engine access by gracefully stopping it, preserving the
  inaccessible socket directories under timestamped backup names, and restarting.
  The exact paths and upstream report are in [ENVIRONMENT.md](../ENVIRONMENT.md).
  No factory reset, Docker data deletion, settings change, or host package install.
- Repaired the verification image: explicit constrained CPU PyTorch installation,
  pinned Ruff, missing entrypoint/documentation/constraints copies, and checkout
  imports for subprocesses. Excluded unrelated local build metadata and personal
  rules. Production corpus, run and checkpoint directories remain excluded.
- Resolved 94 Ruff findings: 23 unused imports, 6 assigned lambdas, 56 statement-layout
  findings, 2 unused locals, and 7 narrowly documented bootstrap-import exceptions.
  Training settings and numerical behavior were not changed.
- Corrected a test that required a historical Windows CUDA wheel label on CPU
  Linux. Historical facts remain bound to the recorded constraints; actual runtime
  public-version checks and the full environment checker remain enforced.
- The first complete container run passed 714 tests and failed one because its
  loader-binding test required ignored production data. That test now uses scoped
  data fixtures while executing real frozen-config validation, protocol/tokenizer
  checks, loader resolution, `prepare`, and `build_identity`. No test was skipped.
  The focused container runner suite passed all 14 tests before the final rebuild.

Initial source verification matched all 105 files in the G3 repair manifest.
The final source scope additionally includes `generate.py`, configs, constraints,
and container tooling. Twenty-seven Python files changed during lint/test repairs.
Independent agent review found no actionable integrated-diff issue; root reviewed
the final data-fixture repair. This is not the required human teammate approval.

## Input and machine evidence

All 129 stable-training and 4 development shard payloads passed hash/size checks.
Four manifest byte identities, the two active manifest content hashes, and both
schedules were checked. The 141 checks were repeated with identical observations.
Final/reserved payloads remained closed.

Actual production preparation resolves runner ID `baseline-9d50a02e9d741daf`.
CPU batch reads passed at cursors 0 to 8, across the component boundary
537108 to 537116, at 976632 to 976640, and on development input. The later source
revision changes only the self-contained test; production code and input hashes
remain those verified by this preflight.

Windows uses CPython 3.12.6, torch 2.5.1+cu124, and RTX 4070 SUPER. CPU Docker
verification uses Python 3.11 and torch 2.5.1+cpu. Environment facts do not certify
sustained performance or device headroom. Docker startup also restarted existing
non-training application/database containers; recheck competing load before profiling.

## Remaining G4 work

Run the sustained 30–60 minute production profile, account for full runtime costs,
complete applicable recovery/takeover evidence, confirm the dated baseline budget,
and obtain teammate freeze approval. No production training, sustained profile,
full benchmark scoring or model publication was performed. The verification run itself
made no commit/push; the later user request authorized Git publication of these changes.
Historical G3 blocked reports are preserved; this successor report resolves only
the verification prerequisites actually measured here.
