# Current G4 handoff

Updated 2026-09-09. Section 1 PASS; sections 2–7 have not started. The user requested
one chat per section and authorized committing/pushing the completed verification
work plus this guide. No later section is launched by that request.

Branch: `codex/g4-verification`; verified implementation and guide commit
`201bf38f7a1507f9bc139d9ef41c10e9d16c3302` (after G3 `d5016f0`). Later coordination-only
commits do not change that implementation. Inspect branch HEAD and local changes.
Source/config/tooling proof is the 148-file manifest
`runs/verification/g4-part1-20260909/final-v2/source-manifest.json`, SHA-256
`0b20afde26466900ad23ae78345f16e8873f1e281d149f003269fe89fc9f6da1`.
The guide changes documentation only after that tested snapshot.

Evidence: [VERIFICATION.md](VERIFICATION.md) and local
`runs/verification/g4-part1-20260909/report.json`. Docker and Windows each passed
715 tests with no skips; Ruff/compile/build/dependencies passed; 34 installed modules
match source; 141 production-input checks and actual CPU input-opening reads passed.
Docker restored by preserving stale socket directories. Lint and two test portability
issues repaired; fixed configs, constraints and production inputs unchanged.

Next: [section 2](02-profile-preparation.md). The existing historical profiler uses
different horizon/schedule/validation controls; prepare a tested wrapper around the
current baseline runner before section 3's sustained measurement. No trustworthy
current sustained profile or full-campaign forecast exists. Recovery/takeover, budget
and actual teammate approval remain. G2's any-machine amendment does not waive G4.

Refresh source/environment, process/load and disk observations before GPU work. No
training is left running by section 1. Preserve unrelated build/egg-info/RULES and
historical evidence. `data/` and `runs/` are ignored and require separate transfer or
reproduction; a fresh checkout alone is insufficient. No automatic commit/push or
baseline launch is authorized for later sections by this handoff.
