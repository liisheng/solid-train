# Current G4 handoff

Publication authorized 2026-09-12T12:16+08:00 on `codex/g4-selected-recipe`;
172 tested source/input files rechecked unchanged. Earlier uncommitted/no-push
statements below describe the verification checkpoint before publication.

Updated 2026-09-12T12:13+08:00. Selected CONTROL integration is verified. Sections
1–2 remain historical PASS; section 3 is next. G4 itself is not passed.

Active branch `codex/g4-selected-recipe`, base HEAD
`7f85c87246c511471a2009ad3cba0d4ffecf1029`; integration changes are uncommitted.
No training/profile/full scoring, commit or push occurred.

Selected recipe: `configs/training/baseline_reduced_v2.yaml`, LR 0.0006,
base 70/20/7/3, fresh seed 1337, 3815 updates/1,000,079,360 tokens. Normalized SHA:
`dcd4d623a8f826f5e007ed33eea4bf2b653414c2088f4354a9b7e6e69a8a3f12`.
V1 stays immutable. Both exposure component files match v1 byte-for-byte; the new
contract binding changes exposure/training identity. Experiment weights cannot initialize G5.

Changed: successor contract/selection receipt, exposure digest registration, runner
and profiler selection, trainer startup guard, integration utility, tests and guides.
See [integration](SELECTED_RECIPE.md) for exact reproduction commands and custody.
The runner CLI/profiler default to v2; old Python harness defaults still use v1.

Evidence: `runs/verification/g4/selected-recipe-20260912/verification.json`,
`source-custody.json`, `integration.json`, `prepare.json`, `input-reads.json`.
The 247-file frozen verification manifest SHA is
`0c5d28389175ceaf76489f522be654030956bda84d1b42a45f2ac7639019065d`.
Docker 826 tests/97.05s and isolated Windows 826/194.52s PASS; Ruff, compile, builds
and dependencies PASS. 172 image source/input files and 38 installed modules match.
Production preparation identifies `baseline-f23398e32dc3100e`; CPU start/boundary/end
reads pass. No baseline run directory exists.

C0/C1 and failed-smoke files are consolidated. Teammate ledger is separate.
Extracted return is archived under `runs/handoff`; the root ZIP remains locked.
Docker is repaired; failed attempts and preserved runtime directories remain.

Next safe command: `Get-Content docs/g4/03-sustained-profile.md`. Refresh live
load/space/environment and regenerate the v2 profile plan in a fresh section-3
engineering directory before execution. Reuse only unaffected input proof. Sustained
profile, current-source recovery, target takeover, budget and two-person approval remain.
