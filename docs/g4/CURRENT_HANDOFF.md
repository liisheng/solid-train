# Current G4 handoff

Current coordination amendment: the approved pre-campaign experiment stage now
precedes section 3. Follow [the experiment handoff](../experiments/README.md),
record its outcome and integrate the successor baseline recipe before resuming
G4 sections 3–7. The section 1–2 evidence below is historical and may be reused
only where its source, inputs and settings remain applicable. Current branch is
`codex/g3.5-pre-campaign-experiments`; no experiment or main training has started.

Updated 2026-09-09. **Sections 1–2 PASS; section 3 NEXT.** No sustained profile,
baseline, full scoring or later-section execution occurred. G4 itself is not passed.

Branch `codex/g4-verification`, HEAD `4e27266e5afafd50ddc3bb7c840e358bde3c74ec`;
section-2 changes are **uncommitted**. Previous verified implementation was `201bf38`.
Current 151-file working-tree identity:
`63e83aaac746a7752208d27fbd964e8202193192c22eaeca29a6857f917da656`.

Changed: new `scripts/profile_baseline_training.py`, `scripts/profile_telemetry.py`
and matching test files; lightweight timers in `train.py`, CPU assertions in
`tests/test_metric_ledger.py`; profile plan and coordination docs. Model/optimizer,
reader, frozen inputs and unrelated build/egg-info/RULES are preserved. Section-1
input-opening proof is reusable; historical timing is not current-source evidence.

Evidence root: `runs/verification/g4/section-02/20260909-1127/`.
`report.json` SHA-256:
`2f24bafcb22962a0505062857bd1f66be7e401fcc1c400e2cfa006e3beccc226`.
The source manifest is `dry-run/source_manifest.json`; input manifest SHA-256:
`dfefb12cdd17aba1779f9c99d919926c130ae908d6cb1133a0bd4f963deebc44`.
See [PROFILE_PLAN.md](PROFILE_PLAN.md) for exact commands and remaining hashes.

Verified: Docker build, **756 tests/zero skips**, Ruff, compile, dependencies;
Windows profiler tests **41 passed**; environments 98/97 checks. All 149 copied
image files and 34 installed modules match. Rechecked 133 payloads and metadata;
real dry-run retains runner `baseline-9d50a02e9d741daf`, captures telemetry, launches
nothing. Sol/Terra approve; root checked final hashes. All agents finished.

Next safe command: `Get-Content docs/g4/03-sustained-profile.md`. A separately
requested section 3 uses the reviewed 2400 optimizer-second / 800-update / 3600
wall-second bounds, excluding 38 updates and requiring ≥1800 valid seconds. Full
3815-update recipe remains intact. No automatic launch, retry, commit or push.

No collector blocker remains. Refresh load/space/custody before launch: Docker-build
telemetry showed memory pressure and competing GPU use. Final check found no Python
processes. Sustained rate gap, recovery/takeover, budget and human approval remain.
Ignored `data/` and evidence require separate transfer; Git alone is insufficient.
