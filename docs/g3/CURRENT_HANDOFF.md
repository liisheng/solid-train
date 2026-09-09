# Current G3 handoff

- **2026-09-09 repair completion supersedes the open findings below:** R1–R3 fixed;
  separate independent reviewer found no unresolved actionable issue. Root full suite
  715 passed; independent focused suite 90 passed; compile/build/diff passed and all
  34 wheel modules match. See [REVIEW_FIXES.md](REVIEW_FIXES.md) for behavior, identities
  and successor evidence. Branch `codex/g3-baseline-recipe`; this reviewed repair checkpoint
  follows `9078430`, with commit/push authorized at 2026-09-09T10:33:39+08:00.
  No baseline/full scoring started. Docker/Ruff remain unavailable.
- New trusted-digest identity fields intentionally change runner IDs. Do not resume old
  engineering manifests through the repaired runner or relabel final-v2 as current-source
  proof. Continue [G4_HANDOFF.md](G4_HANDOFF.md) with successor evidence, tool checks,
  sustained changed-input profiling and applicable recovery/human review.

## Historical checkpoints

- **2026-09-09 checkpoint:** user authorized committing/pushing the G3 implementation and
  review on `codex/g3-baseline-recipe`, based on `f643e9f`. The uncommitted/no-push statements
  below describe the historical section-6 verification, not this publication request.
  R1–R3 remain open; generated build/egg-info, user RULES and ignored run/data artifacts are excluded.

- **2026-09-09 review supersedes the tools-only next step below:** changes requested in
  [PRE_G4_REVIEW.md](PRE_G4_REVIEW.md), R1–R3 (metrics rollback, full-evaluation mode, runtime
  digest enforcement). Fresh675 tests and original-evidence postverification pass; all103
  source files still match. No implementation changed. Fix/regress these issues before clean
  G4 sign-off; preserve existing evidence and the remaining Docker/lint/profile/review tasks.

- Section 6 implementation/local verification complete; readiness is
  `REDUCED_BASELINE_RECIPE_BLOCKED_FOR_G4` because Docker and Ruff could not run.
  Canonical G3–G6 remain NOT_RUN. Branch `codex/g3-baseline-recipe`, HEAD
  `f643e9f09c0700686222712eb58d90e4a0fe1de2`; no commit/push.
- Current report: `runs/verification/g3-section6-integration-final-v2/reduced_g3_report.json`,
  SHA `a587cca114ee8e6444770117ab2a96c2f3ee1b21aaf0c524c03ff8cf840209ef`.
  Fresh postverification passed 157 checks, SHA
  `43f69a23d11d52991ec46a35a8918d846e9c66da4636321740e8a8a443919188`.
  The 103-file source/test manifest is
  `runs/verification/g3-section6-root/final-source-manifest.json`, SHA
  `8ebf3d6ec5f863c94a0807bd1967d86841d9d25f4c5c7b28928a8153bba5dbdd`.
- Exact real-model/BF16 8 versus 4-plus-resume-to-8, full-dev coverage, corruption rejection,
  early-export rejection, and separate positive completed-decay fixture passed. All three
  invocations retain timing/history. Fixed horizon remains 3,815; no baseline was launched.
- Final2 suite: 675 passed, zero failures/skips. Compile/build/eligibility/diff and wheel
  source checks passed; environment 98/98. Sol/Terra approved source/evidence and aggregate
  with tool blockers. Root independently verified artifacts. Prior failed/incomplete attempts
  are preserved; only final-v2 is current.
- Changes: trainer efficiency/timing and composite opening; fail-closed integration verifier
  and tests; root-script Docker copies; audit scope/regressions; coordination/evidence docs.
- Next: [G4_HANDOFF.md](G4_HANDOFF.md). Resolve Docker/lint, measure one sustained production
  profile, check applicable recovery/takeover and obtain teammate approval. Short rate was
  ~51k versus historical ~67k tok/s; no speedup/current campaign forecast. Agent review is
  not teammate approval. Preserve frozen inputs/RULES and unrelated dirty files; transfer
  or reproduce ignored artifacts separately. Details: [INTEGRATION_EVIDENCE.md](INTEGRATION_EVIDENCE.md).
