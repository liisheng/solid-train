# Project status

Updated 2026-09-09T10:33:39+08:00. This is a coordination snapshot; artifact checks and
immutable scope contracts determine gate outcomes.

| Snapshot | Value |
|---|---|
| Current milestone | Reduced G3 repairs R1–R3 complete; independent source review clear; G4 readiness checks remain |
| Branch | `codex/g3-baseline-recipe`; reviewed repair checkpoint follows `9078430`; user authorized commit/push |
| Current verified source identity | 105-file manifest `af6651cea79b899c78bd183dd8e3c08547bd887b9213c85fc9441ac11cc56bfc`; [repair evidence](g3/REVIEW_FIXES.md) |
| Next task | Continue G4 tool checks and successor production evidence/profile; applicable recovery and human review |
| Detailed evidence | [Integration evidence](g3/INTEGRATION_EVIDENCE.md) / [short handoff](g3/CURRENT_HANDOFF.md) |

Accepted scope remains `configs/campaign/submission_scope_v1.yaml`: fixed 1B baseline,
conditional 3–5B, at most one comparison after baseline export and full evaluation.
The recipe is fresh seed 1337, 3,815 updates, batch 8 x 32 x 1,024, WSD 38/3395/382,
1,000,079,360 loss tokens, full-dev monitoring and recovery cadence 100 plus completion.
Frozen configs and the user's RULES are unchanged.

## Gate snapshot

| Gate | Verified state |
|---|---|
| G0 | Foundation/external approval requirements remain separate. |
| G1 | Reduced aggregate PASS, 69/69, 550,094,903 distinct stable tokens; original full-scale G1 unpassed. |
| G2 | `REDUCED_SCOPE_G2_PASS` under any-machine amendment; canonical full-scale G2 NOT_RUN. |
| G3 | Six sections implemented; local correctness PASS; aggregate `REDUCED_BASELINE_RECIPE_BLOCKED_FOR_G4`; original G3 NOT_RUN. |
| G4 | NOT_RUN; required tools, sustained changed-input profile, applicable recovery/takeover and teammate review remain. |
| G5 | NOT_RUN; no main baseline launched. |
| G6 | NOT_RUN; baseline full evaluation, verified release assets and human approvals remain. |

## Review repair verification

- Three agents fixed R1–R3; a separate reviewer found no unresolved actionable issue.
  Root full suite: 715 passed in 124.08s. Independent focused suite: 90 passed.
- Compilation, isolated build and whitespace checks pass; all 34 wheel modules byte-match
  source. Frozen inputs and unrelated user files remain unchanged. No agents remain active.
- New runner identity fields deliberately change IDs; old manifests and the section-6
  report below are historical. This checkpoint contains the verified repairs after `9078430`.

## Historical final section-6 evidence

- Aggregate: `runs/verification/g3-section6-integration-final-v2/reduced_g3_report.json`,
  SHA-256 `a587cca114ee8e6444770117ab2a96c2f3ee1b21aaf0c524c03ff8cf840209ef`.
- Fresh postverification PASS: 157 checks; successor SHA-256
  `43f69a23d11d52991ec46a35a8918d846e9c66da4636321740e8a8a443919188`.
- Final model/BF16 real-input 8 versus 4-plus-resume-to-8: exact model/optimizer/RNG/
  validation state, counters/cursor/order; all 753 dev references; expected corruption and
  early-export rejection; separate positive completed-decay fixture. Full timing history retained.
- Final tree: 675 tests passed, zero failures/skips; compilation/build/eligibility/diff
  checks passed. All 32 wheel modules matched source. Environment: 98 checks passed.
- Sol and Terra independently approved final source/evidence and the blocked aggregate;
  this does not replace G4 teammate approval. Luna implemented the initial rehearsal;
  Sol hardened verifier/timing and composite opening; Terra repaired audit scope; root
  ran final checks/rehearsal and verified artifacts.
- Fixed per-microbatch loss scalarization, duplicate checkpoint writes, timing overwrite,
  composite launch defects and audits incorrectly scanning ignored run outputs. Protected
  history is preserved; formerly failing alignment tests now pass.

## Blockers and efficiency limits

- **Review findings R1–R3 resolved:** metrics rollback preserves superseded work, full
  evaluation requires trusted complete coverage, and runtime digest guards enforce the
  fixed recipe. See [repairs](g3/REVIEW_FIXES.md). CPU-adapted trainer regression is not
  successor CUDA/production evidence; G4 must bind new evidence to the repaired source.
- **Docker:** build could not connect to `dockerDesktopLinuxEngine`; container tests did
  not run. Owner: G4 verification-environment operator. Restore a working approved engine
  and run the documented build/test against the bound source.
- **Lint:** Ruff unavailable; no lint PASS. Owner: G4 verification-environment operator.
  Supply an authorized existing/container runtime, run lint and resolve findings. No host
  packages were installed. These remain blocking requirements, not waived checks.
- The short final rehearsal measured 50,989–51,287 tok/s after its first update and
  5.445 GiB allocated VRAM. Historical reduced profile was 66,989.59 weighted tok/s.
  G4 must resolve the rate gap in its one sustained changed-input profile; no speedup or
  current total-campaign runtime claim is made. Trainer-main and outer process times are
  separate; complete accounting must include all invocations and later export/evaluation.
- Organizer scoring details/harness Git provenance remain provisional; local evaluation
  content and revisions are pinned. Full CUDA/BF16 benchmark runtime is unmeasured.
- Final-v2 is the retained historical section-6 rehearsal; no current-source production
  rehearsal was launched for these repairs. Earlier failed attempts are preserved.
  Existing unrelated build/metadata files and RULES remain; data/runs are ignored local
  artifacts that require separate transfer or reproduction. The user authorized committing
  and pushing these reviewed G3 repairs on 2026-09-09; this does not
  authorize a model release or baseline launch.

Sections 1–5 evidence remains linked from [the section index](g3/README.md). Frozen
historical PENDING fields are superseded by external evidence, never edited into passes.
