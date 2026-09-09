# Project status

Updated 2026-09-09T11:25:00+08:00. This is a coordination snapshot; artifact checks and
immutable scope contracts determine gate outcomes.

| Snapshot | Value |
|---|---|
| Current milestone | G4 part 1 PASS: Docker/container tests, lint, source/input/environment preflight complete |
| Branch / verified implementation commit | `codex/g4-verification`; `201bf38f7a1507f9bc139d9ef41c10e9d16c3302` (part 1 and guide) |
| Current verified source identity | 148-file manifest `0b20afde26466900ad23ae78345f16e8873f1e281d149f003269fe89fc9f6da1` |
| Next task | G4 section 2: prepare/test the production profiler; one chat per section |
| Guide and evidence | [Seven-section guide](g4/README.md) / [current handoff](g4/CURRENT_HANDOFF.md) / [verification](g4/VERIFICATION.md) |

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
| G3 | Six sections implemented; local correctness PASS; historical blocked aggregate preserved, tool blockers superseded by G4 part 1; original G3 NOT_RUN. |
| G4 | No gate PASS; part 1 PASS. Sustained profile, applicable recovery/takeover, budget and teammate review remain. |
| G5 | NOT_RUN; no main baseline launched. |
| G6 | NOT_RUN; baseline full evaluation, verified release assets and human approvals remain. |

## G4 part-1 verification

- Final CPU container suite: 715 passed in 81.59s; final Windows suite: 715 passed
  in 133.90s. Ruff 0.11.13, compilation, dependency consistency and diff checks pass.
- Docker startup repaired; CPU image supplies all test inputs without local corpus.
  All 34 installed modules match source. Both environments pass their pin checks.
- 141 production artifact checks pass; real prepare and CPU start/boundary/end/dev
  reads pass. No training detected; about 214 GB free C: and 1.37 TB D: at preflight.
- Lint and two test portability repairs change source identities. Frozen configs,
  constraints, production inputs and unrelated user files are preserved. Evidence in
  `runs/verification/g4-part1-20260909/`; agents finished, implementation committed
  as `201bf38`. User authorized pushing the G4 branch; no later section has started.

## Historical G3 review repair verification

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
- **Docker and lint blockers resolved:** G4 part-1 successor checks pass; original
  failed/unavailable logs remain historical. No host packages were installed.
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
