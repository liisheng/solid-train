# Project status

2026-09-12T14:39+08:00: User states intent to proceed and requests publication of
the G4 scope/budget/freeze package to `origin/codex/g4-selected-recipe`.
All494 bound files and bundle digest reverified unchanged. Publication does not
record a missing approval receipt, resolve the documented gaps or launch G5.
Git attributes preserve the new hashed artifacts' bytes on checkout.

2026-09-12T14:35+08:00: **G4 section7 package complete; amended G4 BLOCKED.**
Root completed [freeze review](g4/FREEZE_REVIEW.md),494-file hashed bundle and
[G5 handoff](g4/G5_HANDOFF.md). Bundle SHA256
`e5d5e32c08547692d94a5e30463cd76d633558ebb6bd363a8a90984994a17462`.
Branch `codex/g4-selected-recipe`, HEAD `d1080576789794cb35f8839d6789f540223b7653`;
documentation/scope overlay uncommitted, production172-file source unchanged.
Fresh custody462 unique files PASS; successful launch plan only, no G5 directory.
User confirms continuous machine availability. Backup destination/custodian/restore,
measured budget disposition and two actual matching human approvals remain blockers.
Canonical G4/G5/G6 unpassed; no training/scoring/backup/commit/push or active script.
Next: operator closes review blockers, then two humans approve the resulting bundle.
This supersedes earlier section-next/availability entries below.

2026-09-12T14:03+08:00: **G4 section 6 complete; budget conditional, section 7 NEXT.**
[Budget](g4/BASELINE_BUDGET.md) and calculation project 4.406/4.753h training,
53.702/54.057h with provisional reserves. One candidate; evaluation p90 NOT_RUN,
measured fit BLOCKED. Availability, deadline conflict, backup custody and two-person
approval remain. Sections 1–4 complete; 5 not applicable under g4_scope_v2.
Branch `codex/g4-selected-recipe`, HEAD `d1080576789794cb35f8839d6789f540223b7653`;
root completed planning. All 172 source hashes and 14 input hashes match; arithmetic
passes. Prior 826-test/lint/build evidence applies. Docs uncommitted; no training,
scoring, backup or publication. G4 incomplete; older next-section entries superseded.

2026-09-12T13:54+08:00: **Single-machine scope approved; section 6 NEXT.**
User chose remaining training, evaluation and export on the RTX 4070 SUPER.
`configs/operations/g4_scope_v2.yaml` replaces active cross-machine takeover with
verified local recovery; section 5 is NOT_APPLICABLE_UNDER_AMENDED_SCOPE.
See [scope decision](g4/SINGLE_MACHINE_SCOPE.md). Sections 1–4 evidence remains;
budget, evaluation-runtime evidence and two-person freeze approval remain outstanding.
Original canonical G4/takeover remain unpassed. Branch `codex/g4-selected-recipe`,
HEAD `d1080576789794cb35f8839d6789f540223b7653`; root owns this local scope/docs update.
No training, backup, scoring, commit or push performed. Earlier next-section and
target-machine blocker statements below are historical and superseded by this entry.

2026-09-12T13:44+08:00: **G4 section 4 PASS — local recovery**. Exact CUDA BF16
8 versus4+resume and checkpoint4/log8 rollback match; replay archive and retry,
corruption and early-export rejection pass. Review20checksPASS,90artifact hashes
and172source hashes match. No training remains. [Evidence](g4/RECOVERY_EVIDENCE.md).
Branch `codex/g4-selected-recipe`, HEAD `1cd448e`; documentation uncommitted,
production source unchanged. Section3 checkpoint/timing and826-test verification
remain applicable. **Next section5 takeover**; G4 itself, budget and approvals remain.
Supersedes section4 running state below; no later execution or publication.

2026-09-12T13:30+08:00: **G4 section 4 IN PROGRESS**, controller PID16416,
`runs/verification/g4/section-04/20260912-primary-02`. Selected v2 bounded exact
recovery and rollback rehearsal; source unchanged on `codex/g4-selected-recipe`
HEAD `1cd448e`. Attempt01 failed before training on sandbox corpus access and is
preserved. User observes with `section-04/observe-recovery.ps1` and reports completion.
Inspect active_process.json before any relaunch. [Coverage](g4/RECOVERY_EVIDENCE.md).
No section4/G4 PASS yet; no later section or publication.

2026-09-12T13:24+08:00: Section-3 commit `1cd448e` successfully pushed to
`origin/codex/g4-selected-recipe`; remote SHA matches. User explicitly approved
the payload/destination after the review block. Supersedes blocked-publication
status below. This local status update remains uncommitted. Next: section 4.

2026-09-12T13:21+08:00: Section-3 documentation committed locally as `1cd448e`
(five files). Push dry-run passed; actual push was blocked by automatic approval
review pending explicit approval of this payload and GitHub destination. No push
success claimed. This publication-status update is local and uncommitted.

2026-09-12T13:19+08:00: User authorized publication of the verified section-3
documentation to `origin/codex/g4-selected-recipe`. This checkpoint contains five
coordination/evidence documents; raw runs/checkpoints remain local ignored evidence.
The uncommitted/no-publication wording below describes the verification checkpoint.

2026-09-12T13:09+08:00: **G4 section 3 PASS (RTX 4070 SUPER source lane)**.
Single profile exited 0 at 13:05:04, 590 updates / 154,664,960 tokens; 37.392 valid
minutes, **64,498.66 weighted tokens/s**, 31.827% sampled device headroom.
All 17 postverification checks pass, including checkpoint and 133 payload hashes;
172 tested source files unchanged. Ending checkpoint SHA `efcfdd30…7562dc`, cursor
151,040. No training remains. Whole-machine RAM briefly fell to 98.574 MiB free;
the historical rate-gap cause remains unresolved. [Evidence](g4/PROFILE_EVIDENCE.md).
**Next: section 4 local recovery**; G4 itself, target profile/takeover, budget and
two-person approval remain incomplete. Branch `codex/g4-selected-recipe`, HEAD
`6293354`; section-3 documentation uncommitted, no source edits or publication.
Earlier running/next-section-3 entries below are historical and superseded.

2026-09-12: User takes over observation of the running section-3 profile with
`runs/verification/g4/section-03/observe-profile.ps1`; Codex monitoring stops.
Training continues. User will report completion, then section-3 verification and
final evidence/handoff remain to be completed. Do not launch another profile.

2026-09-12T12:24+08:00: **G4 section 3 IN PROGRESS**, root owns the single
bounded selected-v2 engineering profile on `codex/g4-selected-recipe`, HEAD
`6293354`. Evidence: `runs/verification/g4/section-03/20260912-primary-01`;
inspect `active_process.json` and logs before any relaunch. Preflight: 172 tested
source files unchanged, 133 payload hashes PASS, environment 98 PASS, telemetry
available. Bounds: 2400 optimizer seconds / 800 updates / 3600 outer seconds;
exclude updates 1–38, require at least 1800 valid seconds. No G4 PASS or G5 launch.
No source changes, later-section execution, commit or push authorized in this task.

2026-09-12T12:19+08:00: integration commit `6293354` (20 files) is pushed to
`origin/codex/g4-selected-recipe`; remote SHA matches local. User explicitly approved
the destination/payload after the initial approval block. This local publication-status
update is uncommitted. Implementation verification is unchanged; G4 section 3 is next.

Publication authorized 2026-09-12T12:16+08:00 for the verified selected-recipe
integration on `codex/g4-selected-recipe`. All 172 tested source/input files were
rechecked without drift. The uncommitted/no-publication wording below describes
the earlier verification checkpoint; G4 section 3 remains next.

2026-09-12T12:13+08:00: **selected CONTROL integration verified** on
`codex/g4-selected-recipe` (base `7f85c872`, changes uncommitted). Canonical C0/C1
placement and original-source final analysis are complete. V2 contract/exposure,
production preparation and CPU start/boundary/end reads pass. **826 tests pass in
Docker; 826 pass in isolated Windows**, plus Ruff/compile/build/dependencies.
172 container source/input files and 38 installed modules match. Docker startup
was repaired by preserving stale socket directories; no images/data were reset.
See [integration](g4/SELECTED_RECIPE.md) and [handoff](g4/CURRENT_HANDOFF.md).
**Next: G4 section 3**, fresh profiler plan and sustained measurement. G4 sections
3–7 remain; no gate pass, production training, commit or push. Earlier integration,
branch and next-step notes below are historical. The root return ZIP remains locked
by another process; extracted return is archived and all required files are placed.

Deadline check 2026-09-12: the official Devpost rules page header shows October 1,
but its rules body still specifies September 21 at 23:45 UTC+8. Extension is
unconfirmed; retain September 21 for planning until organizer clarification.

C0/C1 return reviewed 2026-09-12T10:22:06+08:00: final analysis independently reproduces the returned
report exactly; outcome CONTROL (LR0.0006, base70/20/7/3). C1 global improvement
0.2891% misses0.3%; broad-general/narrative regressions1.3608%/2.1181% exceed1%.
Both completed382updates/100139008tokens with verified endpoints. Initial C0
headroom failure and successful retry are preserved/accounted. Experiment selection
is complete; supersedes screening-only/pending-confirmation notes below. Next:
selected-settings integration, then remaining G4sections3–7. No G4/G5 pass or launch.
Review: [C0/C1 return](experiments/C0C1_RETURN_REVIEW.md). Evidence:
`runs/verification/g3.5-c0c1-return-20260912/selection.reverified.json`.

RTX4070 SCREEN COMPLETE, verified2026-09-10T06:52+08:00. All three jobs
completed382updates/100,139,008tokens and passed endpoint verification. SMIX
finished2026-09-09T17:40:56+08:00, invocation1724.047s including verification.
Fresh screening analysis selects SMIX (LR0.0006, edu mixture); NLL4.971697876
versus S0 5.118348216 and SLR5.410160987. SMIX improves global loss ~2.87%
and all four protected slices; SLR does not qualify.
Evidence: `runs/pre_campaign/v2-advisory/selection.screen.json` (all endpoints
reverified by analyzer). No training processes remain. Your4070 lane is done.
Next: transfer complete updated bundle to teammate; C0/C1 confirmation at seed1002
on her verified GPU, then final analysis. No final recipe selection or G3.5 gate
PASS yet. Historical pending/active notes below are superseded by this checkpoint.

SMIX smoke PASS, 2026-09-09T17:06+08:00: user authorized two-update smoke;
524,288tokens/full-dev771,072targets/all4slices, VRAM headroom PASS,
peak reserved6.006GiB, exit0. Child28.266s; invocation38.485s including
verification. Full SMIX estimate3973.404s (~66min), advisory with50%margin;
awaiting full-run decision. Evidence `runs/pre_campaign/v2-advisory/smoke/SMIX/`;
plan `runs/verification/g3.5-s0-20260909/smix-plan-after-smoke.json`.
Supersedes SMIX-smoke NOT_RUN below; full SMIX NOT_RUN, selection pending.

SLR COMPLETE, checked2026-09-09T17:01+08:00: finished16:45:45+08:00,
exit0/runner endpoint verification TRAINING_COMPLETE_NOT_SELECTED;
382updates/100,139,008tokens, full-dev771,072targets, finalNLL5.410160987.
Invocation1606.235s including verification (~26m46s); no Python training remains.
Evidence `runs/pre_campaign/v2-advisory/jobs/SLR/invocations/*/verification.json`.
S0 NLL5.118348216 is lower; no final selection until screen is complete.
Next SMIX smoke requires operator decision. Supersedes full-SLR pending/active
notes below; SMIX NOT_RUN and no selection/gate PASS.

SLR smoke PASS, 2026-09-09T16:17+08:00: user authorized two-update smoke;
524,288 loss tokens, full-dev771,072 targets/all4 slices, VRAM headroom PASS
(peak reserved6.006GiB), exit0. Child30.610s; invocation39.641s including
verification. Full SLR estimate4196.735s (~70min), advisory with50%margin;
awaiting full-run decision. Evidence `runs/pre_campaign/v2-advisory/smoke/SLR/`;
plan `runs/verification/g3.5-s0-20260909/slr-plan-after-smoke.json`.
Lane idle, no unknown attempts. Supersedes SLR-smoke NOT_RUN below;
full SLR/SMIX NOT_RUN and no selection/gate PASS.

S0 COMPLETE, checked 2026-09-09T16:11+08:00: finished16:09:29+08:00;
runner endpoint verification `TRAINING_COMPLETE_NOT_SELECTED`, exit0,
382 updates /100,139,008 loss tokens. Final full-dev NLL5.118348216,
771,072 scored targets/all4 protected slices. Full invocation1596.407s including
verification (child1585.719s), versus advisory4557.817s. No training process remains.
Evidence: `runs/pre_campaign/v2-advisory/jobs/S0/invocations/*/verification.json`.
Fresh endpoint recheck PASS: `runs/verification/g3.5-s0-20260909/endpoint-recheck.json`.
Next: SLR two-update smoke after operator decision; SLR/SMIX NOT_RUN.
No recipe selection or G3.5 gate PASS. Prior S0 pending/active notes below are historical.

S0 smoke PASS, 2026-09-09T15:41+08:00: user authorized launch; two BF16 updates /
524,288 loss tokens verified on RTX4070SUPER, full-dev771,072 targets/all4 slices,
VRAM headroom PASS (peak reserved6.006GiB). Child36.75s; runtime ledger47.407s
including verification, no active/unknown attempts. Evidence under
`runs/pre_campaign/v2-advisory/smoke/S0/invocations/` (verification.json).
Full S0 forecast4557.82s (~76min), advisory with50% margin; awaiting operator
full-run decision. This supersedes smoke-pending state below; full S0 NOT_RUN.

S0 operator preflight, 2026-09-09T15:37+08:00: root verified the requested
`codex/g3.5-pre-campaign-experiments` checkout at `7f85c87`. Windows environment
98 checks PASS; actual RTX 4070 SUPER 12GB; existing `v2-advisory` production
input/source check PASS and documented bundle SHA matches. S0 plan PASS, no
measured runtime estimate yet. No active/unknown lane attempts or Python training
processes observed. Only ~1.4GiB host RAM free; close unused apps before smoke.
Awaiting operator's two-update smoke decision under LLM_HANDOFF launch policy;
no GPU training launched. Evidence: `runs/verification/g3.5-s0-20260909/`.

Teammate ZIP placement instructions added to `docs/experiments/LLM_HANDOFF.md`:
six transfer items, exact destinations, post-copy check and later screening update.
Documentation publication on the current branch authorized 2026-09-09;
runtime implementation `664c617` is pushed.

Runtime amendment verified locally on `codex/g3.5-pre-campaign-experiments`: user removed
hard experiment limits in favor of advisory estimates and an operator yes/no choice.
Policy `configs/campaign/experiment_execution_v1.json` supersedes old budget/cutoff
controls. All 821 frozen-image CPU tests and lint/build checks pass; independent
Sol-medium review approved. Use the verified `runs/pre_campaign/v2-advisory` bundle;
old identities remain historical. Jobs/dependencies and GPU NOT_RUN status unchanged.
See [runtime amendment](experiments/ADVISORY_RUNTIME.md). Publication of this verified
amendment to the current branch was authorized on 2026-09-09.

Coordination override 2026-09-09T14:28:00+08:00: active publication milestone is
`codex/g3.5-pre-campaign-experiments`, based on `4e27266`. User authorized committing
and pushing the verified package plus its existing G4 tooling dependencies.
Verified implementation commit `21a0eccffb39db2254a42719960aa95d3634171a` is
pushed to origin on that branch; the remote hash was checked after publication.
User approved Astra's replacement:
three final49M screen jobs plus conditional two-run confirmation (the original
12 aggregate GPUh limit is superseded by the runtime amendment above),
Sep18 submission target. **Experiment package verified, experiments NOT_RUN**:
795 container tests, lint/compile/build/dependencies, real-input CPU prepare and
six parsed trainer identity checks pass; final image source/tests and36 installed
modules match. Astra planned, Luna-medium authored, root integrated, Sol/Terra-medium
independently approved. [Verification](experiments/VERIFICATION.md) binds final hashes.
Next: [operator guide](experiments/OPERATOR_GUIDE.md) and
[LLM handoff](experiments/LLM_HANDOFF.md); obtain actual per-machine smoke evidence,
run/close experiments, integrate the recipe, then G4 sections3–7. Earlier seven-run
draft is retired. Scope amendment: `configs/campaign/submission_scope_v2.yaml`.
The G4 section-2 checkpoint below remains historical; no new gate
pass or production training/scoring is authorized by this publication task.

Historical G4 snapshot, recorded 2026-09-09T12:01:00+08:00. Artifact checks and
immutable scope contracts determine gate outcomes.

| Snapshot | Value |
|---|---|
| Current milestone | G4 sections 1–2 PASS: bounded production profiler prepared, reviewed and tested |
| Branch / implementation checkpoint | `codex/g4-verification`; HEAD `4e27266e5afafd50ddc3bb7c840e358bde3c74ec`; verified section-2 changes uncommitted (prior implementation `201bf38`) |
| Current verified source identity | 151-file manifest `63e83aaac746a7752208d27fbd964e8202193192c22eaeca29a6857f917da656` |
| Active ownership | Root completed verification/handoff; Sol and Terra final reviews approve; all agents finished |
| Next task | Section 3 only on a separate request; refresh competing load/space/custody before sustained profile |
| Guide and evidence | [Seven-section guide](g4/README.md) / [current handoff](g4/CURRENT_HANDOFF.md) / [verification](g4/VERIFICATION.md) |

The historical scope was `configs/campaign/submission_scope_v1.yaml`: fixed 1B baseline,
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
| G4 | No gate PASS; sections 1–4 complete, 5 not applicable under g4_scope_v2, 6 conditional budget complete. Evaluation timing, availability, backup policy and two-person freeze remain. |
| G5 | NOT_RUN; no main baseline launched. |
| G6 | NOT_RUN; baseline full evaluation, verified release assets and human approvals remain. |

## G4 section-2 profile preparation

- [Profile plan/evidence](g4/PROFILE_PLAN.md): fixed 3815-update recipe with 2400 optimizer-second,
  800-update and 3600 wall-second bounds; 38 updates excluded, ≥1800 valid seconds required.
- Docker build and **756 tests** pass; Windows profiler/telemetry **41 tests** pass.
  Ruff/compile/dependencies and environments 98/97 checks pass; copied source/modules match.
- Real dry-run is PLAN_ONLY, runner `baseline-9d50a02e9d741daf`; required telemetry available.
  New timing fields invalidate old timing evidence; unchanged input-reader proof is reused.
- Evidence `runs/verification/g4/section-02/20260909-1127/report.json`, SHA-256
  `2f24bafcb22962a0505062857bd1f66be7e401fcc1c400e2cfa006e3beccc226`.
  No training/profile/scoring/commit/push. Build-time memory/GPU load needs a fresh check.

## Historical G4 part-1 verification

- Final CPU container suite: 715 passed in 81.59s; final Windows suite: 715 passed
  in 133.90s. Ruff 0.11.13, compilation, dependency consistency and diff checks pass.
- Docker startup repaired; CPU image supplies all test inputs without local corpus.
  All 34 installed modules match source. Both environments pass their pin checks.
- 141 production artifact checks pass; real prepare and CPU start/boundary/end/dev
  reads pass. No training detected; about 214 GB free C: and 1.37 TB D: at preflight.
- Lint and two test portability repairs change source identities. Frozen configs,
  constraints, production inputs and unrelated user files are preserved. Evidence in
  `runs/verification/g4-part1-20260909/`; agents finished, implementation committed
  as `201bf38`. User authorized pushing that checkpoint; later sections were then unstarted.

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
