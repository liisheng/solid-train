# G4: one chat per section

The [pre-campaign experiment stage](../experiments/C0C1_RETURN_REVIEW.md) is complete:
CONTROL (LR 0.0006, base 70/20/7/3). The successor is
`configs/training/baseline_reduced_v2.yaml`; see [selected-recipe integration](SELECTED_RECIPE.md)
for evidence, reproduction and current verification status. Earlier sections 1–2
remain historical evidence; recheck anything affected by the integration.

Updated 2026-09-12. This is a workflow guide, not a new gate contract or permission
to execute all sections at once. Sections 1–4 are complete on the source lane;
start the next chat at section 5. See [recovery evidence](RECOVERY_EVIDENCE.md).

G4 establishes a reproducible, measured, recoverable setup for the fresh 1B baseline.
G5 runs that baseline; G6 performs its full evaluation and release. These seven chat
sections expand the earlier five-part explanation so tooling, long execution, and
external-machine coordination do not compete for one context window.

## Sections and dependencies

| Chat | Brief | Deliverable | Prerequisite |
|---|---|---|---|
| 1 — complete | [Verification](01-verification.md) | Working container/lint and verified source/inputs/environment | Reduced G3 |
| 2 — complete | [Profile preparation](02-profile-preparation.md) | Tested bounded profiler, timing/telemetry plan, exact dry-run command; [evidence](PROFILE_PLAN.md) | 1 |
| 3 — complete | [Sustained production profile](03-sustained-profile.md) | 37.4 valid minutes, 64,499 weighted tokens/s; [evidence](PROFILE_EVIDENCE.md) | 2 |
| 4 — complete | [Local recovery](04-local-recovery.md) | Exact CUDA resume/rollback and archived replay costs; [evidence](RECOVERY_EVIDENCE.md) | 2; normally after 3 |
| 5 | [Target-machine takeover](05-takeover.md) | Verified source-to-target custody and resumed training | 4; target available |
| 6 | [Baseline horizon and budget](06-budget.md) | Measured dated budget, explicit reserves and unresolved inputs | 3–5 |
| 7 | [Freeze and G5 handoff](07-freeze.md) | One freeze-bundle digest, two-person review, evidence-backed disposition | 1–6 |

Default order is sequential. If target-machine availability blocks section 5, section 6
may prepare a clearly provisional budget; section 7 cannot convert missing evidence into
a pass. A section may end BLOCKED with useful deliverables and a named next action.
Return to that same section in a new chat when its prerequisite is available.

## Start each chat

Paste the prompt at the bottom of that section's brief. The prompt requests that section
only. Read `AGENTS.md`, `.agent/CONTINUITY.md`, `docs/STATUS.md`, this index,
`CURRENT_HANDOFF.md`, and the selected brief. Then open only its named source files and
relevant tests, using symbol searches for large modules. Do not read all seven briefs,
all of G3, the continuity archive, or entire raw logs as a routine starting step.

The governing documents are `configs/campaign/submission_scope_v2.yaml` (inherits the accepted reduced
scope), `configs/operations/measurement_v1.yaml` (canonical gates/measurements), and
`configs/training/baseline_reduced_v2.yaml` (selected recipe). The original campaign's
`freeze_bundle` section supplies its component/approval requirements. Treat this guide
as a paraphrase and implementation plan; resolve conflicts against the contracts.
Do not reinstate the superseded mandatory proxy/three-arm campaign.

## Shared execution rules

1. Use the active G4 milestone branch in `CURRENT_HANDOFF.md` (`codex/g4-selected-recipe`
   for the successor integration), checking its HEAD and local changes
   first. Do not create another branch for every chat. Preserve unrelated `build/`,
   `src/tinybench_lm.egg-info/`, and the user's `docs/RULES.md`.
2. Complete only the selected section. GPU work is limited to the rehearsals specified
   in its brief once the user asks to execute that section. No G5 baseline, full scoring,
   new data acquisition, optional comparison, model release, or automatic next section.
   Commit/push only when the current session authorizes it; this guide is not standing
   authorization to publish future changes or send messages to teammates.
3. Reuse existing runner, checkpoint, exposure, validation, and verifier code. Keep fixed
   files unchanged. A necessary semantic change requires a versioned successor and a
   recorded impact decision, not a rewritten hash sidecar or a weakened assertion.
4. Use Docker for code verification and the documented Windows CUDA exception for real
   GPU checks. Do not install host packages. For source edits run relevant tests, lint,
   compilation and build; use the full suite when changes cross modules or failures justify
   it. Reuse passing evidence when the bound files and relevant environment are unchanged.
5. Use unique local evidence directories below `runs/verification/g4/section-NN/<attempt>/`
   for new sections. Preserve failed attempts. These proposed paths are not existing
   artifacts. Do not reuse the final baseline's run directory for engineering work.
6. Log exact commands, exit codes, source/input hashes, run IDs, timing and machine facts
   to files. Show summaries in chat: counts, selected measurements, failure reasons, and
   paths. Never dump checkpoint tensors, corpus text, large schedules or full telemetry.
7. Run one primary sustained profile. Repeat only if a specific invalidation or failed
   requirement warrants it. No optimizer/backend/mixture search is included. A slower
   truthful measurement can pass; an unsupported speedup cannot.
8. Missing evidence is NOT_RUN; an external blocker is BLOCKED with owner and next action;
   an observed failed check is FAIL. PASS needs inspected evidence. Agent review does not
   satisfy two-person teammate approval. The G2 any-machine amendment does not amend G4.

## Keep each chat's context bounded

Every section ends with one short tracked evidence summary named in its brief and updates
to `docs/STATUS.md`, `.agent/CONTINUITY.md`, and `docs/g4/CURRENT_HANDOFF.md`. Keep the last
file around 300 words. It must contain: completed/current section, branch and commit plus
working-tree identity, changed files, artifact paths and hashes, tests, remaining blocker,
next section, and exact safe next command when known.

During a long run, also retain PID/container identity, output directory, stop condition,
start time and ownership in the handoff. If context is interrupted, the next chat inspects
that process and its files before launching anything. Do not mistake a lost chat for a
failed process or start a duplicate job. If work grows beyond the brief, stop at a durable
checkpoint with the same section marked incomplete; do not hide it in the next section.

Suggested reading budget is the common handoff plus one brief and targeted source excerpts.
Large reports should be summarized programmatically. A helper agent, when authorized,
gets only its bounded task and required paths; it should return findings and evidence
locations, not another whole-repository briefing.

## Fixed baseline to carry across chats

Fresh seed 1337; 49,658,368 parameters; BF16; batch 8 × 32 × 1,024; LR 0.0006;
3,815 updates / 1,000,079,360 loss tokens; WSD 38/3395/382; final cursor 976,640.
Full-dev monitoring has 40 events; scheduled recovery has 39 saves, plus best saves.
The composite exposure preserves 550,094,903 distinct stable tokens with intentional
reuse. An engineering stop retains the 3,815-update horizon; its early checkpoint is not
a submission endpoint. The baseline starts fresh even if an engineering run is resumable.

Section-1 proof: [VERIFICATION.md](VERIFICATION.md), 715 tests each in CPU Docker and
Windows, Ruff PASS, 141 input checks PASS. Historical ~51k versus ~67k tokens/s remains
unresolved. Neither figure is a measured current campaign runtime.

Section-2 proof: [PROFILE_PLAN.md](PROFILE_PLAN.md), 756 CPU Docker tests and 41
Windows profiler tests pass; real production preparation is PLAN_ONLY. The source
manifest is refreshed for instrumentation. Section 3 requires a separate request.

## Evidence invalidation

After a change, name the affected evidence before rerunning anything. Timing/reader/
training/validation/checkpoint/backend changes can invalidate profiling. Resume/ledger/
checkpoint changes can invalidate recovery. A target-machine change can invalidate its
takeover proof. Documentation-only edits do not by themselves invalidate GPU measurements.
Bind a new source manifest after fixes and retain the old report as historical. Never
assert that a Git commit alone transfers ignored data, checkpoints, or verification logs.
