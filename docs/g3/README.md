# Reduced G3: one chat per section

Prepared 2026-09-07. This is an execution proposal, not a gate-pass claim.

Updated 2026-09-08: sections 1–6 are implemented and locally verified. Section 6 passed
675 tests and exact real-shard recovery; required Docker/lint checks remain blocked.
Use [integration evidence](INTEGRATION_EVIDENCE.md) and the [G4 handoff](G4_HANDOFF.md).
This does not claim READY or a canonical G3/G4 pass.

The priority is to reach a reproducible, fully evaluated 1B baseline, then spend remaining
compute on the authorized conditional 3–5B training and at most one controlled comparison.
Keep architecture, tokenizer, and source families fixed while completing these sections.
The accepted scope is `configs/campaign/submission_scope_v1.yaml`: reduced G3 records the
baseline recipe; G4 checks launch readiness; G5 trains. Canonical G3 remains `NOT_RUN`.

## Order and deliverables

Run these chats sequentially on one dedicated G3 milestone branch. Sections 2–4 share
training/schedule code; parallel editing is more likely to create integration work than save it.

| Chat | Brief | Main deliverable | Depends on |
|---|---|---|---|
| 1 | [Baseline contract and preflight](01-contract.md) | Fixed recipe choices, input inventory, acceptance checklist | Existing reduced G2 |
| 2 | [Exact training exposure](02-exposure.md) | Verified finite composite schedule and reader | 1 |
| 3 | [Complete development validation](03-validation.md) | Full-dev replay with correct final partial batch | 2 |
| 4 | [Baseline runner and export eligibility](04-runner.md) | Validated launch/resume/export paths and run manifest | 1–3 |
| 5 | [Evaluation identity and rehearsal](05-evaluation.md) | Pinned provisional scoring bundle and bounded smoke evidence | 4 |
| 6 | [Integration and independent review](06-integration.md) | Verified reduced-G3 recipe package and G4 handoff | 1–5 |

## Instructions shared by all six chats

1. Read root `AGENTS.md`, `.agent/CONTINUITY.md`, `docs/STATUS.md`, this index,
   your numbered brief, and [CURRENT_HANDOFF.md](CURRENT_HANDOFF.md). The continuity archive
   is historical reference; read it only for a specific unresolved fact.
2. Confirm the active branch and existing changes. Chat 1 creates the dedicated milestone
   branch; later chats continue it. Keep unrelated generated metadata/build files and the
   user's `docs/RULES.md` out of implementation changes. Do not automatically commit/push
   from these briefs; follow the user's current authorization.
3. Read only the brief's targeted implementation files, dependencies needed to understand
   them, and applicable tests. Use function/text searches before reading large modules.
   Parse large JSON artifacts with bounded summary output; never paste schedules, checkpoint
   state, corpus text, or full logs into the conversation.
4. Complete this section and its checks. Do not start another numbered section, the original
   proxy campaign, a main baseline, a long profile, new data acquisition, or public publishing.
   Existing training authorization is preserved; these task boundaries keep each chat bounded.
5. Prefer the existing training, checkpoint, provenance, schedule, and evaluation machinery.
   Add only the adapter/validation support this baseline needs. Keep frozen files unchanged;
   publish versioned successors or external bindings when their semantics change.
6. For code edits, run the relevant tests plus the repository-required compilation/build/lint
   checks that apply. Use the documented CPU container path when available and the recorded
   Windows CUDA exception for GPU work. Do not install host packages. A full suite is normally
   run once in chat 6, or earlier if broad changes/failures justify it. Record unavailable tools
   and real failures; never turn them into PASS or repeatedly rerun an unchanged failure.
7. Keep the existing division of work: the root coordinates; Luna handles bounded implementation
   and Terra handles difficult schedule/integrity questions and final independent review.
   Give an agent only its relevant brief and inputs. Routine sections do not need duplicate
   whole-repository audits; reserve independent review effort for the integrated result.
8. Update `docs/STATUS.md` and the short continuity with meaningful evidence. Replace
   `CURRENT_HANDOFF.md` with a handoff of at most about 300 words: completed section, branch,
   commit/diff identity, changed files, artifact paths/hashes, checks, limitations, and exact
   next section. Link evidence rather than copying logs. A claim of completion needs evidence.
9. If the section expands materially, finish a coherent checkpoint and record the remaining
   issue rather than broadening into a framework redesign. Do not consume the next chat's
   scope to hide an unfinished prerequisite.

New implementation/artifact paths in the briefs are recommendations, not claims that those
files already exist. Keep names consistent once chat 1 records them.

## Baseline choices to carry forward

The 3,815-update horizon and 262,144 loss tokens/update are already in the accepted scope.
The remaining settings below are proposed choices to record in chat 1, not tuning results:

- Final 49,658,368-parameter model; existing 12,288-token tokenizer; fresh seed 1337.
- Microbatch 8, accumulation 32, sequence length 1,024; measured BF16 policy.
- Existing AdamW/weight-decay/clipping contract; peak LR 0.0006.
- WSD: 38 warmup, 3,395 stable, 382 linear-decay updates, ending at zero.
- Full development validation every 100 updates and at completion, preserving the documented
  initial validation convention; recovery saves every 100 completed updates and at completion.
- No benchmark-driven training selection; no pilot checkpoint initialization.

## After these six chats

Use separate later chats for G4 launch readiness, G5 baseline execution, and G6 full evaluation
and export verification. G4 should reuse valid evidence and run only the changed-input checks
needed: a bounded exact-recovery rehearsal and one measured production-shard profile if the
new schedule is not already covered. Do not repeat long profiles in each G3 section.

Secure the baseline export and full evaluation before optional work. Choose any 3–5B extension
using development loss, distinct-data exposure, measured runtime, fallback and evaluation
reserve. Declare a new horizon/run identity before changing a decayed baseline. At most one
comparison remains authorized, with its rule fixed before observing its outcome. Package
submission evidence and demonstration in a separate release task; public writes still need
their own authorization. None of these later tasks is started by this planning package.
