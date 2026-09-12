# Documentation index

Use [project status](STATUS.md) for the current milestone, branch and next actions.
The baseline is trained and its full provisional evaluation is complete. Release
requirements remain pending; the proposed 3B candidate is not approved for execution.

## Current reference

| Document | Purpose |
| --- | --- |
| [Project README](../README.md) | Model, current results, generation command and verification workflow |
| [Status](STATUS.md) | Concise coordination snapshot |
| [Rules](RULES.md) | User-provided competition rules; retained as supplied |
| [Configuration index](../configs/README.md) | Configuration families; consult current status for execution scope |
| [G5 completion](g5/COMPLETION.md) | Training, export, counters and backup evidence |
| [Generation diagnostic](g5/GENERATION_DIAGNOSTIC.md) | Generation weaknesses and diagnostic limits |
| [G6 results](g6/RESULTS.md) | Full provisional scores, environment and timing |
| [G6 release checklist](g6/README.md) | Remaining requirements and artifact identity |
| [Competitiveness plan](experiments/COMPETITIVENESS_PLAN.md) | Draft proposal for one fresh 3B candidate; no execution authorization |
| [Next-run handoff](next_run/README.md) | Detailed candidate specification, preparation checklist, runbook and readiness gaps |

G5's dated completion record says G6 had not run at that time. The later G6 results
supersede that status. Use dated records for evidence and STATUS.md for present state.

## Reproduction and historical evidence

These files preserve earlier implementation, scope and verification records. Old
“next”, “pending”, “running” and launch instructions apply to the recorded date;
they do not authorize new work. Hash-bound documents remain unchanged so their
historical receipts can still be checked.

| Area | Entry points |
| --- | --- |
| Environment | [Recorded environments](ENVIRONMENT.md); G6's actual interpreter is in [G6 results](g6/RESULTS.md) |
| Data pipeline | [G1 corpus pipeline](G1_CORPUS_PIPELINE.md), [streaming verifier](G1_STREAMING_VERIFIER.md), [corpus reproduction](REPRODUCING_THE_CORPUS.md) |
| Reduced scope | [Reduced campaign](REDUCED_CAMPAIGN.md), [expansion plan](REDUCED_EXPANSION_PLAN.md) |
| G2 engineering | [Implementation brief](G2_IMPLEMENTATION_BRIEF.md), [handoff](G2_HANDOFF.md) |
| G3 implementation | [Section index](g3/README.md), [integration evidence](g3/INTEGRATION_EVIDENCE.md), [review fixes](g3/REVIEW_FIXES.md) |
| Recipe experiments | [Original index](experiments/README.md), [runtime amendment](experiments/ADVISORY_RUNTIME.md), [final C0/C1 review](experiments/C0C1_RETURN_REVIEW.md) — CONTROL selected |
| G4 readiness | [Section index](g4/README.md), [selected recipe](g4/SELECTED_RECIPE.md), [readiness record](g4/READINESS_20260912.md), [owner approval](g4/approvals/owner-20260912-readiness.json) |
| Early pilots | [Pilot report](PILOT_REPORT.md), [research plan](RESEARCH_PLAN.md), [phase 1](PHASE_1_BASELINE.md), [verified slice](VERIFIED_SLICE_PILOT.md) |
| Coordination history | [Archived status entries](STATUS_HISTORY_20260913.md), [agent continuity](../.agent/CONTINUITY.md) |

The canonical [G0–G6 definitions](../configs/operations/measurement_v1.yaml) and
[release matrix](../configs/release/evidence_matrix_v1.yaml) retain original requirements.
Reduced or amended outcomes must keep their scope labels. No G7 is defined. A new
candidate reopens the checks affected by its changed inputs.

## Release templates

These are templates, not completed or approved release artifacts:

- [Submission package](templates/SUBMISSION_PACKAGE.md)
- [Model card](templates/MODEL_CARD.md)
- [Data card](templates/DATA_CARD.md)
- [Built With](templates/BUILT_WITH.md)
- [AI assistance disclosure](templates/AI_ASSISTANCE_DISCLOSURE.md)

## Maintenance

Keep current state in STATUS.md and detailed decisions in `.agent/CONTINUITY.md`.
Update this index when adding a milestone. Preserve frozen configuration and evidence
bytes; create a successor record when facts change. Keep historical launch commands
out of current quick-start instructions. Local corpus, checkpoints and `runs/` evidence
are ignored artifacts and are not transferred by cloning the repository.
