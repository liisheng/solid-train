# Project status

Updated 2026-09-13. Start with the [documentation index](README.md).
This is a coordination snapshot; outcomes require named evidence and scope.

| Item | State |
| --- | --- |
| Branch | `g6-release` |
| Publication checkpoint | G6 fix/results and next-run documentation package on this branch; prior G5 checkpoint `5eebdad`. Resolve the package commit with `git log -1`. |
| Baseline | 49,658,368 parameters; 1,000,079,360 loss tokens; G5 PASS under amended scope |
| Evaluation | All five provisional tasks completed; saved coverage and bundle verification PASS |
| Release | G6 BLOCKED on official settings, harness provenance, public artifacts and human approvals |
| Active ownership | Next-run documentation complete; user: release/development decisions and future long-job observation |
| Jobs | No active training/evaluation reported in the completed G6 handoff |
| Further development | Fresh 3B candidate is a proposal only; no new training or scope amendment approved |

## Evidence to read

- [G5 completion](g5/COMPLETION.md): provenance, export hashes, counters and backup.
- [G6 results](g6/RESULTS.md): scores, coverage, environment and timing accounting.
- [G6 release checklist](g6/README.md): remaining work and completed-attempt evidence.
- [Generation diagnostic](g5/GENERATION_DIAGNOSTIC.md): repetition and quality limitations.
- [Competitiveness proposal](experiments/COMPETITIVENESS_PLAN.md): draft data work and fresh 3B run.
- [Next-run handoff](next_run/README.md): exact candidate specification, implementation tasks and operator runbook; documentation ready, training NOT_READY.

G6 code verification: Docker build and Ruff PASS; 76 evaluation tests PASS in 7.47 s;
17 Windows coverage tests PASS in 11.14 s. The earlier 826-test baseline evidence
does not mean the entire suite was rerun for G6.

## Scope and next actions

The [canonical gates](../configs/operations/measurement_v1.yaml) and original
full-campaign requirements remain preserved. Reduced G1/G2 evidence, CONTROL
selection and amended G4/G5 outcomes do not promote all original gates automatically.

| Owner | Next action |
| --- | --- |
| User / organizer | Resolve official settings and the recorded deadline conflict before scheduling release. |
| Release implementer | Resolve harness commit provenance; assemble final assets and reconcile total campaign costs. |
| User / required approvers | Authorize publication and approve the exact release hash; verify public access after publication. |
| User | Decide whether to adopt the separate competitiveness proposal. |

Old entries are preserved in [status history](STATUS_HISTORY_20260913.md).
Technical decisions and agent handoff detail remain in [continuity](../.agent/CONTINUITY.md).
