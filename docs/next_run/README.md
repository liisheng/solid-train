# Next run: preparation handoff

Prepared 2026-09-13. **Documentation ready for preparation; training NOT_READY.**
The user requested comprehensive next-run documentation. This does not record approval
of a new training run, acquisition job or frozen scope amendment. The
[competitiveness proposal](../experiments/COMPETITIVENESS_PLAN.md) remains a draft.

## Read in this order

1. [Current status](../STATUS.md) and [baseline completion](../g5/COMPLETION.md).
2. [Candidate specification](CANDIDATE_SPEC.md): exact proposed recipe and data quotas.
3. [Preparation and implementation](PREPARATION.md): work needed before launch.
4. [Operator runbook](RUNBOOK.md): readiness, observation, recovery and final review.

The working recommendation is one fresh approximately 3B-token CONTROL candidate,
with a larger, cleaner corpus and the existing 49,658,368-parameter architecture.
Keep the completed baseline as fallback. Do not run the old 1B launcher with a new
output directory and assume it implements this proposal.

## Readiness register

| Requirement | Present state | Owner / next action |
| --- | --- | --- |
| Documented recipe, stages and checks | READY_AS_DOCUMENTATION | Implementer follows the linked specification. |
| New scope / candidate contract | NOT_CREATED | Implementer prepares a versioned successor for review. |
| Corpus audit and expanded clean pool | NOT_RUN | Data implementer measures audit, retention, throughput and storage. |
| 3B-compatible runner/exposure support | NOT_IMPLEMENTED | Implementer extends version-bound checks with regression coverage. |
| Baseline per-slice reference / 40 prompts | NOT_PREPARED | Implementer records fixed dev references and diagnostic prompts. |
| Materialized schedule / complete input custody | NOT_CREATED | Produce only after data verification. |
| New-input correctness / recovery / profile | NOT_RUN | Run on the exact candidate input bundle after implementation. |
| Launch command / observer / freeze receipt | NOT_CREATED | Produce from verified successor tools, then review the bundle. |
| Runtime / available resources / deadline | UNCONFIRMED_FOR_NEW_INPUTS | Remeasure and disclose before each long job. |
| New training and benchmark scoring | NOT_RUN | Only after readiness and applicable operator decisions. |

The existing production runner accepts only `baseline_reduced_v1.yaml` and
`baseline_reduced_v2.yaml`; the current corpus-expansion wrapper fixes the old 5%
scope. These are verified implementation gaps, not optional documentation polish.

## First task for the next contributor

Prepare P0/P1: adopt the documented specification into a reviewable successor design,
inventory the existing corpus and its journals without mutation, define the data audit,
and identify the smallest implementation changes listed in PREPARATION.md. Do not
acquire a large corpus or start training as part of a documentation/preparation task.
Existing session authorization for later work, if supplied, takes precedence.

Read `.agent/CONTINUITY.md` first, inspect the actual branch and uncommitted changes,
and preserve unrelated files. G6 code/results and these docs belong to the publication
package on `g6-release`; verify the actual commit before using another checkout.
Use a dedicated milestone branch when implementation is requested and update STATUS.md.

Finish preparation with a report listing exact file hashes, completed checks, remaining
NOT_RUN items, measured estimates and the next executable command. Do not claim
TRAINING_READY because a document or checklist exists.

## Documentation verification

Checked 2026-09-13: all 203 local Markdown links resolve; four existing CLI help
commands exit zero; quota/token arithmetic and LR boundaries match the existing
`WSDSchedule` implementation; 13 historical G4-bound Markdown hashes still match.
Evidence: `runs/verification/next-run-docs/report.json`. No production source changed
in this documentation task, so no new build or training test result is claimed.

## What is reused versus repeated

| Gate | New candidate work |
| --- | --- |
| G0 | Reuse architecture/parameter-count implementation; refresh environment and hardware facts. |
| G1 | Rebuild evidence for changed corpus, filtering, decontamination, shards and schedules. |
| G2 | Repeat targeted end-to-end, loss alignment and recovery checks on new inputs. |
| G3 | Keep confirmed CONTROL; no repeat mixture/LR screen. Version the new acceptance rules. |
| G4 | Freeze new source/config/data identities; verify profile, resources, recovery, budget and approval. |
| G5 | One complete fresh training run and independent export/counter verification. |
| G6 | Evaluate after development selection, then finish release requirements. |

Original evidence remains historical and usable within its scope. These stages add
no new canonical gate and do not overwrite old contracts.
