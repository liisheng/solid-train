# Section 7 — Freeze review and G5 handoff

## Goal and prerequisites

Assemble the exact setup and evidence for review, obtain real approval where available,
and issue an honest G4 disposition. Read the common handoff/index and the six section
summaries; inspect raw evidence only to validate their claims. Read the canonical G4
requirements and `freeze_bundle` in `configs/campaign/preregistration_v1.yaml`, together
with the reduced scope that supersedes the original experiment campaign and
`configs/operations/g4_scope_v2.yaml`. Section 5 is not applicable; use
`SINGLE_MACHINE_SCOPE.md` as its disposition summary.

## Work

1. Build a requirement-to-evidence matrix for two-person freeze approval, sustained real
   throughput, local recovery and no bypassed correctness checks. Record takeover as
   NOT_APPLICABLE_UNDER_AMENDED_SCOPE and bind the exact g4_scope_v2 digest. Recheck hashes,
   source applicability, unresolved findings, failed-attempt accounting, and budget
   prerequisites. Distinguish reduced-scope readiness from an original canonical pass.
2. Create a reviewable freeze manifest binding all applicable components: code,
   environment, model, tokenizer, corpus, mixture, optimizer, schedule, evaluation,
   parent-selection rule, branch calendar, local recovery and downtime policy, and
   checkpoint backup destination/custody/restore procedure. No backup is yet verified
   by the scope decision; record actual evidence and any remaining action. Map superseded
   parent/branch requirements to explicit reduced-scope dispositions, citing the accepted
   amendment. Do not invent completed proxy runs, parent checkpoints or final-model hashes.
   Never edit the original frozen file to turn a pending field into evidence.
3. Bind exact final source, config/tokenizer/input and local report digests; include
   reproduction/transfer instructions for ignored artifacts. A source change after
   profiling must have an explicit evidence-applicability review. Run only invalidated
   checks. Critical gaps remain blockers even if all tests happen to pass.
4. Hash the stable bundle deterministically. Keep approval receipts outside that payload
   to avoid a self-referential hash; receipts identify the same bundle digest, distinct
   approvers, timestamps and scope. Present the concrete bundle for the required two
   human approvals. AI reviewers do not count; do not send messages without authorization.
   Missing approval is BLOCKED. Changing the bundle requires new matching approvals.
5. Prepare a G5 handoff containing fixed launch command, fresh run directory, stop/recovery
   policy, resource calendar, required files/hashes, logging/accounting responsibilities,
   endpoint eligibility/export checks and failure escalation. Confirm the baseline has
   no pilot/rehearsal initialization. Keep the execution flag out of the default command
   shown during this section; the reviewed executable form belongs in the G5 instructions.
6. Record G4_PASS_UNDER_AMENDED_SCOPE only if every applicable requirement is proven.
   The canonical operations validator remains unchanged and is not an amended-scope
   assessor; use an explicit evidence matrix without fabricated takeover labels.
   Keep original canonical
   gates unpassed where their scope is unmet. If blocked, finish the bundle and handoff
   anyway, with a precise owner/next action for each remaining item. Do not run G5 merely
   because the review documents are complete.

## Deliverables and completion

Write `docs/g4/FREEZE_REVIEW.md`, a hashed freeze-bundle manifest, separate approval
receipts (when actually supplied), and `docs/g4/G5_HANDOFF.md`. Mark each gate requirement
with evidence and scope. Update the short handoff and status to the actual disposition.

A complete section can deliver a blocked review package; only complete valid evidence
and actual matching approvals justify PASS. No production baseline, full evaluation,
public model release, or automated following section is executed here.

## New-chat prompt

```text
Execute only G4 section 7, docs/g4/07-freeze.md. Read the common G4 handoff/index,
section summaries and governing contracts. Verify applicability, assemble the
single-hash freeze bundle and requirement matrix, and prepare the G5 handoff.
Present the concrete bundle for any still-required human approvals. Do not invent
approval or mark unmet requirements passed. Do not launch G5, full scoring or
publication. Record the actual G4 disposition and remaining owners/actions.
```
