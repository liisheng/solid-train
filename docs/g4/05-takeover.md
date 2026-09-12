# Section 5 — Rehearse target-machine takeover

**2026-09-12: NOT APPLICABLE UNDER AMENDED SCOPE.** The user chose a
[single-machine campaign](SINGLE_MACHINE_SCOPE.md), recorded in
`configs/operations/g4_scope_v2.yaml`. Skip this section and continue to
[section 6](06-budget.md). The original brief below is retained for history and
does not authorize execution or require another machine for the active scope.

## Goal and prerequisite

Demonstrate the fallback path from the 4070 mainline lane to the 3070 target lane.
Read the common handoff, `RECOVERY_EVIDENCE.md`, the operations contract's
`takeover_rehearsal`, and the existing checkpoint/provenance transfer-verification code.
Inspect `scripts/g2_handoff.py` for reusable custody patterns, not as proof of G4 approval.

The G2 any-compatible-machine amendment does not waive this G4 requirement. Identify
the actual target machine/operator and access method. If unavailable, prepare the full
transfer checklist and record BLOCKED with that owner and next action. Do not pretend a
second local process is another machine or invent a teammate action. A scope amendment
needs an explicit user decision and versioned record before relying on it.

## Work

1. Check section-4 evidence and identify the latest verified engineering checkpoint,
   its sidecars, runner identity, provenance, source commit/manifest, tokenizer,
   frozen configs, schedules and required shard files. Create a transfer manifest with
   relative paths, byte sizes and hashes. Never rely on Git alone to supply ignored data.
2. Prepare or perform scoped transfer using an authorized target/access method. Copy only
   necessary artifacts; preserve originals and existing target work. Do not send messages
   or publish artifacts externally merely because this brief mentions teammate review.
3. Reproduce the target environment and verify the transferred hashes there. Capture
   actual GPU identity, precision support, available space and device headroom. Reuse a
   valid target profile if applicable; otherwise keep its sustained rate unmeasured.
4. Rehearse optional-work cessation: record the target optional jobs paused/stopped, or
   the observed absence of such jobs. Affect only task-owned or explicitly authorized
   processes. Identify the latest verified source checkpoint and verify it on target.
5. Resume that same engineering lineage for a declared small number of whole updates.
   Preserve run identity/horizon/data position. Record exact before/after counters,
   finite behavior, validation/checkpoint results and recovery duration. Do not silently
   change microbatch, precision or optimizer settings to fit the target.
6. Verify a new durable target checkpoint and retain operator/machine-specific receipts.
   Compare transferred checkpoint bytes exactly. Do not promise bitwise equality of
   subsequent computation on different GPUs unless the applicable contract requires and
   evidence demonstrates it; do not weaken a required equality after observing a mismatch.
7. Account for transfer, preparation, verification and resumed training costs. If anything
   fails, record the concrete error and repair boundary; do not call a transfer alone PASS.

## Deliverables and completion

Write `docs/g4/TAKEOVER_EVIDENCE.md` and a portable operator checklist with actual paths,
commands, expected hashes, receipt locations and failure instructions. The evidence must
support all four required steps: optional work stopped, latest verified checkpoint
identified, checkpoint verified on target, training resumed on target. A validator fed
four unchecked labels is not evidence that these events happened.

PASS means the applicable cross-machine rehearsal occurred and was verified. If the
source-to-target engineering rehearsal does not satisfy an unchanged original-mainline
wording, document that scope limitation explicitly for section 7. No G5 training here.

## New-chat prompt

```text
Execute only G4 section 5, docs/g4/05-takeover.md. Read the common handoff/index
and RECOVERY_EVIDENCE.md. Establish the actual authorized target/access method,
prepare the transfer bundle, and complete the bounded takeover rehearsal if the
target is available. Verify custody and actual resumed training there. If blocked,
finish the portable checklist and name the owner/next action; do not waive the
requirement. Do not launch the baseline or send unauthorized teammate messages.
Update TAKEOVER_EVIDENCE.md and the short handoff.
```
