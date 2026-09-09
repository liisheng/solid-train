# Chat 6 — Integration evidence and independent reduced-G3 review

## Outcome and purpose

Verify the five preceding deliverables together and produce a concrete G4 handoff. This
section decides reduced recipe readiness; it does not claim canonical G3/G4 passed or start G5.

## Read after the shared entry instructions

- Chat 1's contract, current handoff and section completion/evidence references in STATUS.
- Composite accounting, validation checks, runner manifest, evaluation binding and smoke bundle.
- Changed code only as required to assess these interfaces. Use relevant earlier G2 evidence
  as a regression baseline, not as proof that newly changed paths passed.

## Work

1. Check each requirement against actual artifacts: digests, scope, source quotas, ordered
   components, cursor semantics, full-dev coverage, WSD phases, completed-checkpoint eligibility,
   loader pins and provisional labels. Detect stale evidence and path substitution. Resolve
   contradictions in the human status snapshot against verified reports.
2. Run the full project suite once on the integrated tree, plus required build/lint/container
   checks. Reuse earlier focused results unless new changes affect them. Preserve the known
   protected-residue alignment failures if still present; diagnose their disposition without
   deleting protected history or weakening checks. Record tool unavailability honestly.
3. Run one bounded real-shard correctness rehearsal, using the final model and final recipe
   controls in a separate engineering run. Reuse the actual composite/dev inputs and original
   3815-update horizon with an explicit early-stop limit (for example eight updates). Compare
   uninterrupted versus interrupted/resumed durable state, cursor and ordered input hashes;
   deliberate corruption must fail. This early checkpoint must fail baseline-export eligibility.
   Small fixtures from chat 2 cover the distant component boundary; do not train thousands of
   updates merely to cross it during G3 verification.
4. Use a short, separate test-only completed-decay contract/fixture with its own identity to
   verify the positive eligible-export logic; it is not evidence that the 3815-update baseline
   completed and must not weaken the production baseline horizon check. Do not
   replace the stopped 3815-update run's planned horizon to manufacture eligibility. Preserve
   clear engineering labels for both fixtures. If the exact integrated rehearsal already exists
   and matches current identities, reuse verified evidence instead of launching a duplicate.
5. Assemble a machine-readable reduced-G3 report binding every artifact and its check result.
   Readiness must fail on missing evidence; leave missing/failed requirements explicit. Proposed
   status: `REDUCED_BASELINE_RECIPE_READY_FOR_G4`; canonical G3 remains `NOT_RUN`.
6. Terra performs one independent review of the final diff and evidence. Routine fixes remain
   narrow and get relevant checks; a redesign goes back to its owning section with a precise
   handoff. Agent approval does not replace teammate approval required for G4.
7. Produce the G4 handoff: exact launch/dry-run/resume/export commands, branch/diff identity,
   fixed 3815-update horizon, observed resource use, disk requirements, evaluation command,
   outstanding review items, and the one necessary changed-input production profile/recovery
   check. The historical weighted rate projects about 4.15 optimizer-hours for 1B; record it as
   conditional and exclude evaluation/checkpoint/preparation time until measured.

## Done

The reduced recipe package is complete, integrated checks and independent review have concrete
results, STATUS and continuity agree with those results, and G4 can act without rediscovering
the implementation. If a blocker remains, report the exact artifact, owner and next action;
do not invent a READY status. Preserve a short final handoff rather than six chat transcripts.

## Stop boundary

No full baseline, optional comparison, final-holdout/full benchmark scoring, public release,
or repeated long profiling. G4 is a separate task. G3 is not expanded to include winning every
benchmark before the model has even been trained.
