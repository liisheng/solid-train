# G2 implementation brief

## Goal

Complete the user-authorized reduced `reduced_5pct_v1` G2 check under
`configs/operations/g2_scope_v2.yaml` on any compatible machine, while preserving the
frozen `measurement_v1.yaml` contract and keeping canonical full-scale G1/G2 `NOT_RUN`.
The result must be independently reviewable from machine-readable evidence.

## Assumptions

- Existing reduced shards, schedules, source export, and step-zero provenance are the
  pinned inputs; do not rebuild them or launch baseline training.
- A fresh process and fresh output directory are sufficient for the amended artifact
  requirement; machine identity is recorded but is not itself a gate.
- Existing valid evidence may be reused only after checking hashes, sizes, schema, and
  provenance consistency.

## Acceptance criteria

The combined report is `REDUCED_SCOPE_G2_PASS` and maps all five requirements to PASS:

1. Real-shard training produces finite validation loss with a strict decrease.
2. Interrupted/resumed and uninterrupted runs match model, optimizer/scaler, RNG/data
   RNG, counters, run identity, best-validation state, cursor, schedule hash, and ordered
   nonempty batch-reference hashes.
3. Deliberately corrupted checkpoint loading is rejected closed.
4. Export/reload passes the fixed release checks.
5. A fresh process evaluates the exact pinned original source export, with matching
   export/provenance/scope/environment identities and deterministic inference.

Also require a passing environment report, BF16 support plus measured stability, strict
deterministic execution with TF32 disabled, exact scope/export/provenance size and SHA-256
identities, and fail-closed behavior for missing, malformed, tampered, or substituted
evidence. Preserve explicit canonical G1/G2 `NOT_RUN` fields.

## Edge cases

- Missing or non-object JSON, duplicate or missing requirement fields, empty batch
  references, incomplete resume fields, non-finite/equal losses, altered scope sidecar,
  altered export/provenance, path substitution, and stale evidence must fail closed.
- Do not delete protected historical residue when reproducing prior alignment-audit
  checks; use a new safe temporary path and record permission failures.
- Keep source profile/efficiency observations separate from G2 correctness and canonical
  G4 claims.

## Non-goals

Do not change `measurement_v1.yaml`, reinterpret canonical gates, claim cross-machine
transfer, run baseline/full-scale training, run a long profile, download data, weaken
determinism or audit checks, delete protected artifacts, commit/push, or replace valid
historical evidence.

## Recommended approach

1. Read the continuity, status, handoff, and scope contract.
2. Validate the existing aggregate, recovery, source-verification, environment, export,
   provenance, and schedule identities.
3. If required, run one bounded fresh-process recovery in a new output directory using
   the pinned reduced shards/schedules.
4. Verify the original source export and provenance with the existing release verifier,
   then generate the combined report through `scripts/g2_handoff.py`.
5. Run focused negative-path tests, the full suite, byte-compilation, a package wheel
   build, configured lint, and the documented Docker verification attempt. Record
   unavailable tooling honestly.
6. Update `docs/STATUS.md` and `.agent/CONTINUITY.md` with evidence paths, results,
   blockers, and review targets. Leave Terra to perform read-only review.

## Tests Luna must add or run

- Focused G2 handoff, recovery/training-recipe, release-evidence, and provenance tests.
- Negative tests for malformed JSON, missing/duplicate fields, altered scope/export/
  provenance identities, empty batch references, incomplete resume state, and corrupted
  checkpoints.
- Full `pytest` suite, Python byte-compilation, package wheel build, configured lint,
  and the repository's CPU Docker verification command when Docker is available.

## Unresolved questions

- Terra's final read-only review is complete and approved. The full suite still has two
  protected-residue alignment failures; Docker and Ruff remain unavailable, which are
  verification limitations to retain in reports.
- Baseline training is outside this G2 task; consult the recorded campaign
  authorization before a separate baseline task. Any scope change that would promote
  reduced evidence to canonical full-scale G1/G2/G4 requires an explicit decision.

## Copy-paste prompt for Luna

> Complete reduced `reduced_5pct_v1` G2 under `configs/operations/g2_scope_v2.yaml`.
> Preserve `measurement_v1.yaml`, historical evidence, and canonical full-scale G1/G2
> `NOT_RUN`. Verify pinned shard/schedule/export/provenance identities, run at most one
> bounded fresh-process recovery in a new output directory, verify the original source
> export with deterministic release checks, and generate a combined report. Require all
> five amended requirements, BF16 stability, strict deterministic execution, nonempty
> ordered batch references, exact resume state/cursor/hash identity, and fail-closed
> malformed/missing/tampered/substituted evidence. Run focused and full tests, byte-
> compilation, a package wheel build, configured lint, and the documented Docker
> attempt; update STATUS and
> CONTINUITY with exact paths/results. Do not run baseline or long profiles, download,
> delete protected residue, weaken checks, commit, or push. Return a compact evidence-
> based report for Terra's read-only review.
