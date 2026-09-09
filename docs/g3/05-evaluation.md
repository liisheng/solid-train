# Chat 5 — Pinned evaluation identity and bounded rehearsal

## Outcome and purpose

Ensure the eventual baseline can produce reproducible full benchmark results without a late
dataset/adapter surprise. Depends on chat 4's manifest/export interface. This is evaluation
engineering, not a model-quality experiment and not final benchmark scoring.

## Read after the shared entry instructions

- Baseline manifest interface and `configs/evaluation/evaluation_provisional_v2.yaml`.
- Active `configs/data/decontam_v3.yaml`, benchmark input evidence and existing immutable
  revision registry needed to reconcile the required tasks.
- `evaluate.py`, targeted functions in `src/tinybench_lm/evaluation_protocol.py` and
  `src/tinybench_lm/evaluation_tasks.py`.
- `tests/test_evaluation_protocol.py`, `tests/test_evaluation_tasks.py`, relevant adapter tests.

## Work

1. Inventory the actual installed harness identity and all five required task definitions,
   dataset revisions, prompts, shot counts, metrics, context and scoring policies. Reuse
   verified existing pins where applicable; do not update a dependency just to call it current.
2. Create a versioned successor protocol or explicit runtime binding, preserving v2's corrected
   WikiText-103 semantics and existing history. Reconcile task identities with decontamination
   v3, including the same pinned quarantine datasets. Fill locally resolvable pending identities.
3. Enforce the pins at the loader/runtime boundary, or verify the effective loaded revisions;
   copying SHA strings into a report is not enough. Record a concrete blocker if the harness
   cannot establish which dataset it actually scored. Use targeted primary documentation only
   when the implementation requires it; avoid another broad competition/literature review.
4. Preserve explicit provisional labels where organizer settings are unresolved. Do not wait
   for organizer answers to complete a reproducible provisional baseline path. Do not invent
   official settings, treat a smoke score as a result, or use benchmark outcomes for training.
5. Exercise the five-task bundle with a bounded smoke, at most 100 examples per required task,
   using an existing verified engineering export if compatible. This is an engineering-export
   rehearsal, not a completed-baseline eligible export. Use development custody rules; do not
   score the private validation-final split or expand the public WikiText smoke into a full run.
6. Verify raw outputs, sample counts, revisions, tokenizer/checkpoint hashes, runtime settings,
   protocol digest and evidence manifest. Record measured smoke time only; it is not a reliable
   full-suite runtime estimate. Fill the baseline manifest's evaluation identity from chat 4.

## Meaningful tests and done

Test wrong dataset/checkpoint/protocol identities fail, WikiText resolves 103 rather than the
legacy alias, effective loader pins agree with the declared binding, and provisional labels
are retained. Reuse bundle integrity checks and ensure the bounded smoke passes them.
Run focused evaluation tests and applicable build/lint checks.

Done means a reviewer can tell exactly what each future score means, reproduce the command,
and verify the smoke output without treating it as a quality measurement. Hand off the pinned
full-evaluation command with limits removed for later G6 use, but do not execute it now.

## Stop boundary

No full baseline training/evaluation, prompt search, best-checkpoint selection from benchmark
scores, organizer messaging, remote publishing, or unrelated secondary tasks.
