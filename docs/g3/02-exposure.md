# Chat 2 — Exact finite training exposure

## Outcome and purpose

Implement and verify the smallest finite composite schedule needed to train the accepted
baseline at its intended source mix, with exact recovery across the component boundary.
Depends on chat 1's fixed contract and interface decisions.

## Read after the shared entry instructions

- The new baseline contract and chat 1's inventory.
- `src/tinybench_lm/repeated_schedule.py`; targeted schedule builder/reader/hash/verification
  functions in `src/tinybench_lm/schedule.py` and `scripts/build_schedule.py`.
- `tests/test_repeated_schedule.py`, relevant parts of `tests/test_mixture_schedule.py`.
- `train.py` schedule-loading and cursor integration only if the new reader requires wiring.
- Summarize metadata/counts from `data/schedules/reduced_5pct_v1/stable_train.json` and the
  associated stable manifest; do not emit its 537,112 references into chat.

## Work

1. Recheck the base hash and source counts: fineweb_edu 375907, dclm 107439,
   openwebmath 37579, narrative 16187; total 537112. If actual verified values differ, reconcile
   the input identity before constructing outputs rather than silently using stale arithmetic.
2. Keep that schedule unchanged as the first component. Build a second component, unique
   within that component (not globally across both passes),
   of 439528 sequences from the same verified stable shards, with quotas:
   fineweb_edu 307741, dclm 87889, openwebmath 30786, narrative 13112.
3. Join the components in a versioned exposure plan. Record both file/content identities,
   order, counts, quotas, token accounting and one absolute cursor. Intentional reuse between
   components is allowed and reported; duplicates within each component remain rejected.
   Reuse existing quota builders and reader APIs. Extend the repeat wrapper narrowly or add
   a small composite wrapper; do not weaken or rewrite frozen base-schedule semantics.
4. Verify stable-only source/split membership, shard bounds, total supply, ordered reference
   identities, exact rounded 70/20/7/3 exposure, and exhaustion at 976640 sequences. Preserve
   within-component ordering/locality guarantees and document component-level reuse.
5. Materialize outputs in a new baseline-specific directory. Bind them to the contract from
   chat 1 and publish a compact accounting report. Record consumed tokens separately from
   the 550094903 distinct selected corpus tokens, about 1.818 effective passes.

## Meaningful tests and done

Use tiny two-component fixtures to test the last reference of component 1, first of component
2, interruption/resume at and around the boundary, final exhaustion, wrong component hash,
altered order, wrong source quotas, and forbidden duplicate references inside a component.
Test both stream behavior and metadata binding; a recomputed total alone is insufficient.
Run existing schedule/repeat tests plus the applicable build/lint checks. Validate the real
generated artifact with bounded-memory verification and emit only its counts/hashes/results.

Done means the reader can supply the exact baseline inputs, its cursor resumes identically,
and original G2 artifacts remain available. Record any required runner integration in the
handoff to chat 3 and chat 4.

## Stop boundary

No training or profiling. No new corpus, reshards, infinite wrapping, globally-unique claim
for repeated exposure, general curriculum framework, or unrelated schedule optimizations.
