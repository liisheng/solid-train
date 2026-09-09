# Chat 3 — Complete development validation

## Outcome and purpose

Make development loss a trustworthy signal for baseline monitoring and later quality
decisions. Score every scheduled development sequence once, regardless of batch divisibility.
Depends on chat 2 so shared training changes are applied sequentially.

## Read after the shared entry instructions

- The baseline contract and current handoff.
- `train.py`: validation-stream creation, `evaluate`, logging, best-state tracking, and
  absolute-update validation cadence.
- `src/tinybench_lm/schedule.py`: `ScheduledTokenStream` batching and cursor behavior.
- Relevant entry-point tests in `tests/test_training_recipe.py`; add a focused validation
  test file if it keeps the change clearer.
- Metadata of `data/schedules/reduced_5pct_v1/validation_dev.json` and its manifest.

## Work

1. Confirm the development schedule has 753 references. Current batch-limited validation
   wraps: 95 batches of 8 score 760 references; 94 score only 752. Add an explicit full-dev
   mode instead of choosing another approximate batch count.
2. Replay the schedule without wrap and request `min(batch_size, remaining)` for the last
   batch. Aggregate summed loss and scored-token counts, rather than averaging a one-sequence
   batch equally with an eight-sequence batch. Record sequence/token count and schedule hash.
3. Preserve sampled engineering validation as an explicitly labeled mode if still required by
   existing tools. Baseline commands must select full-dev explicitly. Repeated evaluation must
   reset dev state consistently and must not change the training cursor or training RNG behavior.
4. Wire the baseline's declared cadence and completion validation while preserving absolute
   update numbering across resume. Clearly distinguish the existing post-first-update
   validation event from a true pretraining measurement; do not relabel one as the other.
5. Prepare the deterministic validation-final schedule and record its identity if needed for
   the later evaluator. Preparing/verifying references is not scoring the final holdout. Keep
   it out of training monitoring, checkpoint selection, and the G3 smoke run.

## Meaningful tests and done

Test 753 references at batch size 8 gives 94 full batches plus one reference, each scored once.
Use unequal synthetic losses to catch incorrect batch weighting. Cover divisible and
non-divisible sizes, repeat replay equality, empty/mismatched schedules, preserved train
state, and unchanged validation-event positions after resume. Compare the aggregated result
against a direct reference computation within the expected numeric tolerance.

Run focused tests and applicable build/lint checks. A full GPU pass is not needed for this
section's acceptance; integrated real-shard execution belongs in chat 6. Handoff the explicit
full-dev invocation and output fields to chat 4.

## Stop boundary

No benchmark-driven tuning, full baseline training, final-holdout scoring, new evaluation
dashboard, model change, or unrelated loader rewrite.
