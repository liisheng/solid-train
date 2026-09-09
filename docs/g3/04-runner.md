# Chat 4 — Baseline runner, identity, and eligible export

## Outcome and purpose

Make the fixed recipe executable and resumable, and prevent an incomplete or undecayed
checkpoint from being presented as the secured baseline export. Depends on chats 1–3.

## Read after the shared entry instructions

- Baseline contract, exposure accounting, validation interface, and current handoff.
- `train.py` argument validation, run identity, stopping controls, checkpoints, and logging.
- Relevant functions in `src/tinybench_lm/training_recipe.py`, especially release eligibility;
  `src/tinybench_lm/provenance.py` export/reload; and existing checkpoint role APIs.
- `scripts/verify_real_training.py`, `tests/test_durable_checkpoints.py`,
  `tests/test_provenance.py`, `tests/test_release_evidence.py` as needed.

## Work

1. Add a thin baseline prepare/verify/launch/export entry point, or extend an existing suitable
   one. Suggested name: `scripts/run_reduced_baseline.py`. Avoid a generic orchestration system.
   Produce a dry-run/plan that checks inputs and shows the exact command without starting training.
2. Bind scope/recipe digests, model/tokenizer identities, both composite components, dev
   schedule, precision/environment facts, seed, LR phases, cadence and exact counters into the
   run manifest. Reject substituted identities. The manifest must later bind the selected
   evaluation protocol from chat 5; the dry-run reports that field as PENDING now. Actual G5
   launch requires all training identities and the resolved evaluation binding; bounded
   engineering fixtures remain explicitly separate from baseline launch readiness.
3. Integrate the composite stream and full-dev mode. Preserve the 3815-update horizon under
   bounded stopping controls. Resume requires identical semantic fields and the correct absolute
   cursor; changing seed, horizon, LR, inputs, or relevant validation policy must fail or issue
   a separately declared identity rather than silently mutate the baseline.
4. Preserve step-zero provenance from fresh initialization. Reuse durable checkpointing and
   save policy. Record the completed baseline endpoint and its fallback role separately from
   an arbitrary earlier best-dev checkpoint. Inspect actual retention disk requirements; reuse
   safe retention support, without adding automatic deletion or removing historical artifacts.
5. Before baseline export, verify the source checkpoint itself completed 3815 updates,
   1000079360 tokens, and the frozen cursor; inspect the bound WSD schedule and apply the existing
   release eligibility check. Positive configured decay alone is insufficient: early stopped
   checkpoints must not pass merely because their planned eventual LR is zero.
6. Export via the proven provenance path, re-run reload/cap/weight checks, and bind export hash
   to the eligible source checkpoint and recipe evidence. Keep engineering exports separately
   labeled so earlier G2 fixtures can still exercise their intended paths.

## Meaningful tests and done

Test launch refusal for missing/drifted inputs, insufficient supply, invalid phase/counter
arithmetic, and semantic resume changes. Test export rejection for early stopping, zero decay,
nonzero final LR, wrong source hash and mismatched cursor. Test a valid completed-decay fixture
passes export/reload. Use tiny models/fixtures for full endpoints; do not train 1B to test the
eligibility code. Run checkpoint/provenance regressions and applicable build/lint checks.

Done means the launch path is reviewable and dry-runnable, recovery semantics are preserved,
and export eligibility is enforced. Hand off the runner, required identity fields, and exact
bounded-rehearsal controls to chat 5/6.

## Stop boundary

No main baseline launch, long profile, arbitrary checkpoint deletion, canonical gate promotion,
optional experiment, or rewrite of working G2 checkpoint machinery without a demonstrated need.
