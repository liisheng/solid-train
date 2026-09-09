# Chat 1 — Baseline contract and preflight

## Outcome and purpose

Produce one fixed reduced-baseline specification and a concise input inventory. Every later
chat should implement these decisions rather than reopen model, LR, or scope discussions.
This section is primarily configuration/documentation; avoid building a generic campaign system.

## Read after the shared entry instructions

- `configs/campaign/submission_scope_v1.yaml` and its sidecar.
- `configs/final_49m.json`, `configs/training/recipe_v1.yaml`,
  `configs/training/checkpoint_v1.yaml`.
- `docs/G2_HANDOFF.md`; selected status/identity fields from
  `runs/reduced_campaign/g2_report_luna_final.json` and
  `runs/reduced_campaign/reduced_5pct_v1/aggregate.json`.
- `runs/reduced_campaign/reduced_5pct_v1/profile/measurement.json` and the associated
  `profile/train/run_config.json`. Inspect existing recipe validators only as needed.

## Work

1. Check branch, dirty files, artifact availability, and active project jobs. Create a dedicated
   branch such as `codex/g3-baseline-recipe` from the verified current work; preserve unrelated
   changes. Update the milestone/branch in STATUS as reduced G3 preparation, not a gate pass.
2. Record the index's baseline choices in a new versioned config, suggested
   `configs/training/baseline_reduced_v1.yaml`, with digest sidecar. Distinguish the authorized
   horizon from proposed seed/LR/decay choices. Record the reason for each choice, including
   that 0.0006 has engineering evidence but has not won a controlled LR comparison.
3. Fix the exact arithmetic: 3,815 updates, 262,144 loss tokens/update, 1,000,079,360 consumed
   loss tokens, 976,640 sequences, 38/3,395/382 WSD phases. Fix source labels and quotas:
   fineweb_edu 683648; dclm 195328; openwebmath 68365; narrative 29299.
4. Define the interfaces later sections must supply: a versioned two-component exposure plan,
   complete development replay, source checkpoint eligibility, evaluation binding, and run
   manifest. Reuse existing types where possible. Store generated artifact hashes in a separate
   readiness/run manifest once they exist; do not invent hashes to make a config look complete.
5. Inventory the current tokenizer, shard, schedule, scope and environment identities using
   existing verification evidence. Note that `dclm` is the retained internal ID for general
   FineWeb. Record the BF16/determinism evidence, validation custody, and checkpoint policy.
6. Provide a readiness checklist owned by sections 2–6. Fix immutable recipe choices now;
   mark missing generated artifacts PENDING. Final bundle readiness is decided only in chat 6.

## Verification and done

Validate config parsing, digest consistency, horizon/batch arithmetic, and LR phase legality
through existing validators where available. A docs/config-only change does not need a new
test framework. The specification must be usable without reading this conversation and must
not label original G3 passed. Finish with the shared short handoff to chat 2.

## Stop boundary

Do not run training, profiles, benchmarks, corpus rebuilding, or original P/F proxy experiments.
Do not redesign architecture, optimizer, or the frozen canonical gate system. Input identities
that are not yet produced stay explicitly pending for their owning section.
