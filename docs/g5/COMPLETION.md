# G5 completion

Recorded 2026-09-12T22:49+08:00. **G5_PASS_UNDER_AMENDED_SCOPE** for the
single-machine selected CONTROL reduced baseline. This is an evidence-based
paraphrase of the active contracts and receipts; the original full campaign and
canonical frozen status fields are not promoted. G6 remains NOT_RUN.

## Gate evidence

| G5 requirement | Result and evidence |
| --- | --- |
| Stable lineage and valid fallback | PASS: fresh seed1337 initialization reproduced; completed/latest/best checkpoints independently verified at3815 updates, cursor976640 and zero final optimizer LR. All three contain identical endpoint bytes. |
| Primary set and confirmation complete or marked incomplete | PASS for amended scope: one selected CONTROL 1B baseline complete. Prior C0/C1 confirmation and deterministic selection are recorded in `docs/experiments/C0C1_RETURN_REVIEW.md`. Original later A/B/C campaign is not run under this scope; optional extension/comparison not run. |
| Counters reconcile | PASS: all3815 consecutive metric rows,1,000,079,360 loss tokens,40 complete dev evaluations; one fresh invocation, no resume or replay. Optimizer duration matches phase history. |
| No frozen artifact changed | PASS: completion job rechecked494 historical freeze entries plus10 successor entries against the owner-approved digest; original lineage hashes unchanged after export. |

The endpoint is a49,658,368-parameter BF16 model trained with the fixed CONTROL
recipe. Final full-dev loss3.287145695, perplexity26.766355;753 sequences,
771,072 scored targets and95 batches. These are development results, not submission
benchmark scores.

## Artifacts and timing

- Run: `runs/reduced_campaign/reduced_baseline_v2/run`.
- Runner identity: `baseline-f23398e32dc3100e`; training identity: `run-146ddcb9c11da742`.
- Completed checkpoint SHA256: `11109b8071ad86018554acc4c802bea494b403016f25e6b58fb3b59fb7e81f18`.
- Export: `baseline_export.pt`, SHA256 `89438ee3165a64bd3c15b3ba1f4aa6658841a423a554164061c2d4c0ac95c288`.
- Export reload, parameter recount and provenance verification PASS.
- Training command ended2026-09-12T19:14:30+08:00, exit0, outer15,767.193s (4h22m47s).
- Optimizer15,414.290s; trainer process15,703.980s; launch controller15,839.281s.
- Completion verification/export/backup controller113.254s, charged separately.
  Launch plus completion controllers total15,952.534s; nested timers are not added.
  This total excludes earlier G3/G4 work, this subsequent review, and future G6 work.

Versioned completion backup:
`D:\SWE\LLM BACKUP\g5-completion-20260912-224817`.
Includes all run artifacts and launch evidence with a hash manifest; previous backup
generation retained. This is a same-disk byte backup, not disk-failure protection
or a restored-copy training claim. Root review rechecked49 lineage/backup hashes.

Evidence directory: `runs/verification/g5/20260912-completion-01`:
`receipt.json`, `endpoints.json`, `counters.json`, `export.json`, `frozen-hashes.json`,
`lineage-before.json`, `lineage-after.json`, `backup.json`, and `review.json`.
Launch receipts/logs remain in `runs/verification/g5/20260912-launch-01`.
All evidence and model binaries are local ignored artifacts; Git alone cannot transfer them.

Production source was unchanged; the previously verified826-test Docker and Windows
suites/build/lint remain applicable. The completion helper passed syntax validation;
its actual six-stage execution passed with empty stderr. No new model training,
benchmark evaluation, commit, push or publication occurred during completion review.

## G6 handoff

Use only the verified export above. Next work is fresh-environment full benchmark
evaluation with complete coverage, measured runtime and separate cost accounting,
then efficiency/release documentation and applicable release approvals. The accepted
24-hour evaluation allowance remains provisional; organizer-setting and harness
provenance gaps from G4 still require disposition. Do not tune training settings
using benchmark results. Model publication and G6 release approval remain separate.
