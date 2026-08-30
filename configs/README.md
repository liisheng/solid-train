# Model configuration purposes

- `final_49m.json`: canonical final competition architecture and the default for final-facing model tools.
- `baseline_49m.json`: compatibility alias for the canonical final architecture; retained for existing commands and audit references.
- `pilot_12m.json`: smaller pipeline and bounded-training pilot; not a final model candidate.
- `deep_thin_gqa_49m.json`: rejected architecture-research candidate retained for reproducibility; not a final model candidate.

# Frozen data protocols

`configs/data/` holds versioned, immutable data-safety protocols. Their content is pinned by
SHA-256 in `src/tinybench_lm/data_protocols.py` (`FROZEN_PROTOCOL_SHA256`), so loading fails
closed if a threshold is edited. A protocol change means publishing the next version
(`*_v2.yaml`) with a recorded reason, never editing `v1` in place.

- `data/dedup_v1.yaml`: matching normalization, exact SHA-256 dedup, 512-character mirror check,
  5-word/128-permutation MinHash near dedup at estimated Jaccard 0.85, and train/reserved/validation
  cluster isolation.
- `data/decontam_v1.yaml`: the three benchmark quarantine rules plus the frozen required and
  secondary task identities. Dataset revisions and the harness commit are `PENDING_PIN`/`BLOCKED`;
  fixture calibration is allowed while blocked, a real-corpus scan is not.

Both protocols were calibrated on planted fixtures only (`tests/test_data_protocols.py`). No
real-corpus removal or quarantine rate has been measured.

# Frozen source registry and integrity filters

`configs/data/sources_v1.yaml` and `configs/data/filters_v1.yaml` are pinned by SHA-256 in
`src/tinybench_lm/source_manifest.py` (`FROZEN_CORPUS_PROTOCOL_SHA256`) under the same
fail-closed, publish-a-new-version rule.

- `data/sources_v1.yaml`: the streaming manifest schema (source ID, revision, stable document ID,
  URL or a recorded reason it is withheld, raw hash, license, filter decisions, accepted token
  count), the allowed stable and reserved sources with their shares and token targets, validation
  splits, corpus profiles, and every prohibition from the plan's do-not-use and hosted-assistant
  rules.
- `data/filters_v1.yaml`: the frozen filter evaluation order and thresholds for English
  probability, empty/binary/malformed/length records, credential and personal-contact dumps, and
  OpenWebMath prose retention.

Calibrated on planted local fixtures only (`tests/test_source_manifest.py`). No source revision is
pinned, no license review has been recorded, and no accepted token count has been measured with the
final tokenizer: `scripts/audit_source_policy.py` reports those as `BLOCKED`/`DEFERRED`, and
`assert_ready_for_real_corpus_acquisition` refuses a real acquisition pass until they are cleared.
