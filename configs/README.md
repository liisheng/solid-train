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
- `data/decontam_v2.yaml`: production G1 successor with the same matching rules and task set,
  pinned to the public dataset revisions and lm-evaluation-harness commit resolved on
  2026-09-05. This protocol is `READY` for a real-corpus scan. The downloaded benchmark-item
  body remains local; its compact count and SHA-256 evidence is committed at
  `docs/evidence/decontamination/benchmark_inputs.json`.

The matching rules were calibrated on planted fixtures only (`tests/test_data_protocols.py`).
The production benchmark inputs have been acquired, but no real training-corpus removal or
quarantine rate has been measured yet.

# Frozen source registry and integrity filters

`configs/data/sources_v4.yaml` (active), the superseded `sources_v3.yaml`, `sources_v2.yaml`
and `sources_v1.yaml`, and `configs/data/filters_v1.yaml` are pinned by SHA-256 in
`src/tinybench_lm/source_manifest.py` (`FROZEN_CORPUS_PROTOCOL_SHA256`) under the same
fail-closed, publish-a-new-version rule. Every superseded version stays in the tree and keeps
verifying, because a superseded protocol is still evidence of what was frozen and when.

**`data/sources_v4.yaml` is the active source registry.** The chain:

- **v1** was written before the Track 01 rules text was available and required a per-title
  licence review the rules do not ask for — the rule is *"Credit everything you use in Built
  With and your README"* — which blocked all corpus acquisition.
- **v2** replaced that with a recorded declared licence per source plus an attribution
  obligation, and pinned all nine source revisions to immutable Hugging Face commits.
- **v3** resolved the one licence v2 left open and closed the README half of the attribution
  requirement.
- **v4** replaced two sources that could not be read. `storytracer/US-PD-Books` carries no
  text at all — it is a catalogue of archive.org URLs — and `mlfoundations/dclm-baseline-1.0`
  needs the unpinned `zstandard` package. Both had been registered on licence metadata without
  an availability check. They are now `sedthh/gutenberg_english` (MIT) and
  `HuggingFaceFW/fineweb` (ODC-By), each streamed at its pinned revision and confirmed to
  yield text. v4 adds an `availability_policy` section so a source cannot be registered on
  licence metadata alone again.

The synthetic-text and full-Pile prohibitions are retained deliberately even though the rules
permit both; each carries a recorded `retention_reason`.

- `data/sources_v1.yaml` *(superseded)*: the streaming manifest schema (source ID, revision, stable document ID,
  URL or a recorded reason it is withheld, raw hash, license, filter decisions, accepted token
  count), the allowed stable and reserved sources with their shares and token targets, validation
  splits, corpus profiles, and every prohibition from the plan's do-not-use and hosted-assistant
  rules.
- `data/filters_v1.yaml`: the frozen filter evaluation order and thresholds for English
  probability, empty/binary/malformed/length records, credential and personal-contact dumps, and
  OpenWebMath prose retention.

Calibrated on planted local fixtures only (`tests/test_source_manifest.py`). Every source revision
is now pinned and every declared licence recorded, so `assert_ready_for_real_corpus_acquisition`
no longer refuses an acquisition pass. Two things remain outstanding and are reported honestly by
`scripts/audit_source_policy.py`:

- `accepted_token_measurement` is `DEFERRED` — the final 12,288-ID tokenizer does not exist, so no
  accepted token count has been measured.
- `attribution` splits three ways: `readme_credits` and `built_with_template` are `PASS` (the README
  Credits section and the Built With table both list every source with its licence and pinned
  revision), while `built_with_submitted` stays `NOT_RUN` because nobody has submitted to Devpost.

v3 resolved the one licence v2 left open: OpenWebMath is ODC-By 1.0. v2 had recorded it as
`PENDING_CARD_READ` because the Hub API's top-level `cardData` carried no `license` field — a lookup
error, since the field is nested under `dataset_info:` and the card's License section states it
outright. FineWeb-Edu and OpenWebMath both derive from Common Crawl, so `additional_terms` records
the Common Crawl terms-of-use obligation.
