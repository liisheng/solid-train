# Data card — TinyBench-LM training corpus (template)

> **Template.** Every count here must come from `scripts/audit_source_policy.py` over a real
> acquisition pass. No source revision is pinned and no license review is recorded, so every
> measured field is `NOT_RUN`. Never fill a removal rate, token count, or overlap rate from a
> fixture: the fixtures are planted test cases, not corpus measurements.

## Corpus profile

| Field | Value | Evidence |
|---|---|---|
| Selected profile | `NOT_RUN` — `full_v1` (>= 11B) or dated `degraded_v1` (>= 8B) | `configs/data/shards_v1.yaml` |
| Accepted stable tokens | `NOT_RUN` | `src/tinybench_lm/shards.py` |
| Reserved pool tokens | `NOT_RUN` (>= 390.1M margin required) | `src/tinybench_lm/shards.py` |
| `validation_dev` tokens | `NOT_RUN` (10-20M) | `src/tinybench_lm/shards.py` |
| `validation_final` tokens | `NOT_RUN` (10-20M) | `src/tinybench_lm/shards.py` |
| Tokenizer | 12,288 IDs, byte-level BPE with byte fallback | `configs/data/tokenizer_v1.yaml` |
| Scope-reduction decision | `NOT_RUN` — dated and recorded if `degraded_v1` is used | `configs/data/pipeline_bench_v1.yaml` |

## Sources

One row per source. A source may not enter the corpus until its revision is pinned and its
license is reviewed and recorded.

| Source ID | Revision | License | Share | Accepted tokens | URL | Attribution |
|---|---|---|---|---|---|---|
| `NOT_RUN` | `PENDING_PIN` | `NOT_RUN` | `NOT_RUN` | `NOT_RUN` | `NOT_RUN` | `NOT_RUN` |

Mixture definitions (`M-base` 70/20/7/3 and `M-edu` 85/5/7/3 over
FineWeb-Edu / general web / OpenWebMath / narrative) are frozen in
`configs/campaign/preregistration_v1.yaml`. Which mixture the final run uses is `NOT_RUN`
until the P8 screen completes under its preregistered rule.

## Excluded by policy

The corpus never contains, and the filters reject:

- synthetic, generated, or hosted-model-rewritten text;
- teacher logits, probabilities, rankings, or hidden states;
- official benchmark examples, or benchmark-targeted retrieval and generation;
- large code corpora;
- unfiltered raw Common Crawl;
- the full Pile;
- text labeled, rewritten, or scored by a hosted model.

Evidence: `configs/data/sources_v1.yaml`, `configs/data/filters_v1.yaml`,
`src/tinybench_lm/eligibility.py`.

## Filtering

| Filter | Threshold | Documents rejected | Reason code |
|---|---|---|---|
| English probability | `configs/data/filters_v1.yaml` | `NOT_RUN` | — |
| Empty / binary / malformed / length | `configs/data/filters_v1.yaml` | `NOT_RUN` | — |
| Credential and contact dumps | `configs/data/filters_v1.yaml` | `NOT_RUN` | — |
| OpenWebMath prose retention | `configs/data/filters_v1.yaml` | `NOT_RUN` | — |

## Deduplication and decontamination

| Step | Setting | Measured effect |
|---|---|---|
| Exact document dedup | SHA-256 over NFKC/lowercase/whitespace-normalized text | `NOT_RUN` |
| First-512-character mirror | SHA-256 | `NOT_RUN` |
| Near-duplicate | 5-word shingles, 128-permutation MinHash, estimated Jaccard 0.85 | `NOT_RUN` |
| Benchmark quarantine | all three rules in `configs/data/decontam_v1.yaml` | `NOT_RUN` |
| Split isolation | no near-duplicate cluster crosses stable / reserved / dev / final | `NOT_RUN` |

Protocols were calibrated on planted fixtures only. Matching normalization is applied to the
comparison text and never mutates stored training text.

## Repetition

Effective passes over the corpus: `TBD`. Report the measured value, not the plan's target.

## Protected reporting slices

`broad_general`, `educational_science`, `narrative_coreference`, `math_technical` — frozen
before proxy training in `configs/data/shards_v1.yaml`. A candidate may not regress any slice
by more than 1% relative.
