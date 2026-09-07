# Verified v3 slice pilot — 2026-09-06

This is an engineering pilot, not a G1/G2/G4 gate pass or a release candidate.
The user authorized inspecting the slice and running a small verified dataset through
training and evaluation. Full training budget and campaign scope remain undecided.

## Corpus audit

All 124,150 filter-passing cluster members completed v3 decontamination:
123,754 KEEP and 396 QUARANTINE (219 contiguous overlap, 86 complete-field
substring, 91 shingle coverage). These are document decisions before final duplicate
representative selection, not a guarantee of zero contamination.

The scan was observed complete at 2026-09-06T01:16:13+08:00, no later than
2h47m49s after launch. The exact scan timer was not flushed before the subsequent
selection pause; do not present this upper bound as an exact runtime.

Selection exposed a missing cluster lookup index: EXPLAIN showed a correlated
`SCAN member` per candidate. Adding `dedup_decisions(cluster_id)` replaces that
with an indexed search without changing eligibility. The resumed selection stopped
on three genuine shortages:

| Selection | Selected tokens | Required slice tokens |
|---|---:|---:|
| Reserved textbook | 561,907 | 975,250 |
| Stable mathematics | 4,443,709 | 7,700,000 |
| Stable narrative | 1,054,961 | 3,300,000 |

Stable assignments total 104,501,376 tokens across four sources. The final accepted
bundle was not published, and full G1 remains incomplete. Preserve the SQLite state;
unchanged reruns cannot resolve insufficient supply. A source top-up or explicitly
versioned scope change is required. Whole-document selection can overshoot small
validation quotas substantially (one narrative development document has 303,177
tokens), so small slices are not exact miniature mixtures.

## Training evidence

The independent pilot selected the first 22,000 eligible representative keys from
the main sources, which are DCLM-only because keys sort by source. Every member
of each selected duplicate cluster had a committed KEEP decision. A deterministic
cluster hash assigns approximately 2% to validation. Train and validation cluster
sets are disjoint. Original text was encoded with the existing final 12,288-ID
tokenizer and EOS ID 1 into verified uint16 streams.

| Measurement | Result |
|---|---:|
| Training corpus tokens | 16,696,140 |
| Validation corpus tokens | 324,486 |
| Model parameters | 49,658,368 |
| Optimizer updates | 64 |
| Consumed training tokens, including repeated random samples | 16,777,216 |
| Optimizer time, excluding validation/checkpoint overhead | 285.56 seconds |
| Weighted throughput after four warmup updates | 59,006.58 tokens/second |
| Peak PyTorch allocated GPU memory | 5.40 GiB |
| Validation loss after first update / final | 9.1137 / 6.8032 |
| Validation perplexity after first update / final | 9,078.46 / 900.70 |

Hardware: local RTX 4070 SUPER. Configuration: `configs/final_49m.json`, BF16,
sequence length 1024, microbatch 8, accumulation 32, peak LR 0.0006, four warmup
updates and eight decay updates. Fresh initialization was recorded; best/latest
checkpoints passed the existing retention integrity inventory. Other corpus work
ran concurrently, and rates varied; this five-minute flat-stream run does not replace
the required 30–60-minute production-shard measurement. Training/validation losses
show that learning occurs, not that the model is competitive. Do not resume this
pilot checkpoint as the final campaign: its data boundaries and sampler are pilot-only.

Local artifacts are under `runs/verified_slice_pilot`: `prepare.py`, hashed data
metadata and document ledger, `train/metrics.jsonl`, checkpoints and provenance.
Compact persistent evidence is under `docs/evidence/pilots/verified_slice_v3`.

## Evaluation evidence

The saved checkpoint reloaded successfully. The four required accuracy tasks each
scored 100 examples: HellaSwag accuracy 27% / normalized 28%; ARC-Easy 26% / 24%;
PIQA 50% / 45%; WinoGrande 44%. These small-sample results are broadly near chance
and establish execution, not competitive quality. No benchmark result is being used
to tune or select a training configuration.

The combined harness run took 135.24 seconds and passed 12 artifact integrity checks.
However, inspecting its resolved configuration exposed a semantic mismatch: harness
`wikitext` is `EleutherAI/wikitext_document_level`, `wikitext-2-raw-v1`, not WikiText-103.
Its 62-document word perplexity of 10,770.10 must not be called WikiText-103. The
existing provisional protocol's task mapping needs correction in a new version before
release; artifact integrity checks alone do not establish correct dataset identity.

A separate explicit WikiText-103 pilot scored the first 100 nonempty test rows from
the already pinned `Salesforce/wikitext`, `wikitext-103-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. It scored 10,665 tokenizer tokens in
1.90 seconds, with token perplexity 2,186.30. It uses raw acquired text, no
detokenization, and the adapter's rolling scoring; this token-denominator result is
not comparable to the harness word-perplexity number or an official scoring protocol.
The selected row IDs, text hash, dataset revision and checkpoint hash are preserved
in `docs/evidence/pilots/verified_slice_v3/wikitext103_pilot.json`.

All evaluation here is **PILOT_ONLY / PROVISIONAL_NOT_OFFICIAL**. Full accuracy
splits, the final held-out scoring definition, production-shard training, exact resume,
cross-machine takeover and clean export checks remain outside this bounded pilot.

## Budget and scope options (proposals, not frozen decisions)

Using 50,000–60,000 tokens/second as a provisional planning range:

| Consumed training tokens | Pure training time | With an assumed 30% overhead allowance |
|---|---:|---:|
| 1 billion | 4.6–5.6 hours | 6.0–7.2 hours |
| 3 billion | 13.9–16.7 hours | 18.1–21.7 hours |
| 5 billion | 23.1–27.8 hours | 30.1–36.1 hours |
| 10 billion | 46.3–55.6 hours | 60.2–72.2 hours |

The allowance is an assumption, not a measured end-to-end estimate. Data acquisition,
cleaning, full benchmark evaluation, failures, and submission preparation are additional.
At the current 104.5M-token stable allocation, 1B consumed tokens would be about
9.6 corpus passes and 5B about 47.8: expand distinct clean data rather than treating
repetition as equivalent to new data. Final budgets must follow a real-shard profile.

Recommended staged option: secure one fully evaluated 1B-token model first, with
enough distinct clean data to control repetition; extend the planned training horizon
to 3–5B only after preparation throughput and a release fallback are secured.
Choose the schedule/horizon and fallback strategy explicitly before the real run;
do not silently change a frozen run's identity or extend a fully decayed checkpoint.

Scope choices: (A) one model and complete evaluation/documentation/demo, lowest risk;
(B) that baseline plus one controlled comparison if time remains, recommended;
(C) the original proxy and multi-branch campaign, highest preparation/analysis burden.
Any departure from frozen campaign definitions requires a dated new version, never an
in-place edit or a claim that omitted canonical gates passed.

The [official rules](https://gibc-v2.devpost.com/rules), checked 2026-09-06, allow
public datasets and require fresh weights, at most 50M parameters, specified benchmark
reporting and submission assets. They state no minimum training-token budget.
Deadline: 2026-09-21T23:45:00+08:00. Completion is plausible with reduced scope,
conditional on resolving data supply and retaining time for full evaluation and assets.

## Implementation verification

The cluster-index change passed 29 relevant tests and a wheel build. Docker build
was attempted but the Linux engine is unavailable; Ruff was attempted but is absent.
No system packages were installed. Existing v3 changes remain uncommitted on
`g1-evidence`. Both the limited harness run and explicit WikiText-103 sample completed;
their results and limitations are recorded above. No training or evaluation process remains.
