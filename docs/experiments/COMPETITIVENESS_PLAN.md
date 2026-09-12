# Draft: improve the 49M baseline

2026-09-13T01:51+08:00. **PROPOSAL ONLY — not an approved training or scope change.**
User requested a draft. This document does not amend frozen contracts or launch work.

The [next-run handoff](../next_run/README.md) expands this proposal into a candidate
specification, implementation checklist and operator runbook. It resolves the 3B label
to 11,445 updates / 3,000,238,080 loss tokens and lists the remaining runtime gaps.
Use that package to prepare the run; the proposal is not an executable launch config.

## Recommendation

Build one fresh 3B-token candidate with the existing 49,658,368-parameter architecture,
12,288-token tokenizer and confirmed CONTROL recipe, using a larger, cleaner corpus.
Preserve the verified 1B baseline as fallback. Prioritize data quality and useful
exposure before architecture changes, chat tuning or another hyperparameter search.

This is a hypothesis, not a promise of higher benchmark scores. Existing diagnostics
found readable data, correct artifact identities and learned next-token prediction,
but poor free generation. They did not establish the dominant cause. The project has
trained on 1,000,079,360 tokens, about 20.14 tokens/parameter, from 550,094,903 distinct
stable tokens. More exposure is plausible, but repeatedly cycling the same material
can have diminishing returns. At unchanged corpus size, 3B and 5B would correspond to
approximately 5.45 and 9.09 aggregate passes; individual source repetition differs.

The previous 85/5/7/3 mixture failed confirmation: global improvement 0.289% versus
required 0.300%, broad-general regression 1.36%, narrative regression 2.12%.
Do not retry that treatment merely because the benchmark results are disappointing.
See [confirmation evidence](C0C1_RETURN_REVIEW.md).

## Proposed stages and decision rules

| Stage | Work | Exit evidence |
| --- | --- | --- |
| P0: freeze the comparison | Bind baseline hashes, development data, candidate recipe, budget and one-comparison scope in a successor contract. | Reviewed immutable plan; original baseline and protocols preserved. |
| P1: audit and expand data | Inspect a fixed random 100 documents/source, extend the existing licensed source revisions, remove measured boilerplate/repetition defects. | Source-level quality report, clean token counts, hashes, licenses, split isolation and decontamination PASS. |
| P2: qualify implementation | Verify actual scheduled input/target alignment, document boundaries/EOS, loss masks, source sampling and gradient accumulation; run short correctness/recovery checks on new inputs. | Tests and real-input checks pass; measured memory/throughput; complete data and schedule identities. |
| P3: train one candidate | Fresh initialization, fixed 3B budget, fixed recipe and full-dev monitoring. | Complete decay endpoint, verified counters, export reload, training and cost ledger. |
| P4: select on development data | Apply the preregistered NLL rule below; record generation diagnostics separately. | Candidate decision fixed before its benchmark evaluation. |
| P5: evaluate and release | One full benchmark evaluation of the selected new candidate, then complete G6 release requirements. | Same-protocol comparison, all raw results, efficiency disclosure and release approvals. |

These are proposed development stages, not new canonical G7–G11 gates. Changed data
reopens the relevant G1/G2 checks; changed run contracts require G4 readiness before G5
training. Existing proof should be reused only where its bound inputs remain unchanged.

### P1: better and more distinct data

Keep target shares 70% FineWeb-Edu / 20% general FineWeb / 7% OpenWebMath / 3% narrative.
The internal `dclm` label denotes general FineWeb, not the DCLM dataset. Expand within
the existing approved source families and pinned revisions first. Do not silently use
reserved experimental pools. Aim for 3B usable distinct tokens; require at least 1.5B
with source-specific exposure at most 2 passes for the proposed 3B run. Inability to meet
that condition returns to planning rather than silently relaxing the data gate.

Audit repeated paragraphs, navigation, malformed extraction, long whitespace runs,
non-English material and short/low-content documents. Choose filter thresholds from
training-only samples, record retention by source, and re-audit a separate random
sample after filtering. Do not indiscriminately strip repeated text from mathematical
notation or dialogue. Existing deduplication and benchmark decontamination must rerun
on expanded inputs; old checks cannot certify new documents. Keep dev/final splits
out of training, including document-level and near-duplicate matches.

Human spot checks diagnose data, not the entire corpus. Report counts and uncertainty;
the eight prefixes already inspected are insufficient evidence for a corpus-wide claim.
Do not add teacher-generated answers, distillation or synthetic instruction data as
part of this proposal. Existing rules prohibit distillation; ambiguity around external
synthetic corpora is unnecessary for this candidate.

### P2–P3: preserve known working choices

Retain model/tokenizer, BF16, context 1024, batch 8 x accumulation 32 and peak LR 0.0006.
Retain optimizer settings unless a verified correctness defect requires a separate fix.
Use seed 1337 and a materialized, source-balanced schedule. Freeze exact whole-update
counts and warmup/stable/decay counts in the successor contract before any training;
the proposed proportions remain 1% / 89% / 10%, rounded with explicit accounting.
The [exact draft specification](../next_run/CANDIDATE_SPEC.md) uses half-up rounding:
114 warmup, 10,186 stable and 1,145 decay updates. Runtime adoption still requires a
new versioned contract and verification.

Start fresh. The existing endpoint has LR zero; the current resume identity guards
cannot be bypassed to turn it into an unplanned longer run. A continuation would need
a separate restart schedule and lineage experiment and is not the recommended path.

The short preflight verifies execution and provides estimates; it does not select a
mixture or checkpoint. Keep complete full-dev loss and all four slice losses, save
recoverable checkpoints, preserve failed attempts, and disclose actual wall time and
approximate FLOPs. User observes long jobs. Estimates are advisory; elapsed time never
silently kills training. If instability or data corruption is detected, pause for diagnosis.

### P4: meaningful development improvement

Proposed engineering adoption threshold: endpoint full-dev token NLL at least 2% below
baseline 3.287145695 (at most 3.221402781), with no protected slice more than 1% worse.
First reproduce the baseline per-slice values using the same fixed dev schedule and
evaluation precision; the baseline's logged global value alone is not enough.
Use complete sample counts and token-weighted losses, not the best minibatch.
These thresholds are proposed practical rules, not significance claims. Predeclare
paired document-level uncertainty reporting, including caveats about the small narrative
holdout; do not invent confidence from one initialization.

Freeze 40 original development prompts before the candidate, balanced across general,
educational, technical and narrative text. Use fixed seeds and greedy / temperature 0.8
decoding; compare repeated 4-grams and a written coherence rubric. No benchmark text
or paraphrases. These are diagnostic reports, not a second candidate-selection signal
under the existing NLL-only policy. Bad generation must remain visible even if NLL wins.

If the candidate fails adoption, retain the baseline and report the failed experiment.
Do not automatically start another run. A second seed, 5B run, new mixture or architecture
search requires a separately reviewed scope decision after development evidence.

### P5: benchmark improvement without test-set tuning

The current benchmark scores are already exposed. A successor cannot claim its design
was blind to them. Keep training decisions tied to the predeclared development process,
do not inspect benchmark errors to curate training documents, and evaluate only after
the candidate decision is frozen. Do not switch models, retune, or try multiple endpoints
based on the new scores. Preserve negative results as well as improvements.

Report baseline and candidate accuracy AND normalized accuracy, WikiText denominators,
stderr and complete coverage under the same frozen protocol. If organizer settings
change, rerun both artifacts under one new protocol. Published external model cards
are context, not a matched ranking. No numerical benchmark gains are forecast here;
benchmark improvement is the desired outcome, not a training-selection threshold.

## Budget and sequence

Projection uses the actual G5 outer training time 15,767.193 s for 1,000,079,360 tokens.
It assumes unchanged throughput and similar overhead per token; remeasure on new shards.

| Option | Estimated training time | With 25% planning margin | Decision |
| --- | ---: | ---: | --- |
| Fresh 3B candidate | 13.1 h | 16.4 h | Recommended after data gate |
| Fresh 5B candidate | 21.9 h | 27.4 h | Defer; not additive to 3B and not authorized |

CPU data download/filter/tokenization, implementation checks, preflight, storage/backup,
failures and release work are additional and currently unmeasured. Measure a fixed input
sample and extrapolate with retention uncertainty before launching full data preparation.
Approximate 6ND compute for 3B is 8.94e17 FLOPs (estimate, not measured hardware FLOPs).
One measured G6 evaluation took 192.305 s outer; reserve 1 h for evaluation/review as a
planning choice, not a p90 estimate. Final release preparation needs a separate calendar
buffer of at least one day; verify the organizer deadline before accepting a schedule.

Sequence after approval: data/preflight with measured ETA, one overnight-scale training
run, then development decision and final evaluation/release. Do not promise a fixed
calendar finish until data preparation is measured. Training efficiency is scored:
include prior experiments and the original baseline in total costs, and show marginal
candidate cost separately. More training may improve quality while worsening efficiency;
the unknown competition weights prevent a guaranteed overall ranking gain.

## Evidence and research basis

Local evidence: [G5 completion](../g5/COMPLETION.md),
[generation diagnostic](../g5/GENERATION_DIAGNOSTIC.md),
[G6 results](../g6/RESULTS.md), [rules](../RULES.md).

Research checked 2026-09-13:

- [SmolLM, published 2024-07-16](https://huggingface.co/blog/smollm): primary account
  of curated data for small models. Supports investigating data quality; its synthetic
  corpus is not adopted here.
- [SmolLM2, submitted 2025-02-04](https://arxiv.org/abs/2502.02737): reports substantial
  training exposure and data refinement. Its results do not establish that this 49M
  model will improve at 3B tokens. The proposed scale and thresholds are our engineering
  choices based on local resources, not claims from that paper.

Next action if this draft is accepted: prepare the versioned P0/P1 contract and data
audit, with no training until the dataset and measured run budget are reviewable.
