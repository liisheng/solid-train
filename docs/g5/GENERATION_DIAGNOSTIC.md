# Generation diagnostic

2026-09-12T23:16+08:00. Local diagnostic of the completed baseline, following
user-reported repetition. This is not G6 evaluation or a release approval.

## Findings

- Export SHA256 matches the G5 completion record, and every exported model tensor
  exactly equals the completed training checkpoint. All weights are finite.
- Tokenizer SHA256 `2103d520df7a23490054cff474f5c0e0f241bb56b51051284ab92889a683c635`
  matches the frozen runner identity; vocabulary size is 12,288. The shard
  manifest's tokenizer digest identifies the protocol, not the tokenizer file.
- Four prompts, seed 1337, 80 new tokens each, top-k 50, temperatures 0 and 0.8:
  all greedy samples develop strong repetition. Sampling reduces exact repetition
  but produces semantic drift and false statements. Lower temperature is not a fix.
- FP32 and BF16 first-token argmax agrees on all four prompts; mean absolute logit
  differences range 0.0128–0.0189. This is a spot check, not full precision equivalence.
- No generated sample contains EOS, so failure to stop at EOS in the existing
  generator does not explain these samples.
- Eight 512-target data spot checks (first shard prefix in each of four sources,
  training and development) decode as readable text. Correct next-token loss is
  2.95–4.67 versus 9.58–11.32 for unshifted targets. This argues against an obvious
  tokenizer or copy-current-token failure; it is not a complete data pipeline audit.
- Prefixes include web boilerplate and repeated page furniture. This supports
  investigating data quality, but eight nonrandom prefixes cannot establish its
  prevalence or prove it caused the generated repetition.
- Recorded full-dev loss improved from 9.3476 to 3.2871 during training; final
  training batch loss is 3.1816. Learning occurred, but these losses do not establish
  factual accuracy, instruction following, or coherent long-form generation.

## Conclusion and next step

Update 2026-09-13: the subsequently completed [G6 evaluation](../g6/RESULTS.md)
provides provisional benchmark results. The recommendation below records the next
step at the time of this diagnostic; no additional benchmark run is implied.

The poor generation is reproducible with the verified artifact. No artifact mismatch
or obvious inference arithmetic defect was found. The evidence points toward a
model-quality limitation, with the relative contributions of capacity, data quality,
and training duration unresolved. The model is not ready to be presented as a useful
chat assistant. Training completion remains valid; conversational quality is a separate
unmet objective.

Keep this baseline immutable and perform the planned G6 evaluation for objective
benchmark evidence before selecting further training work. Any quality improvement
should use a separate experiment with a fixed development generation set, broader
data-quality sampling, and measured comparisons. Repetition penalties could suppress
surface loops but would not demonstrate improved knowledge or reasoning.

Evidence: `runs/verification/generation-diagnostic/{check.py,report.json,data_check.py,data-report.json}`.
Eight generations completed in 5.47 seconds including loading and integrity checks;
the data check ran separately. Existing Windows CUDA environment used under the
repository's CUDA workflow exception. No production code, model, tokenizer or frozen
configuration changed; no training, benchmark scoring, commit or publication performed.
