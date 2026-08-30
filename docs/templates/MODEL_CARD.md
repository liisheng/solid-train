# Model card — TinyBench-LM (template)

> **Template.** Every `TBD` is a value only a completed run can supply, and every `BLOCKED`
> needs an external answer. Do not replace a placeholder with an estimate, a target, or a
> number from the pilot: an unmeasured figure presented as a result is the one failure this
> template exists to prevent. Fill a field only when its named evidence path exists.

## Model summary

| Field | Value | Evidence |
|---|---|---|
| Name | TinyBench-LM | — |
| Track | GIBC V2 Track 01, 50M parameter cap | — |
| Architecture | decoder-only Transformer, pre-norm RMSNorm, SwiGLU, RoPE, GQA | `configs/final_49m.json` |
| Unique trainable parameters | 49,658,368 | `scripts/count_params.py` |
| Cap headroom | 341,632 | `scripts/count_params.py` |
| Layers / width / heads | 14 / 512 / 8 query, 4 KV | `configs/final_49m.json` |
| FFN width | 1,504 | `configs/final_49m.json` |
| Vocabulary | 12,288 | `configs/data/tokenizer_v1.yaml` |
| Context length | 1,024 | `configs/final_49m.json` |
| Embedding tying | input and output share one parameter tensor | `scripts/count_params.py` |
| Initialization | random; no pretrained weights, fine-tuning, or distillation | `src/tinybench_lm/provenance.py` |
| Step-zero weight hash | `NOT_RUN` | `src/tinybench_lm/provenance.py` |
| Release commit / tag | `NOT_RUN` | — |
| Release artifact hash | `NOT_RUN` | — |
| Rollback artifact hash | `NOT_RUN` | — |

## Intended use and limitations

**Intended.** Research into small-scale from-scratch language-model training under a hard
parameter cap; reproducing the training and evaluation pipeline; the competition's required
benchmark evaluation.

**Not intended.** Production deployment, factual question answering, advice of any kind, or any
setting where a confidently-wrong answer causes harm. A 49.66M-parameter model trained on a
bounded public corpus is not a general assistant.

**Known limitations.** `TBD` — complete from measured evaluation, not from expectation.

## Training data

See `docs/templates/DATA_CARD.md`. Corpus profile, accepted token count, source revisions, and
licenses are `NOT_RUN` until acquisition completes.

## Training procedure

| Field | Value | Evidence |
|---|---|---|
| Optimizer | AdamW, betas 0.9/0.95, eps 1e-8 | `configs/training/recipe_v1.yaml` |
| Weight decay | 0.1, excluding embeddings and all normalization weights | `configs/training/recipe_v1.yaml` |
| Gradient clip | global norm 1.0 | `configs/training/recipe_v1.yaml` |
| LR schedule | WSD: warmup, stable peak, linear decay to zero | `configs/training/recipe_v1.yaml` |
| Peak learning rate | `NOT_RUN` — selected by P4 and F1/F2 | `configs/campaign/preregistration_v1.yaml` |
| Global batch | 262,144 loss tokens per update | `configs/training/recipe_v1.yaml` |
| Precision | BF16, with FP16 + GradScaler fallback | `configs/training/recipe_v1.yaml` |
| Consumed tokens | `TBD` | `src/tinybench_lm/operations.py` |
| Effective passes | `TBD` | `src/tinybench_lm/operations.py` |
| Tokens per parameter | `TBD` | `src/tinybench_lm/operations.py` |
| Total wall time | `TBD` | `src/tinybench_lm/operations.py` |
| Active GPU-hours | `TBD` | `src/tinybench_lm/operations.py` |
| Approximate FLOPs (6ND estimate) | `TBD` | `src/tinybench_lm/operations.py` |

> 20B consumed tokens and 402.8 tokens/parameter are **stretch targets, not results**. Report
> them as targets until a measurement reaches them.

## Hardware

| Machine | GPU | VRAM | Role | Median / p10 / p90 throughput | Peak VRAM | Peak RAM |
|---|---|---|---|---|---|---|
| Mainline | `TBD` | `TBD` | continuous stable lineage, full-size LR safety | `TBD` | `TBD` | `TBD` |
| Data / branches | `TBD` | `TBD` | preprocessing, proxies, A/B/C, evaluation | `TBD` | `TBD` | `TBD` |

## Evaluation

Protocol: `configs/evaluation/evaluation_provisional_v1.yaml`. Values below are `NOT_RUN`.
Anything scored under the provisional protocol must be labeled provisional until organizer
answers are received.

| Task | Metric | Value | Protocol hash |
|---|---|---|---|
| HellaSwag | `NOT_RUN` | `NOT_RUN` | `NOT_RUN` |
| ARC-Easy | `NOT_RUN` | `NOT_RUN` | `NOT_RUN` |
| PIQA | `NOT_RUN` | `NOT_RUN` | `NOT_RUN` |
| WinoGrande | `NOT_RUN` | `NOT_RUN` | `NOT_RUN` |
| WikiText-103 perplexity | `NOT_RUN` | `BLOCKED` — exact slice and settings await organizer answers | `NOT_RUN` |

Secondary reasoning tasks are labeled non-official and never influence training.

## Reserved-pool experiment

One sentence, written only after `validation_final` is opened and the frozen analysis has run:
`NOT_RUN`. Null, harmful, and incomplete are valid outcomes and must be reported as such.

## Reproduction

```
# CPU load
NOT_RUN

# Evaluation
NOT_RUN
```

## Disclosure

AI assistance: see `docs/templates/AI_ASSISTANCE_DISCLOSURE.md`.
Built With: see `docs/templates/BUILT_WITH.md`.
