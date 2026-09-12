# Proposed candidate specification

Status: DRAFT_FOR_IMPLEMENTATION, 2026-09-13. Exact values below resolve details left
open in the [proposal](../experiments/COMPETITIVENESS_PLAN.md). They must be bound into
a new machine-readable contract before execution; this Markdown is not a runtime config.

## Model and training

| Field | Proposed value |
| --- | --- |
| Candidate label | `control_3b_v1` (new namespace, not yet an existing run) |
| Model | `configs/final_49m.json`; 49,658,368 unique trainable parameters |
| Tokenizer | Existing `data/tokenizer_final/tokenizer.json`; 12,288 IDs; EOS 1; no prepended BOS |
| Initialization | Fresh random seed 1337; no baseline, pilot or pretrained weights |
| Hardware | Single RTX 4070 SUPER; refresh measured availability/memory |
| Precision | BF16 with existing deterministic execution policy; no silent fallback |
| Batch | 8 sequences × 32 accumulation × 1,024 loss tokens |
| Loss tokens per update | 262,144 |
| Updates | 11,445 = ceiling(3,000,000,000 / 262,144) |
| Exact exposure | 3,000,238,080 loss tokens; 2,929,920 scheduled sequences |
| Optimizer | AdamW, beta1 0.9, beta2 0.95, epsilon 1e-8, weight decay 0.1 |
| Gradient clipping | Global norm 1.0; preserve existing decay/no-decay groups |
| Peak LR | 0.0006; WSD schedule |
| Warmup / stable / decay | 114 / 10,186 / 1,145 updates |
| Final LR | Exactly zero on update index 11,444 |
| Dev cadence | Completed update 1, every 100 completed updates, and final update |
| Recovery cadence | Every 100 completed updates and completion; retain interruption evidence |

Warmup and decay use half-up rounding of 1% and 10% of 11,445, with stable updates
as the remainder. Decay starts at zero-based update 10,300 (completed update 10,301).
Use the existing LR function and test first/last updates and both phase transitions;
do not rewrite its arithmetic in an observer or launcher.

The 3B label is approximate; reports and verifiers must use the exact token count.
Keeping model, tokenizer, LR and mixture does not make this a controlled attribution
study: both data and duration change relative to the baseline. Report it as a candidate
comparison, not proof that one individual treatment caused a gain.

## Source quotas and distinct data

Allocate the 2,929,920 sequences at 70/20/7/3 using integer largest-remainder rounding.
For equal fractional remainders, use source order as listed below. A sequence contributes
1,024 loss targets; do not count the additional input/lookahead token twice.

| Source ID | Sequences | Loss-token exposure | Minimum distinct usable sequences |
| --- | ---: | ---: | ---: |
| `fineweb_edu` | 2,050,944 | 2,100,166,656 | 1,025,472 |
| `dclm` (general FineWeb) | 585,984 | 600,047,616 | 292,992 |
| `openwebmath` | 205,094 | 210,016,256 | 102,547 |
| `narrative` | 87,898 | 90,007,552 | 43,949 |
| Total | 2,929,920 | 3,000,238,080 | 1,464,960 |

Require both at least 1,500,119,040 distinct usable loss-token positions across these
source pools and the per-source minimums above. Aim for one full run's worth of
distinct data. Distinct means positions in deduplicated accepted documents, not
distinct vocabulary values. Exclude dev/final/reserved pools from this denominator.

Materialize two schedule components: per-source component A gets ceiling(quota/2),
component B gets the remainder. Each component must have no duplicate reference.
Prefer unused references in B; repeat only when the verified source pool is insufficient
for a full pass. No reference or underlying target position may be exposed more than
twice across components. Verify overlaps explicitly; changing window offsets must not
evade the cap. Component count, quota tables, order, seed and all hashes are contractual.

Use the existing pinned four source families and tokenizer. Freeze new filtering and
acquisition rules as successors if their semantics change; no reserved-pool borrowing,
synthetic answers, distillation, benchmark training material or tokenizer retraining.

## Development selection

Use the exact existing `validation_dev` manifest and schedule. Do not rebuild a larger
dev set and compare its loss to the old scalar. Recompute the baseline's complete global
and four protected-slice losses with the candidate evaluator before launch; record
token sums, counts, precision, device and identities. Freeze the measured reference.

Candidate qualifies when `(baseline_nll - candidate_nll) / baseline_nll >= 0.02`
and every slice satisfies `(candidate_nll - baseline_nll) / baseline_nll <= 0.01`.
Use unrounded values. The recorded baseline 3.287145695 suggests a threshold near
3.221402781; a materially different reproduction requires diagnosis before freeze.
Require all 753 dev sequences / 771,072 scored targets and complete slice coverage.

Select the completed endpoint, not the best intermediate checkpoint. Missing evidence,
nonfinite values or failed correctness makes the candidate ineligible. If it fails,
keep the baseline; do not automatically try 5B or another mixture.

Freeze 40 original diagnostic prompts (10/source category), sampling seed 1337,
80 new tokens, top-k 50, temperatures 0 and 0.8, yielding 80 samples per model.
Record token-level repeated 4-gram fraction on generated text only and a fixed rubric
for topic adherence, coherence, repetition and checkable factual claims. Reset RNG per
sample. Report all samples. This diagnostic does not override NLL-only selection.
Any uncertainty estimates require validated grouping/attribution; packed sequences
are not automatically independent documents. No seed-significance claim from one run.

## Identities to preserve

- Baseline export SHA-256: `89438ee3165a64bd3c15b3ba1f4aa6658841a423a554164061c2d4c0ac95c288`.
- Tokenizer SHA-256: `2103d520df7a23490054cff474f5c0e0f241bb56b51051284ab92889a683c635`.
- Model config file SHA-256: `69db94fe2bd0fef99b7a8ee183678b3bc2fc8098a9cdea8fd71adf1149bb1782`.
- Development schedule content hash: `50ddce9f4f12c5f70e6369170332d168c9a1783ee530c61ab4857c1b16ca61fd`.

Reverify these against local artifacts. File-byte hashes and canonical content hashes
are different identity types and must be labelled correctly in the successor bundle.
