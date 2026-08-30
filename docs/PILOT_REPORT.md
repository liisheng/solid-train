# TinyBench-LM local pilot report

> **Historical — superseded by `GIBC-Track01-Final-Plan-v5.md`.**
>
> This is a dated record of a real pilot, and its measurements stay valid *as pilot
> measurements*: the throughput, peak VRAM, and hardware facts below were observed on the
> machine and date named here. They are not final-campaign results, and no figure in this
> document may be reported as one.
>
> What the final plan supersedes is the **recommendation**, not the measurement. The
> 49,295,872-parameter architecture reported here (12 layers, 8 heads, 1,536 FFN, 16,384
> vocabulary, 704,128 margin) was replaced by the final 49,658,368-parameter architecture
> (14 layers, 8 query / 4 KV heads, 1,504 FFN, 12,288 vocabulary, 341,632 margin) in
> `configs/final_49m.json`. Throughput measured on the old architecture does not transfer to
> the new one.
>
> Current architecture: `configs/final_49m.json`. Current status of every claim:
> `configs/release/evidence_matrix_v1.yaml`.

Date: 26 August 2026  
Machine: AMD Ryzen 7 7700, 16 GB system RAM, NVIDIA RTX 4070 SUPER 12 GB  
Working directory: `D:\SWE\benchmark-50m-lm`

## Outcome

The complete local path is operational: public-data streaming, text cleaning, custom
BPE training, document-level splitting, packed-token storage, BF16 training,
validation, checkpoint save/load, generation, parameter-cap verification, and
`lm-evaluation-harness` scoring.

The recommended baseline contains **49,295,872 trainable parameters**, including the
tied embedding/output tensor. The competition cap test leaves a **704,128 parameter**
safety margin.

## Measured hardware performance

| Architecture | Context | Micro-batch | Throughput | Peak VRAM |
|---|---:|---:|---:|---:|
| 12 × 512 baseline | 512 | 4 | 66,592 tok/s | 1.82 GiB |
| 12 × 512 baseline | 1,024 | 8 | **76,457 tok/s** | 5.21 GiB |
| 12 × 512 baseline | 1,024 | 16 | 70,331 tok/s | 9.71 GiB |
| 24 × 384 deep/thin GQA | 1,024 | 4 | 45,702 tok/s | 3.69 GiB |
| 24 × 384 deep/thin GQA | 1,024 | 8 | 47,185 tok/s | 6.72 GiB |

Micro-batch 8 at 1,024 tokens is the best measured full-size baseline setting. A
larger micro-batch is slower and leaves too little VRAM safety margin for desktop use.

## Real-data tests

The bounded FineWeb-Edu pilot downloaded 6,000 documents (28.0 MiB of UTF-8 text),
trained an 8,192-token byte-level BPE, and produced:

- 7,532,275 training tokens;
- 46,550 validation tokens;
- deterministic document-level separation between training and validation.

The 12.8M pilot model processed 16,384,000 tokens. Validation loss fell from 9.0520
to 4.6456, while steady-state training reached approximately 212,000 tok/s.

The full 49.3M baseline processed 1,638,400 tokens. Validation loss fell from 9.7042
to 6.2871 at approximately 75,500 tok/s. Its saved checkpoint also completed a live
ARC-Easy smoke evaluation through `lm-evaluation-harness`.

At the same token count and optimizer settings, the 49.4M deep/thin candidate reached
validation loss 6.5154 at approximately 47,500 tok/s. It is therefore not the leading
candidate: it was both slower and worse in this early-data regime. A longer controlled
ablation could revisit it, but the baseline is the prudent final-run default.

## Revised full-run estimates

The estimates below add practical allowance for validation, checkpoint writes, data
movement, and occasional desktop interference.

| Training budget | Pure measured compute | Practical allocation |
|---|---:|---:|
| 1B tokens | 3.6–3.7 hours | 4–5 hours |
| 3B tokens | 10.9–11.0 hours | 12–14 hours |
| 5B tokens | 18.2–18.4 hours | 20–23 hours |

These times apply to the current 12-layer architecture and software stack. Final data
curation and evaluation are additional calendar work, but not continuous GPU training.

## Current storage

| Component | Size |
|---|---:|
| Python/CUDA environment | 4.71 GiB |
| Download/package cache | 2.70 GiB |
| Pilot runs and checkpoints | 2.49 GiB |
| Public pilot data | 0.04 GiB |
| **Current total** | **about 9.94 GiB** |

The package cache can be removed after the environment is stable. The run directory
currently contains redundant experimental checkpoints that can also be archived or
removed after the architecture decision. Neither cleanup is required yet because D:
has ample capacity.

## Interpretation

This pilot validates engineering, not model capability. The dataset is far too small
and narrow for competition scoring, and the 8K pilot tokenizer leaves half of the
full model's 16K output vocabulary unused. The real run requires a fresh 16K tokenizer,
a much larger documented data mixture, evaluation-set decontamination, and billions of
training tokens.

