# TinyBench-LM

TinyBench-LM is a from-scratch, decoder-only language model designed for the GIBC V2
50-million-parameter track. The final architecture contains **49,658,368 unique trainable
parameters** — every parameter counted once, including the token embedding, which is the
same tensor as the output head rather than a second copy of it. That leaves **341,632**
parameters of headroom under the 50,000,000 cap. Verify it yourself with
`scripts\count_params.py`; the number is enforced by test, not by assertion.

**Random initialization.** No pretrained weights, no fine-tuning of another model, and no
distillation. The eligibility scan (`src\tinybench_lm\eligibility.py`) fails closed on
`from_pretrained()`, remote weight fetching, or a teacher dependency in the training path,
and `src\tinybench_lm\provenance.py` records a step-zero weight hash before the first
optimizer step.

**What it is not.** A general assistant. A 49.66M-parameter model trained on a bounded
public corpus is a research artifact for a parameter-capped track, not a source of facts or
advice.

## Status

No final training run has happened. Benchmark scores, perplexity, throughput, memory,
training time, and compute are therefore **not reported here** — reporting them would mean
inventing them. The machine-readable status of every submission claim, with its evidence
path and verifier, is `configs/release/evidence_matrix_v1.yaml`; render it with
`src\tinybench_lm\release.py`. Nothing in that matrix is `PASS` unless a named verifier has
actually run.

This repository currently provides:

- a RoPE + RMSNorm + SwiGLU causal Transformer using PyTorch SDPA;
- exact parameter-cap verification;
- bounded, streaming public-data acquisition;
- a custom byte-level BPE tokenizer;
- document-level train/validation splitting and packed `uint16` token shards;
- BF16 training, gradient accumulation, validation, logs, and resumable checkpoints;
- local text generation and a hardware throughput profiler.
- an `lm-evaluation-harness` adapter for the competition benchmarks.

`docs/PILOT_REPORT.md` and `docs/RESEARCH_PLAN.md` are **historical**. Their measurements
were real, but they describe a superseded 49,295,872-parameter architecture; both carry a
banner saying so. Treat the authoritative execution plan and the frozen configs under
`configs/` as current.

## Mental model

Training repeatedly performs the same cycle:

```text
public text -> clean -> tokenize -> make input/next-token pairs
            -> predict -> measure error -> backpropagate -> update weights
```

At inference time, the trained model predicts one token, appends it to the prompt,
and repeats. Benchmark multiple-choice scoring instead compares the likelihood of
each candidate continuation.

## Architecture

The final architecture in `configs/final_49m.json` uses **14 layers**, a **512**-wide
residual stream, **8 query heads with 4 key/value heads** (grouped-query attention), a
**1,504**-wide SwiGLU feed-forward network, a **12,288**-token vocabulary, and a **1,024**-token
maximum context. Attention is pre-norm RMSNorm with RoPE at theta 10,000; there are no
biases and dropout is zero. Input and output embeddings are one shared parameter tensor,
which is why the counter enumerates unique `Parameter` objects rather than state-dict
entries.

```text
49,658,368 = 6,291,456 embedding + 14 × 3,097,600 per layer + 512 final norm
```

`configs/README.md` records what each config is for. `baseline_49m.json` is a compatibility
alias for the same final architecture; `pilot_12m.json` is for fast pipeline validation only;
`deep_thin_gqa_49m.json` is a rejected research candidate kept for reproducibility. Neither
pilot nor rejected config is a final model candidate.

## Local environment

The pilot environment and caches live on `D:\SWE\benchmark-50m-lm` to avoid filling
the Windows system drive. Runtime dependencies are pinned exactly and bounded by a
constraints file that was generated from a working install, so a fresh checkout cannot
silently resolve a different tokenizer, dataset, or evaluation-harness protocol. From
PowerShell in this directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]" -c constraints\verified-py311-windows.txt
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe scripts\count_params.py
.\.venv\Scripts\python.exe -m pytest
```

`scripts\check_environment.py` is the dependency check: it compares the declared pins,
`constraints\verified-py311-windows.txt`, and the versions actually installed, and exits
non-zero on any unpinned, missing, or divergent dependency. Test tooling (`pytest`,
`hypothesis`) is the separate `test` extra, so runtime and release installs omit it.
GPU/backend choices stay optional and documented; the check reports backend facts as
information only and never changes model semantics. Verified versions, platform facts,
and the CUDA wheel option are recorded in `docs/ENVIRONMENT.md`.

Prepare a small public pilot corpus:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py `
  --dataset HuggingFaceFW/fineweb-edu `
  --subset sample-10BT `
  --max-docs 6000 `
  --max-bytes 30000000 `
  --vocab-size 8192 `
  --output-dir data\processed\fineweb_edu_pilot
```

The existing pilot uses the public
[FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
whose dataset card lists the ODC-By license. Synthetic TinyStories text is excluded
because this project does not use commercial model-generated training data.

Run a short pilot:

```powershell
.\.venv\Scripts\python.exe train.py `
  --config configs\pilot_12m.json `
  --data-dir data\processed\fineweb_edu_pilot `
  --run-dir runs\fineweb_edu_pilot `
  --steps 1000
```

Generate text from the best checkpoint:

```powershell
.\.venv\Scripts\python.exe generate.py `
  --checkpoint runs\fineweb_edu_pilot\best.pt `
  --tokenizer data\processed\fineweb_edu_pilot\tokenizer.json `
  --prompt "Once upon a time"
```

Run the competition's four multiple-choice benchmarks:

```powershell
.\.venv\Scripts\python.exe evaluate.py `
  --checkpoint runs\full\best.pt `
  --tokenizer data\processed\final\tokenizer.json `
  --tasks hellaswag,arc_easy,piqa,winogrande `
  --output runs\evaluation\core_results.json
```

`paloma_wikitext_103` is available as a separate harness task for development.
The final submission must use the exact WikiText-103 held-out slice and evaluation
settings specified or supplied by the organizers.

## Competition integrity

The final corpus must be licensed and documented. Evaluation samples and close
duplicates from HellaSwag, ARC-Easy, PIQA, WinoGrande, and the held-out WikiText-103
slice must be excluded. Public benchmark training splits will not be added unless
their use is explicitly confirmed as acceptable and disclosed.

These are not aspirations. The data-safety protocols under `configs/data/` are frozen and
pinned by SHA-256, so a threshold cannot be edited after a scan; the eligibility scan fails
closed on pretrained weights, distillation, or a teacher dependency; and the campaign's
decision thresholds were frozen in `configs/campaign/preregistration_v1.yaml` before any
outcome exists. What has *not* happened is the measurement: no corpus has been acquired, no
decontamination rate has been measured, and no run has started. Every such item is `NOT_RUN`
or `BLOCKED` in the evidence matrix rather than quietly omitted.

## Credits

Every dataset, framework, and tool used, as the Track 01 rules require. Datasets are pinned
to immutable revisions in `configs/data/sources_v4.yaml`; the revision column is the exact
Hugging Face commit this project uses.

### Training corpus

| Dataset | Licence | Pinned revision | Role |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ODC-By 1.0 | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | 70% stable — educational English; also the reserved science and top-decile pools |
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | ODC-By 1.0 | `9bb295ddab0e05d785b879661af7260fed5140fc` | 20% stable — general-language diversity |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | ODC-By 1.0 | `fde8ef8de2300f5e778f56261843dab89f230815` | 7% stable — mathematical and scientific prose; also the reserved math pool |
| [Project Gutenberg English](https://huggingface.co/datasets/sedthh/gutenberg_english) | MIT | `28973b04f28fd7be4a6186a042bc26159d4366ca` | 3% stable — long-form narrative and co-reference |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | CC BY-SA 3.0 / GFDL | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | reserved pool — general article text |
| [open-text-books](https://huggingface.co/datasets/izumi-lab/open-text-books) | CC BY-SA 4.0 | `1245fefd628d37483366b8e707fdc5650fd3c48e` | reserved pool — textbook and instructional prose |

FineWeb-Edu, FineWeb, and OpenWebMath all derive from Common Crawl; users of those datasets should also
observe the [Common Crawl terms of use](https://commoncrawl.org/terms-of-use/). Wikipedia text
is share-alike, and any redistribution of that portion carries CC BY-SA obligations.

**Not used:** no synthetic or model-generated corpus (TinyStories and similar), no teacher
logits or rankings, no benchmark examples, no large code corpora, no unfiltered Common Crawl,
and no hosted model labelling, rewriting, or scoring training documents. The prohibitions are
enforced in `configs/data/sources_v4.yaml` and scanned by `src/tinybench_lm/eligibility.py`.

### Frameworks and tools

| Component | Role |
|---|---|
| [PyTorch](https://pytorch.org) | model, training, SDPA attention, mixed precision |
| [Hugging Face Datasets](https://github.com/huggingface/datasets) | streaming pinned public datasets |
| [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) | the original 12,288-token BPE |
| [safetensors](https://github.com/huggingface/safetensors) | release weights and hash verification |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | required benchmark evaluation |
| [NumPy](https://numpy.org) | uint16 token storage, deterministic bootstrap |
| [pytest](https://pytest.org) + [Hypothesis](https://hypothesis.works) | property and contract tests |

Exact pinned versions are in `constraints/verified-py311-windows.txt`, verified by
`scripts\check_environment.py`.

AI assistance is disclosed in `docs/templates/AI_ASSISTANCE_DISCLOSURE.md`. No hosted model
performs inference for the submitted model, labels or rewrites training text, or is a required
evaluation dependency.

## Evidence and status

`configs/release/evidence_matrix_v1.yaml` maps every competition contract item and every
G0–G6 gate to a path, a verifier, a status, and a failure policy:

- **`PASS`** — a named verifier ran and the artifact exists. Only the parameter cap and the
  no-pretrained-weights scan currently qualify.
- **`BLOCKED`** — an organizer, teammate, or host must act first; the owner and next action
  are recorded. Personal eligibility and the exact WikiText-103 slice are here.
- **`NOT_RUN`** / **`TBD`** — the artifact or measurement does not exist yet.

`src\tinybench_lm\release.py` enforces that a `PASS` names a verifier and a path that exist
and that a `BLOCKED` names an owner, so ticking a box early fails a test. Submission
templates — model card, data card, AI-assistance disclosure, Built With, and the submission
package — are in `docs/templates/`.

## Stop conditions and fallback

No additional architecture, data family, optimization method, dashboard, frontend, or
hosted-model workflow will be added unless measured evidence creates a specific need and the
critical schedule is ahead. If the branch experiment fails or is incomplete, the submission
ships the best valid ordinary-decay or stable fallback and reports the experiment as null or
incomplete. An undecayed peak-LR mainline checkpoint is never released as the fallback.
