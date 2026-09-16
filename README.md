# TinyBench-LM

A **49,658,368-parameter language model trained from scratch** for GIBC V2 Track 01.
It continues English text; it is a base model, not an instruction-tuned chatbot.
The completed baseline trained on **1,000,079,360 loss tokens**. The optional fresh
3B-token run was abandoned due to time constraints and is not part of this release.

**[Download the trained model and evidence — v1.0.0](https://github.com/liisheng/solid-train/releases/tag/v1.0.0)**

## Start here: run the trained model

You need Git, Docker Desktop (Linux containers) or Docker Engine, an internet
connection for setup, and enough disk space for the Python/PyTorch image. No API
key, training corpus, or retraining is required. The model download is about 199 MB.
The commands below work in PowerShell and Bash and run inference on CPU.

```sh
git clone --branch v1.0.0 https://github.com/liisheng/solid-train.git
cd solid-train
docker build -t tinybench-lm:verify .
docker volume create tinybench-model
docker run --rm -v tinybench-model:/model tinybench-lm:verify python release_tools/download_release.py --output-dir /model
docker run --rm tinybench-lm:verify python scripts/count_params.py
docker run --rm -v tinybench-model:/model:ro tinybench-lm:verify python generate.py --checkpoint /model/baseline_export.pt --tokenizer /model/tokenizer.json --prompt "Software engineering is the practice of" --max-new-tokens 40 --temperature 0 --top-k 1
```

The downloader checks the size and SHA-256 of every model asset against the
[release manifest](configs/release/submission_v1.json). Parameter counting should
print **49,658,368**, below the 50,000,000 cap by **341,632**. Generation prints the
prompt followed by its continuation. The example uses near-zero temperature and top-k 1 for an argmax continuation.
Do not expect a factual answer or conversation: repetition and factual errors are
known limitations. See [generation diagnostics](docs/g5/GENERATION_DIAGNOSTIC.md).

The named Docker volume keeps the model across container runs. Repeating the
download command verifies existing files and skips unchanged downloads. GitHub's
automatic source ZIP does not include release assets; use the downloader or the
individual asset links on the release page.

## Test and evaluate

Run the self-contained code tests and lint checks without downloading any corpus:

```sh
docker run --rm tinybench-lm:verify
docker run --rm tinybench-lm:verify python -m ruff check src scripts release_tools tests train.py generate.py evaluate.py
```

For a small **CPU smoke evaluation**, score one example from each of the five tasks:

```sh
docker volume create tinybench-evaluation
docker volume create tinybench-cache
docker run --rm -v tinybench-model:/model:ro -v tinybench-evaluation:/results -v tinybench-cache:/cache tinybench-lm:verify python evaluate.py --checkpoint /model/baseline_export.pt --tokenizer /model/tokenizer.json --device cpu --precision float32 --smoke --limit 1 --batch-size 1 --bootstrap-iters 0 --cache-dir /cache --output /results/smoke/results.json --bundle /results/smoke/bundle
```

This is an installation check, **not a benchmark score**. First use downloads the
pinned public benchmark datasets; the one-example limit does not limit dataset
download size. Results persist in the `tinybench-evaluation` Docker volume. To copy
them to a local folder:

```sh
docker create --name tinybench-results -v tinybench-evaluation:/results tinybench-lm:verify
docker cp tinybench-results:/results ./evaluation-results
docker rm tinybench-results
```

The Windows guide includes `python scripts\check_environment.py` to check the
installed versions against `constraints/verified-py311-windows.txt`.

For the full CUDA evaluation and a native Windows installation, follow
[the judge guide](docs/SUBMISSION.md). Full CPU scoring is possible but slower;
the recorded GPU timing is not a CPU estimate. Published raw outputs are in
`evaluation-evidence.zip` on the release page. A compact
[benchmark summary](results/benchmark-summary.json) is also in Git.

## Results and training cost

All scores below are **PROVISIONAL_NOT_OFFICIAL**. The exact organizer WikiText
slice and official evaluation settings have not been confirmed. Our run used
lm-evaluation-harness 0.4.12, zero-shot, seed 1234, batch 16, CUDA/BF16, and
1,000 bootstrap iterations, with complete verified example coverage.

| Benchmark | Examples | Accuracy | Normalized accuracy |
| --- | ---: | ---: | ---: |
| HellaSwag | 10,042 | 27.30% | 27.52% |
| ARC-Easy | 2,376 | 41.33% | 37.50% |
| PIQA | 1,838 | 58.65% | 58.11% |
| WinoGrande | 1,267 | 49.96% | Not emitted |

WikiText-103: **word perplexity 235.1593**, evaluated over 2,891 nonempty test
paragraphs. This differs from the token-based development perplexity and must not
be compared directly with it. WinoGrande accuracy is near chance.

| Completed baseline measurement | Value |
| --- | --- |
| Training hardware | NVIDIA GeForce RTX 4070 SUPER, 12 GB |
| Training precision | BF16 |
| Loss tokens / optimizer updates | 1,000,079,360 / 3,815 |
| Training command wall time | 15,767.193 seconds (4 h 22 min 47 s) |
| Optimizer time | 15,414.290 seconds |
| Peak allocated GPU memory in training logs | 5.4454 GiB |
| Approximate training compute, `6 × parameters × tokens` | 2.97974 × 10^17 FLOPs (0.298 EFLOP) |
| Full evaluation process wall time | 192.305 seconds |

The FLOP estimate is a rough dense-model approximation, not measured hardware
FLOPs. Training wall time corresponds to about 4.38 single-GPU wall-hours, not
measured GPU-active time. These figures cover the completed baseline only; earlier
experiments, profiling, failed data preparation, verification, and evaluation are
excluded from the training subtotal. Whole-project compute is not fully reconciled.
See [G5 evidence](docs/g5/COMPLETION.md), [G6 evidence](docs/g6/RESULTS.md), and
[the machine-readable training summary](results/training-summary.json).

## Model and reproducibility

The [configuration](configs/final_49m.json) defines 14 layers, width 512, 8 query heads and
4 key/value heads, SwiGLU width 1,504, a 12,288-token BPE vocabulary, and
1,024-token context. The model uses RoPE, RMSNorm, tied embeddings/output weights,
and PyTorch SDPA. Unique parameters are counted once, including the tied tensor.

Training began with seeded random initialization, not pretrained weights,
fine-tuning, or distillation. The [step-zero provenance](results/step_zero_provenance.json)
records the seed, initial weight digest, source identity, and parameter count.
See the [model card](docs/MODEL_CARD.md) for intended use and limitations.
The published checkpoint is the original verified export; packaging does not
change its weights or tokenizer.

Full retraining is optional for reviewers. It needs separate corpus acquisition,
cleaning, decontamination, packing, and appropriate GPU/storage resources. Start
with [the reduced baseline contract](docs/g3/BASELINE_CONTRACT.md),
[the selected recipe](configs/training/baseline_reduced_v2.yaml), and
[the training handoff](docs/g4/G5_HANDOFF.md). Historical local evidence paths are
provenance references, not files that appear automatically after cloning.
[The documentation index](docs/README.md) distinguishes completed work from
historical proposals. No 3B-run preparation is needed to use this release.

The historical [evidence matrix](configs/release/evidence_matrix_v1.yaml) retains
original campaign requirements and their `NOT_RUN` or blocked states. Publishing
this baseline does not claim those larger campaign gates have passed.

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

AI coding assistance (OpenAI Codex) was used for implementation, tests, debugging, research summaries, and documentation. Project decisions and submitted claims remain the entrant's responsibility. No hosted model
performs inference for the submitted model, labels or rewrites training text, or is a required
evaluation dependency.
