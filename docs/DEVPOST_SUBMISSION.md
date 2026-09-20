# Devpost submission copy — TinyBench-LM (GIBC V2, Track 01)

Paste-ready text for Devpost fields 01 (Project Description) and 04 (Built With).
Every number here is taken from committed evidence in this repository; see
`results/training-summary.json`, `results/benchmark-summary.json`,
`results/step_zero_provenance.json`, and `configs/training/baseline_reduced_v2.yaml`.

---

## 00 | Tagline (Devpost pitch field, 200-character maximum)

Recommended:

> A language model that fits under 50M parameters and under your desk. Trained from random
> noise on one RTX 4070 in 4h22m. No pretrained weights. No teacher. No API key. (167)

Alternates:

- Anyone can download a model. We built one from random noise — 49.6M parameters, 1B
  tokens, 4h22m on a gaming PC — and published the hash of step zero to prove it. (162)
- 341,632 parameters to spare. Trained from random noise on a gaming PC in 4h22m — and it
  publishes the hash of its own starting weights, so "from scratch" is checkable. (167)
- Everyone says "trained from scratch." We published the hash of our random weights so you
  can check. 49.6M params, 1B tokens, 4 hours, one gaming GPU. (149)

Do not use phrasing that claims the corpus is certified free of benchmark contamination.
`configs/data/decontam_v3.yaml` records the opposite under `limitations`: paraphrases and
short copied questions are outside the detector's guarantees.

---

## 01 | Project Description

### TinyBench-LM — a 49,658,368-parameter language model trained from scratch, under a hard 50M cap

**Elevator version.** TinyBench-LM is an English causal language model built and trained
from random initialization on a single consumer GPU (RTX 4070 SUPER, 12 GB) in 4 hours 22
minutes. It has 49,658,368 unique trainable parameters — 341,632 under the Track 01 cap,
counting embeddings and the tied output head. It was trained on 1,000,079,360 loss tokens
drawn from a corpus we acquired, filtered, deduplicated, and decontaminated ourselves. No
pretrained weights, no distillation, no teacher model, no synthetic data, no hosted API in
the training or inference path. A judge can download the weights and reproduce a
generation in five commands.

### The problem

The interesting constraint in Track 01 is not "can you write a transformer." Reference
implementations are everywhere. The constraint is that 50M parameters is a *budget*, and a
budget forces every decision to be defended: how many layers versus how wide, how big a
vocabulary (every vocab entry costs `d_model` parameters twice if untied), how many tokens
to spend, what those tokens should be.

The second, less-discussed problem is that small-model results are extremely easy to
fake — accidentally. Train on a corpus scraped from the open web and you have almost
certainly ingested HellaSwag and ARC items. Initialize from a pilot checkpoint and your
"from scratch" claim is gone. Pick the best of twenty runs by benchmark score and you have
tuned on the test set. None of these leave a visible trace unless you build the machinery
to prevent them. So this project treats *provenance* as a first-class deliverable
alongside the model.

### What we built

**1. The model.** 14 layers, `d_model` 512, 8 query heads with 4 key/value heads
(grouped-query attention), SwiGLU feed-forward of width 1,504, RoPE positional encoding
(theta 10,000), RMSNorm, no biases, tied input/output embeddings, 1,024-token context, and
a 12,288-token byte-level BPE vocabulary. Attention runs through PyTorch SDPA.

The parameter budget drove each of these. Tied embeddings alone free ~6.3M parameters
(12,288 × 512) to spend on depth. The small vocabulary is a deliberate trade: fewer
embedding parameters and `uint16` token storage, at the cost of longer sequences per unit
of text. GQA (8 query heads, 4 KV heads) cuts KV-cache and projection cost without the
quality drop of full multi-query. The result is a deep-and-narrow model — 14 layers at
width 512 — which is where the parameter budget is best spent for a fixed 50M.

**2. The corpus.** `scripts/prepare_corpus.py` is a six-stage, restartable pipeline:
acquire/filter → global near-duplicate removal → benchmark-index construction →
benchmark decontamination → boundary assignment → atomic publication.

- Sources are streamed from **pinned Hugging Face commit revisions**, not branch names:
  FineWeb-Edu 70%, FineWeb 20%, OpenWebMath 7%, Project Gutenberg English 3%.
- Deduplication is exact + mirror + 128-row MinHash at a 0.85 Jaccard threshold, indexed
  as 32 bands of 4 rows (a passing pair can differ in at most 19 rows, so at least one
  band must collide — the index is provably complete at that threshold). Connected
  near-duplicate components are merged to one canonical representative.
- Decontamination scans every kept document against an index of the benchmark suites under
  three rules: a complete benchmark field of ≥13 normalized words appearing as a substring,
  a ≥50-word contiguous overlap, and shingle coverage (≥3 distinct 13-word shingles
  covering ≥10% of the document). We document what this does *not* catch — paraphrases and
  short copied questions — rather than claiming the corpus is certified clean.
- Every stage's state is disk-backed SQLite and resumable; a resumed run must match the
  acquisition, source, filter, dedup, and tokenizer identity hashes or it fails closed.

The decontamination rules themselves have a bug-fix history worth reading: v2 quarantined
**every one of its first 7,767 documents**, because standalone benchmark answers like "2",
"4", and "8" matched rule 1. v3 added the 13-word minimum. Both versions are frozen in the
repo; v2 is kept as historical evidence instead of being edited away.

**3. The training run.** AdamW (β 0.9/0.95, ε 1e-8, weight decay 0.1, global grad-norm
clip 1.0), a WSD (warmup–stable–decay) learning-rate schedule with peak 6e-4: 38 warmup
updates, 3,395 stable, 382 linearly decayed to exactly zero, totalling 3,815 updates.
Batch is 256 sequences per update (micro-batch 8 × gradient accumulation 32) = 262,144
loss tokens per update. BF16, seed 1337, strict CUDA determinism (deterministic
algorithms, TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`).

1.00B tokens against 49.7M parameters is **~20.1 tokens per parameter** — deliberately at
the Chinchilla compute-optimal ratio, since the parameter count was fixed by the rules and
the token budget was the only free variable.

Measured cost: 15,767.193 s wall (4 h 22 min 47 s), 15,414.290 s in the optimizer, 5.4454
GiB peak allocated VRAM, ≈2.98 × 10^17 FLOPs by the `6 × params × tokens` approximation
(a dense-model estimate, not measured hardware FLOPs).

**4. The verification layer.** This is the part we would point a judge at first.

- `src/tinybench_lm/eligibility.py` scans the source registry and **fails closed** on
  prohibited data: synthetic/model-generated corpora, teacher logits or rankings, benchmark
  examples, unfiltered Common Crawl, hosted-model labelling or rewriting.
- Frozen protocol files (sources, tokenizer, dedup, decontamination, shards, training
  recipe) are pinned by SHA-256 over their bytes with CRLF normalized to LF, with the
  digest registered *in code*. Changing a protocol requires publishing a new version file
  with a recorded reason; in-place edits break loading.
- `results/step_zero_provenance.json` records the seed, the **digest of the initial random
  weights**, and the parameter count — so "trained from scratch" is an auditable claim, not
  an assertion.
- The training recipe was frozen *before* the run and forbids benchmark-driven selection
  (`benchmark_driven_training_selection: false`) and pilot-checkpoint initialization. The
  selected configuration is the pre-registered CONTROL; the challenger recipe (C1) failed
  its pre-declared adoption criteria and was not adopted.
- Release assets are size- and SHA-256-checked against `configs/release/submission_v1.json`
  by the downloader itself.
- The repo ships property-based and contract tests (pytest + Hypothesis) and a Dockerfile
  that pins the entire transitive dependency set from `constraints/verified-py311-windows.txt`.

### Results (all PROVISIONAL — the organizers' exact WikiText slice and official settings
are not yet confirmed)

Evaluated with lm-evaluation-harness 0.4.12, zero-shot, seed 1234, batch 16, CUDA/BF16,
1,000 bootstrap iterations, full example coverage:

| Benchmark | Examples | Accuracy | Normalized accuracy |
| --- | ---: | ---: | ---: |
| PIQA | 1,838 | 58.65% | 58.11% |
| ARC-Easy | 2,376 | 41.33% | 37.50% |
| HellaSwag | 10,042 | 27.30% | 27.52% |
| WinoGrande | 1,267 | 49.96% | not emitted |

WikiText-103 word perplexity: **235.1593** over 2,891 non-empty test paragraphs.

Read honestly: PIQA and ARC-Easy are meaningfully above chance, HellaSwag is roughly at
chance (25%), and WinoGrande is at chance (50%). That is the expected shape for a 50M
model trained on 1B tokens — commonsense-completion benchmarks are the last thing to come
online at this scale, and we report the near-chance numbers rather than dropping the
tasks. The model repeats, hallucinates facts, and has no instruction tuning; the
limitations are written into the model card and a dedicated generation-diagnostics
document instead of being left for a judge to discover.

### How a judge runs it

Docker, CPU, no GPU and no API key required. Five commands: build the image, download the
~199 MB model (hash-verified against the release manifest), print the parameter count
(**49,658,368**), and generate. A CPU smoke evaluation over one example per task is
documented separately as an installation check — explicitly labelled *not* a benchmark
score. The full CUDA evaluation path and a native Windows setup are in `docs/SUBMISSION.md`.

### What we would do differently

The largest lesson is in `sources_v4.yaml`: two data sources in v3 were registered on
their *license metadata* and never actually read. One (`storytracer/US-PD-Books`) contains
no text at all — it is a catalogue of archive.org URLs. The other required a compression
library outside the pinned environment. Both were caught by an availability check before
any tokenizer was built on them, but the rule now recorded in the protocol is blunt: *a
licence check is not an availability check.* Registering a source requires streaming one
document from the pinned revision and confirming a non-empty text field.

Second: a planned 3-billion-token run was scoped, budgeted, and then **abandoned for time**
rather than half-finished and reported as complete. The evidence matrix in
`configs/release/evidence_matrix_v1.yaml` keeps those gates in their real `NOT_RUN` state.
Publishing this baseline does not claim they passed. Given more compute, extending the
token horizon at this exact parameter count is the single highest-value change — the model
is compute-limited, not capacity-limited.

**Repository:** https://github.com/liisheng/solid-train
**Release (weights + evidence):** https://github.com/liisheng/solid-train/releases/tag/v1.0.0

---

## 04 | Built With

### Devpost tag list (paste into the "Built With" field)

```
python, pytorch, cuda, docker, numpy, sqlite, hugging-face-datasets,
hugging-face-tokenizers, safetensors, transformers, pyarrow,
lm-evaluation-harness, pytest, hypothesis, ruff, git, github-releases,
bfloat16, rope, rmsnorm, swiglu, grouped-query-attention, adamw, minhash,
byte-level-bpe, fineweb-edu, fineweb, openwebmath, project-gutenberg,
rtx-4070-super, openai-codex
```

### Full inventory

**Languages and runtime**

| Item | Version | Role |
| --- | --- | --- |
| Python | 3.11 (Docker image); 3.12.6 on the training machine | all code |
| PyTorch | 2.5.1 (+cu124 on the training machine) | model, training loop, SDPA attention, BF16 autocast |
| CUDA | 12.4 build | GPU training and full evaluation |
| Docker | `python:3.11-slim` base | reproducible judge-facing environment |

**Libraries**

| Library | Version | Role |
| --- | --- | --- |
| Hugging Face `datasets` | 3.2.0 | streaming the pinned public corpora |
| Hugging Face `tokenizers` | 0.20.3 | training and loading the 12,288-token byte-level BPE |
| `transformers` | 4.46.3 | transitive dependency of the evaluation harness |
| `safetensors` | 0.8.0 | release weight serialization and hash verification |
| NumPy | 1.26.4 | `uint16` token shard storage, deterministic bootstrap |
| `lm-eval` (lm-evaluation-harness) | 0.4.12 | HellaSwag, ARC-Easy, PIQA, WinoGrande, WikiText scoring |
| PyArrow | 25.0.1 | dataset I/O |
| PyYAML | 6.0.2 | frozen protocol files |
| `tqdm` | 4.67.1 | progress reporting |
| SQLite | Python stdlib `sqlite3` | restartable corpus-pipeline state, filter/dedup/decontamination ledgers |
| `xxhash` | 4.0.1 | fast content hashing in the pipeline |
| pytest | 8.3.5 | contract and unit tests |
| Hypothesis | 6.130.5 | property-based tests |
| Ruff | 0.11.13 | lint |

Full transitive set with exact pins: `constraints/verified-py311-windows.txt`, verified by
`python scripts/check_environment.py`.

**Datasets** (all pinned to immutable Hugging Face commit revisions in
`configs/data/sources_v4.yaml`)

| Dataset | Licence | Pinned revision | Role |
| --- | --- | --- | --- |
| HuggingFaceFW/fineweb-edu | ODC-By 1.0 | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | 70% of the stable mixture (educational English); also the reserved science and top-decile pools |
| HuggingFaceFW/fineweb | ODC-By 1.0 | `9bb295ddab0e05d785b879661af7260fed5140fc` | 20% (general-language diversity) |
| open-web-math/open-web-math | ODC-By 1.0 | `fde8ef8de2300f5e778f56261843dab89f230815` | 7% (mathematical/scientific prose); also the reserved math pool |
| sedthh/gutenberg_english | MIT | `28973b04f28fd7be4a6186a042bc26159d4366ca` | 3% (long-form narrative, co-reference) |
| wikimedia/wikipedia | CC BY-SA 3.0 / GFDL | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | reserved pool (not in the trained mixture) |
| izumi-lab/open-text-books | CC BY-SA 4.0 | `1245fefd628d37483366b8e707fdc5650fd3c48e` | reserved pool (not in the trained mixture) |

FineWeb-Edu, FineWeb, and OpenWebMath all derive from Common Crawl; the
[Common Crawl terms of use](https://commoncrawl.org/terms-of-use/) also apply. Wikipedia
text is share-alike and any redistribution of that portion carries CC BY-SA obligations.

**Benchmark datasets** (evaluation only, and explicitly excluded from training by the
decontamination pass): HellaSwag, ARC-Easy, PIQA, WinoGrande, WikiText-103.

**Explicitly NOT used** — enforced by `configs/data/sources_v4.yaml` and scanned by
`src/tinybench_lm/eligibility.py`: no synthetic or model-generated corpus (TinyStories and
similar), no teacher logits or rankings, no benchmark examples in training, no large code
corpora, no unfiltered Common Crawl, no pretrained weights, no distillation, and no hosted
model labelling, rewriting, or scoring any training document.

**Architecture techniques**: grouped-query attention (8 Q / 4 KV heads), SwiGLU
feed-forward, RoPE (theta 10,000), RMSNorm (ε 1e-5), tied input/output embeddings,
bias-free linear layers, PyTorch scaled-dot-product attention, AdamW, WSD
(warmup–stable–decay) learning-rate schedule, BF16 mixed precision with gradient
accumulation, MinHash LSH near-duplicate detection (128 rows, 32 bands × 4, 0.85 Jaccard),
13-word shingle benchmark decontamination.

**Hardware**: NVIDIA GeForce RTX 4070 SUPER (12 GB) for training and full evaluation;
peak allocated VRAM 5.4454 GiB. Any CPU for inference and the smoke evaluation.

**Infrastructure**: Git and GitHub (source), GitHub Releases (weights, tokenizer, config,
evaluation-evidence archive, `SHA256SUMS.txt`), Docker for the reviewer environment.

**AI tools**: OpenAI Codex was used for implementation, tests, debugging, research
summaries, and documentation. Project decisions and all submitted claims remain the
entrant's responsibility. No hosted model performs inference for the submitted model,
labels or rewrites training text, or is a required evaluation dependency.
