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

Devpost renders these seven headings on the project page. Paste each section under its
matching heading.

### Inspiration

Getting a transformer to train is not hard anymore. Reference implementations are
everywhere, and you can have one emitting tokens in an afternoon. So a 50-million-parameter
cap is not really a coding challenge — it is a *budgeting* challenge. Every decision has to
be defended out of a fixed account: depth or width, vocabulary size (every entry costs
`d_model` parameters, twice if you don't tie the output head), how many tokens to spend,
and what those tokens should be.

The second thing that pushed us was quieter. Small-model results are extremely easy to
fake — and usually by accident. Train on web text you didn't filter and you have almost
certainly swallowed HellaSwag and ARC items. Warm-start from a pilot checkpoint and your
"from scratch" claim is gone. Run twenty configs and submit the best benchmark score and
you have tuned on the test set. None of those leave a visible trace. Nobody can catch them
from the outside, which means the only person who can keep the result honest is the person
who built it.

So we decided the provenance would be a deliverable, equal in weight to the weights.

### What it does

TinyBench-LM is an English causal language model with **49,658,368 trainable parameters** —
341,632 under the Track 01 cap, counting embeddings and the tied output head. It continues
English text. It is a base model, not an instruction-tuned chatbot: give it a prefix and it
writes the next tokens.

It was trained from random initialization on **1,000,079,360 tokens** on a single consumer
GPU (RTX 4070 SUPER, 12 GB) in **4 hours 22 minutes**. No pretrained weights, no
distillation, no teacher model, no synthetic corpus, and no hosted API anywhere in the
training, evaluation, or inference path.

A judge can run it in five commands on a laptop with no GPU: build the Docker image,
download the ~199 MB model (every asset size- and SHA-256-checked against a release
manifest), print the parameter count, and generate. The whole flow is CPU-only and needs no
API key and no corpus download.

### How we built it

**The model.** 14 layers, `d_model` 512, 8 query heads with 4 key/value heads
(grouped-query attention), SwiGLU feed-forward of width 1,504, RoPE (theta 10,000),
RMSNorm, no biases, tied input/output embeddings, 1,024-token context, 12,288-token
byte-level BPE vocabulary, attention through PyTorch SDPA.

The budget drove every one of those. Tying the embeddings frees ~6.3M parameters
(12,288 × 512) to spend on depth instead. The small vocabulary is a deliberate trade —
fewer embedding parameters and `uint16` token storage, paid for with longer sequences per
unit of text. GQA cuts KV and projection cost without the quality drop of full multi-query.
The result is deep and narrow, 14 layers at width 512, which is where a fixed 50M is best
spent.

**The corpus.** `scripts/prepare_corpus.py` is a six-stage restartable pipeline:
acquire/filter → global near-duplicate removal → benchmark-index construction → benchmark
decontamination → boundary assignment → atomic publication.

Sources stream from **pinned Hugging Face commit revisions**, never branch names:
FineWeb-Edu 70%, FineWeb 20%, OpenWebMath 7%, Project Gutenberg English 3%. Deduplication
is exact + mirror + 128-row MinHash at a 0.85 Jaccard threshold, indexed as 32 bands of 4
rows — a passing pair can differ in at most 19 rows, so at least one band must collide,
which makes the index provably complete at that threshold. Decontamination scans every kept
document against an index built from the benchmark suites under three rules: a complete
benchmark field of ≥13 normalized words appearing as a substring, a ≥50-word contiguous
overlap, and shingle coverage (≥3 distinct 13-word shingles covering ≥10% of the document).
All pipeline state is disk-backed SQLite and resumable, and a resumed run must match the
acquisition, source, filter, dedup, and tokenizer identity hashes or it fails closed.

**The training run.** AdamW (β 0.9/0.95, ε 1e-8, weight decay 0.1, global grad-norm clip
1.0) under a WSD (warmup–stable–decay) learning-rate schedule with peak 6e-4: 38 warmup
updates, 3,395 stable, 382 decayed linearly to exactly zero — 3,815 updates total. Batch is
256 sequences per update (micro-batch 8 × gradient accumulation 32) = 262,144 loss tokens
per update. BF16, seed 1337, strict CUDA determinism with TF32 disabled.

1.00B tokens against 49.7M parameters is **~20.1 tokens per parameter**, deliberately at
the Chinchilla compute-optimal ratio — the parameter count was fixed by the rules, so the
token budget was the only free variable worth optimizing.

Measured: 15,767.193 s wall, 15,414.290 s in the optimizer, 5.4454 GiB peak allocated VRAM,
≈2.98 × 10^17 FLOPs by the `6 × params × tokens` approximation (a dense-model estimate, not
measured hardware FLOPs).

**The verification layer.** This is the part we would point a judge at first.
`src/tinybench_lm/eligibility.py` scans the source registry and **fails closed** on
prohibited data — synthetic corpora, teacher logits or rankings, benchmark examples,
unfiltered Common Crawl, hosted-model labelling. Every frozen protocol file (sources,
tokenizer, dedup, decontamination, shards, recipe) is pinned by SHA-256 over its bytes with
CRLF normalized to LF, and the digest is registered *in code*, so an in-place edit breaks
loading. `results/step_zero_provenance.json` records the seed, the parameter count, and the
**digest of the initial random weights** — which turns "trained from scratch" from an
assertion into something you can check. The recipe was frozen before the run and forbids
benchmark-driven selection (`benchmark_driven_training_selection: false`) and
pilot-checkpoint initialization.

**Results** (all PROVISIONAL — the organizers' exact WikiText slice and official settings
are unconfirmed). lm-evaluation-harness 0.4.12, zero-shot, seed 1234, batch 16, CUDA/BF16,
1,000 bootstrap iterations, full example coverage:

| Benchmark | Examples | Accuracy | Normalized accuracy |
| --- | ---: | ---: | ---: |
| PIQA | 1,838 | 58.65% | 58.11% |
| ARC-Easy | 2,376 | 41.33% | 37.50% |
| HellaSwag | 10,042 | 27.30% | 27.52% |
| WinoGrande | 1,267 | 49.96% | not emitted |

WikiText-103 word perplexity: **235.1593** over 2,891 non-empty test paragraphs.

Read honestly: PIQA and ARC-Easy are meaningfully above chance, HellaSwag is roughly at
chance (25%), WinoGrande is at chance (50%). That is the expected shape for 50M parameters
on 1B tokens — commonsense completion is the last capability to come online at this scale.
We report the near-chance numbers rather than dropping the tasks.

### Challenges we ran into

**Our contamination filter quarantined the entire corpus.** The v2 decontamination scan
threw out **every one of its first 7,767 documents**. The cause was rule 1: standalone
benchmark answers like "2", "4", and "8" are complete benchmark fields, and they appear in
approximately all prose ever written. v3 added a 13-word minimum to that rule, matching the
existing shingle resolution. v2 is still in the repo, frozen, as historical evidence rather
than quietly deleted — and we wrote down what the fix costs us: short copied questions and
paraphrases now fall outside the detector's guarantees, and we say so instead of claiming a
clean corpus.

**Two data sources we had registered turned out to be unreadable.** `storytracer/US-PD-Books`
contains no text at all — its schema is a catalogue of `ocaid`, `title`, `author`,
`full_text_url`, and the actual books live at archive.org behind those URLs. And
`mlfoundations/dclm-baseline-1.0` cannot be read in our pinned environment at all
("Compression type zstd not supported"), because `zstandard` isn't in the verified
constraints set and adding a dependency to satisfy one source would have invalidated the
environment we had measured. Both had been registered on their *licence metadata*. An
availability check caught them before any tokenizer was built on top of them.

**12 GB of VRAM.** The batch size we wanted didn't fit. Micro-batch 8 with gradient
accumulation 32 reproduces a 256-sequence update inside 5.4 GiB, which is what made a
1B-token run possible on hardware that also has to run a desktop.

**The model generates repetitive text, and we could not find a bug to blame.** We ran a
full diagnostic: the exported checkpoint's hash and every tensor match the training
checkpoint, the tokenizer matches the runner identity, all weights are finite, FP32 and
BF16 agree on the first argmax across all four test prompts, and next-token loss on real
training and dev prefixes is 2.95–4.67 versus 9.58–11.32 unshifted — so the model is
reading its context. Four prompts under both greedy and sampled decoding reproduce the
repetition. There is no inference defect we can find; the cause is model quality at this
scale. We published the diagnostic with its conclusion unresolved rather than shipping a
vaguer claim.

**We ran out of time for the run we wanted.** A 3-billion-token run was scoped, budgeted at
13.1 hours (16.4 with margin), and then abandoned. It is recorded as `NOT_RUN` rather than
partially executed and reported as done.

### Accomplishments that we're proud of

- **49,658,368 parameters — 341,632 under the cap**, with the counting script in the repo
  so anyone can confirm it in one command.
- A **1B-token run on one consumer GPU in 4h22m**, at 5.4 GiB peak VRAM. No cluster, no
  cloud credits, no rented A100.
- **"From scratch" is auditable, not asserted.** The seed and the digest of the initial
  random weights are published. That is a claim you can falsify, which is the only kind
  worth making.
- **A corpus pipeline that actually ran**, with a restart-safe SQLite ledger recording the
  filter, dedup, quarantine, and assignment decision for every distinct candidate document.
- **Full-coverage evaluation** — all 10,042 HellaSwag examples, all 2,376 ARC-Easy, no
  sampling, no limit flags — with the raw harness outputs published.
- **A five-command reproduction** on CPU, with hash-verified downloads, that we rehearsed
  from a clean machine.
- **Reporting our near-chance scores and our unresolved generation diagnostic**, because a
  submission that only documents its wins isn't documenting anything.

### What we learned

**A licence check is not an availability check.** This is now written into the source
protocol as a rule: registering a source requires streaming one document from the pinned
revision and confirming a non-empty text field. Metadata will happily describe a dataset
that contains nothing you can train on.

**Precision failures in a filter look exactly like success.** A decontamination pass that
quarantines everything is *technically* preventing contamination. We only caught it because
the pipeline reports yield counters per stage, and 7,767-out-of-7,767 is a number that
stops you. Instrument the thing that is supposed to be protecting you.

**Freeze first, or you will never know.** Pinning the recipe before the run, with
benchmark-driven selection explicitly forbidden, is uncomfortable — you cannot fix your
choices once the loss curve disappoints. That discomfort is the entire point. Our
pre-registered challenger recipe (C1) failed its adoption criteria and was not adopted,
which is only a meaningful statement because the criteria existed beforehand.

**At 50M parameters, we are compute-limited, not capacity-limited.** 20.1 tokens per
parameter is compute-optimal for a fixed budget, but "optimal" at this scale still lands
below the threshold where commonsense benchmarks come alive. The bottleneck is tokens seen,
not parameters available.

**Byte-exactness is fragile in ordinary ways.** Git's end-of-line conversion would have
rewritten our tokenizer on checkout, silently breaking every digest for anyone cloning on
Windows. One `.gitattributes` line prevents it. Small infrastructure details quietly
invalidate large evidence chains.

### What's next for TinyBench-LM

**Extend the token horizon at this exact parameter count.** The next run is already fully
specified in `docs/next_run/`: 11,445 updates, 3,000,238,080 tokens, 2,929,920 sequences,
WSD phases of 114/10,186/1,145 — identical 49.6M architecture, identical tokenizer,
identical CONTROL recipe. Only the data and the horizon change. Since the model is
compute-limited, this is the single highest-value change available.

**Expand the clean corpus.** That run needs ≥1.5B distinct tokens with no source used more
than twice, which means running the acquisition pipeline at a scale the current 5% slice
hasn't reached.

**Pre-registered success criteria, again.** The candidate is only adopted on a ≥2% global
dev-NLL improvement with no slice regressing more than 1%, decided on the development split
with exactly one post-selection benchmark run. Same discipline, larger budget.

**Reconcile whole-project compute.** We report the baseline training subtotal precisely and
say plainly that earlier experiments, profiling, failed data preparation, and verification
are excluded and not fully reconciled. A complete accounting is honest work still owed.

**Close the organizer-dependent gaps.** Every score we publish is labelled
`PROVISIONAL_NOT_OFFICIAL` until the exact WikiText slice and official harness settings are
confirmed. Those labels come off when the numbers are re-run under the confirmed protocol —
not before.

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
