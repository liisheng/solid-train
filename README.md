# TinyBench-LM

TinyBench-LM is a from-scratch, decoder-only language model designed for the GIBC V2
50-million-parameter track. The primary baseline contains **49,295,872 trainable
parameters**, including its token embeddings and tied output head.

This repository is an early pilot. It currently provides:

- a RoPE + RMSNorm + SwiGLU causal Transformer using PyTorch SDPA;
- exact parameter-cap verification;
- bounded, streaming public-data acquisition;
- a custom byte-level BPE tokenizer;
- document-level train/validation splitting and packed `uint16` token shards;
- BF16 training, gradient accumulation, validation, logs, and resumable checkpoints;
- local text generation and a hardware throughput profiler.
- an `lm-evaluation-harness` adapter for the competition benchmarks.

Measured local results and the evidence-to-experiment plan are in
`docs/PILOT_REPORT.md` and `docs/RESEARCH_PLAN.md`.

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

The baseline in `configs/baseline_49m.json` uses 12 layers, a 512-wide residual
stream, 8 attention heads, a 1,536-wide SwiGLU feed-forward network, a 16,384-token
vocabulary, and a 1,024-token maximum context. Input and output embeddings share the
same parameter tensor. The smaller pilot config is intended only for fast pipeline
validation.

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

The decontamination report, final data mixture, and full-run configuration remain to
be produced before a competition training run.
