# Reproducing the corpus and artifacts

The 2 GiB tokenizer sample is **not** in the repository, and deliberately so: four of its
files exceed GitHub's hard 100 MB per-file limit, the set is 4.3 GB against a ~1.6 MB
repository, and republishing FineWeb- and Wikipedia-derived text is redistribution, which
carries CC BY-SA share-alike and Common Crawl terms-of-use obligations that merely reading
it does not.

What is committed is the **recipe**, which regenerates the sample byte-for-byte. Two
independent runs were verified to select identical document ids.

## What you get without running anything

| artifact | path | size |
|---|---|---|
| The final 12,288-ID tokenizer | `data/tokenizer_final/tokenizer.json` | 839 KB |
| Sample manifest (what was drawn) | `docs/evidence/tokenizer/sample_manifest.json` | 3 KB |
| Build record summary | `docs/evidence/tokenizer/build_record_summary.json` | 5 KB |
| G1 pipeline benchmark | `docs/evidence/pipeline/slice_1pct.*.json` | 17 KB |
| Benchmark quarantine-input manifest | `docs/evidence/decontamination/benchmark_inputs.json` | 5 KB |
| Hardware inventory | `docs/evidence/hardware/rtx_3070.json` | — |

The tokenizer is tracked `-text` in `.gitattributes`, so its bytes survive checkout on
Windows. If you clone and its digest does not match, that attribute was bypassed.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]" -c constraints\verified-py311-windows.txt
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe -m pytest
```

`check_environment.py` must report `RESULT: PASS`. If your torch build differs from the
pinned one, that is itself a G0 finding: "environments reproduce on both machines" is a gate
requirement, not a formality.

## Redrawing the 2 GiB sample

```powershell
.\.venv\Scripts\python.exe scripts\draw_tokenizer_sample.py 1.5
```

Streams ~3.2 GB from the four pinned revisions and selects 2,148,137,287 bytes in about
3.5 minutes on a 25 MB/s connection. Output lands in `data/tokenizer_sample/` (gitignored).

The draw is deterministic. It is fixed by the pinned source revisions, the frozen selection
salt `tokenizer_v1_stratified_sample`, and the pool factor. The second argument overrides the
represented size for a quick smoke, e.g. `... 1.5 8000000` for 8 MB.

Verify against the committed manifest:

```powershell
.\.venv\Scripts\python.exe -c "import json;a=json.load(open('data/tokenizer_sample/sample_manifest.json'));b=json.load(open('docs/evidence/tokenizer/sample_manifest.json'));print('match:', a['manifest_sha256']==b['manifest_sha256'])"
```

## Rebuilding the tokenizer

Only needed if you want to confirm the committed `tokenizer.json`; otherwise use it directly.

```powershell
# concatenate the four per-source files first
.\.venv\Scripts\python.exe scripts\build_tokenizer.py --mode final `
  --corpus data\tokenizer_sample\combined.jsonl `
  --output-dir data\tokenizer_final
```

Verify an existing artifact without rebuilding:

```powershell
.\.venv\Scripts\python.exe scripts\build_tokenizer.py --mode verify --output-dir data\tokenizer_final
```

All nine required checks must PASS.

## Streaming shard production

For a real accepted corpus, shard production reads JSONL one row at a time and retains only
the current source shard plus on-disk duplicate-ID/cluster/shard-record indexes. The input
must be sorted by `(boundary, source_id, document_id)` in the frozen boundary order. Sorting and global
near-deduplication belong to the upstream filtering stage; this step does not silently
materialize or remix an arbitrary corpus. Upstream isolation evidence is mandatory:

```powershell
.\.venv\Scripts\python.exe scripts\build_shards.py --documents data\accepted.jsonl `
  --tokenizer-dir data\tokenizer_final --output-dir data\shards `
  --scale FINAL --token-counter-id final_tokenizer_v1 `
  --streaming --isolation-verified
```

The builder rejects malformed rows, duplicate IDs, unsorted input, invalid source/boundary
tags, undeclared validation slices, and explicit cluster crossings. It writes into a sibling
staging directory and publishes the complete shard tree atomically. A non-empty output
directory is refused on retry; choose a new output directory for a fresh, auditable restart.
The frozen 268,435,456-token shard ceiling is enforced as a soft boundary (an individual
document is never split, so one unusually long document may exceed it). Manifest records are
indexed on disk and exposed lazily; requesting a complete manifest necessarily materializes
its contract-mandated document-boundary arrays. The CLI verifies one manifest at a time and
defers full mixture/profile reconciliation until a streaming aggregate verifier is supplied.
The resulting manifests and shards preserve the existing source-tagged uint16/EOS contract.
This stage does not claim global near-dedup/isolation evidence.

## Rebuilding the benchmark quarantine inputs

The production decontamination protocol is `configs/data/decontam_v2.yaml`. It pins all
required and secondary public benchmark datasets plus the lm-evaluation-harness commit used
for G1. Recreate the local, gitignored benchmark-item body with:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_decontamination_inputs.py
```

The command writes `data/decontamination/benchmark_items.jsonl` and refreshes the compact
committed evidence manifest. A matching run contains 1,362,239 usable rows and has SHA-256
`976f81cfef5ef3d6ff9d3d608c482b348e1e9606acec1dfd261d2f235536fa4d`.
Blank WikiText rows are excluded and counted in the manifest; no other benchmark fields or
metadata are added implicitly. PIQA, LogiQA, and MathQA require their reviewed dataset loader
code, whose expected SHA-256 values are frozen in the protocol.

## Running a pipeline slice

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline_slice.py 0.01 slice_1pct runs\bench\slice_1pct.measurements.json
.\.venv\Scripts\python.exe scripts\pipeline_benchmark.py --measurements runs\bench\slice_1pct.measurements.json --artifact runs\bench\slice_1pct.artifact.json
```

The 1% slice takes about 5 minutes and forecasts the full 11B run. Measured on an RTX 3070 Ti
host: 27,419 s (~7.6 h) single-threaded, of which `within_source_near_dedup` is 55%. That
stage is embarrassingly parallel and is the obvious thing to speed up first.

## What is deliberately not shared

No model weights exist yet, because no training run has happened. Weight hosting stays
`BLOCKED` in `configs/release/evidence_matrix_v1.yaml` until organizers answer the
Section 2.2 question about uploading self-trained weights.

If the corpus itself ever needs to move between machines, prefer a Hugging Face dataset repo
or a GitHub Release attachment over the git history, and record the licensing decision first.
