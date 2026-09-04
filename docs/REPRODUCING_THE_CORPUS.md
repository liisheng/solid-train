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
