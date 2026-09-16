# Judge setup and release contents

The submission model is the completed 1B-token baseline. Release `v1.0.0` binds
the unchanged verified export to its tokenizer and configuration. The abandoned
3B candidate is not needed for inference, testing, or evaluation.

## Download manually

Open <https://github.com/liisheng/solid-train/releases/tag/v1.0.0> and download:

- `baseline_export.pt`: trained model export, about 199 MB.
- `tokenizer.json`: matching 12,288-token tokenizer.
- `final_49m.json`: architecture configuration.
- `MODEL_CARD.md`: model description, intended use, and limitations.
- `SHA256SUMS.txt`: hashes for the model assets, model card, and evidence archive.
- `evaluation-evidence.zip`: original evaluation bundle, environment/verification
  receipts, training metrics, and initial-weight provenance.

The README downloader checks model assets automatically. On Windows, manual
downloads can be checked with `Get-FileHash <file> -Algorithm SHA256`; on Linux use
`sha256sum -c SHA256SUMS.txt` in a folder containing all five payload files.
The model SHA-256 is
`89438ee3165a64bd3c15b3ba1f4aa6658841a423a554164061c2d4c0ac95c288`.

No access token or approval should be needed. The source archives that GitHub
generates automatically contain code only; download the model assets separately.

## Native Windows CUDA setup

The original full evaluation used Python 3.12.6, PyTorch 2.5.1+cu124, Windows,
and an RTX 4070 SUPER. Install Python 3.12 and a compatible NVIDIA driver first.
Run these commands in PowerShell from the release checkout:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r constraints/verified-py311-windows.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe release_tools/download_release.py
.\.venv\Scripts\python.exe scripts/count_params.py
.\.venv\Scripts\python.exe generate.py --checkpoint runs/release/baseline_export.pt --tokenizer runs/release/tokenizer.json --prompt "The purpose of science is" --max-new-tokens 40 --temperature 0 --top-k 1
```

The constraints filename is historical; it does not mean that Python 3.11 was the
interpreter used for the published full evaluation. Editable installation and
explicit `PYTHONPATH` keep repository-bound protocol paths in this checkout.

## Full provisional evaluation

After the native setup, run:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/release/baseline_export.pt --tokenizer runs/release/tokenizer.json --protocol configs/evaluation/evaluation_provisional_v2.yaml --device cuda --precision bfloat16 --full --batch-size 16 --bootstrap-iters 1000 --cache-dir .cache --output runs/judge-full/results.json --bundle runs/judge-full/bundle
```

All five required tasks are selected by default, with zero-shot settings and seed
1234 from the protocol. There is no example limit. Outputs include raw JSON,
coverage checks, hashes, runtime metadata and stderr. The first run requires
internet access and downloads public benchmark datasets. Cached evaluation took
192.305 seconds on our GPU; setup/download time and other machines can differ.

For full CPU evaluation, use the README Docker evaluation command with `--full`,
remove `--smoke --limit 1`, and set `--bootstrap-iters 1000`. Keep CPU/float32 and
batch size 1 to limit memory. This differs from the recorded CUDA/BF16 execution;
minor numerical differences are possible. Use a separate output directory.

The evaluator verifies the installed harness content and task definitions. If an
identity check fails, keep the error and use the pinned dependencies; do not
disable the guard. The original harness package is version 0.4.12 with a recorded
content digest; its upstream Git commit provenance is still unresolved. This
release does not convert that fact into an official provenance claim.

## Evidence and limitations

`results/benchmark-summary.json` retains the harness summary fields but omits
per-example samples and task configuration bodies; it is explicitly an extract.
The complete original output is in the release evidence ZIP. Historical command
paths inside that archive describe the original machine and are not commands to
copy. Use the commands above to reproduce from your own checkout.

Training metrics in the archive can be used to plot loss, learning rate, throughput,
and development loss. The training summary covers the completed baseline only.
Whole-project compute, including earlier experiments and failed preparation,
has not been fully reconciled. Official organizer settings, especially the exact
held-out WikiText slice, remain unconfirmed; all scores remain provisional.

## Competition entry

The GitHub package supplies source, model, instructions and evaluation evidence.
Devpost still needs the project description, a 2–5 minute English demo video,
at least three screenshots, complete Built With credits, and team information.
The [official rules](https://gibc-v2.devpost.com/rules), checked 2026-09-16,
give the deadline as 2026-10-01 at 23:45 UTC+8. The rules require disclosure of AI
coding assistance; the README credits contain that disclosure without authorship
bylines. Public source/model access does not itself complete a Devpost submission.
