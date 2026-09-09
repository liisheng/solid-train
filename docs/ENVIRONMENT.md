# Reproducible environment

The competition contract requires a public, reproducible setup. Open-ended dependency
ranges break that: a fresh install months later can resolve a different tokenizer,
`datasets`, or `lm-evaluation-harness` version and therefore a different evaluation
protocol than the one that produced our recorded numbers. Every runtime dependency is
therefore pinned exactly in `pyproject.toml`, cross-checked against
`constraints/verified-py311-windows.txt`, and verified by one command.

## Install

```powershell
.\.venv\Scripts\python.exe -m pip install -e . -c constraints\verified-py311-windows.txt
```

Optional test tooling (property and unit suites) is a separate extra, so a runtime-only
or release install never pulls it in:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]" -c constraints\verified-py311-windows.txt
```

## Dependency check command

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
```

The command compares the declared pins, the verified constraints file, and the versions
actually installed in the interpreter that runs it. It exits non-zero when a runtime
requirement is unpinned, missing, or divergent. Useful flags:

- `--json` emit the machine-readable report.
- `--output <path>` write the JSON report for evidence bundles.
- `--write-constraints <path>` regenerate the constraints file from what is installed now.
- `--no-facts` skip the informational GPU/backend section.

Regenerating the constraints file is the only sanctioned way to change it: the contents
must remain observed facts rather than aspirations.

## Verified facts

Everything in this section was produced by
`.\.venv\Scripts\python.exe scripts\check_environment.py` on the RTX 3070 Ti lane. Values
were not transcribed from documentation or assumed.

- Result: `PASS`, 100 checks, 0 failures.
- Python: 3.11.4 (CPython), `requires-python = ">=3.11"` satisfied.
- Platform: `Windows-10-10.0.26200-SP0`, `AMD64`.

| Runtime dependency | Pin | Installed |
|---|---|---|
| datasets | 3.2.0 | 3.2.0 |
| lm-eval | 0.4.12 | 0.4.12 |
| numpy | 1.26.4 | 1.26.4 |
| pyyaml | 6.0.2 | 6.0.2 |
| tokenizers | 0.20.3 | 0.20.3 |
| torch | 2.5.1 | 2.5.1+cu124 |
| tqdm | 4.67.1 | 4.67.1 |

| Test extra | Pin | Installed |
|---|---|---|
| hypothesis | 6.130.5 | 6.130.5 |
| pytest | 8.3.5 | 8.3.5 |

## GPU and backend options are optional

The check command reports backend facts as information only. It never selects a backend,
enables a fused kernel, or changes model semantics; nothing in the training or evaluation
path reads these facts.

Observed on this machine:

- `torch_version`: `2.5.1+cu124`
- `torch_cuda_build`: `12.4`
- `torch_cuda_available`: `True`
- `cuda_devices`: `['NVIDIA GeForce RTX 3070 Ti']`
- `backend_promotion`: `NOT_RUN` — sustained-throughput measurement and backend promotion
  are deferred to the hardware-measurement work; no throughput, memory, or
  backend-superiority claim is made here.

The pinned `torch==2.5.1` matches any local build of 2.5.1, so a CPU-only wheel and the
verified `2.5.1+cu124` CUDA 12.4 wheel both satisfy the contract. Installing the CUDA
wheel requires the vendor wheel index and is an explicit, documented operator choice:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

## Section 6 local environment check

On 2026-09-08, the existing Windows CUDA environment passed 98 checks with no
failures. The saved report is
`runs/verification/g3-section6-root/environment.json`: CPython 3.12.6,
Windows 11 build 26200, torch 2.5.1+cu124, CUDA 12.4 available, and an NVIDIA
GeForce RTX 4070 SUPER. This supersedes the previously unrecorded second-lane
environment status; it does not promote a backend or replace the required G4
sustained production profile. No packages were installed.

The section-6 container build could not connect to the Docker Desktop Linux
engine. Ruff was absent from the environment. These checks remain unavailable,
not passed; their recorded output is under `runs/verification/g3-section6-root/`.

## CPU container verification

The repository Docker image uses Python 3.11 and the official PyTorch 2.5.1 CPU
wheel. Both installation steps apply the existing public-version constraints;
the historical CUDA build label in that file is a comment, not a CPU requirement.
Runtime pins and the Windows CUDA environment are unchanged. Ruff 0.11.13 is
installed only in the verification image.

```powershell
docker build -t tinybench-lm:verify .
docker run --rm tinybench-lm:verify
docker run --rm tinybench-lm:verify python -m ruff check src scripts tests train.py generate.py evaluate.py
```

The image includes the entrypoints, constraints, and documentation required by
the full test suite. Local corpus, runs, checkpoints, generated build metadata,
and the untracked personal rules document are excluded. The existing checked-in
tokenizer fixture remains included. No host package installation is required.

On 2026-09-09 Docker Desktop initially failed on an inaccessible `dockerInference`
socket. After a graceful stop, the runtime directories were preserved as
`%LOCALAPPDATA%\Docker\run.g4-backup-20260909-1059` and
`%LOCALAPPDATA%\docker-secrets-engine.g4-backup-20260909-1059`; restarting restored
engine access. No factory reset or Docker data deletion was performed. This
matches the [upstream startup report](https://github.com/docker/desktop-feedback/issues/460).
Detailed G4 part-1 results are recorded in `docs/g4/VERIFICATION.md`.

## Unresolved

- Python versions other than the measured 3.11.4 and 3.12.6 environments remain
  supported by declaration (`>=3.11`), with no wider compatibility claim.
