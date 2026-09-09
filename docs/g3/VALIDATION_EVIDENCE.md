# Reduced G3 section 3 validation evidence

Recorded 2026-09-08 on `codex/g3-baseline-recipe`. This is implementation evidence for
the reduced G3 validation section; it is not a canonical G3 gate result. The focused tests
use synthetic models only; no real baseline or validation-final model scoring occurred.

## Implemented contract

`train.py --validation-mode full-dev` replays the materialized `validation_dev` schedule
from its beginning, requests `min(micro_batch_size, remaining)` for the final batch, and
requires the schedule to be finite and nonempty. The mode rejects non-scheduled data and
any split other than `validation_dev`, so a `validation_final` schedule cannot enter
training monitoring. The existing `sampled` mode remains available and is explicitly
labeled in its result metadata.

Each batch contributes `mean_loss * valid_target_count` to a device-side numerator, excluding
the model's ignore-index targets. The result
divides by the total scored target-token count and records mode, sequence count, batch
count, scored-token count, schedule ID, and schedule content hash in `metrics.jsonl`.
For the active development schedule, the expected full-dev coverage is 753 references,
94 full batches of 8 plus one final batch of 1, and 753 × 1024 scored target tokens.
The schedule identity is content hash `50ddce9f…16ca61fd` (file SHA-256
`1174495e…27d65f2`), with the full values bound by the existing schedule metadata.

Evaluation snapshots and restores the validation cursor, stream wrap policy, model training
flag, Python/NumPy/CPU Torch/CUDA RNG streams, including exception paths. Repeated replay
therefore does not perturb training state or resume behavior. The first monitoring event
remains post-completed-update 1 (`step == 0`); subsequent events use absolute completed
updates at the configured interval and horizon completion, so resume does not create a
second first-update event.

## Reproduction and checks

The baseline invocation supplied to section 4 must select full-dev explicitly and set the
declared recovery cadence, for example:

```text
python train.py ... --validation-mode full-dev --eval-interval 100 --save-interval 100
```

The `...` must bind the reduced baseline's materialized composite train schedule and the
`validation_dev` manifest/schedule. The final schedule was prepared and verified as metadata only; no final-holdout model scoring occurred.
Section 4 must also bind validation mode, validation schedule ID/hash, and the fixed
validation/recovery cadence into run identity and reject those changes on resume.

Focused verification used the existing Windows virtual environment because Docker Desktop's
Linux engine was unavailable and no packages were installed:

```text
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest -q tests/test_validation.py tests/test_training_recipe.py tests/test_mixture_schedule.py
66 passed in 4.56s (independent Sol rerun)
.\.venv\Scripts\python.exe -m compileall -q train.py src/tinybench_lm tests/test_validation.py
PASS
.\.venv\Scripts\python.exe -m pip wheel . --no-deps --no-build-isolation --wheel-dir runs/verification/g3-section3/wheels
PASS; wheel `tinybench_lm-0.1.0-py3-none-any.whl`, SHA-256
`64a951445e417256739ade87e3ab76347c31182f6f763875d05df1f13a35ea13`
```

The required container attempt was blocked before execution by the unavailable Docker
Desktop Linux engine (`dockerDesktopLinuxEngine` pipe). Ruff was unavailable, so no lint
result is claimed. Terra prepared and verified the deterministic validation-final schedule
without scoring it: `data/schedules/reduced_5pct_v1/validation_final.json`, 858 references /
878,592 tokens, sequence length 1,024, seed 1337. File SHA-256 is
`ae57a04210a1140485ab6179c98610ca9ebffa36adba384091690e5b1e716e44`; schedule content
hash is `dba4dad35c6b22b4f859c51cf5174e7afd7712d5483f4b0774665849a822795b`.
Reproduction used `scripts/build_schedule.py` with the reduced shard root, final manifest,
output path, `--sequence-length 1024`, and `--seed 1337`; `verify_schedule` passed with no
scoring. Sol independently approved the implementation after the 66-test check.

Final reviewed file hashes: `train.py` `a943cfaa7d0b74ae5ada4406bce089c7bfe5de3dc71781946435ec0ab4f2adc5`,
`tests/test_validation.py` `efa30a255ce622a73e5cbcbc8fab7e9861bb4c7af0f08d82968c4b7017cd6240`,
and the evidence file was refreshed with these final review results.
Sol and Terra independently approved the same implementation revision; `git diff --check`
passed. The canonical G3 gate remains `NOT_RUN`.

Exact metadata-only schedule reproduction:

```powershell
.\.venv\Scripts\python.exe scripts/build_schedule.py --shard-root data/shards/reduced_5pct_v1 --manifest data/shards/reduced_5pct_v1/validation_final.manifest.json --output data/schedules/reduced_5pct_v1/validation_final.json --sequence-length 1024 --seed 1337
```

Final manifest file SHA-256:
`79a704b8e84d26d55e97338bf07e6150832a78f057fd708945ee2561eb2a6015`;
content hash `2a8ebcd80a9f6da6bfa1cd71a7d615ba8d5f98a052dfa4bc6a7284be4a695be6`.
Root independently verified the schedule/manifest file hashes and sequence/token arithmetic.
Terra independently passed 38 validation/recipe tests and compilation on the same final code.
Review confirmed no per-batch scalar synchronization or full integrity scans; actual full-dev
runtime and integrated phase timing remain unmeasured. Generated schedules are ignored local
artifacts and require separate reproduction or transfer.
