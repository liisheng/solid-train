# Reduced G3 section 5 evaluation evidence

Updated 2026-09-08T22:12:00+08:00. This is a reproducible provisional evaluation
rehearsal. Reduced G3 and canonical G3 remain NOT_RUN, and none of the smoke values is a
model-quality or official competition result.

`src/tinybench_lm/evaluation_binding.py` resolves all five required tasks against frozen
`configs/data/decontam_v3.yaml`, injects every dataset revision at the loader boundary,
and rejects task, preprocessing, or installed harness content drift before scoring. This
corrects the installed 0.4.12 alias for PIQA (`baber/piqa`) to the quarantined
`ybisk/piqa` revision `2e8ac2dffd59bac8c3c6714948f4c551a0848bb0`. The binding
hashes 15,090 installed non-bytecode harness files (12,248,112 bytes), the selected task
YAML and helper files, the project WikiText implementation, and decontamination v3. The
installed wheel exposes no git/direct-url commit provenance, so
`harness_commit_status=BLOCKED_UNOBSERVABLE_FROM_WHEEL` remains an explicit blocker for
an official result.

PIQA is loaded with `trust_remote_code=true`. The effective cached `piqa.py` SHA-256 was
independently checked as `99107ad59e6f5ffc26923f98bb4cb6e2029b0e368005e9639ea7eadc86d2d46b`,
matching decontamination v3. The immutable dataset revision remains the loader-bound source
identity on a clean machine.

The installed-content check measured 4.23 seconds cold on this Windows filesystem and is
cached after the first call in a process. It runs only while preparing a run or evaluation,
never in the training or evaluation per-example path. The runner records binding digest
`cb75ff8713a14244279dc62683840b559ec212e9aa03133da8316ca830351ad7`
in its semantic identity.

The final bounded rehearsal used the verified G2 recovery engineering export. It scored
exactly one public example from each required task and took 44.245 seconds on CPU. The
persisted verifier re-opened the bundle, verified its manifest and provisional protocol,
cross-checked raw and metadata sample counts, required all five tasks in order, matched
the current binding, and matched the reviewed checkpoint and tokenizer hashes. The release
export independently passed all 20 reload/provenance checks.

The immutable v2 metadata still lists its historical `model_revision` and
`tokenizer_revision` placeholders as unpinned. For this engineering rehearsal, the local
artifacts are nevertheless resolved by the exact runtime SHA-256 values below. Future
release revision names remain pending and this smoke does not promote them.

- Bundle: `runs/verification/g3-section5-smoke-final-v2/bundle`
- Bundle manifest SHA-256: `a1bb2a99309170e42702257777db6642fdf4c02061529208f38d9f1c5b32f23d`
- Raw results SHA-256: `5f67602325c2877179308df982d67c93e3fbb3fae9e5a9765c34e2fa9142b2e7`
- Run metadata SHA-256: `71a0a972be5e27ef4318a37e39fbbaf1daf2f349e55ae4561636e1ffc8e0500f`
- Engineering export SHA-256: `5b3f165cc7432631557eb02c17d44e7c3f311728a0b57c17ed825e9f62f63802`
- Tokenizer SHA-256: `2103d520df7a23490054cff474f5c0e0f241bb56b51051284ab92889a683c635`
- Evaluation v2 SHA-256: `60eb9750bd70e74e2d5db868af068bf5cf2c578704cc1da2be12b9f2b536976b`
- Decontamination v3 SHA-256: `66abe996a08ede1e09f4562c5bbbc42a94adf506be2d035751170b41c2c8c041`
- Installed harness content SHA-256: `0a9482a5184dcc3d06938523615c0c760f433730497afcdcc1e42f42f02f13a7`

Final source custody:

| File | SHA-256 |
|---|---|
| `evaluate.py` | `8205e76768f7a6ef3be3519d4ad281ac486360dda7c3e9f015393ad66b8a2bb7` |
| `src/tinybench_lm/evaluation_binding.py` | `46c56a85880a49896a08b334bfaf4a287aa55d914eebad4aa17c00709b804df2` |
| `src/tinybench_lm/evaluation_tasks.py` | `45b336fe6c25b0903697752ca3371260bf8180faee3366d876f62c2c15f025a1` |
| `scripts/run_reduced_baseline.py` | `4c9143225fed1d4568b771ce886cd8b162245634de37e28477a500a7620e470a` |
| `tests/test_evaluation_binding.py` | `ec2a69bb7ff2fbd49640e24869836e2fbf80d0026d90695d626488fb886e7980` |
| `tests/test_reduced_baseline_runner.py` | `9d9e3fa11c8d7afbf23638ff8c6c3180870ab9b391ffcac00db91eef2a320279` |

The bounded command recorded in `bundle/command.txt` was:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/reduced_campaign/reduced_5pct_v1/g2_recovery_luna_fix3/engineering_export.pt --tokenizer runs/reduced_campaign/g2_handoff_v2/artifacts/data/tokenizer_final/tokenizer.json --device cpu --precision float32 --limit 1 --smoke --output runs/verification/g3-section5-smoke-final-v2/results.json --bundle runs/verification/g3-section5-smoke-final-v2/bundle
```

Re-open and verify the persisted evidence without scoring again:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from tinybench_lm.evaluation_binding import verify_smoke_bundle; from tinybench_lm.evaluation_protocol import load_evaluation_protocol; p=load_evaluation_protocol(Path('configs/evaluation/evaluation_provisional_v2.yaml')); print(verify_smoke_bundle(Path('runs/verification/g3-section5-smoke-final-v2/bundle'), p, expected_checkpoint_sha256='5b3f165cc7432631557eb02c17d44e7c3f311728a0b57c17ed825e9f62f63802', expected_tokenizer_sha256='2103d520df7a23490054cff474f5c0e0f241bb56b51051284ab92889a683c635'))"
```

The later G6 full provisional command removes the smoke limit explicitly:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint <eligible-baseline-export> --tokenizer runs/reduced_campaign/g2_handoff_v2/artifacts/data/tokenizer_final/tokenizer.json --device cuda --precision bfloat16 --full --output runs/evaluation/baseline-provisional/results.json --bundle runs/evaluation/baseline-provisional/bundle
```

G6 must record its own runtime/device/precision and verify its bundle. This CPU float32 smoke
does not establish full-suite runtime or numerical equivalence with the future CUDA BF16 run.

The combined focused binding, runner, protocol, and task suite passed 57 tests in 49.27
seconds:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_binding.py tests/test_evaluation_protocol.py tests/test_evaluation_tasks.py tests/test_reduced_baseline_runner.py -q
```

Compilation passed. The reviewed wheel is
`runs/verification/g3-section5-final-reviewed/wheels/tinybench_lm-0.1.0-py3-none-any.whl`
(256,182 bytes; SHA-256
`1d5c6c55282fa057562dbd071f4213d040ffedfb1db485ccfdbe04731f000211`).
Sol and Terra independently approved the same final source hashes, and Terra independently
verified the final-v2 bundle. Docker was unavailable and Ruff was not installed.
No training, full evaluation, private final-holdout scoring, install, commit, or push was
performed.
