# Reduced G3 section 2 exposure evidence

Recorded 2026-09-08 on `codex/g3-baseline-recipe`. This section materializes the finite
two-component training exposure only; it does not train, profile, launch, export, or wire
the phase 4 runner.

## Reproduction

From the repository root, using the existing local environment:

```powershell
.\.venv\Scripts\python.exe scripts/build_reduced_exposure.py
```

The command loads `data/shards/reduced_5pct_v1/stable_train.manifest.json` and the frozen
`data/schedules/reduced_5pct_v1/stable_train.json` as component 1. It deterministically builds
component 2 with seed 1337, sequence length 1024, local shuffle buffer 1024, and quotas
`fineweb_edu=307741`, `dclm=87889`, `openwebmath=30786`, `narrative=13112`.

## Result

Artifacts are in `runs/reduced_campaign/reduced_baseline_v1/`:

| item | value |
|---|---|
| plan content hash | `6b1b747e47ea907810bf68f9ff941f61319ba7dbc7a034e3e2a746775666b0e3` |
| component 1 content hash | `820bd7f695643b2df82c5ea8c9816dcd763b375382a9a28f94656ad43529b828` |
| component 2 content hash | `0ff9b56492d60317de42a251a5fa55b583e5a63aff8b59345bb7aaad7672889c` |
| component 1 file SHA-256 | `ee174527f0a00a23f6d13d88e723dea1988f7df44fd4bb6c3aaf5f1300f4052e` |
| component 2 file SHA-256 | `6f3907104dd286af8298e234c6845509e6f18f4ab32391d9ac31dfc26c245245` |
| plan file SHA-256 | `76f16bfc98620227e2070645e5e901d8a4a8811c3970509b92769ebd84f71f8f` |
| manifest content hash | `bd1b29919f46ef6391f7b465699dd260c58befbbef44810698a681b26a3b22e7` |
| recipe normalized-LF SHA-256 | `1d4901c4c8d83c4c4ffd17e953de2b23f5a762f2af8eb8bb57fbc6fd8e26c541` |
| contract document normalized-LF SHA-256 | `38e2fd234decbca9ecad544b1889bbbafa4996fbf47890ee3371f337242c74c0` |

Accounting reports 976,640 sequences, 1,000,079,360 consumed loss tokens, final absolute
cursor 976,640, aggregate source counts `683648 / 195328 / 68365 / 29299`, and 439,528
cross-component repeated references. Duplicates are rejected within each component.

## Phase 4 interface

Phase 4 should call `load_exposure_plan(plan_path, (component_1_path, component_2_path))`,
then run `verify_exposure(exposure, manifest)` before opening
`CompositeTokenStream(shard_root, manifest, exposure)`. This validates the persisted plan,
ordered component identities, pinned recipe, exact quotas, and schedule invariants before
the reader is used. Persist only
`stream.state_dict()` at accumulation boundaries. The state contains one absolute
`schedule_cursor` bound to the composite plan hash; the reader fails closed at 976,640 and
does not wrap. `validate_components=True` (the default) runs the existing per-schedule
manifest, bounds, locality, quota, protocol, and deterministic rebuild checks once at open.

The hot path retains lazy per-shard mmap reads and only splits a microbatch when it crosses
the 537,112-entry component boundary. No training or runner integration is claimed here.

## Checks and limits

`.venv` tests passed: `tests/test_exposure.py` (5), `tests/test_repeated_schedule.py` plus
`tests/test_mixture_schedule.py` (30), and Python compilation passed. The fallback wheel
build also passed with `python -m pip wheel . --no-deps --no-build-isolation --wheel-dir
runs/verification/g3-section2/wheels`. Ruff was unavailable. Docker was unavailable because
the Linux engine pipe was not running. The artifact command itself completed with
`status=PASS`; no training,
profiling, benchmark, or canonical G3/G4/G5/G6 claim follows from this evidence.

## Independent review

At 2026-09-08T00:35:00+08:00, Sol and Terra approved the same final implementation.
Sol independently ran the 35-test suite and checked stream recovery at boundary minus one,
boundary, and boundary plus one; final exhaustion; identity, quota, count, hash, order and
swapped-component rejection; and within-component duplicates. Terra independently loaded
the real plan, ran full component validators, read across cursor 537111, resumed into
component 2, and checked exhausted-cursor rejection. Root independently checked artifact
file hashes, unchanged base file, frozen recipe sidecar, and accounting arithmetic.

Reviewed implementation file SHA-256 values:

| file | SHA-256 |
|---|---|
| `src/tinybench_lm/exposure.py` | `dcfbe3ad061ad5a9c308eab0571bab9f81e6e74be37d87bdb5cfe5757089b418` |
| `scripts/build_reduced_exposure.py` | `7ca99892adadb418b8845b6900a497f801e95111c7bda074262f5db2076a30e7` |
| `tests/test_exposure.py` | `12b51253809f97c2822262f4c6f9281d9785a9a8c6510df01442d6d30a3fc842` |

Static efficiency review confirms cached identities, a shared lazy mmap cache, no full
integrity scans during batch/seek operations, and reuse of the underlying batch hash except
when a batch crosses components. This is not measured throughput or peak-memory evidence.
Full component validation remains startup work. Consumed exposure is about 1.818 effective
passes over the 550,094,903 selected distinct stable tokens; repeated tokens are not new data.
Generated artifacts are ignored local files and must be transferred or reproduced separately.
