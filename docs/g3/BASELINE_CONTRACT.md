# Reduced G3 baseline contract

This document is the human-readable companion to
[`configs/training/baseline_reduced_v1.yaml`](../../configs/training/baseline_reduced_v1.yaml).
It fixes the reduced 1B baseline recipe for the remaining G3 sections and records which
inputs are verified, which artifacts are still pending, and what later sections must return.
It does not claim canonical G3, G4, G5, or G6 completion.

## Fixed recipe

The accepted submission scope authorizes a 3,815-update baseline at 262,144 loss tokens per
optimizer update. That is exactly 1,000,079,360 loss tokens and 976,640 sequences at sequence
length 1,024. The source quotas are fixed as follows:

| source label | quota sequences |
| --- | ---: |
| `fineweb_edu` | 683,648 |
| `dclm` | 195,328 |
| `openwebmath` | 68,365 |
| `narrative` | 29,299 |

`dclm` is the retained internal label for the general FineWeb source. It does not identify a
DCLM dataset. The exposure implementation in section 2 must account for the exact quotas and
the intentional reuse required by the consumed horizon; the 550,094,903 distinct stable tokens
from reduced G2 are not the same quantity as the 1,000,079,360 consumed training tokens.

Section 2 must preserve the existing 537,112-sequence stable schedule as component 1 and add
component 2 with 439,528 sequences and source quotas `fineweb_edu=307,741`, `dclm=87,889`,
`openwebmath=30,786`, and `narrative=13,112`. The proposed artifact directory is
`runs/reduced_campaign/reduced_baseline_v1/`, containing `exposure_plan.json`,
`component_1.json`, `component_2.json`, and `exposure_accounting.json`. The final absolute
cursor must be 976,640. These are paths and requirements only; no future artifact hash is
invented here.

The model is the existing 49,658,368-parameter configuration in `configs/final_49m.json`,
with the final 12,288-ID tokenizer and a fresh random initialization at seed 1337. The batch
is microbatch 8 with accumulation 32 and sequence length 1,024. The optimizer is the frozen
AdamW contract from `configs/training/recipe_v1.yaml`: betas 0.9/0.95, epsilon 1e-8, weight
decay 0.1 with the existing decay groups, and global gradient clipping at 1.0.

The learning-rate schedule is WSD with peak LR `0.0006`, 38 warmup updates, 3,395 stable
updates, and 382 linear-decay updates to exactly zero. These phase lengths sum to 3,815 and
the decay begins at completed update 3,434. The horizon is the user-authorized part of the
scope. Seed, peak LR, and phase choices are fixed engineering choices in this recipe package;
`0.0006` has engineering evidence in reduced G2, but no controlled LR comparison has selected
it. No benchmark result is allowed to revise these choices in G3.
The 382-update decay is approximately 10% of the horizon, retained conservatively rather than
optimized by a comparison.

BF16 is the preferred measured policy, with no GradScaler. The existing reduced G2 run config
records BF16 support and stability, strict deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG`
`:4096:8`, TF32 disabled, cuDNN benchmarking disabled, and highest float32 matmul precision.
FP16 with a GradScaler remains the declared fallback only when BF16 is unsupported or measured
unstable.

## Validation and checkpoint custody

Development monitoring uses `validation_dev` and its 753-reference schedule. Section 3 must
implement full-dev replay: every reference is scored once, the final partial batch is included,
and the loss is aggregated by scored-token count. `validation_final` remains closed and is not
used for training monitoring or checkpoint selection.

The existing runner's validation convention is preserved precisely: the first event occurs
after completed update 1 (`step == 0` in the zero-based loop), then after completed updates that
are absolute multiples of 100, and again after completed update 3,815. There is no pre-training
validation event. Resume keeps absolute update numbering, so it must not create a second
first-update event. Recovery checkpoints are written at completed-update multiples of 100 and at
completion, only at accumulation boundaries; an improving development result may also write the
best-validation checkpoint under its existing selected-endpoint role.

The existing G2 recovery report also binds BF16 stability and the deterministic execution
policy to the active evidence, including strict deterministic algorithms, `:4096:8`, TF32
disabled, and cuDNN benchmarking disabled. The reduced profile is retained as efficiency
evidence, while the recovery report is the custody for this correctness/policy fact.

An eligible baseline source checkpoint is a future artifact, not the existing reduced G2
engineering export. Section 4 must prove fresh initialization provenance, exactly 3,815
completed updates, exactly 1,000,079,360 consumed loss tokens, final LR zero, the composite
schedule hash and cursor, and the frozen checkpoint manifest before export. An early-stopped
checkpoint remains ineligible even though the planned recipe has a decay phase.

## Input inventory

The machine-readable config records the full hashes. The important verified identities are:

- accepted scope: `configs/campaign/submission_scope_v1.yaml`, SHA-256
  `87630f9d…6891f24d`;
- model file: `configs/final_49m.json`, file SHA-256
  `69db94fe…49bb1782`, canonical model hash `d0c60218…47d96e3c`;
- tokenizer protocol: `configs/data/tokenizer_v3.yaml`, normalized-LF SHA-256
  `f8e0f397…46a38686`; tokenizer artifact SHA-256
  `2103d520…89a683c635`; token counter `final_tokenizer_v1`;
- reduced shard protocol: normalized-LF SHA-256
  `b93ba07f…5760015d`; source protocol identity
  `5c697f01…58052b5b`; stable, dev, final, and reserved manifest content hashes are recorded
  in the config; active decontamination v3 is
  `66abe996…c2c8c041`, with dedup v1 `81ede480…9fb80f7` and filters v1
  `a22f631f…f2453a`;
- existing stable schedule content hash `820bd7f6…43529b828` (file SHA-256
  `ee174527…0f4052e`) and development schedule content hash
  `50ddce9f…16ca61fd` (file SHA-256 `1174495e…27d65f2`);
- frozen recipe normalized-LF SHA-256 `fdd5f9ab…1250c0f8` and checkpoint protocol
  normalized-LF SHA-256 `7d2c33ee…33639bc1`.

The active measured environment in the G2 source evidence is machine `DESKTOP-0S78ISR` with
an NVIDIA GeForce RTX 4070 SUPER, CPython 3.12.6, and torch 2.5.1+cu124 (CUDA 12.4). The
environment report SHA-256 is `d15f8780…9c2b9`. This is historical G2 evidence and does not
claim a new runtime check in section 1.

## Timing and cost accounting

The retained reduced profile measured 1,901.9477939001517 optimizer seconds, 1,886.1647485001595
post-warmup optimizer seconds, and 1,932.232609700004 total wall seconds. Its weighted
optimizer-only throughput was 66,989.592558 tokens/s, with p10 67,267.738991, median
67,686.290218, p90 67,810.699752, and peak allocation 5.4454140663 GiB. The profile sampled
eight validation batches and is explicitly `ENGINEERING_PROFILE_NOT_SUBMISSION_MODEL`.

Optimizer-only throughput divides consumed loss tokens by optimizer seconds. Each training,
export, and evaluation process reports its own launch-to-exit wall cost. A later campaign
manifest aggregates those separately measured process wall times; it must not imply that a
training process captured export/evaluation time. Later run manifests must report these phase timings separately:
`preparation_seconds`, `training_optimizer_seconds`, `validation_seconds`,
`checkpoint_seconds`, `export_seconds`, `evaluation_seconds`, and `total_wall_seconds`.
The historical optimizer timing must not be turned into a total-cost forecast or a G4 promotion
claim. The composite exposure and full-dev validation require fresh timing.

## Interfaces and readiness

Sections 2–6 own the following outputs:

| owner | required output | current state |
| --- | --- | --- |
| 2 | versioned two-component exposure plan, exact quotas/counts, ordered hashes, one absolute cursor, within-component duplicate checks | `PENDING` |
| 3 | full-dev replay over all 753 references, partial-batch and token-weighted accounting, repeat/reset evidence | `PENDING` |
| 4 | dry-run/runner, semantic run identity, resume checks, eligibility checks and export path; actual completed baseline checkpoint remains G5/G6 custody | `PENDING` |
| 5 | effective evaluation loader binding, provisional labels, bounded smoke evidence and identity manifest | `PENDING` |
| 6 | integrated report, independent review, phase timing report and G4 handoff; baseline actual checkpoint/export remain later G5/G6 custody | `PENDING` |

Generated exposure, checkpoint, export, evaluation, run-manifest, and phase-timing hashes are
left `PENDING` until their owning section produces them. Section 6 decides final reduced-G3
readiness; the canonical G3 gate remains `NOT_RUN`.

## Section 1 verification record

Recorded 2026-09-08T00:16:02+08:00 on `codex/g3-baseline-recipe` with the local Python 3.12.6
environment. The following bounded check was run from the repository root; it uses the
existing recipe and WSD validators and verifies the contract sidecar:

```powershell
@'
from pathlib import Path
import hashlib, yaml
from tinybench_lm.training_recipe import (
    load_training_recipe, WSDSchedule, tokens_for_updates,
    updates_for_tokens, warmup_updates_for_horizon,
)
p = Path("configs/training/baseline_reduced_v1.yaml")
c = yaml.safe_load(p.read_bytes())
h, lr, b = c["horizon"], c["learning_rate"], c["batch"]
assert h["total_updates"] * h["loss_tokens_per_update"] == 1000079360
assert h["consumed_loss_tokens"] // h["sequence_length"] == 976640
assert sum(lr[k] for k in ("warmup_updates", "stable_updates", "decay_updates")) == 3815
assert b["micro_batch_size"] * b["gradient_accumulation"] * b["sequence_length"] == 262144
r = load_training_recipe()
assert r["_digest"] == c["optimizer"]["recipe_sha256_normalized_lf"]
assert tokens_for_updates(3815, protocol=r) == 1000079360
assert updates_for_tokens(1000079360, protocol=r) == 3815
assert warmup_updates_for_horizon(3815, protocol=r) == 38
s = WSDSchedule(3815, 38, 382, 6e-4)
assert s.stable_updates == 3395 and s.phase(3433) == "linear_decay" and s.learning_rate(3814) == 0.0
assert hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == Path(str(p)+".sha256").read_text().strip()
print("config parse, arithmetic, recipe, LR, sidecar: PASS")
'@ | .\.venv\Scripts\python.exe -
```

Result: `config parse, arithmetic, recipe, LR, sidecar: PASS`; contract digest
`1d4901c4c8d83c4c4ffd17e953de2b23f5a762f2af8eb8bb57fbc6fd8e26c541`.
The inventory hash pass additionally checked the normalized-LF identities for tokenizer v3,
recipe v1, checkpoint v1, shard v1, sources v4, decontam v3, and the four reduced shard
manifest files (10 files total). Docker preflight was attempted with
`docker info --format '{{.ServerVersion}}'` and failed because the Docker Desktop Linux
engine pipe was unavailable; no container check was claimed.
