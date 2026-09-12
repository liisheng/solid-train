from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import replace
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tinybench_lm import LOSS_IGNORE_INDEX, ModelConfig, TinyBenchLM
from tinybench_lm.checkpointing import (
    ROLE_SELECTED_ENDPOINT,
    BestValidationState,
    CheckpointCounters,
    CheckpointIntegrityError,
    assert_accumulation_boundary,
    build_checkpoint_payload,
    load_checkpoint_protocol,
    format_retention_plan,
    frozen_config_hashes,
    inventory_from_directory,
    load_verified_checkpoint,
    manifest_path_for,
    read_manifest,
    plan_retention,
    restore_durable_state,
    save_durable_checkpoint,
)
from tinybench_lm.data import PackedTokenDataset, TrainingSource, load_data_metadata
from tinybench_lm.exposure import CompositeTokenStream, load_exposure_plan, verify_exposure
from tinybench_lm.metric_ledger import reconcile_metrics
from tinybench_lm.shards import load_split_manifest
from tinybench_lm.provenance import record_step_zero_provenance, write_step_zero_provenance
from tinybench_lm.schedule import CURSOR_STATE_KEY, ScheduledTokenStream, open_scheduled_stream, training_order_hash
from tinybench_lm.repeated_schedule import RepeatedScheduledStream
from tinybench_lm.training_recipe import (
    SCOPE_FINAL,
    SCOPE_PILOT,
    BatchPlan,
    RunSemantics,
    WSDSchedule,
    adamw_parameter_groups,
    adamw_settings,
    assert_finite,
    assert_run_id_unchanged,
    assert_update_record,
    assert_valid_token_ids,
    batch_plan_violations,
    build_run_semantics,
    model_config_hash,
    build_update_record,
    load_training_recipe,
    plan_batch,
    release_candidate_violations,
    select_precision_policy,
    warmup_updates_for_horizon,
)

STEP_ZERO_PROVENANCE_FILENAME = "step_zero_provenance.json"
RUN_IDENTITY_FILENAME = "run_identity.json"
PHASE_TIMING_HISTORY_FILENAME = "phase_timing_history.jsonl"
EXPERIMENT_PROTECTED_SLICES = frozenset((
    "broad_general", "educational_science", "narrative_coreference", "math_technical",
))

#: Preserved bounded-pilot accumulation. A final run derives the accumulation that hits the
#: frozen 262,144-loss-token global batch instead; a pilot smoke run stays small on purpose.
PILOT_GRADIENT_ACCUMULATION = 4

#: Argument groups that select a batch source. Final training uses a materialized index
#: schedule (Plan Section 5.4); the flat-stream sampler is pilot-only.
SCHEDULE_ARGUMENTS = (
    "shard_root",
    "train_manifest",
    "train_schedule",
    "validation_manifest",
    "validation_schedule",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyBench-LM from random initialization")
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_12m.json"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/pilot"),
        help="PILOT ONLY: flat train.bin/validation.bin directory for random-sample smoke runs",
    )
    parser.add_argument("--shard-root", type=Path, help="root of the source-tagged uint16 shard namespaces")
    parser.add_argument("--train-manifest", type=Path, help="split manifest for the training split")
    parser.add_argument("--train-schedule", type=Path, help="materialized index schedule for training")
    parser.add_argument("--exposure-plan", type=Path, help="persisted two-component baseline exposure plan")
    parser.add_argument("--exposure-component-1", type=Path, help="first exposure component schedule")
    parser.add_argument("--exposure-component-2", type=Path, help="second exposure component schedule")
    parser.add_argument("--train-epochs", type=int, default=1,
                        help="explicit finite passes over the verified schedule; changes run identity")
    parser.add_argument("--validation-manifest", type=Path, help="split manifest for validation_dev")
    parser.add_argument("--validation-schedule", type=Path, help="materialized index schedule for validation_dev")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--steps", type=int, default=1_000, help="optimizer updates in the horizon (K)")
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        help="microbatches per update; omit to derive the accumulation that hits the frozen "
        "262,144-loss-token global batch exactly",
    )
    parser.add_argument("--learning-rate", type=float, default=6e-4, help="WSD peak learning rate")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        help="WSD warmup updates; omit to use the frozen ~1%% of the horizon",
    )
    parser.add_argument(
        "--decay-updates",
        type=int,
        default=0,
        help="WSD linear-decay updates ending at exactly zero LR. 0 keeps the peak LR to the "
        "end, which is a legal parent lineage but never a release fallback (Plan Section 15)",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--bf16-stability",
        choices=["stable", "unstable"],
        help="recorded outcome of the BF16 stability measurement. Omit only for pilot runs: a "
        "final run refuses to start without it (Plan Section 7)",
    )
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument(
        "--validation-mode",
        choices=("sampled", "full-dev"),
        default="sampled",
        help="validation policy; full-dev scores every validation_dev schedule reference exactly once",
    )
    parser.add_argument("--experiment-slice-reporting", action="store_true",
                        help="experiment-only: report losses for declared dev slices")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--runner-identity", type=Path, help="immutable section-4 runner identity manifest")
    parser.add_argument(
        "--stop-after-updates", type=int,
        help="save and stop at this completed update count without changing the declared training horizon",
    )
    parser.add_argument("--stop-after-training-seconds", type=float,
                        help="profiling: stop at an update boundary after this optimizer-time window")
    return parser.parse_args()


def write_phase_timing(
    run_dir: Path,
    timing: dict[str, float],
    *,
    started_at_update: int,
    completed_updates: int,
    resume_checkpoint: Path | None,
    invocation_id: str | None = None,
) -> dict[str, object]:
    """Persist latest timing and append one immutable record per process invocation."""
    history_path = run_dir / PHASE_TIMING_HISTORY_FILENAME
    existing_records = []
    if history_path.is_file():
        existing_records = [
            json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    record: dict[str, object] = {
        "schema": "training_phase_timing_invocation_v1",
        "invocation_index": len(existing_records),
        "started_at_update": int(started_at_update),
        "completed_updates": int(completed_updates),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "metric_invocation_id": invocation_id,
        **{key: float(value) for key, value in timing.items()},
    }
    with history_path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
    with (run_dir / "phase_timing.json").open("w", encoding="utf-8", newline="\n") as output:
        json.dump(timing, output, indent=2, sort_keys=True)
        output.write("\n")
    return record


def use_materialized_schedule(args: argparse.Namespace) -> bool:
    """True when a materialized index schedule was supplied for final training.

    All five schedule arguments are required together: a half-configured mixture would fall
    back to random flat-stream sampling and silently produce an unreproducible exposure.
    """
    exposure = [getattr(args, name, None) for name in ("exposure_plan", "exposure_component_1", "exposure_component_2")]
    if any(item is not None for item in exposure):
        if not all(item is not None for item in exposure):
            raise ValueError("exposure plan and both component schedules are required together")
        return True
    supplied = [name for name in SCHEDULE_ARGUMENTS if getattr(args, name, None) is not None]
    if not supplied:
        return False
    if len(supplied) != len(SCHEDULE_ARGUMENTS):
        missing = sorted(set(SCHEDULE_ARGUMENTS) - set(supplied))
        raise ValueError(
            "a materialized schedule needs every one of "
            f"{list(SCHEDULE_ARGUMENTS)}; missing {missing}. "
            "Refusing to fall back to pilot random sampling for a final run."
        )
    return True


def run_scope(args: argparse.Namespace) -> str:
    """A materialized-schedule run is a final-scope run; the flat sampler is pilot scope."""
    return SCOPE_FINAL if use_materialized_schedule(args) else SCOPE_PILOT


def should_validate(completed_update: int, horizon: int, interval: int) -> bool:
    """Absolute completed-update cadence, including update 1 and horizon completion."""
    completed = int(completed_update)
    if completed < 1 or int(horizon) < 1 or int(interval) < 1:
        raise ValueError("completed update, horizon, and validation interval must be positive")
    return completed == 1 or completed % int(interval) == 0 or completed == int(horizon)


def open_batch_sources(args: argparse.Namespace) -> tuple[TrainingSource, TrainingSource, dict[str, object]]:
    """Open the training and validation batch sources plus the data facts to record.

    Final training reads a materialized ``(shard_id, token_offset, length)`` schedule whose
    resume state is one integer ``schedule_cursor``. The pilot random flat-stream sampler is
    retained for bounded smoke runs only and is labeled as such in the run record.
    """
    exposure_args = tuple(
        getattr(args, name, None)
        for name in ("exposure_plan", "exposure_component_1", "exposure_component_2")
    )
    if any(item is not None for item in exposure_args):
        if not all(item is not None for item in exposure_args):
            raise ValueError("exposure plan and both component schedules are required together")
        if any(getattr(args, name, None) is None for name in ("shard_root", "validation_manifest", "validation_schedule")):
            raise ValueError("composite exposure requires shard root and validation inputs")
        manifest = load_split_manifest(
            Path(args.train_manifest)
            if args.train_manifest
            else Path("data/shards/reduced_5pct_v1/stable_train.manifest.json")
        )
        exposure = load_exposure_plan(args.exposure_plan, (args.exposure_component_1, args.exposure_component_2))
        verify_exposure(exposure, manifest)
        train_data = CompositeTokenStream(args.shard_root, manifest, exposure, validate_components=False)
        validation_data = open_scheduled_stream(args.shard_root, args.validation_manifest, args.validation_schedule, wrap=True)
        if validation_data.manifest.split_id != "validation_dev" or validation_data.schedule.split_id != "validation_dev":
            train_data.close()
            validation_data.close()
            raise ValueError("training validation inputs must be the validation_dev manifest and schedule")
        facts = {
            "batch_source": "composite baseline exposure",
            "shard_root": str(args.shard_root),
            "train_schedule_content_hash": exposure.content_hash,
            "train_schedule_id": "baseline_exposure_v1",
            "train_scheduled_sequences": exposure.sequence_count,
            "train_epochs": 1,
            "train_total_scheduled_sequences": exposure.sequence_count,
            "train_sequences_per_source": exposure.source_counts,
            "exposure_plan": str(args.exposure_plan),
            "exposure_component_hashes": [component.content_hash() for component in exposure.components],
            "validation_schedule_content_hash": validation_data.content_hash,
            "validation_schedule_id": validation_data.schedule.schedule_id,
        }
        return train_data, validation_data, facts
    if not use_materialized_schedule(args):
        metadata = load_data_metadata(args.data_dir)
        train_data = PackedTokenDataset(args.data_dir / "train.bin", args.seed)
        validation_data = PackedTokenDataset(args.data_dir / "validation.bin", args.seed + 1)
        return train_data, validation_data, {"batch_source": "PILOT_ONLY random flat stream", **metadata}

    train_data = open_scheduled_stream(args.shard_root, args.train_manifest, args.train_schedule)
    epochs = getattr(args, "train_epochs", 1)
    if epochs > 1:
        train_data = RepeatedScheduledStream(train_data, epochs)
    # Validation replays its own schedule from the start of every evaluation pass, so a short
    # validation schedule is reused rather than exhausted mid-run.
    validation_data = open_scheduled_stream(
        args.shard_root, args.validation_manifest, args.validation_schedule, wrap=True
    )
    if getattr(args, "validation_mode", None) is not None and (
        getattr(validation_data.manifest, "split_id", None) != "validation_dev"
        or getattr(validation_data.schedule, "split_id", None) != "validation_dev"
    ):
        validation_data.close()
        raise ValueError("training validation inputs must be the validation_dev manifest and schedule")
    facts = {
        "batch_source": "materialized index schedule",
        "shard_root": str(args.shard_root),
        "train_schedule_content_hash": train_data.content_hash,
        "train_schedule_id": train_data.schedule.schedule_id,
        "train_scheduled_sequences": train_data.schedule.sequence_count,
        "train_epochs": epochs,
        "train_total_scheduled_sequences": train_data.schedule.sequence_count * epochs,
        "train_sequences_per_source": train_data.schedule.sequences_per_source,
        "validation_schedule_content_hash": validation_data.content_hash,
        "validation_schedule_id": validation_data.schedule.schedule_id,
    }
    return train_data, validation_data, facts


def build_batch_plan(args: argparse.Namespace, scope: str, recipe: dict[str, object]) -> BatchPlan:
    """Derive or validate the accumulation for the frozen 262,144-loss-token global batch.

    An explicit ``--gradient-accumulation`` is always honored and then checked. When it is
    omitted, a final-scope run derives the accumulation that hits the frozen target exactly,
    and a pilot-scope run keeps its small preserved default.
    """
    if args.gradient_accumulation is not None:
        plan = BatchPlan(args.micro_batch_size, args.sequence_length, args.gradient_accumulation)
    elif scope == SCOPE_FINAL:
        plan = plan_batch(args.micro_batch_size, args.sequence_length, protocol=recipe)
    else:
        plan = BatchPlan(args.micro_batch_size, args.sequence_length, PILOT_GRADIENT_ACCUMULATION)
    problems = batch_plan_violations(plan, scope=scope, protocol=recipe)
    if problems:
        raise ValueError("; ".join(problems))
    return plan


def build_lr_schedule(args: argparse.Namespace, recipe: dict[str, object]) -> WSDSchedule:
    """The WSD schedule: linear warmup, stable peak, linear decay to exactly zero."""
    warmup = (
        warmup_updates_for_horizon(args.steps, protocol=recipe)
        if args.warmup_steps is None
        else int(args.warmup_steps)
    )
    return WSDSchedule(
        total_updates=int(args.steps),
        warmup_updates=warmup,
        decay_updates=int(args.decay_updates),
        peak_lr=float(args.learning_rate),
    )


def bf16_measured_stable(args: argparse.Namespace) -> bool | None:
    """Tri-state: measured stable, measured unstable, or not measured at all."""
    if args.bf16_stability is None:
        return None
    return args.bf16_stability == "stable"


def resolve_run_identity(
    run_dir: Path, semantics: RunSemantics, recipe: dict[str, object]
) -> str:
    """Issue or re-verify this run directory's ID.

    Plan Section 15: a learning-rate or semantic change creates a new run ID and never mutates
    an existing lineage. The recorded ID is compared against a freshly computed one, so an
    edited LR, horizon, batch, optimizer setting, precision policy, seed, schedule hash, or
    recipe digest fails closed instead of continuing under the old identity.
    """
    run_id = semantics.run_id(recipe)
    path = run_dir / RUN_IDENTITY_FILENAME
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        assert_run_id_unchanged(
            str(recorded["run_id"]),
            semantics,
            recorded_semantics=RunSemantics(**recorded["semantics"]),
            protocol=recipe,
        )
        return run_id
    with path.open("w", encoding="utf-8") as output:
        json.dump({"run_id": run_id, "semantics": semantics.to_dict()}, output, indent=2, sort_keys=True)
        output.write("\n")
    return run_id


def assert_runner_identity(args, *, config, plan, lr_schedule, precision, data_facts) -> None:
    """Bind direct final training to the section-4 manifest before model allocation."""
    if args.runner_identity is None:
        return
    try:
        identity = json.loads(args.runner_identity.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"runner identity cannot be read: {error}") from error
    if identity.get("identity_schema") == "pre_campaign_runner_v2":
        from tinybench_lm.experiments import validate_identity
        validate_identity(identity, args)
        if not getattr(args, "experiment_slice_reporting", False):
            raise ValueError("pre_campaign_runner_v2 requires experiment slice reporting")
    elif identity.get("identity_schema") == "pre_campaign_runner_v1":
        raise ValueError("obsolete pre_campaign_runner_v1 identity; use pre_campaign_runner_v2")
    elif identity.get("identity_schema") != "reduced_baseline_runner_v1":
        raise ValueError("unknown runner identity schema")
    if identity.get("scope") == "baseline_reduced_v2":
        from tinybench_lm.baseline_contract import validate_selected_identity
        validate_selected_identity(identity)
    expected = {
        "validation_mode": args.validation_mode,
        "eval_interval": int(args.eval_interval), "save_interval": int(args.save_interval),
        "seed": int(args.seed), "total_updates": int(args.steps),
        "peak_lr": float(lr_schedule.peak_lr), "warmup_updates": int(lr_schedule.warmup_updates),
        "decay_updates": int(lr_schedule.decay_updates),
        "micro_batch_size": int(plan.micro_batch_size), "gradient_accumulation": int(plan.gradient_accumulation),
        "sequence_length": int(plan.sequence_length),
        "train_schedule_content_hash": str(data_facts.get("train_schedule_content_hash")),
        "validation_schedule_hash": str(data_facts.get("validation_schedule_content_hash")),
    }
    for key, value in expected.items():
        observed = identity.get(key)
        if isinstance(value, float):
            if observed is None or abs(float(observed) - value) > 1e-12:
                raise ValueError(f"runner identity field {key} differs from resolved training")
        elif str(observed) != str(value):
            raise ValueError(f"runner identity field {key} differs from resolved training")
    canonical = model_config_hash(config.to_dict())
    if identity.get("model_config_canonical_hash") != canonical:
        raise ValueError("runner identity model configuration differs from resolved training")
    precision_identity = identity.get("precision", {})
    if precision_identity.get("dtype") != precision.dtype_name or bool(precision_identity.get("grad_scaler")) != bool(precision.use_grad_scaler):
        raise ValueError("runner identity precision policy differs from resolved training")


def open_resume_payload(
    path: Path,
    *,
    expected_run_id: str,
    expected_frozen_config_hashes: dict[str, str],
    expected_schedule_content_hash: str,
) -> dict[str, object]:
    """Load a resume artifact, verifying it first whenever it claims to be durable.

    Plan Section 7.2 requires a checksum, a load test, and a hash comparison before a resume.
    A durable checkpoint always ships a sidecar manifest, so its checksum, counters, run ID,
    frozen config hashes, and schedule binding are all re-derived and any mismatch refuses the
    resume. A pre-existing format-v2 pilot checkpoint has no manifest and no durable envelope;
    it still resumes through the same proven RNG/sampler path, with the missing durability
    evidence stated rather than assumed.
    """
    if manifest_path_for(path).is_file():
        return load_verified_checkpoint(
            path,
            expected_frozen_config_hashes=expected_frozen_config_hashes,
            expected_run_id=expected_run_id,
            expected_schedule_content_hash=expected_schedule_content_hash,
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("durable_checkpoint_format_version") is not None:
        raise CheckpointIntegrityError(
            f"{path} is a durable checkpoint but {manifest_path_for(path).name} is absent, so its "
            "bytes cannot be verified. Refusing to resume from an unverified artifact."
        )
    print(
        f"WARNING: {path} is a legacy format-v2 checkpoint with no checksum manifest, no scaler "
        "state, no schedule cursor, and no run-ID binding. Resume uses the proven RNG/sampler "
        "path, but its durability and lineage are unverified."
    )
    return payload


def report_retention(run_dir: Path) -> None:
    """Print the retention proposal for a run directory. Nothing is deleted here."""
    # `latest.pt` is a rolling recovery state; `best.pt` holds the best-validation endpoint and
    # is treated as protected evidence rather than something retention may propose deleting.
    entries = inventory_from_directory(run_dir, roles={"best.pt": ROLE_SELECTED_ENDPOINT})
    if not entries:
        return
    print(format_retention_plan(plan_retention(entries)))


class ValidationResult:
    """Token-weighted development replay result and auditable coverage metadata."""

    def __init__(self, loss: float, mode: str, sequence_count: int, scored_token_count: int,
                 batch_count: int, schedule_content_hash: str | None, schedule_id: str | None,
                 slice_metrics: dict[str, dict[str, float | int]] | None = None) -> None:
        self.loss = loss
        self.mode = mode
        self.sequence_count = sequence_count
        self.scored_token_count = scored_token_count
        self.batch_count = batch_count
        self.schedule_content_hash = schedule_content_hash
        self.schedule_id = schedule_id
        self.slice_metrics = slice_metrics

    def to_dict(self) -> dict[str, object]:
        result = {
            "validation_loss": self.loss,
            "validation_mode": self.mode,
            "validation_sequence_count": self.sequence_count,
            "validation_scored_token_count": self.scored_token_count,
            "validation_batch_count": self.batch_count,
            "validation_schedule_content_hash": self.schedule_content_hash,
            "validation_schedule_id": self.schedule_id,
        }
        if self.slice_metrics is not None:
            result["validation_slice_metrics"] = self.slice_metrics
            result["validation_slices"] = {name: values["loss"] for name, values in self.slice_metrics.items()}
        return result


def _rng_snapshot() -> tuple[object, object, object, list[torch.Tensor] | None]:
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda


def _restore_rng(snapshot: tuple[object, object, object, list[torch.Tensor] | None]) -> None:
    py, numpy_state, torch_state, cuda = snapshot
    random.setstate(py)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)


@torch.no_grad()
def evaluate_result(
    model: TinyBenchLM,
    dataset: TrainingSource,
    args: argparse.Namespace,
    device: torch.device,
    autocast_context,
) -> ValidationResult:
    """Run sampled or complete development validation while preserving training state."""
    mode = str(getattr(args, "validation_mode", "sampled"))
    if mode == "full-dev" and not isinstance(dataset, (ScheduledTokenStream, CompositeTokenStream)):
        raise ValueError("full-dev validation requires the materialized validation_dev schedule")
    if mode == "full-dev" and getattr(getattr(dataset, "manifest", None), "split_id", "validation_dev") != "validation_dev":
        raise ValueError("full-dev validation is restricted to the validation_dev split")
    state = dataset.state_dict()
    original_wrap = getattr(dataset, "wrap", None)
    was_training = model.training
    rng = _rng_snapshot()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    scored_token_total = torch.zeros((), dtype=torch.int64, device=device)
    slice_sums: dict[str, float] = {}
    slice_counts: dict[str, int] = {}
    slice_enabled = bool(getattr(args, "experiment_slice_reporting", False))
    slice_by_shard: dict[str, str] = {}
    if slice_enabled:
        manifest = getattr(dataset, "manifest", None)
        for shard in getattr(manifest, "shards", ()):
            declared = tuple(getattr(shard, "protected_slices", ()))
            if len(declared) != 1:
                raise ValueError("experiment slice reporting requires exactly one protected slice per shard")
            slice_by_shard[str(shard.shard_id)] = str(declared[0])
    sequence_count = 0
    batches = 0
    try:
        if isinstance(dataset, (ScheduledTokenStream, CompositeTokenStream)):
            dataset.rewind()
            if mode == "full-dev":
                dataset.wrap = False
                requested = dataset.schedule.sequence_count
            else:
                requested = args.eval_batches * args.micro_batch_size
        else:
            requested = args.eval_batches * args.micro_batch_size
        model.eval()
        remaining = requested
        while remaining > 0:
            batch_size = min(args.micro_batch_size, remaining) if mode == "full-dev" else args.micro_batch_size
            inputs, targets = dataset.get_batch(batch_size, args.sequence_length, device)
            with autocast_context():
                logits, loss = model(inputs, targets)
            assert loss is not None
            token_count = (targets != LOSS_IGNORE_INDEX).sum()
            total_loss += loss.detach().double() * token_count
            scored_token_total += token_count
            if slice_enabled:
                entries = tuple(getattr(dataset, "last_batch_entries", ()))
                if len(entries) != batch_size:
                    raise ValueError("validation reader did not expose one reference per scored sequence")
                per_token = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                    ignore_index=LOSS_IGNORE_INDEX, reduction="none",
                ).reshape(targets.shape)
                for row, entry in enumerate(entries):
                    name = slice_by_shard.get(str(entry.shard_id))
                    if name is None:
                        raise ValueError("validation reference shard has no declared protected slice")
                    kept = targets[row] != LOSS_IGNORE_INDEX
                    count = int(kept.sum().item())
                    if count:
                        slice_sums[name] = slice_sums.get(name, 0.0) + float(per_token[row][kept].sum().item())
                        slice_counts[name] = slice_counts.get(name, 0) + count
            sequence_count += batch_size
            batches += 1
            remaining -= batch_size
    finally:
        try:
            dataset.load_state_dict(state)
        finally:
            try:
                if isinstance(dataset, (ScheduledTokenStream, CompositeTokenStream)) and original_wrap is not None:
                    dataset.wrap = bool(original_wrap)
            finally:
                try:
                    model.train(was_training)
                finally:
                    _restore_rng(rng)
    scored_tokens = int(scored_token_total.item())
    if sequence_count == 0 or scored_tokens == 0:
        raise ValueError("validation schedule produced no scored tokens")
    slice_metrics = None
    if slice_enabled:
        if (set(slice_sums) != EXPERIMENT_PROTECTED_SLICES
                or set(slice_counts) != EXPERIMENT_PROTECTED_SLICES
                or sum(slice_counts.values()) != scored_tokens):
            raise ValueError("validation slice coverage does not reconcile with global token coverage")
        if any(count <= 0 or not math.isfinite(total) for name, total in slice_sums.items() for count in (slice_counts[name],)):
            raise ValueError("validation slice totals/counts are non-finite or empty")
        slice_metrics = {
            name: {"loss_sum": total, "token_count": slice_counts[name], "loss": total / slice_counts[name]}
            for name, total in sorted(slice_sums.items())
        }
    return ValidationResult(
        loss=float((total_loss / scored_tokens).item()),
        mode=mode,
        sequence_count=sequence_count,
        scored_token_count=scored_tokens,
        batch_count=batches,
        schedule_content_hash=getattr(dataset, "content_hash", None),
        schedule_id=(getattr(getattr(dataset, "schedule", None), "schedule_id", None)),
        slice_metrics=slice_metrics,
    )


@torch.no_grad()
def evaluate(
    model: TinyBenchLM,
    dataset: TrainingSource,
    args: argparse.Namespace,
    device: torch.device,
    autocast_context,
) -> float:
    """Compatibility wrapper returning only the validation loss."""
    return evaluate_result(model, dataset, args, device, autocast_context).loss


def configure_strict_cuda_determinism() -> dict[str, object]:
    """Enable CUDA's strict deterministic contract before the first model operation.

    PyTorch raises if an invoked CUDA operation has no deterministic implementation.  That
    failure is intentional: exact recovery evidence cannot silently downgrade to a
    numerically nondeterministic execution policy.
    """
    requested_workspace = ":4096:8"
    # A checkpoint resume always receives this one canonical cuBLAS policy, even if its
    # parent process carried another legal deterministic workspace size.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = requested_workspace
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "strict_deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }


def main() -> None:
    process_started = time.perf_counter()
    args = parse_args()
    if args.runner_identity is not None:
        if not args.runner_identity.is_file():
            raise ValueError(f"runner identity is missing: {args.runner_identity}")
        args.runner_identity_hash = __import__("hashlib").sha256(args.runner_identity.read_bytes()).hexdigest()
    if args.train_epochs < 1 or (args.train_epochs != 1 and not use_materialized_schedule(args)):
        raise ValueError("train-epochs must be positive and requires a materialized schedule")
    if args.stop_after_updates is not None and not 0 < args.stop_after_updates <= args.steps:
        raise ValueError("stop-after-updates must be positive and no greater than steps")
    if args.stop_after_training_seconds is not None and args.stop_after_training_seconds <= 0:
        raise ValueError("stop-after-training-seconds must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    deterministic_execution = configure_strict_cuda_determinism()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training configuration expects an NVIDIA GPU")

    recipe = load_training_recipe()
    checkpoint_protocol = load_checkpoint_protocol()
    scope = run_scope(args)
    if scope == SCOPE_FINAL:
        if args.runner_identity is None:
            raise ValueError("final-scope training requires the section-4 runner identity manifest")
        if args.validation_mode != "full-dev":
            raise ValueError("final-scope training requires --validation-mode full-dev")
        if args.eval_interval != 100:
            raise ValueError("final-scope validation cadence is fixed at --eval-interval 100")
        if args.save_interval != 100:
            raise ValueError("final-scope recovery cadence is fixed at --save-interval 100")
    precision = select_precision_policy(
        bf16_supported=torch.cuda.is_bf16_supported(),
        bf16_measured_stable=bf16_measured_stable(args),
        scope=scope,
        protocol=recipe,
    )
    amp_dtype = precision.torch_dtype()

    config = ModelConfig.from_json(args.config)
    if args.sequence_length > config.max_seq_len:
        raise ValueError("sequence-length exceeds the model configuration")

    plan = build_batch_plan(args, scope, recipe)
    args.gradient_accumulation = plan.gradient_accumulation
    lr_schedule = build_lr_schedule(args, recipe)
    args.warmup_steps = lr_schedule.warmup_updates

    train_data, validation_data, data_facts = open_batch_sources(args)
    data_facts["execution_policy"] = deterministic_execution
    if "actual_vocab_size" in data_facts and int(data_facts["actual_vocab_size"]) > config.vocab_size:
        raise ValueError("Tokenizer vocabulary is larger than the model vocabulary")
    if data_facts["batch_source"] == "PILOT_ONLY random flat stream":
        print(
            "PILOT ONLY: training from the random flat-stream sampler. A final run must pass "
            "--shard-root/--train-manifest/--train-schedule so the consumed mixture and "
            "exposure order are reproducible from one schedule hash and one integer cursor."
        )
    train_schedule_hash = str(data_facts.get("train_schedule_content_hash", "PILOT_ONLY_NO_SCHEDULE"))
    assert_runner_identity(
        args, config=config, plan=plan, lr_schedule=lr_schedule, precision=precision, data_facts=data_facts
    )
    available = data_facts.get("train_total_scheduled_sequences")
    needed = args.steps * plan.loss_tokens_per_update // args.sequence_length
    if available is not None and int(available) < needed:
        raise ValueError(f"Training horizon needs {needed} scheduled sequences; only {available} supplied")

    model = TinyBenchLM(config).to(device)
    parameter_count = model.count_parameters()
    if parameter_count > 50_000_000:
        raise RuntimeError(f"Parameter cap exceeded: {parameter_count:,}")
    raw_model = model
    if args.compile:
        model = torch.compile(model)

    # Plan Section 7: weight decay 0.1 excluding embeddings and all normalization weights.
    # The tied embedding/output Parameter is enumerated once and lands in the no-decay group.
    settings = adamw_settings(recipe)
    parameter_groups = adamw_parameter_groups(raw_model, weight_decay=args.weight_decay, protocol=recipe)
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=lr_schedule.peak_lr,
        betas=settings["betas"],
        eps=settings["epsilon"],
        fused=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision.use_grad_scaler)

    # The run identity is resolved before any resume, because a resume must be checked against
    # the run ID and the frozen artifact hashes it claims to continue (Plan Sections 7.2, 15).
    args.run_dir.mkdir(parents=True, exist_ok=True)
    semantics = build_run_semantics(
        model_config=config.to_dict(),
        schedule=lr_schedule,
        plan=plan,
        precision=precision,
        weight_decay=args.weight_decay,
        gradient_clip_global_norm=args.grad_clip,
        seed=args.seed,
        train_schedule_content_hash=train_schedule_hash,
        protocol=recipe,
    )
    run_id = resolve_run_identity(args.run_dir, semantics, recipe)
    artifact_hashes = frozen_config_hashes(
        model_config_hash=semantics.model_config_hash, recipe=recipe
    )

    first_step = 0
    best_validation_loss = float("inf")
    best_validation = BestValidationState.unevaluated()
    if args.resume:
        checkpoint = open_resume_payload(
            args.resume,
            expected_run_id=run_id,
            expected_frozen_config_hashes=artifact_hashes,
            expected_schedule_content_hash=train_schedule_hash,
        )
        if args.runner_identity is not None:
            recorded_runner_hash = checkpoint.get("training_args", {}).get("runner_identity_hash")
            if recorded_runner_hash != args.runner_identity_hash:
                raise CheckpointIntegrityError("resume runner identity manifest differs from checkpoint custody")
        resume_state = restore_durable_state(
            checkpoint,
            raw_model,
            optimizer,
            train_data,
            validation_data,
            scaler=scaler,
        )
        first_step = resume_state.first_update_index
        best_validation_loss = resume_state.best_validation_loss
        if isinstance(checkpoint.get("best_validation_state"), dict):
            best_validation = BestValidationState.from_dict(checkpoint["best_validation_state"])
        if not resume_state.reproducible:
            print(
                "WARNING: legacy checkpoint has no RNG/sampler state; "
                "resume is functional but not exactly reproducible"
            )
        print(
            f"Resumed at update {first_step} (scaler_restored={resume_state.scaler_restored}, "
            f"schedule_cursor={resume_state.schedule_cursor})"
        )

    previous_record, metric_invocation_id = reconcile_metrics(
        args.run_dir, completed_updates=first_step, run_id=run_id,
        schedule=lr_schedule, plan=plan, precision=precision,
        schedule_content_hash=train_schedule_hash,
        checkpoint_cursor=train_data.state_dict().get(CURSOR_STATE_KEY),
        resume_checkpoint=args.resume,
    )
    provenance_path = args.run_dir / STEP_ZERO_PROVENANCE_FILENAME
    if args.resume:
        # A resumed run inherits its lineage; re-recording would overwrite frozen evidence
        # with weights that are no longer step zero.
        print(f"Resumed run: step-zero evidence stays at {provenance_path}")
    else:
        step_zero = record_step_zero_provenance(
            raw_model,
            config,
            seed=args.seed,
            config_path=args.config,
            optimizer=optimizer,
        )
        write_step_zero_provenance(provenance_path, step_zero)
        print(f"Step-zero weight hash: {step_zero.weight_sha256}")

    with (args.run_dir / "run_config.json").open("w", encoding="utf-8") as output:
        json.dump(
            {
                "model_config": config.to_dict(),
                "training_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "parameter_count": parameter_count,
                "device": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "amp_dtype": str(amp_dtype),
                "run_id": run_id,
                "run_scope": scope,
                "run_semantics": semantics.to_dict(),
                "recipe_digest": recipe["_digest"],
                "lr_schedule": lr_schedule.to_dict(),
                "lr_schedule_fingerprint": lr_schedule.fingerprint(),
                "batch_plan": plan.to_dict(),
                "precision_policy": precision.to_dict(),
                "parameter_groups": {
                    str(group["group_name"]): {
                        "tensors": len(group["params"]),
                        "elements": sum(int(parameter.numel()) for parameter in group["params"]),
                        "weight_decay": float(group["weight_decay"]),
                    }
                    for group in parameter_groups
                },
                "data_metadata": data_facts,
            },
            output,
            indent=2,
            sort_keys=True,
        )
        output.write("\n")

    log_path = args.run_dir / "metrics.jsonl"
    tokens_per_step = plan.loss_tokens_per_update
    def autocast_context():
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    def write_checkpoint(path: Path, completed_update: int, pending_microbatches: int) -> None:
        """Write one durable checkpoint at an accumulation boundary (Plan Section 7.2).

        The boundary is asserted rather than assumed: a payload whose counters describe a
        completed update while microbatches of the next one are already accumulated would
        double-count or drop data on resume, so it is refused before anything is written.
        """
        checkpoint_started = time.perf_counter()
        assert_accumulation_boundary(pending_microbatches, plan.gradient_accumulation)
        payload = build_checkpoint_payload(
            model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            args=args,
            train_data=train_data,
            validation_data=validation_data,
            counters=CheckpointCounters.at_update(
                completed_update, plan, microbatches_completed_in_update=pending_microbatches
            ),
            run_id=run_id,
            frozen_config_hashes=artifact_hashes,
            schedule_content_hash=train_schedule_hash,
            best_validation=best_validation,
            protocol=checkpoint_protocol,
        )
        save_durable_checkpoint(path, payload, protocol=checkpoint_protocol)
        phase_timing["checkpoint_seconds"] += time.perf_counter() - checkpoint_started

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Run ID: {run_id} ({scope} scope)")
    print(f"Precision: {precision.dtype_name} scaler={precision.use_grad_scaler} [{precision.status}]")
    if precision.status != "PASS":
        print(f"  {precision.reason}")
    print(
        f"WSD: warmup={lr_schedule.warmup_updates} stable={lr_schedule.stable_updates} "
        f"decay={lr_schedule.decay_updates} peak_lr={lr_schedule.peak_lr:g}"
    )
    for problem in release_candidate_violations(lr_schedule):
        print(f"  NOT A RELEASE CANDIDATE: {problem}")
    print(f"Tokens/optimizer update: {tokens_per_step:,}")
    torch.cuda.reset_peak_memory_stats()
    model.train()

    measured_training_seconds = 0.0
    phase_timing = {
        "training_optimizer_seconds": 0.0,
        "data_wait_seconds": 0.0,
        "validation_seconds": 0.0,
        "checkpoint_seconds": 0.0,
    }
    completed_updates = first_step
    for step in range(first_step, args.steps):
        started = time.perf_counter()
        update_started_seconds = started - process_started
        lr = lr_schedule.learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        # Keep the logging reduction on device until the explicit update-boundary
        # synchronization below; scalarizing every microbatch adds avoidable host/device
        # synchronization to the optimizer hot path.
        accumulated_loss_device = torch.zeros((), dtype=torch.float32, device=device)
        # Counted, not assumed: this is what makes "save only at accumulation boundaries"
        # checkable instead of a comment.
        pending_microbatches = 0
        update_batch_entries = []
        # Plan Section 15 fails closed on invalid token IDs. Shard verification already bounds
        # every stored ID, so the in-loop check samples update boundaries instead of adding a
        # device synchronization to every microbatch of a multi-day run.
        check_tokens = step == first_step or (step + 1) % args.eval_interval == 0
        for _ in range(plan.gradient_accumulation):
            data_wait_started = time.perf_counter()
            inputs, targets = train_data.get_batch(args.micro_batch_size, args.sequence_length, device)
            phase_timing["data_wait_seconds"] += time.perf_counter() - data_wait_started
            update_batch_entries.extend(getattr(train_data, "last_batch_entries", ()))
            if check_tokens:
                assert_valid_token_ids(inputs, config.vocab_size, name="input_ids")
                assert_valid_token_ids(targets, config.vocab_size, name="targets", allow_ignore_index=True)
            with autocast_context():
                _, loss = model(inputs, targets)
                assert loss is not None
                scaled_loss = loss / plan.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            accumulated_loss_device += loss.detach().float() / plan.gradient_accumulation
            pending_microbatches += 1
        # Preserve the fail-closed pre-step finite-loss check while keeping its one scalar
        # transfer per optimizer update rather than one transfer per microbatch.
        assert_finite("train_loss", accumulated_loss_device)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        # The optimizer step closed the accumulation window, so the state is now saveable.
        pending_microbatches = 0
        torch.cuda.synchronize()
        accumulated_loss = float(accumulated_loss_device.item())
        optimizer_finished_seconds = time.perf_counter() - process_started
        elapsed = optimizer_finished_seconds - update_started_seconds
        measured_training_seconds += elapsed
        phase_timing["training_optimizer_seconds"] += elapsed

        update_record = assert_update_record(
            build_update_record(
                run_id=run_id,
                update_index=step,
                schedule=lr_schedule,
                plan=plan,
                precision=precision,
                loss=accumulated_loss,
                grad_norm=grad_norm,
                schedule_content_hash=train_schedule_hash,
                schedule_cursor=train_data.state_dict().get(CURSOR_STATE_KEY),
            ),
            schedule=lr_schedule,
            plan=plan,
            previous=previous_record,
        )
        previous_record = update_record
        record = {
            **update_record.to_dict(),
            "invocation_id": metric_invocation_id,
            "step": step,
            "train_loss": update_record.loss,
            "tokens": update_record.consumed_loss_tokens,
            "tokens_per_second": tokens_per_step / elapsed,
            "step_seconds": elapsed,
            "update_started_seconds": update_started_seconds,
            "optimizer_finished_seconds": optimizer_finished_seconds,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        # The hash must cover every scheduled sequence consumed by this optimizer
        # update.  A last-microbatch hash cannot demonstrate the full exposure order.
        if update_batch_entries:
            record["train_batch_reference_hash"] = training_order_hash(update_batch_entries)

        # Validation cadence is expressed in absolute completed updates.  The event after
        # completed update 1 is post-update monitoring, never a pre-training measurement;
        # a resume must not recreate it.
        should_eval = should_validate(step + 1, args.steps, args.eval_interval)
        if should_eval:
            validation_started = time.perf_counter()
            validation = evaluate_result(model, validation_data, args, device, autocast_context)
            phase_timing["validation_seconds"] += time.perf_counter() - validation_started
            record.update(validation.to_dict())
            if getattr(args, "experiment_slice_reporting", False):
                # Bind the endpoint slice evidence into the durable checkpoint's immutable
                # training_args envelope before the checkpoint is written.  The runner can
                # therefore reject a metrics.jsonl-only edit during independent verification.
                args.experiment_endpoint_validation = validation.to_dict()
            validation_loss = validation.loss
            record["validation_perplexity"] = math.exp(min(20.0, validation_loss))
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation = BestValidationState(
                    loss=best_validation_loss,
                    update_index=step,
                    relative_path="best.pt",
                )
                write_checkpoint(args.run_dir / "best.pt", step, pending_microbatches)
        if getattr(args, "experiment_slice_reporting", False):
            record["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
            record["peak_reserved_vram_gib"] = torch.cuda.max_memory_reserved() / 2**30
        with log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        if step == first_step or (step + 1) % args.log_interval == 0 or should_eval:
            suffix = f" val={record['validation_loss']:.4f}" if "validation_loss" in record else ""
            print(
                f"step={step + 1}/{args.steps} phase={update_record.phase} lr={lr:.3e} "
                f"loss={accumulated_loss:.4f}{suffix} "
                f"tok/s={record['tokens_per_second']:,.0f} vram={record['peak_vram_gib']:.2f}GiB"
            )
        completed_updates = step + 1
        checkpoint_written_this_update = False
        if (step + 1) % args.save_interval == 0 or step + 1 == args.steps:
            write_checkpoint(args.run_dir / "latest.pt", step, pending_microbatches)
            checkpoint_written_this_update = True
        if args.stop_after_updates is not None and step + 1 >= args.stop_after_updates:
            if not checkpoint_written_this_update:
                write_checkpoint(args.run_dir / "latest.pt", step, pending_microbatches)
            print(f"Stopped safely after {step + 1} updates; declared horizon remains {args.steps}.")
            break
        if args.stop_after_training_seconds is not None and measured_training_seconds >= args.stop_after_training_seconds:
            if not checkpoint_written_this_update:
                write_checkpoint(args.run_dir / "latest.pt", step, pending_microbatches)
            print(f"Profiling window complete: {measured_training_seconds:.2f} optimizer seconds.")
            break

    # Preserve a separately named completed endpoint/fallback role. This is a byte-for-byte
    # custody copy of the final durable latest checkpoint and its manifest; no deletion occurs.
    if args.stop_after_updates is None and args.stop_after_training_seconds is None:
        latest = args.run_dir / "latest.pt"
        if latest.is_file():
            checkpoint_started = time.perf_counter()
            shutil.copy2(latest, args.run_dir / "completed.pt")
            latest_manifest = manifest_path_for(latest)
            if latest_manifest.is_file():
                completed_manifest = replace(read_manifest(latest), checkpoint_filename="completed.pt")
                manifest_path_for(args.run_dir / "completed.pt").write_text(
                    json.dumps(completed_manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            phase_timing["checkpoint_seconds"] += time.perf_counter() - checkpoint_started
    phase_timing["process_wall_seconds"] = time.perf_counter() - process_started
    write_phase_timing(
        args.run_dir,
        phase_timing,
        started_at_update=first_step,
        completed_updates=completed_updates,
        resume_checkpoint=args.resume,
        invocation_id=metric_invocation_id,
    )
    report_retention(args.run_dir)


if __name__ == "__main__":
    main()
