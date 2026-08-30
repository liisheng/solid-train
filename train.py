from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from tinybench_lm import ModelConfig, TinyBenchLM
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
    plan_retention,
    restore_durable_state,
    save_durable_checkpoint,
)
from tinybench_lm.data import PackedTokenDataset, TrainingSource, load_data_metadata
from tinybench_lm.provenance import record_step_zero_provenance, write_step_zero_provenance
from tinybench_lm.schedule import CURSOR_STATE_KEY, open_scheduled_stream
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
    build_update_record,
    load_training_recipe,
    plan_batch,
    release_candidate_violations,
    select_precision_policy,
    warmup_updates_for_horizon,
)

STEP_ZERO_PROVENANCE_FILENAME = "step_zero_provenance.json"
RUN_IDENTITY_FILENAME = "run_identity.json"

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
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def use_materialized_schedule(args: argparse.Namespace) -> bool:
    """True when a materialized index schedule was supplied for final training.

    All five schedule arguments are required together: a half-configured mixture would fall
    back to random flat-stream sampling and silently produce an unreproducible exposure.
    """
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


def open_batch_sources(args: argparse.Namespace) -> tuple[TrainingSource, TrainingSource, dict[str, object]]:
    """Open the training and validation batch sources plus the data facts to record.

    Final training reads a materialized ``(shard_id, token_offset, length)`` schedule whose
    resume state is one integer ``schedule_cursor``. The pilot random flat-stream sampler is
    retained for bounded smoke runs only and is labeled as such in the run record.
    """
    if not use_materialized_schedule(args):
        metadata = load_data_metadata(args.data_dir)
        train_data = PackedTokenDataset(args.data_dir / "train.bin", args.seed)
        validation_data = PackedTokenDataset(args.data_dir / "validation.bin", args.seed + 1)
        return train_data, validation_data, {"batch_source": "PILOT_ONLY random flat stream", **metadata}

    train_data = open_scheduled_stream(args.shard_root, args.train_manifest, args.train_schedule)
    # Validation replays its own schedule from the start of every evaluation pass, so a short
    # validation schedule is reused rather than exhausted mid-run.
    validation_data = open_scheduled_stream(
        args.shard_root, args.validation_manifest, args.validation_schedule, wrap=True
    )
    facts = {
        "batch_source": "materialized index schedule",
        "shard_root": str(args.shard_root),
        "train_schedule_content_hash": train_data.content_hash,
        "train_schedule_id": train_data.schedule.schedule_id,
        "train_scheduled_sequences": train_data.schedule.sequence_count,
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


@torch.no_grad()
def evaluate(
    model: TinyBenchLM,
    dataset: TrainingSource,
    args: argparse.Namespace,
    device: torch.device,
    autocast_context,
) -> float:
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        inputs, targets = dataset.get_batch(args.micro_batch_size, args.sequence_length, device)
        with autocast_context():
            _, loss = model(inputs, targets)
        assert loss is not None
        losses.append(loss.detach().float())
    model.train()
    return torch.stack(losses).mean().item()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training configuration expects an NVIDIA GPU")

    recipe = load_training_recipe()
    checkpoint_protocol = load_checkpoint_protocol()
    scope = run_scope(args)
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
    if "actual_vocab_size" in data_facts and int(data_facts["actual_vocab_size"]) > config.vocab_size:
        raise ValueError("Tokenizer vocabulary is larger than the model vocabulary")
    if data_facts["batch_source"] != "materialized index schedule":
        print(
            "PILOT ONLY: training from the random flat-stream sampler. A final run must pass "
            "--shard-root/--train-manifest/--train-schedule so the consumed mixture and "
            "exposure order are reproducible from one schedule hash and one integer cursor."
        )
    train_schedule_hash = str(data_facts.get("train_schedule_content_hash", "PILOT_ONLY_NO_SCHEDULE"))

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
    autocast_context = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)

    def write_checkpoint(path: Path, completed_update: int, pending_microbatches: int) -> None:
        """Write one durable checkpoint at an accumulation boundary (Plan Section 7.2).

        The boundary is asserted rather than assumed: a payload whose counters describe a
        completed update while microbatches of the next one are already accumulated would
        double-count or drop data on resume, so it is refused before anything is written.
        """
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

    previous_record = None
    for step in range(first_step, args.steps):
        started = time.perf_counter()
        lr = lr_schedule.learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        # Counted, not assumed: this is what makes "save only at accumulation boundaries"
        # checkable instead of a comment.
        pending_microbatches = 0
        # Plan Section 15 fails closed on invalid token IDs. Shard verification already bounds
        # every stored ID, so the in-loop check samples update boundaries instead of adding a
        # device synchronization to every microbatch of a multi-day run.
        check_tokens = step == first_step or (step + 1) % args.eval_interval == 0
        for _ in range(plan.gradient_accumulation):
            inputs, targets = train_data.get_batch(args.micro_batch_size, args.sequence_length, device)
            if check_tokens:
                assert_valid_token_ids(inputs, config.vocab_size, name="input_ids")
                assert_valid_token_ids(targets, config.vocab_size, name="targets", allow_ignore_index=True)
            with autocast_context():
                _, loss = model(inputs, targets)
                assert loss is not None
                scaled_loss = loss / plan.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            accumulated_loss += loss.detach().float().item() / plan.gradient_accumulation
            pending_microbatches += 1
        assert_finite("train_loss", accumulated_loss)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        # The optimizer step closed the accumulation window, so the state is now saveable.
        pending_microbatches = 0
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

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
            "step": step,
            "train_loss": update_record.loss,
            "tokens": update_record.consumed_loss_tokens,
            "tokens_per_second": tokens_per_step / elapsed,
            "step_seconds": elapsed,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }

        should_eval = step == first_step or (step + 1) % args.eval_interval == 0 or step + 1 == args.steps
        if should_eval:
            validation_loss = evaluate(model, validation_data, args, device, autocast_context)
            record["validation_loss"] = validation_loss
            record["validation_perplexity"] = math.exp(min(20.0, validation_loss))
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation = BestValidationState(
                    loss=best_validation_loss,
                    update_index=step,
                    relative_path="best.pt",
                )
                write_checkpoint(args.run_dir / "best.pt", step, pending_microbatches)
        with log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        if step == first_step or (step + 1) % args.log_interval == 0 or should_eval:
            suffix = f" val={record['validation_loss']:.4f}" if "validation_loss" in record else ""
            print(
                f"step={step + 1}/{args.steps} phase={update_record.phase} lr={lr:.3e} "
                f"loss={accumulated_loss:.4f}{suffix} "
                f"tok/s={record['tokens_per_second']:,.0f} vram={record['peak_vram_gib']:.2f}GiB"
            )
        if (step + 1) % args.save_interval == 0 or step + 1 == args.steps:
            write_checkpoint(args.run_dir / "latest.pt", step, pending_microbatches)

    report_retention(args.run_dir)


if __name__ == "__main__":
    main()
