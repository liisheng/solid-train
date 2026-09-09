"""Reconcile an append-only attempt log with the checkpoint's canonical lineage.

This is startup work only. Superseded work is archived before atomic replacement;
archive names are content addressed so a crash between those writes is idempotent.
Archive optimizer seconds supplement canonical rows, not invocation wall timings.
An abrupt crash can still lose unlogged work; external process timing is required.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from .training_recipe import (
    BatchPlan, PrecisionPolicy, TrainingIntegrityError, UpdateRecord, WSDSchedule,
    assert_update_record,
)


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def reconcile_metrics(
    run_dir: Path, *, completed_updates: int, run_id: str,
    schedule: WSDSchedule, plan: BatchPlan, precision: PrecisionPolicy,
    schedule_content_hash: str, checkpoint_cursor: int | None,
    resume_checkpoint: Path | None,
) -> tuple[UpdateRecord | None, str]:
    """Validate all existing rows; retain exactly the restored checkpoint prefix.

    Legacy rows without invocation IDs remain byte-identical and receive an explicit
    source-log segment identity in archives. A resume requires the complete prefix;
    a checkpoint alone is insufficient evidence for a missing training ledger.
    """
    path = run_dir / "metrics.jsonl"
    raw = path.read_bytes() if path.exists() else b""
    lines = raw.splitlines(keepends=True)
    previous = None
    retained = None
    rows = []
    try:
        for index, line in enumerate(lines):
            row = json.loads(line)
            record = UpdateRecord(**{field.name: row[field.name] for field in fields(UpdateRecord)})
            for name in ("update_index", "consumed_loss_tokens", "loss_tokens_per_update"):
                if type(getattr(record, name)) is not int:
                    raise ValueError(f"{name} must be an integer")
            if record.schedule_cursor is not None and type(record.schedule_cursor) is not int:
                raise ValueError("schedule_cursor must be an integer or null")
            if record.update_index != index:
                raise ValueError("metrics have missing, duplicate, or out-of-order updates")
            if record.run_id != run_id or record.schedule_content_hash != schedule_content_hash:
                raise ValueError("metrics identity does not match restored run")
            if record.precision_dtype != precision.dtype_name or type(record.grad_scaler) is not bool or record.grad_scaler != precision.use_grad_scaler:
                raise ValueError("metrics precision does not match restored run")
            if checkpoint_cursor is not None and record.schedule_cursor != (index + 1) * plan.sequences_per_update:
                raise ValueError("metrics schedule cursor does not match exact batch plan")
            if checkpoint_cursor is None and record.schedule_cursor is not None:
                raise ValueError("metrics cursor is present for a checkpoint without a schedule cursor")
            if "step" in row and (type(row["step"]) is not int or row["step"] != index):
                raise ValueError("metrics step alias disagrees")
            if "tokens" in row and (type(row["tokens"]) is not int or row["tokens"] != record.consumed_loss_tokens):
                raise ValueError("metrics token alias disagrees")
            if "train_loss" in row and row["train_loss"] != record.loss:
                raise ValueError("metrics loss alias disagrees")
            seconds = row["step_seconds"]
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("metrics optimizer time must be finite and positive")
            if "invocation_id" in row and (not isinstance(row["invocation_id"], str) or not row["invocation_id"]):
                raise ValueError("metrics invocation identity is malformed")
            previous = assert_update_record(record, schedule=schedule, plan=plan, previous=previous)
            if index + 1 == completed_updates:
                retained = record
            rows.append(row)
        if type(completed_updates) is not int or completed_updates < 0 or completed_updates > len(rows):
            raise ValueError("metrics prefix is missing updates required by checkpoint")
        if resume_checkpoint is None and rows:
            raise ValueError("existing metrics require an explicit checkpoint resume")
        if checkpoint_cursor is not None and checkpoint_cursor != completed_updates * plan.sequences_per_update:
            raise ValueError("restored checkpoint cursor disagrees with completed updates")
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise TrainingIntegrityError(f"METRIC_LEDGER_INVALID: {exc}") from exc

    archive_name = None
    if len(rows) > completed_updates:
        source_hash = hashlib.sha256(raw).hexdigest()
        archive_name = f"{source_hash}-after-{completed_updates}.json"
        archive = {
            "schema": "superseded_metric_segment_v1", "run_id": run_id,
            "segment_id": archive_name[:-5], "source_metrics_sha256": source_hash,
            "restored_completed_updates": completed_updates,
            "resume_checkpoint": str(resume_checkpoint),
            "source_invocation_ids": sorted({row.get("invocation_id", f"legacy-{source_hash}") for row in rows[completed_updates:]}),
            "superseded_optimizer_seconds": sum(row["step_seconds"] for row in rows[completed_updates:]),
            "superseded_loss_tokens": (len(rows) - completed_updates) * plan.loss_tokens_per_update,
            "records": rows[completed_updates:],
        }
        archive_dir = run_dir / "superseded_metrics"
        archive_dir.mkdir(exist_ok=True)
        if any(item.name != archive_name for item in archive_dir.glob(f"{source_hash}-after-*.json")):
            raise TrainingIntegrityError(
                "METRIC_LEDGER_INVALID: finish the previously archived rollback boundary "
                "before selecting a different checkpoint"
            )
        encoded = (json.dumps(archive, sort_keys=True, indent=2) + "\n").encode()
        archive_path = archive_dir / archive_name
        if archive_path.exists():
            if archive_path.read_bytes() != encoded:
                raise TrainingIntegrityError("METRIC_LEDGER_INVALID: existing segment content disagrees")
        else:
            _atomic_write(archive_path, encoded)
        _atomic_write(path, b"".join(lines[:completed_updates]))
    elif raw and not raw.endswith(b"\n"):
        # A complete JSON last row without its delimiter is valid, but cannot be appended to.
        _atomic_write(path, raw + b"\n")

    invocation_id = uuid.uuid4().hex
    invocation_dir = run_dir / "metric_invocations"
    invocation_dir.mkdir(exist_ok=True)
    invocation = {
        "schema": "metric_invocation_v1", "invocation_id": invocation_id,
        "run_id": run_id, "started_at_update": completed_updates,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "superseded_segment": archive_name,
    }
    _atomic_write(invocation_dir / f"{invocation_id}.json", (json.dumps(invocation, sort_keys=True) + "\n").encode())
    return retained, invocation_id
