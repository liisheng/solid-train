"""Durable, verified, schedule-aware checkpoints.

Plan Section 7.2 is explicit about what a checkpoint is: *"Save at accumulation boundaries:
model, optimizer, scaler, all RNG states, schedule cursor and schedule hash, counters, run
ID, config hashes, and best validation state. Write to a temporary file, flush and fsync,
checksum, load-test, then rename."* Plan Section 9 adds the retention rules, and Plan
Section 13 G2 makes a verified resume a gate rather than a habit.

:mod:`tinybench_lm.checkpoint` already writes a format-v2 payload whose RNG and sampler
resume is exact and evidenced, but three things were missing and each one is a way for a
checkpoint to be *acknowledged* without being durable or complete:

1. **State.** The FP16 fallback's ``GradScaler`` state, the schedule content hash and integer
   cursor, the token/update counters, the run ID, the frozen config hashes, and the
   best-validation state were never persisted. A resume therefore restored a model but not
   the lineage: the scale factor restarted, the consumed mixture was recoverable only from a
   bit-generator blob, and nothing could prove the resumed run was still the same run.
2. **Durability.** ``torch.save`` to a temporary path followed by ``replace`` publishes an
   atomically-named file, but the bytes were never flushed to the device, never checksummed,
   and never independently reloaded. A truncated write became the newest "good" checkpoint.
3. **Retention.** Nothing distinguished a rolling recovery state from a branch parent, a
   selected endpoint, or the release fallback, so retention was a manual judgement call.

This module is the replacement writer, backed by one frozen config::

    configs/training/checkpoint_v1.yaml

A durable payload is a strict **superset** of format v2, so the proven
:func:`tinybench_lm.checkpoint.restore_checkpoint_state` path still restores it and the
existing pilot resume evidence stays valid.

Guarantees, mirroring :mod:`tinybench_lm.data_protocols`, :mod:`tinybench_lm.schedule`, and
:mod:`tinybench_lm.training_recipe`:

1. **Immutable.** The config is verified against a pinned SHA-256 digest
   (:data:`FROZEN_CHECKPOINT_PROTOCOL_SHA256`) on every load.
2. **Boundary-only.** A payload whose counters describe a partially accumulated update is
   rejected before a single byte is written.
3. **Durable in a fixed order.** temporary file, flush, ``fsync``, checksum, independent load
   test, rename. The load test reopens the temporary file and re-derives the weight hash and
   the counters, so a partial write fails *before* the rename publishes it.
4. **Verified before resume.** Checksum, byte size, payload completeness, counter arithmetic,
   accumulation boundary, weight hash, frozen config hashes, run ID, and schedule binding are
   all checked first. Corruption and drift fail closed.
5. **Retention that cannot lose evidence.** Protected roles are never deleted, an unverified
   copy is never deleted, and deletion needs a verified successor. Remote copies are never
   deleted by this repository at all.
6. **Absence of evidence is never PASS.** No final-scale checkpoint exists, so its measured
   write time and size stay ``NOT_RUN``, and remote-copy verification stays an operator
   action reported as ``NOT_RUN`` rather than a pass.

Nothing here starts training, measures checkpoint wall time, or verifies a remote copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    capture_rng_state,
    restore_checkpoint_state,
)
from .config import ModelConfig
from .data import TrainingSource
from .data_protocols import ProtocolError, ProtocolNotReadyError, load_protocol
from .environment import CheckResult
from .model import TinyBenchLM
from .provenance import file_sha256, state_dict_sha256
from .schedule import CURSOR_STATE_KEY, canonical_payload_bytes
from .shards import DEFERRED, FAIL, NOT_RUN, PASS
from .training_recipe import (
    TRAINING_RECIPE_DIR,
    BatchPlan,
    UpdateRecord,
    assert_no_hash_drift,
    recipe_digest,
)

CHECKPOINT_PROTOCOL_PATH = TRAINING_RECIPE_DIR / "checkpoint_v1.yaml"

#: SHA-256 of the frozen checkpoint contract, over file bytes with CRLF normalized to LF.
FROZEN_CHECKPOINT_PROTOCOL_SHA256: Mapping[str, str] = {
    "checkpoint_v1.yaml": "7d2c33ee9743cda80380a59dfc4dcd0eb477a20a05734332aec5934933639bc1",
}

#: Version of the durable payload envelope. The base ``checkpoint_format_version`` stays 2.
DURABLE_CHECKPOINT_FORMAT_VERSION = 1

#: Version of the sidecar manifest schema.
CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1

TEMPORARY_SUFFIX = ".tmp"
MANIFEST_SUFFIX = ".manifest.json"

#: Schedule-hash value recorded when the pilot random flat-stream sampler produced the batches.
PILOT_SCHEDULE_HASH_SENTINEL = "PILOT_ONLY_NO_SCHEDULE"

#: Status recorded for a remote copy nobody has verified. Verification is an operator action.
REMOTE_COPY_NOT_VERIFIED = "NOT_RUN: off-machine copy verification is an operator action"

# --------------------------------------------------------------------------------------
# Reason-code vocabulary
# --------------------------------------------------------------------------------------

CHECKPOINT_OK = "CHECKPOINT_OK"
CHECKPOINT_CHECKSUM_MISMATCH = "CHECKPOINT_CHECKSUM_MISMATCH"
CHECKPOINT_CORRUPT = "CHECKPOINT_CORRUPT"
CHECKPOINT_COUNTER_IMPOSSIBLE = "CHECKPOINT_COUNTER_IMPOSSIBLE"
CHECKPOINT_FROZEN_ARTIFACT_MISMATCH = "CHECKPOINT_FROZEN_ARTIFACT_MISMATCH"
CHECKPOINT_INCOMPLETE_STATE = "CHECKPOINT_INCOMPLETE_STATE"
CHECKPOINT_LOAD_TEST_FAILED = "CHECKPOINT_LOAD_TEST_FAILED"
CHECKPOINT_MANIFEST_MISSING = "CHECKPOINT_MANIFEST_MISSING"
CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY = "CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY"
CHECKPOINT_RETENTION_UNSAFE_DELETE = "CHECKPOINT_RETENTION_UNSAFE_DELETE"
CHECKPOINT_RETENTION_UNVERIFIED_RETAINED = "CHECKPOINT_RETENTION_UNVERIFIED_RETAINED"
CHECKPOINT_RUN_ID_MISMATCH = "CHECKPOINT_RUN_ID_MISMATCH"
CHECKPOINT_SCHEDULE_BINDING_MISMATCH = "CHECKPOINT_SCHEDULE_BINDING_MISMATCH"

CHECKPOINT_FAIL_CLOSED_REASON_CODES: frozenset[str] = frozenset(
    {
        CHECKPOINT_CHECKSUM_MISMATCH,
        CHECKPOINT_CORRUPT,
        CHECKPOINT_COUNTER_IMPOSSIBLE,
        CHECKPOINT_FROZEN_ARTIFACT_MISMATCH,
        CHECKPOINT_INCOMPLETE_STATE,
        CHECKPOINT_LOAD_TEST_FAILED,
        CHECKPOINT_MANIFEST_MISSING,
        CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY,
        CHECKPOINT_RETENTION_UNSAFE_DELETE,
        CHECKPOINT_RUN_ID_MISMATCH,
        CHECKPOINT_SCHEDULE_BINDING_MISMATCH,
    }
)

# --------------------------------------------------------------------------------------
# Retention roles (Plan Section 9)
# --------------------------------------------------------------------------------------

ROLE_RECOVERY = "recovery"
ROLE_BRANCH_PARENT = "branch_parent"
ROLE_SELECTED_ENDPOINT = "selected_endpoint"
ROLE_FALLBACK = "fallback"
CHECKPOINT_ROLES: tuple[str, ...] = (
    ROLE_RECOVERY,
    ROLE_BRANCH_PARENT,
    ROLE_SELECTED_ENDPOINT,
    ROLE_FALLBACK,
)
PROTECTED_ROLES: frozenset[str] = frozenset(
    {ROLE_BRANCH_PARENT, ROLE_SELECTED_ENDPOINT, ROLE_FALLBACK}
)

KEEP = "KEEP"
DELETE = "DELETE"


class CheckpointContractError(ProtocolError):
    """The frozen checkpoint contract is malformed, or a payload violates it."""


class CheckpointIntegrityError(CheckpointContractError):
    """A checkpoint is corrupt, incomplete, or does not belong to the lineage resuming it."""


class CheckpointsNotReadyError(ProtocolNotReadyError):
    """A final-scale checkpoint claim needs a measurement or an operator action that does not exist."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def load_checkpoint_protocol(
    path: Path = CHECKPOINT_PROTOCOL_PATH, *, verify: bool = True
) -> dict[str, Any]:
    """Load the frozen checkpoint contract, verifying its pinned digest by default."""
    protocol = load_protocol(path, verify=verify, registry=FROZEN_CHECKPOINT_PROTOCOL_SHA256)
    for section in ("boundary", "payload", "durability", "integrity", "retention", "fail_closed", "readiness"):
        if section not in protocol:
            raise CheckpointContractError(f"checkpoint protocol is missing required section {section!r}")

    boundary = protocol["boundary"]
    if not bool(boundary["save_only_at_accumulation_boundaries"]):
        raise CheckpointContractError(
            f"{CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY}: Plan Section 7.2 saves only at "
            "accumulation boundaries"
        )
    if not bool(boundary["microbatches_completed_in_update_must_be_zero"]):
        raise CheckpointContractError(
            f"{CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY}: a legal save point has no partially "
            "accumulated update"
        )

    payload = protocol["payload"]
    if int(payload["base_format_version"]) != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointContractError(
            f"the contract declares base format v{payload['base_format_version']}, the "
            f"implemented base format is v{CHECKPOINT_FORMAT_VERSION}"
        )
    if int(payload["durable_format_version"]) != DURABLE_CHECKPOINT_FORMAT_VERSION:
        raise CheckpointContractError(
            f"the contract declares durable format v{payload['durable_format_version']}, the "
            f"implementation writes v{DURABLE_CHECKPOINT_FORMAT_VERSION}"
        )
    if not bool(payload["base_format_superset"]):
        raise CheckpointContractError(
            "a durable payload must stay a superset of format v2 so the proven resume path keeps working"
        )
    if str(payload["pilot_schedule_hash_sentinel"]) != PILOT_SCHEDULE_HASH_SENTINEL:
        raise CheckpointContractError(
            f"the contract names the pilot schedule sentinel "
            f"{payload['pilot_schedule_hash_sentinel']!r}, expected {PILOT_SCHEDULE_HASH_SENTINEL!r}"
        )

    durability = protocol["durability"]
    expected_order = (
        "write_temporary_file",
        "flush",
        "fsync",
        "checksum",
        "independent_load_test",
        "rename",
    )
    if tuple(str(step) for step in durability["write_order"]) != expected_order:
        raise CheckpointContractError(
            f"the frozen write order must be exactly {expected_order}, found {tuple(durability['write_order'])}"
        )
    if str(durability["checksum_algorithm"]) != "sha256":
        raise CheckpointContractError("the frozen checksum algorithm is sha256")
    if not bool(durability["sidecar_manifest_required"]):
        raise CheckpointContractError(
            f"{CHECKPOINT_MANIFEST_MISSING}: a checkpoint must carry an independent checksum manifest"
        )

    integrity = protocol["integrity"]
    for flag in (
        "verify_before_resume",
        "verify_checksum_against_manifest",
        "verify_payload_completeness",
        "verify_counter_arithmetic",
        "verify_accumulation_boundary",
        "verify_frozen_config_hashes",
        "verify_run_id",
    ):
        if not bool(integrity[flag]):
            raise CheckpointContractError(f"the frozen integrity contract must enable {flag}")

    retention = protocol["retention"]
    if int(retention["keep_latest_recovery_states"]) < 1:
        raise CheckpointContractError("retention must keep at least one recovery state")
    declared_roles = tuple(str(role["role"]) for role in retention["roles"])
    if tuple(sorted(declared_roles)) != tuple(sorted(CHECKPOINT_ROLES)):
        raise CheckpointContractError(
            f"the contract must declare exactly the roles {sorted(CHECKPOINT_ROLES)}, found {sorted(declared_roles)}"
        )
    protected = {str(role["role"]) for role in retention["roles"] if bool(role["protected"])}
    if protected != set(PROTECTED_ROLES):
        raise CheckpointContractError(
            f"the protected roles must be exactly {sorted(PROTECTED_ROLES)}, found {sorted(protected)}"
        )
    for flag in (
        "never_delete_unverified_local_copy",
        "never_delete_unverified_remote_copy",
        "deletion_requires_verified_self",
        "deletion_requires_verified_successor",
        "remote_copies_are_never_deleted_by_this_repository",
        "plan_only_no_automatic_deletion",
    ):
        if not bool(retention[flag]):
            raise CheckpointContractError(f"the frozen retention contract must enable {flag}")

    if not bool(protocol["fail_closed"]["acknowledge_only_after_verified_rename"]):
        raise CheckpointContractError(
            "a checkpoint may only be acknowledged after a verified, renamed artifact exists"
        )
    return protocol


def _resolved(protocol: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return protocol if protocol is not None else load_checkpoint_protocol()


def checkpoint_protocol_digest(protocol: Mapping[str, Any] | None = None) -> str:
    """The frozen contract's own digest, recorded inside every payload it governs."""
    return str(_resolved(protocol)["_digest"])


def keep_latest_recovery_states(protocol: Mapping[str, Any] | None = None) -> int:
    """Plan Section 9's rolling recovery depth (three)."""
    return int(_resolved(protocol)["retention"]["keep_latest_recovery_states"])


def readiness_results(protocol: Mapping[str, Any] | None = None) -> tuple[CheckResult, ...]:
    """Reportable readiness statuses. An absent measurement or operator action is never PASS."""
    readiness = _resolved(protocol)["readiness"]
    results: list[CheckResult] = []
    for name in (
        "measured_checkpoint_write_seconds",
        "measured_checkpoint_size_bytes",
        "verified_remote_copy",
    ):
        status = str(readiness.get(name, NOT_RUN))
        results.append(
            CheckResult(
                f"checkpoint.readiness.{name}",
                "measured evidence",
                status,
                status if status in {PASS, FAIL, DEFERRED, NOT_RUN} else FAIL,
                str(readiness.get("next_action", "")) if status != PASS else "measured",
            )
        )
    return tuple(results)


def assert_ready_for_final_scale_claim(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: a final-scale checkpoint claim needs measurements that do not exist yet."""
    readiness = _resolved(protocol)["readiness"]
    blocked = [
        name
        for name in (
            "measured_checkpoint_write_seconds",
            "measured_checkpoint_size_bytes",
            "verified_remote_copy",
        )
        if str(readiness.get(name)) != PASS
    ]
    if blocked:
        raise CheckpointsNotReadyError(
            f"final-scale checkpoint evidence does not exist yet: {blocked}. "
            f"blocker={readiness.get('blocker')} owner={readiness.get('owner')} "
            f"next_action={readiness.get('next_action')}"
        )


# --------------------------------------------------------------------------------------
# Counters (Plan Section 7.2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointCounters:
    """Exact token/update counters for a completed update at an accumulation boundary."""

    update_index: int
    updates_completed: int
    consumed_loss_tokens: int
    loss_tokens_per_update: int
    gradient_accumulation: int
    microbatches_completed_in_update: int = 0

    def violations(self) -> tuple[str, ...]:
        """Reason-coded report. Empty means the counters reconcile exactly."""
        problems: list[str] = []
        index = int(self.update_index)
        completed = int(self.updates_completed)
        per_update = int(self.loss_tokens_per_update)
        accumulation = int(self.gradient_accumulation)
        pending = int(self.microbatches_completed_in_update)
        if index < 0:
            problems.append(f"{CHECKPOINT_COUNTER_IMPOSSIBLE}: update_index {index} is negative")
        if per_update < 1:
            problems.append(
                f"{CHECKPOINT_COUNTER_IMPOSSIBLE}: loss_tokens_per_update {per_update} must be positive"
            )
        if accumulation < 1:
            problems.append(
                f"{CHECKPOINT_COUNTER_IMPOSSIBLE}: gradient_accumulation {accumulation} must be positive"
            )
        if completed != index + 1:
            problems.append(
                f"{CHECKPOINT_COUNTER_IMPOSSIBLE}: updates_completed {completed} does not equal "
                f"update_index + 1 = {index + 1}"
            )
        expected_tokens = completed * per_update
        if int(self.consumed_loss_tokens) != expected_tokens:
            problems.append(
                f"{CHECKPOINT_COUNTER_IMPOSSIBLE}: consumed_loss_tokens "
                f"{self.consumed_loss_tokens} does not equal {completed} x {per_update} = {expected_tokens}"
            )
        if pending != 0:
            problems.append(
                f"{CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY}: {pending} microbatches of the next "
                "update are already accumulated, so this is not a save point"
            )
        return tuple(problems)

    @property
    def at_accumulation_boundary(self) -> bool:
        return int(self.microbatches_completed_in_update) == 0

    @property
    def next_update_index(self) -> int:
        """The update a resume from this checkpoint must run first."""
        return int(self.update_index) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_index": int(self.update_index),
            "updates_completed": int(self.updates_completed),
            "consumed_loss_tokens": int(self.consumed_loss_tokens),
            "loss_tokens_per_update": int(self.loss_tokens_per_update),
            "gradient_accumulation": int(self.gradient_accumulation),
            "microbatches_completed_in_update": int(self.microbatches_completed_in_update),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointCounters":
        missing = sorted(
            {
                "update_index",
                "updates_completed",
                "consumed_loss_tokens",
                "loss_tokens_per_update",
                "gradient_accumulation",
                "microbatches_completed_in_update",
            }
            - set(payload)
        )
        if missing:
            raise CheckpointContractError(
                f"{CHECKPOINT_INCOMPLETE_STATE}: counters are missing {missing}"
            )
        return cls(
            update_index=int(payload["update_index"]),
            updates_completed=int(payload["updates_completed"]),
            consumed_loss_tokens=int(payload["consumed_loss_tokens"]),
            loss_tokens_per_update=int(payload["loss_tokens_per_update"]),
            gradient_accumulation=int(payload["gradient_accumulation"]),
            microbatches_completed_in_update=int(payload["microbatches_completed_in_update"]),
        )

    @classmethod
    def at_update(
        cls, update_index: int, plan: BatchPlan, *, microbatches_completed_in_update: int = 0
    ) -> "CheckpointCounters":
        """Derive the counters from the batch plan, so they are computed and not typed."""
        index = int(update_index)
        completed = index + 1
        return cls(
            update_index=index,
            updates_completed=completed,
            consumed_loss_tokens=plan.consumed_loss_tokens(completed),
            loss_tokens_per_update=plan.loss_tokens_per_update,
            gradient_accumulation=int(plan.gradient_accumulation),
            microbatches_completed_in_update=int(microbatches_completed_in_update),
        )

    @classmethod
    def from_update_record(cls, record: UpdateRecord, plan: BatchPlan) -> "CheckpointCounters":
        """Reuse the audited per-update record's counters rather than recomputing them loosely."""
        counters = cls(
            update_index=int(record.update_index),
            updates_completed=int(record.update_index) + 1,
            consumed_loss_tokens=int(record.consumed_loss_tokens),
            loss_tokens_per_update=int(record.loss_tokens_per_update),
            gradient_accumulation=int(plan.gradient_accumulation),
        )
        problems = counters.violations()
        if problems:
            raise CheckpointContractError("; ".join(problems))
        return counters


def assert_accumulation_boundary(
    microbatches_completed_in_update: int, gradient_accumulation: int
) -> None:
    """Fail closed unless the accumulation window is closed and the optimizer step is done."""
    pending = int(microbatches_completed_in_update)
    accumulation = int(gradient_accumulation)
    if accumulation < 1:
        raise CheckpointContractError(
            f"{CHECKPOINT_COUNTER_IMPOSSIBLE}: gradient_accumulation {accumulation} must be positive"
        )
    if pending < 0 or pending >= accumulation:
        raise CheckpointContractError(
            f"{CHECKPOINT_COUNTER_IMPOSSIBLE}: {pending} accumulated microbatches is outside "
            f"[0, {accumulation})"
        )
    if pending != 0:
        raise CheckpointContractError(
            f"{CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY}: {pending} of {accumulation} microbatches "
            "are accumulated; a checkpoint may only be written after a completed optimizer step"
        )


# --------------------------------------------------------------------------------------
# Best-validation state (Plan Section 7.2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BestValidationState:
    """The best validation result the lineage has seen, and where its weights live."""

    loss: float
    update_index: int | None = None
    relative_path: str | None = None
    weight_sha256: str | None = None

    @property
    def evaluated(self) -> bool:
        return self.update_index is not None

    def violations(self) -> tuple[str, ...]:
        problems: list[str] = []
        value = float(self.loss)
        if self.evaluated:
            if not math.isfinite(value):
                problems.append(
                    f"{CHECKPOINT_INCOMPLETE_STATE}: best validation loss is {value} but an "
                    "evaluation was recorded"
                )
            if int(self.update_index) < 0:
                problems.append(
                    f"{CHECKPOINT_COUNTER_IMPOSSIBLE}: best validation update index "
                    f"{self.update_index} is negative"
                )
        elif value != float("inf"):
            problems.append(
                f"{CHECKPOINT_INCOMPLETE_STATE}: no validation pass was recorded, so the best "
                f"loss must stay infinite, found {value}"
            )
        return tuple(problems)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": float(self.loss),
            "update_index": None if self.update_index is None else int(self.update_index),
            "relative_path": self.relative_path,
            "weight_sha256": self.weight_sha256,
            "evaluated": self.evaluated,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BestValidationState":
        return cls(
            loss=float(payload["loss"]),
            update_index=None if payload.get("update_index") is None else int(payload["update_index"]),
            relative_path=payload.get("relative_path"),
            weight_sha256=payload.get("weight_sha256"),
        )

    @classmethod
    def unevaluated(cls) -> "BestValidationState":
        return cls(loss=float("inf"))


# --------------------------------------------------------------------------------------
# Payload construction and completeness
# --------------------------------------------------------------------------------------


def required_payload_keys(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    return tuple(str(key) for key in _resolved(protocol)["payload"]["required_keys"])


def _scaler_state(scaler: Any) -> tuple[dict[str, Any] | None, bool]:
    """(state dict, enabled) for a ``GradScaler``, or ``(None, False)`` when there is none."""
    if scaler is None:
        return None, False
    enabled = bool(scaler.is_enabled()) if hasattr(scaler, "is_enabled") else True
    state = scaler.state_dict()
    return (dict(state) if state is not None else None), enabled


def build_checkpoint_payload(
    *,
    model: TinyBenchLM,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    args: argparse.Namespace | Mapping[str, Any],
    train_data: TrainingSource,
    validation_data: TrainingSource,
    counters: CheckpointCounters,
    run_id: str,
    frozen_config_hashes: Mapping[str, str],
    schedule_content_hash: str,
    best_validation: BestValidationState,
    scaler: Any = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the complete durable payload, refusing an illegal save point.

    The payload keeps every format-v2 key, so
    :func:`tinybench_lm.checkpoint.restore_checkpoint_state` still restores it unchanged, and
    adds the scaler, the schedule binding, the counters, the run ID, the frozen config hashes,
    and the best-validation state that Plan Section 7.2 requires.
    """
    resolved = _resolved(protocol)
    problems = list(counters.violations()) + list(best_validation.violations())
    if problems:
        raise CheckpointContractError("; ".join(problems))

    scaler_state, scaler_enabled = _scaler_state(scaler)
    if scaler_enabled and not scaler_state:
        raise CheckpointContractError(
            f"{CHECKPOINT_INCOMPLETE_STATE}: the GradScaler is enabled but carries no state to persist"
        )

    required_hashes = tuple(str(name) for name in resolved["payload"]["required_frozen_config_hashes"])
    missing = sorted(set(required_hashes) - set(frozen_config_hashes))
    if missing:
        raise CheckpointContractError(
            f"{CHECKPOINT_INCOMPLETE_STATE}: frozen config hashes are missing {missing}"
        )

    train_state = train_data.state_dict()
    cursor = train_state.get(CURSOR_STATE_KEY)
    if cursor is None and str(schedule_content_hash) != PILOT_SCHEDULE_HASH_SENTINEL:
        raise CheckpointContractError(
            f"{CHECKPOINT_SCHEDULE_BINDING_MISMATCH}: the batch source exposes no "
            f"{CURSOR_STATE_KEY!r}, so the schedule hash must be the pilot sentinel "
            f"{PILOT_SCHEDULE_HASH_SENTINEL!r}, not {schedule_content_hash!r}"
        )

    arguments = dict(args) if isinstance(args, Mapping) else vars(args)
    payload: dict[str, Any] = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "durable_checkpoint_format_version": DURABLE_CHECKPOINT_FORMAT_VERSION,
        "protocol_digest": checkpoint_protocol_digest(resolved),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler_state,
        "grad_scaler_enabled": scaler_enabled,
        "model_config": config.to_dict(),
        "training_args": {
            key: str(value) if isinstance(value, Path) else value for key, value in arguments.items()
        },
        "step": int(counters.update_index),
        "best_validation_loss": float(best_validation.loss),
        "best_validation_state": best_validation.to_dict(),
        "rng_state": capture_rng_state(),
        "data_rng_state": {
            "train": train_state,
            "validation": validation_data.state_dict(),
        },
        "schedule_content_hash": str(schedule_content_hash),
        "schedule_cursor": None if cursor is None else int(cursor),
        "counters": counters.to_dict(),
        "run_id": str(run_id),
        "frozen_config_hashes": {str(key): str(value) for key, value in sorted(frozen_config_hashes.items())},
        "weight_sha256": state_dict_sha256(model.state_dict()),
    }
    absent = [key for key in required_payload_keys(resolved) if key not in payload]
    if absent:  # pragma: no cover - the literal above covers every declared key
        raise CheckpointContractError(f"{CHECKPOINT_INCOMPLETE_STATE}: payload is missing {absent}")
    return payload


def payload_violations(
    payload: Mapping[str, Any], *, protocol: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Reason-coded completeness and arithmetic report for a durable payload."""
    resolved = _resolved(protocol)
    contract = resolved["payload"]
    problems: list[str] = []

    absent = [key for key in required_payload_keys(resolved) if key not in payload]
    if absent:
        problems.append(f"{CHECKPOINT_INCOMPLETE_STATE}: payload is missing {absent}")
        return tuple(problems)

    if int(payload["checkpoint_format_version"]) != CHECKPOINT_FORMAT_VERSION:
        problems.append(
            f"{CHECKPOINT_CORRUPT}: base format version is {payload['checkpoint_format_version']}, "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    if int(payload["durable_checkpoint_format_version"]) != DURABLE_CHECKPOINT_FORMAT_VERSION:
        problems.append(
            f"{CHECKPOINT_CORRUPT}: durable format version is "
            f"{payload['durable_checkpoint_format_version']}, expected {DURABLE_CHECKPOINT_FORMAT_VERSION}"
        )

    rng_state = payload["rng_state"]
    if not isinstance(rng_state, Mapping):
        problems.append(f"{CHECKPOINT_INCOMPLETE_STATE}: rng_state is not a mapping")
    else:
        missing_streams = [
            str(name) for name in contract["required_rng_streams"] if str(name) not in rng_state
        ]
        if missing_streams:
            problems.append(
                f"{CHECKPOINT_INCOMPLETE_STATE}: rng_state is missing {missing_streams}"
            )
        if (
            bool(contract["cuda_rng_required_when_available"])
            and torch.cuda.is_available()
            and "torch_cuda" not in rng_state
        ):
            problems.append(f"{CHECKPOINT_INCOMPLETE_STATE}: rng_state is missing torch_cuda")

    data_state = payload["data_rng_state"]
    if not isinstance(data_state, Mapping) or not {"train", "validation"} <= set(data_state):
        problems.append(
            f"{CHECKPOINT_INCOMPLETE_STATE}: data_rng_state must carry train and validation state"
        )

    try:
        counters = CheckpointCounters.from_dict(payload["counters"])
    except CheckpointContractError as error:
        problems.append(str(error))
        counters = None
    if counters is not None:
        problems.extend(counters.violations())
        if int(payload["step"]) != int(counters.update_index):
            problems.append(
                f"{CHECKPOINT_COUNTER_IMPOSSIBLE}: step {payload['step']} disagrees with counter "
                f"update_index {counters.update_index}"
            )

    try:
        best = BestValidationState.from_dict(payload["best_validation_state"])
    except (KeyError, TypeError, ValueError) as error:
        problems.append(f"{CHECKPOINT_INCOMPLETE_STATE}: best_validation_state is unreadable: {error}")
        best = None
    if best is not None:
        problems.extend(best.violations())
        recorded = float(payload["best_validation_loss"])
        if not (math.isinf(recorded) and math.isinf(float(best.loss))) and recorded != float(best.loss):
            problems.append(
                f"{CHECKPOINT_INCOMPLETE_STATE}: best_validation_loss {recorded} disagrees with "
                f"best_validation_state loss {best.loss}"
            )

    if bool(payload["grad_scaler_enabled"]) and bool(contract["scaler_state_required_when_enabled"]):
        scaler_state = payload["scaler"]
        if not isinstance(scaler_state, Mapping) or "scale" not in scaler_state:
            problems.append(
                f"{CHECKPOINT_INCOMPLETE_STATE}: the FP16 fallback's GradScaler is enabled but its "
                "state was not persisted"
            )

    cursor = payload["schedule_cursor"]
    schedule_hash = str(payload["schedule_content_hash"])
    if cursor is None:
        if not bool(contract["schedule_cursor_null_permitted_for_pilot_sampler"]):
            problems.append(f"{CHECKPOINT_INCOMPLETE_STATE}: schedule_cursor is null")
        elif schedule_hash != PILOT_SCHEDULE_HASH_SENTINEL:
            problems.append(
                f"{CHECKPOINT_SCHEDULE_BINDING_MISMATCH}: schedule_cursor is null but the payload "
                f"claims schedule {schedule_hash!r}"
            )
    else:
        if int(cursor) < 0:
            problems.append(f"{CHECKPOINT_COUNTER_IMPOSSIBLE}: schedule_cursor {cursor} is negative")
        if schedule_hash == PILOT_SCHEDULE_HASH_SENTINEL or not schedule_hash:
            problems.append(
                f"{CHECKPOINT_SCHEDULE_BINDING_MISMATCH}: a schedule cursor of {cursor} needs the "
                "schedule content hash it indexes"
            )
        train_state = payload["data_rng_state"]
        if isinstance(train_state, Mapping):
            inner = train_state.get("train")
            if isinstance(inner, Mapping) and CURSOR_STATE_KEY in inner:
                if int(inner[CURSOR_STATE_KEY]) != int(cursor):
                    problems.append(
                        f"{CHECKPOINT_SCHEDULE_BINDING_MISMATCH}: payload cursor {cursor} disagrees "
                        f"with the batch source cursor {inner[CURSOR_STATE_KEY]}"
                    )
                recorded_hash = str(inner.get("schedule_content_hash", ""))
                if recorded_hash and recorded_hash != schedule_hash:
                    problems.append(
                        f"{CHECKPOINT_SCHEDULE_BINDING_MISMATCH}: payload names schedule "
                        f"{schedule_hash!r}, the batch source state names {recorded_hash!r}"
                    )

    hashes = payload["frozen_config_hashes"]
    if not isinstance(hashes, Mapping):
        problems.append(f"{CHECKPOINT_INCOMPLETE_STATE}: frozen_config_hashes is not a mapping")
    else:
        missing_hashes = [
            str(name)
            for name in contract["required_frozen_config_hashes"]
            if not str(hashes.get(str(name), ""))
        ]
        if missing_hashes:
            problems.append(
                f"{CHECKPOINT_INCOMPLETE_STATE}: frozen_config_hashes is missing {missing_hashes}"
            )

    if not str(payload["run_id"]):
        problems.append(f"{CHECKPOINT_INCOMPLETE_STATE}: run_id is empty")

    state = payload["model"]
    if not isinstance(state, Mapping) or not state:
        problems.append(f"{CHECKPOINT_INCOMPLETE_STATE}: model state is absent")
    else:
        observed = state_dict_sha256(state)
        if observed != str(payload["weight_sha256"]):
            problems.append(
                f"{CHECKPOINT_CORRUPT}: model weights hash to {observed}, the payload records "
                f"{payload['weight_sha256']}"
            )
    optimizer_state = payload["optimizer"]
    if not isinstance(optimizer_state, Mapping) or "param_groups" not in optimizer_state:
        problems.append(f"{CHECKPOINT_INCOMPLETE_STATE}: optimizer state is absent")
    return tuple(problems)


def assert_payload_complete(
    payload: Mapping[str, Any], *, protocol: Mapping[str, Any] | None = None
) -> None:
    """Fail closed on any completeness, boundary, counter, or binding violation."""
    problems = payload_violations(payload, protocol=protocol)
    if problems:
        raise CheckpointIntegrityError("; ".join(problems))


# --------------------------------------------------------------------------------------
# Sidecar manifest
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointManifest:
    """Independent checksum/identity record written beside a checkpoint."""

    checkpoint_filename: str
    sha256: str
    bytes: int
    run_id: str
    update_index: int
    updates_completed: int
    consumed_loss_tokens: int
    loss_tokens_per_update: int
    schedule_content_hash: str
    schedule_cursor: int | None
    weight_sha256: str
    frozen_config_hashes: Mapping[str, str]
    best_validation_loss: float
    protocol_digest: str
    durable_checkpoint_format_version: int = DURABLE_CHECKPOINT_FORMAT_VERSION
    schema_version: int = CHECKPOINT_MANIFEST_SCHEMA_VERSION
    remote_copy_verified: str = REMOTE_COPY_NOT_VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "checkpoint_filename": self.checkpoint_filename,
            "sha256": self.sha256,
            "bytes": int(self.bytes),
            "run_id": self.run_id,
            "update_index": int(self.update_index),
            "updates_completed": int(self.updates_completed),
            "consumed_loss_tokens": int(self.consumed_loss_tokens),
            "loss_tokens_per_update": int(self.loss_tokens_per_update),
            "schedule_content_hash": self.schedule_content_hash,
            "schedule_cursor": None if self.schedule_cursor is None else int(self.schedule_cursor),
            "weight_sha256": self.weight_sha256,
            "frozen_config_hashes": dict(sorted(self.frozen_config_hashes.items())),
            "best_validation_loss": float(self.best_validation_loss),
            "protocol_digest": self.protocol_digest,
            "durable_checkpoint_format_version": int(self.durable_checkpoint_format_version),
            "remote_copy_verified": self.remote_copy_verified,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointManifest":
        return cls(
            checkpoint_filename=str(payload["checkpoint_filename"]),
            sha256=str(payload["sha256"]),
            bytes=int(payload["bytes"]),
            run_id=str(payload["run_id"]),
            update_index=int(payload["update_index"]),
            updates_completed=int(payload["updates_completed"]),
            consumed_loss_tokens=int(payload["consumed_loss_tokens"]),
            loss_tokens_per_update=int(payload["loss_tokens_per_update"]),
            schedule_content_hash=str(payload["schedule_content_hash"]),
            schedule_cursor=None
            if payload.get("schedule_cursor") is None
            else int(payload["schedule_cursor"]),
            weight_sha256=str(payload["weight_sha256"]),
            frozen_config_hashes={
                str(key): str(value) for key, value in dict(payload["frozen_config_hashes"]).items()
            },
            best_validation_loss=float(payload["best_validation_loss"]),
            protocol_digest=str(payload["protocol_digest"]),
            durable_checkpoint_format_version=int(payload["durable_checkpoint_format_version"]),
            schema_version=int(payload.get("schema_version", CHECKPOINT_MANIFEST_SCHEMA_VERSION)),
            remote_copy_verified=str(payload.get("remote_copy_verified", REMOTE_COPY_NOT_VERIFIED)),
        )


def manifest_path_for(path: str | Path) -> Path:
    """The sidecar manifest path for a checkpoint artifact."""
    target = Path(path)
    return target.with_name(target.name + MANIFEST_SUFFIX)


def read_manifest(path: str | Path) -> CheckpointManifest:
    """Read a sidecar manifest, failing closed when it is absent."""
    target = manifest_path_for(path)
    if not target.is_file():
        raise CheckpointIntegrityError(f"{CHECKPOINT_MANIFEST_MISSING}: {target} does not exist")
    return CheckpointManifest.from_dict(json.loads(target.read_text(encoding="utf-8")))


def _build_manifest(
    path: Path, payload: Mapping[str, Any], *, sha256: str, size: int
) -> CheckpointManifest:
    counters = CheckpointCounters.from_dict(payload["counters"])
    return CheckpointManifest(
        checkpoint_filename=path.name,
        sha256=sha256,
        bytes=int(size),
        run_id=str(payload["run_id"]),
        update_index=counters.update_index,
        updates_completed=counters.updates_completed,
        consumed_loss_tokens=counters.consumed_loss_tokens,
        loss_tokens_per_update=counters.loss_tokens_per_update,
        schedule_content_hash=str(payload["schedule_content_hash"]),
        schedule_cursor=payload["schedule_cursor"],
        weight_sha256=str(payload["weight_sha256"]),
        frozen_config_hashes=dict(payload["frozen_config_hashes"]),
        best_validation_loss=float(payload["best_validation_loss"]),
        protocol_digest=str(payload["protocol_digest"]),
    )


# --------------------------------------------------------------------------------------
# Durable write (Plan Section 7.2)
# --------------------------------------------------------------------------------------


def _write_and_sync(path: Path, writer) -> int:
    """Write through a handle, flush it, and fsync it. Returns the byte size."""
    with path.open("wb") as handle:
        writer(handle)
        handle.flush()
        os.fsync(handle.fileno())
    return path.stat().st_size


def _fsync_directory(directory: Path) -> bool:
    """Best-effort directory fsync. Windows does not support it; that is never a claim."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except (OSError, AttributeError):
        return False
    try:
        os.fsync(descriptor)
        return True
    except OSError:
        return False
    finally:
        os.close(descriptor)


def save_durable_checkpoint(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
) -> CheckpointManifest:
    """Write a checkpoint durably: temp file, flush, fsync, checksum, load test, rename.

    The independent load test reopens the *temporary* file, re-derives the weight hash, and
    re-runs the completeness checks, so a truncated or partially flushed write is caught
    before the rename makes it the newest acknowledged checkpoint. The sidecar manifest is
    published after the artifact: a crash between the two leaves a checkpoint with no
    manifest, which verification reports as unverified, and retention never deletes an
    unverified copy.
    """
    resolved = _resolved(protocol)
    assert_payload_complete(payload, protocol=resolved)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + TEMPORARY_SUFFIX)

    size = _write_and_sync(temporary, lambda handle: torch.save(dict(payload), handle))
    digest = file_sha256(temporary)

    try:
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
    except Exception as error:  # torch raises a variety of unpickling errors
        temporary.unlink(missing_ok=True)
        raise CheckpointIntegrityError(
            f"{CHECKPOINT_LOAD_TEST_FAILED}: {target.name} could not be reloaded after writing "
            f"({type(error).__name__}: {error})"
        ) from error
    problems = payload_violations(reloaded, protocol=resolved)
    if str(reloaded.get("weight_sha256", "")) != str(payload["weight_sha256"]):
        problems = (
            *problems,
            f"{CHECKPOINT_LOAD_TEST_FAILED}: reloaded weight hash "
            f"{reloaded.get('weight_sha256')} differs from the written {payload['weight_sha256']}",
        )
    if problems:
        temporary.unlink(missing_ok=True)
        raise CheckpointIntegrityError(f"{CHECKPOINT_LOAD_TEST_FAILED}: " + "; ".join(problems))

    manifest = _build_manifest(target, payload, sha256=digest, size=size)
    manifest_target = manifest_path_for(target)
    manifest_temporary = manifest_target.with_name(manifest_target.name + TEMPORARY_SUFFIX)
    _write_and_sync(
        manifest_temporary,
        lambda handle: handle.write(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8") + b"\n"
        ),
    )

    os.replace(temporary, target)
    os.replace(manifest_temporary, manifest_target)
    _fsync_directory(target.parent)
    return manifest


# --------------------------------------------------------------------------------------
# Verification before resume (Plan Sections 7.2, 15)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointReport:
    """A set of auditable checkpoint checks. ``NOT_RUN`` entries are never passes."""

    results: tuple[CheckResult, ...]

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status == FAIL)

    @property
    def ok(self) -> bool:
        return bool(self.results) and not self.failures

    def result(self, check_id: str) -> CheckResult:
        for candidate in self.results:
            if candidate.check_id == check_id:
                return candidate
        raise KeyError(check_id)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "results": [result.__dict__ for result in self.results]}


def _verdict(check_id: str, requirement: Any, observed: Any, ok: bool, reason: str) -> CheckResult:
    return CheckResult(check_id, str(requirement), str(observed), PASS if ok else FAIL, reason)


def verify_checkpoint(
    path: str | Path,
    *,
    expected_frozen_config_hashes: Mapping[str, str] | None = None,
    expected_run_id: str | None = None,
    expected_schedule_content_hash: str | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> CheckpointReport:
    """Re-derive every claim a checkpoint makes, before anything resumes from it."""
    resolved = _resolved(protocol)
    target = Path(path)
    results: list[CheckResult] = [
        _verdict(
            "checkpoint.file_present",
            "the checkpoint artifact exists",
            target if target.is_file() else "missing",
            target.is_file(),
            f"{CHECKPOINT_CORRUPT} unless the renamed artifact exists",
        )
    ]
    if not target.is_file():
        return CheckpointReport(tuple(results))

    manifest_target = manifest_path_for(target)
    results.append(
        _verdict(
            "checkpoint.manifest_present",
            "an independent checksum manifest",
            manifest_target.name if manifest_target.is_file() else "missing",
            manifest_target.is_file(),
            f"{CHECKPOINT_MANIFEST_MISSING} means the artifact is unverified, never that it is fine",
        )
    )
    manifest: CheckpointManifest | None = None
    if manifest_target.is_file():
        try:
            manifest = read_manifest(target)
        except (CheckpointIntegrityError, KeyError, ValueError) as error:
            results.append(
                _verdict(
                    "checkpoint.manifest_readable",
                    "a readable manifest",
                    f"<invalid: {type(error).__name__}>",
                    False,
                    f"{CHECKPOINT_MANIFEST_MISSING}: the manifest cannot be parsed",
                )
            )

    observed_size = target.stat().st_size
    observed_digest = file_sha256(target)
    if manifest is not None:
        results.append(
            _verdict(
                "checkpoint.byte_size",
                manifest.bytes,
                observed_size,
                observed_size == manifest.bytes,
                f"{CHECKPOINT_CORRUPT}: a truncated or extended artifact is not the one that was verified",
            )
        )
        results.append(
            _verdict(
                "checkpoint.checksum",
                manifest.sha256,
                observed_digest,
                observed_digest == manifest.sha256,
                f"{CHECKPOINT_CHECKSUM_MISMATCH}: the bytes changed after the verified write",
            )
        )

    try:
        payload = torch.load(target, map_location="cpu", weights_only=False)
    except Exception as error:
        results.append(
            _verdict(
                "checkpoint.loads",
                "the artifact reloads",
                f"<failed: {type(error).__name__}>",
                False,
                f"{CHECKPOINT_CORRUPT}: {error}",
            )
        )
        return CheckpointReport(tuple(results))
    results.append(
        _verdict("checkpoint.loads", "the artifact reloads", "loaded", True, "the artifact reloads on its own")
    )

    problems = payload_violations(payload, protocol=resolved)
    results.append(
        _verdict(
            "checkpoint.payload_complete",
            "every required state, counter, and binding",
            list(problems[:3]) or "complete",
            not problems,
            f"{CHECKPOINT_INCOMPLETE_STATE}/{CHECKPOINT_COUNTER_IMPOSSIBLE}/"
            f"{CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY} fail closed",
        )
    )

    counters = None
    if not problems:
        counters = CheckpointCounters.from_dict(payload["counters"])
        results.append(
            _verdict(
                "checkpoint.accumulation_boundary",
                "0 accumulated microbatches",
                counters.microbatches_completed_in_update,
                counters.at_accumulation_boundary,
                f"{CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY} unless the optimizer step completed",
            )
        )
        results.append(
            _verdict(
                "checkpoint.consumed_tokens",
                counters.updates_completed * counters.loss_tokens_per_update,
                counters.consumed_loss_tokens,
                True,
                "counters reconcile by exact integer arithmetic",
            )
        )

    if manifest is not None:
        results.append(
            _verdict(
                "checkpoint.manifest_agrees_with_payload",
                f"run {manifest.run_id} update {manifest.update_index} weights {manifest.weight_sha256[:12]}",
                f"run {payload.get('run_id')} update {payload.get('step')} weights {str(payload.get('weight_sha256'))[:12]}",
                manifest.run_id == str(payload.get("run_id"))
                and manifest.update_index == int(payload.get("step", -1))
                and manifest.weight_sha256 == str(payload.get("weight_sha256")),
                f"{CHECKPOINT_CORRUPT}: the manifest and the payload describe different artifacts",
            )
        )

    if expected_run_id is not None:
        observed_run_id = str(payload.get("run_id", ""))
        results.append(
            _verdict(
                "checkpoint.run_id",
                expected_run_id,
                observed_run_id,
                observed_run_id == str(expected_run_id),
                f"{CHECKPOINT_RUN_ID_MISMATCH}: resuming a different lineage would mutate it",
            )
        )
    if expected_frozen_config_hashes is not None:
        recorded = dict(payload.get("frozen_config_hashes", {}))
        drifted = {
            key: (value, recorded.get(key))
            for key, value in expected_frozen_config_hashes.items()
            if str(recorded.get(key)) != str(value)
        }
        results.append(
            _verdict(
                "checkpoint.frozen_config_hashes",
                dict(sorted(expected_frozen_config_hashes.items())),
                drifted or "identical",
                not drifted,
                f"{CHECKPOINT_FROZEN_ARTIFACT_MISMATCH}: a frozen artifact changed under this lineage",
            )
        )
    if expected_schedule_content_hash is not None:
        observed_hash = str(payload.get("schedule_content_hash", ""))
        results.append(
            _verdict(
                "checkpoint.schedule_binding",
                expected_schedule_content_hash,
                observed_hash,
                observed_hash == str(expected_schedule_content_hash),
                f"{CHECKPOINT_SCHEDULE_BINDING_MISMATCH}: exposure order would change silently",
            )
        )

    readiness = resolved["readiness"]
    results.append(
        CheckResult(
            "checkpoint.remote_copy_verified",
            "an off-machine copy with a verified checksum",
            str(readiness["verified_remote_copy"]),
            NOT_RUN,
            f"blocker={readiness['blocker']} owner={readiness['owner']} next_action={readiness['next_action']}",
        )
    )
    return CheckpointReport(tuple(results))


def load_verified_checkpoint(
    path: str | Path,
    *,
    expected_frozen_config_hashes: Mapping[str, str] | None = None,
    expected_run_id: str | None = None,
    expected_schedule_content_hash: str | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a checkpoint, then load it. Corruption and drift raise before any resume."""
    report = verify_checkpoint(
        path,
        expected_frozen_config_hashes=expected_frozen_config_hashes,
        expected_run_id=expected_run_id,
        expected_schedule_content_hash=expected_schedule_content_hash,
        protocol=protocol,
    )
    if not report.ok:
        detail = "; ".join(f"{result.check_id}: {result.observed} [{result.reason}]" for result in report.failures)
        raise CheckpointIntegrityError(f"refusing to resume from {Path(path).name}: {detail}")
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResumeState:
    """What a resume restored, and what the next update must be."""

    first_update_index: int
    best_validation_loss: float
    reproducible: bool
    scaler_restored: bool
    run_id: str | None = None
    schedule_content_hash: str | None = None
    schedule_cursor: int | None = None
    counters: CheckpointCounters | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_update_index": int(self.first_update_index),
            "best_validation_loss": float(self.best_validation_loss),
            "reproducible": bool(self.reproducible),
            "scaler_restored": bool(self.scaler_restored),
            "run_id": self.run_id,
            "schedule_content_hash": self.schedule_content_hash,
            "schedule_cursor": self.schedule_cursor,
            "counters": None if self.counters is None else self.counters.to_dict(),
        }


def restore_durable_state(
    payload: Mapping[str, Any],
    model: TinyBenchLM,
    optimizer: torch.optim.Optimizer,
    train_data: TrainingSource,
    validation_data: TrainingSource,
    *,
    scaler: Any = None,
) -> ResumeState:
    """Restore model, optimizer, scaler, RNG, batch source, counters, and lineage identity.

    The model/optimizer/RNG/batch-source restore delegates to the proven format-v2 path in
    :func:`tinybench_lm.checkpoint.restore_checkpoint_state`, so the exact RNG and sampler
    resume that already has evidence is reused rather than reimplemented. A legacy format-v2
    checkpoint therefore still resumes; it simply reports the durable fields it lacks.
    """
    first_step, best_validation_loss, reproducible = restore_checkpoint_state(
        dict(payload), model, optimizer, train_data, validation_data
    )

    scaler_restored = False
    scaler_state = payload.get("scaler")
    if scaler is not None and isinstance(scaler_state, Mapping) and scaler_state:
        if not hasattr(scaler, "is_enabled") or scaler.is_enabled():
            scaler.load_state_dict(dict(scaler_state))
            scaler_restored = True
    elif scaler is not None and bool(payload.get("grad_scaler_enabled", False)):
        raise CheckpointIntegrityError(
            f"{CHECKPOINT_INCOMPLETE_STATE}: the checkpoint records an enabled GradScaler but "
            "carries no scaler state to restore"
        )

    counters: CheckpointCounters | None = None
    if isinstance(payload.get("counters"), Mapping):
        counters = CheckpointCounters.from_dict(payload["counters"])
        problems = counters.violations()
        if problems:
            raise CheckpointIntegrityError("; ".join(problems))
        if counters.next_update_index != first_step:
            raise CheckpointIntegrityError(
                f"{CHECKPOINT_COUNTER_IMPOSSIBLE}: the counters place the next update at "
                f"{counters.next_update_index}, the restored step counter says {first_step}"
            )

    cursor = payload.get("schedule_cursor")
    return ResumeState(
        first_update_index=first_step,
        best_validation_loss=best_validation_loss,
        reproducible=reproducible,
        scaler_restored=scaler_restored,
        run_id=None if payload.get("run_id") is None else str(payload["run_id"]),
        schedule_content_hash=None
        if payload.get("schedule_content_hash") is None
        else str(payload["schedule_content_hash"]),
        schedule_cursor=None if cursor is None else int(cursor),
        counters=counters,
    )


def assert_resume_matches_frozen_artifacts(
    payload: Mapping[str, Any],
    expected_frozen_config_hashes: Mapping[str, str],
) -> None:
    """Fail closed when a frozen artifact drifted between the save and the resume."""
    assert_no_hash_drift(
        {str(key): str(value) for key, value in expected_frozen_config_hashes.items()},
        {str(key): str(value) for key, value in dict(payload.get("frozen_config_hashes", {})).items()},
    )


def frozen_config_hashes(
    *,
    model_config_hash: str,
    schedule_protocol_digest: str | None = None,
    recipe: Mapping[str, Any] | None = None,
    checkpoint_protocol: Mapping[str, Any] | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The frozen-artifact hashes a checkpoint binds itself to."""
    hashes: dict[str, str] = {
        "recipe_digest": recipe_digest(recipe),
        "model_config_hash": str(model_config_hash),
        "checkpoint_protocol_digest": checkpoint_protocol_digest(checkpoint_protocol),
    }
    if schedule_protocol_digest:
        hashes["schedule_protocol_digest"] = str(schedule_protocol_digest)
    if extra:
        hashes.update({str(key): str(value) for key, value in extra.items()})
    return dict(sorted(hashes.items()))


# --------------------------------------------------------------------------------------
# Retention (Plan Section 9)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointEntry:
    """One local checkpoint copy, its role, and whether anybody has verified it.

    ``remote_copy_verified`` is tri-state: ``True`` when an operator verified an off-machine
    checksum, ``False`` when a copy exists but failed or was never checked, and ``None`` when
    no remote copy is claimed at all. Nothing in this repository deletes a remote copy.
    """

    path: str
    role: str
    update_index: int
    verified: bool = False
    remote_copy_verified: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "update_index": int(self.update_index),
            "verified": bool(self.verified),
            "remote_copy_verified": self.remote_copy_verified,
        }


@dataclass(frozen=True)
class RetentionDecision:
    """What retention proposes for one copy, and the reason it can be audited by."""

    path: str
    role: str
    update_index: int
    action: str
    reason_code: str
    reason: str


@dataclass(frozen=True)
class RetentionPlan:
    """A proposal only. Nothing in this module deletes a file."""

    decisions: tuple[RetentionDecision, ...]
    keep_latest_recovery_states: int
    protocol_digest: str = ""

    @property
    def keep(self) -> tuple[RetentionDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.action == KEEP)

    @property
    def delete(self) -> tuple[RetentionDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.action == DELETE)

    @property
    def keep_paths(self) -> tuple[str, ...]:
        return tuple(decision.path for decision in self.keep)

    @property
    def delete_paths(self) -> tuple[str, ...]:
        return tuple(decision.path for decision in self.delete)

    def decision(self, path: str) -> RetentionDecision:
        for candidate in self.decisions:
            if candidate.path == path:
                return candidate
        raise KeyError(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep_latest_recovery_states": int(self.keep_latest_recovery_states),
            "protocol_digest": self.protocol_digest,
            "decisions": [decision.__dict__ for decision in self.decisions],
        }


def plan_retention(
    entries: Iterable[CheckpointEntry],
    *,
    keep_latest: int | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> RetentionPlan:
    """Propose which local copies to keep. Protected and unverified copies are never deleted.

    Rules, in order:

    1. A branch parent, a selected endpoint, or the fallback is protected and always kept: a
       branch parent is bound by an append-only hash and cannot be regenerated.
    2. An unverified local copy is kept and reported, because it may be the only intact one.
    3. The newest ``keep_latest`` recovery states are kept.
    4. A superseded recovery state is proposed for deletion only when it verified locally and
       at least one kept recovery state also verified.

    Remote copies are never proposed for deletion, and this function performs no deletion.
    """
    resolved = _resolved(protocol)
    depth = keep_latest_recovery_states(resolved) if keep_latest is None else int(keep_latest)
    if depth < 1:
        raise CheckpointContractError("retention must keep at least one recovery state")

    ordered = sorted(entries, key=lambda entry: (-int(entry.update_index), entry.path))
    unknown = sorted({entry.role for entry in ordered} - set(CHECKPOINT_ROLES))
    if unknown:
        raise CheckpointContractError(
            f"unknown checkpoint roles {unknown}; declare a role in configs/training/checkpoint_v1.yaml"
        )

    recovery = [entry for entry in ordered if entry.role == ROLE_RECOVERY]
    retained_recovery = recovery[:depth]
    retained_paths = {entry.path for entry in retained_recovery}
    has_verified_successor = any(entry.verified for entry in retained_recovery)

    decisions: list[RetentionDecision] = []
    for entry in ordered:
        if entry.role in PROTECTED_ROLES:
            decisions.append(
                RetentionDecision(
                    entry.path,
                    entry.role,
                    int(entry.update_index),
                    KEEP,
                    CHECKPOINT_OK,
                    f"{entry.role} is protected evidence and is never deleted",
                )
            )
            continue
        if entry.path in retained_paths:
            decisions.append(
                RetentionDecision(
                    entry.path,
                    entry.role,
                    int(entry.update_index),
                    KEEP,
                    CHECKPOINT_OK,
                    f"one of the latest {depth} recovery states",
                )
            )
            continue
        if not entry.verified:
            decisions.append(
                RetentionDecision(
                    entry.path,
                    entry.role,
                    int(entry.update_index),
                    KEEP,
                    CHECKPOINT_RETENTION_UNVERIFIED_RETAINED,
                    "an unverified copy may be the only intact one, so it is retained for review",
                )
            )
            continue
        if not has_verified_successor:
            decisions.append(
                RetentionDecision(
                    entry.path,
                    entry.role,
                    int(entry.update_index),
                    KEEP,
                    CHECKPOINT_RETENTION_UNVERIFIED_RETAINED,
                    "no retained recovery state has been verified, so nothing is superseded yet",
                )
            )
            continue
        decisions.append(
            RetentionDecision(
                entry.path,
                entry.role,
                int(entry.update_index),
                DELETE,
                CHECKPOINT_OK,
                f"verified recovery state superseded by the latest {depth} verified states",
            )
        )
    return RetentionPlan(tuple(decisions), depth, checkpoint_protocol_digest(resolved))


def retention_plan_violations(
    plan: RetentionPlan, entries: Iterable[CheckpointEntry]
) -> tuple[str, ...]:
    """Reason-coded report that a plan never proposes an unsafe deletion."""
    by_path = {entry.path: entry for entry in entries}
    verified_kept_recovery = any(
        by_path[decision.path].verified
        for decision in plan.keep
        if decision.path in by_path and by_path[decision.path].role == ROLE_RECOVERY
    )
    problems: list[str] = []
    for decision in plan.delete:
        entry = by_path.get(decision.path)
        if entry is None:
            problems.append(
                f"{CHECKPOINT_RETENTION_UNSAFE_DELETE}: {decision.path} is not in the inventory"
            )
            continue
        if entry.role in PROTECTED_ROLES:
            problems.append(
                f"{CHECKPOINT_RETENTION_UNSAFE_DELETE}: {entry.path} is a protected {entry.role}"
            )
        if not entry.verified:
            problems.append(
                f"{CHECKPOINT_RETENTION_UNSAFE_DELETE}: {entry.path} was never verified locally"
            )
        if not verified_kept_recovery:
            problems.append(
                f"{CHECKPOINT_RETENTION_UNSAFE_DELETE}: {entry.path} has no verified retained "
                "recovery state to supersede it"
            )
    return tuple(problems)


def assert_retention_plan_safe(plan: RetentionPlan, entries: Iterable[CheckpointEntry]) -> None:
    """Fail closed on any unsafe proposed deletion."""
    problems = retention_plan_violations(plan, entries)
    if problems:
        raise CheckpointContractError("; ".join(problems))


def inventory_from_directory(
    directory: str | Path,
    *,
    roles: Mapping[str, str] | None = None,
    default_role: str = ROLE_RECOVERY,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[CheckpointEntry, ...]:
    """Build an inventory of a run directory, verifying each checkpoint it finds.

    Verification is what makes a copy eligible for retention decisions at all, so it is done
    here rather than assumed: an artifact whose manifest is missing or whose checksum does not
    match is recorded as ``verified=False`` and can never be proposed for deletion.
    """
    root = Path(directory)
    entries: list[CheckpointEntry] = []
    for path in sorted(root.glob("*.pt")):
        report = verify_checkpoint(path, protocol=protocol)
        update_index = -1
        if report.ok:
            manifest = read_manifest(path)
            update_index = manifest.update_index
        else:
            manifest_target = manifest_path_for(path)
            if manifest_target.is_file():
                try:
                    update_index = read_manifest(path).update_index
                except (CheckpointIntegrityError, KeyError, ValueError):
                    update_index = -1
        entries.append(
            CheckpointEntry(
                path=path.name,
                role=str((roles or {}).get(path.name, default_role)),
                update_index=update_index,
                verified=report.ok,
            )
        )
    return tuple(entries)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def format_checkpoint_report(results: Sequence[CheckResult], title: str = "") -> str:
    """Human-readable summary of checkpoint verification."""
    width = max((len(result.check_id) for result in results), default=0)
    lines = [title] if title else []
    lines.extend(
        f"{result.status:<8} {result.check_id:<{width}}  {result.requirement} -> {result.observed}"
        for result in results
    )
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    failures = [result for result in results if result.status == FAIL]
    lines.append("")
    lines.append("Summary: " + ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))
    lines.append("RESULT: " + ("PASS" if not failures else "FAIL"))
    if failures:
        lines.append("Failures:")
        lines.extend(f"  {result.check_id}: {result.reason}" for result in failures)
    return "\n".join(lines)


def format_retention_plan(plan: RetentionPlan) -> str:
    """Human-readable retention proposal. Nothing here deletes a file."""
    width = max((len(decision.path) for decision in plan.decisions), default=0)
    lines = [f"Retention proposal (keep latest {plan.keep_latest_recovery_states} recovery states):"]
    lines.extend(
        f"{decision.action:<6} {decision.path:<{width}}  update={decision.update_index} "
        f"role={decision.role} [{decision.reason_code}] {decision.reason}"
        for decision in plan.decisions
    )
    lines.append(f"Keep: {len(plan.keep)}  Propose delete: {len(plan.delete)} (no file is deleted here)")
    return "\n".join(lines)


def checkpoint_payload_fingerprint(payload: Mapping[str, Any]) -> str:
    """Identity of a payload's non-tensor lineage fields, for logs and tests."""
    material = {
        "run_id": str(payload.get("run_id", "")),
        "counters": dict(payload.get("counters", {})),
        "schedule_content_hash": str(payload.get("schedule_content_hash", "")),
        "schedule_cursor": payload.get("schedule_cursor"),
        "weight_sha256": str(payload.get("weight_sha256", "")),
        "frozen_config_hashes": dict(payload.get("frozen_config_hashes", {})),
    }
    return hashlib.sha256(canonical_payload_bytes(material)).hexdigest()


__all__ = [
    "BestValidationState",
    "CHECKPOINT_CHECKSUM_MISMATCH",
    "CHECKPOINT_CORRUPT",
    "CHECKPOINT_COUNTER_IMPOSSIBLE",
    "CHECKPOINT_FAIL_CLOSED_REASON_CODES",
    "CHECKPOINT_FROZEN_ARTIFACT_MISMATCH",
    "CHECKPOINT_INCOMPLETE_STATE",
    "CHECKPOINT_LOAD_TEST_FAILED",
    "CHECKPOINT_MANIFEST_MISSING",
    "CHECKPOINT_NOT_AT_ACCUMULATION_BOUNDARY",
    "CHECKPOINT_OK",
    "CHECKPOINT_PROTOCOL_PATH",
    "CHECKPOINT_RETENTION_UNSAFE_DELETE",
    "CHECKPOINT_RETENTION_UNVERIFIED_RETAINED",
    "CHECKPOINT_ROLES",
    "CHECKPOINT_RUN_ID_MISMATCH",
    "CHECKPOINT_SCHEDULE_BINDING_MISMATCH",
    "CheckpointContractError",
    "CheckpointCounters",
    "CheckpointEntry",
    "CheckpointIntegrityError",
    "CheckpointManifest",
    "CheckpointReport",
    "CheckpointsNotReadyError",
    "DELETE",
    "DURABLE_CHECKPOINT_FORMAT_VERSION",
    "FROZEN_CHECKPOINT_PROTOCOL_SHA256",
    "KEEP",
    "MANIFEST_SUFFIX",
    "PILOT_SCHEDULE_HASH_SENTINEL",
    "PROTECTED_ROLES",
    "REMOTE_COPY_NOT_VERIFIED",
    "ROLE_BRANCH_PARENT",
    "ROLE_FALLBACK",
    "ROLE_RECOVERY",
    "ROLE_SELECTED_ENDPOINT",
    "ResumeState",
    "RetentionDecision",
    "RetentionPlan",
    "TEMPORARY_SUFFIX",
    "assert_accumulation_boundary",
    "assert_payload_complete",
    "assert_ready_for_final_scale_claim",
    "assert_resume_matches_frozen_artifacts",
    "assert_retention_plan_safe",
    "build_checkpoint_payload",
    "checkpoint_payload_fingerprint",
    "checkpoint_protocol_digest",
    "format_checkpoint_report",
    "format_retention_plan",
    "frozen_config_hashes",
    "inventory_from_directory",
    "keep_latest_recovery_states",
    "load_checkpoint_protocol",
    "load_verified_checkpoint",
    "manifest_path_for",
    "payload_violations",
    "plan_retention",
    "read_manifest",
    "readiness_results",
    "required_payload_keys",
    "restore_durable_state",
    "retention_plan_violations",
    "save_durable_checkpoint",
    "verify_checkpoint",
]
