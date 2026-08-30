"""Deterministic materialized mixture schedules and the one integer resume cursor.

Plan Section 5.4 is explicit: *"A mixture is a materialized index schedule, not a repacked
dataset. Each schedule contains deterministic ``(shard_id, token_offset, length)``
references and a content hash. Resume restores one integer schedule cursor. Use shard-level
shuffle, sequential reads inside shards, and a bounded local shuffle buffer to avoid
page-cache thrashing on the 16GB host."* Plan Section 7.2 requires the resume state to be
saved at accumulation boundaries, and Plan Section 8.3 hashes exposure lists of
``(doc_id, token_offset, length, multiplicity)`` -- both need an exposure order that is
reproducible from recorded state alone.

Random flat-stream sampling cannot satisfy any of that. :class:`~tinybench_lm.data.PackedTokenDataset`
draws uniform random offsets from one monolithic token file, so the consumed source mixture
is unknown, unreproducible across code changes, and unrecoverable from a checkpoint except
through a bit-generator blob. This module is the replacement for **final training**; the
pilot sampler stays available under an explicit ``PILOT_ONLY`` label for bounded smoke tests
and for the format-v2 resume evidence that already exists.

The contract is backed by one frozen config::

    configs/data/schedule_v1.yaml

Guarantees, mirroring :mod:`tinybench_lm.data_protocols`, :mod:`tinybench_lm.source_manifest`,
and :mod:`tinybench_lm.shards`:

1. **Immutable.** The config is verified against a pinned SHA-256 digest
   (:data:`FROZEN_SCHEDULE_PROTOCOL_SHA256`) on every load. Schedule records are frozen
   dataclasses in a tuple, and every persisted schedule carries a content hash over its
   canonical payload.
2. **Deterministic.** Ordering RNG is derived from the protocol digest, the split manifest's
   content hash, the split ID, the sequence length, and the seed -- never from global
   interpreter RNG state. Identical inputs and seed therefore produce byte-identical
   schedules.
3. **Bounded reads.** Entries inside a shard are non-overlapping and ascending; shard order
   is shuffled; a bounded local shuffle buffer of ``B`` sequences displaces any entry by at
   most ``B - 1`` positions, so reads stay sequential enough for the 16GB host.
4. **One integer cursor.** :class:`ScheduleCursor` state is one nonnegative integer bound to
   the schedule content hash. Schedule hash plus cursor completely determines the consumed
   training input, and a resume against a different schedule fails closed.
5. **Fail closed.** Out-of-bounds references, unknown shards, duplicate references, mixed
   boundaries, quota mismatches, unbounded displacement, and content-hash drift all raise
   rather than degrade into a pass.
6. **Absence of evidence is never PASS.** No schedule over real shards has been
   materialized; :func:`assert_ready_for_real_schedules` keeps that ``NOT_RUN`` explicit.

Nothing here acquires a corpus, produces shards, or starts training.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .data_protocols import DATA_PROTOCOL_DIR, ProtocolNotReadyError, load_protocol
from .environment import CheckResult
from .shards import (
    DEFERRED,
    FAIL,
    NOT_RUN,
    PASS,
    ShardContractError,
    ShardRecord,
    SplitManifest,
)
from .tokenizer import STORAGE_DTYPE

SCHEDULE_PROTOCOL_PATH = DATA_PROTOCOL_DIR / "schedule_v1.yaml"

#: SHA-256 of the frozen schedule contract, over file bytes with CRLF normalized to LF.
FROZEN_SCHEDULE_PROTOCOL_SHA256: Mapping[str, str] = {
    "schedule_v1.yaml": "af095d1bd39baee8d9af8b34bb4bbbaa2ce2949dead7a5f4f3650b00eaa5198a",
}

_SCHEDULE_SCHEMA_VERSION = "schedule_v1"
_CURSOR_STATE_VERSION = 1

#: The one key a checkpoint uses for the integer resume position (Plan Sections 5.4, 7.2).
CURSOR_STATE_KEY = "schedule_cursor"

# --------------------------------------------------------------------------------------
# Reason-code vocabulary
# --------------------------------------------------------------------------------------

SCHEDULE_OK = "SCHEDULE_OK"
SCHEDULE_CONTENT_HASH_MISMATCH = "SCHEDULE_CONTENT_HASH_MISMATCH"
SCHEDULE_CURSOR_OUT_OF_RANGE = "SCHEDULE_CURSOR_OUT_OF_RANGE"
SCHEDULE_REFERENCE_OUT_OF_BOUNDS = "SCHEDULE_REFERENCE_OUT_OF_BOUNDS"
SCHEDULE_SHARD_UNKNOWN = "SCHEDULE_SHARD_UNKNOWN"
SCHEDULE_DUPLICATE_REFERENCE = "SCHEDULE_DUPLICATE_REFERENCE"
SCHEDULE_BOUNDARY_MIXED = "SCHEDULE_BOUNDARY_MIXED"
SCHEDULE_SOURCE_QUOTA_MISMATCH = "SCHEDULE_SOURCE_QUOTA_MISMATCH"
SCHEDULE_SOURCE_SUPPLY_EXHAUSTED = "SCHEDULE_SOURCE_SUPPLY_EXHAUSTED"
SCHEDULE_LOCAL_SHUFFLE_UNBOUNDED = "SCHEDULE_LOCAL_SHUFFLE_UNBOUNDED"
SCHEDULE_EMPTY = "SCHEDULE_EMPTY"

SCHEDULE_FAIL_CLOSED_REASON_CODES: frozenset[str] = frozenset(
    {
        SCHEDULE_CONTENT_HASH_MISMATCH,
        SCHEDULE_CURSOR_OUT_OF_RANGE,
        SCHEDULE_REFERENCE_OUT_OF_BOUNDS,
        SCHEDULE_SHARD_UNKNOWN,
        SCHEDULE_DUPLICATE_REFERENCE,
        SCHEDULE_BOUNDARY_MIXED,
        SCHEDULE_SOURCE_QUOTA_MISMATCH,
        SCHEDULE_SOURCE_SUPPLY_EXHAUSTED,
        SCHEDULE_LOCAL_SHUFFLE_UNBOUNDED,
        SCHEDULE_EMPTY,
    }
)


class ScheduleContractError(ShardContractError):
    """The frozen schedule contract is malformed, or a schedule/cursor violates it."""


class ScheduleResumeError(ScheduleContractError):
    """A resume state does not belong to the schedule it is being restored into."""


class SchedulesNotReadyError(ProtocolNotReadyError):
    """Real-scale schedule construction is gated behind real shard production."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def load_schedule_protocol(
    path: Path = SCHEDULE_PROTOCOL_PATH, *, verify: bool = True
) -> dict[str, Any]:
    """Load the frozen schedule contract, verifying its pinned digest by default."""
    protocol = load_protocol(path, verify=verify, registry=FROZEN_SCHEDULE_PROTOCOL_SHA256)
    required = (
        "entry",
        "identity",
        "ordering",
        "cursor",
        "quotas",
        "bounds",
        "pilot_sampler",
        "final_training_reader",
        "readiness",
    )
    for section in required:
        if section not in protocol:
            raise ScheduleContractError(f"schedule protocol is missing required section {section!r}")
    entry = protocol["entry"]
    if tuple(str(name) for name in entry["reference_fields"]) != ("shard_id", "token_offset", "length"):
        raise ScheduleContractError(
            "the frozen entry reference must be exactly (shard_id, token_offset, length)"
        )
    if int(entry["label_shift"]) < 1:
        raise ScheduleContractError("label_shift must be at least 1 so every entry carries its final target")
    ordering = protocol["ordering"]
    for flag in ("shard_level_shuffle", "sequential_reads_inside_shard", "bounded_local_shuffle"):
        if not bool(ordering[flag]):
            raise ScheduleContractError(f"the frozen ordering contract must enable {flag}")
    cursor = protocol["cursor"]
    if str(cursor["state_key"]) != CURSOR_STATE_KEY:
        raise ScheduleContractError(
            f"schedule protocol names the cursor state key {cursor['state_key']!r}, expected {CURSOR_STATE_KEY!r}"
        )
    if not bool(cursor["binds_schedule_content_hash"]):
        raise ScheduleContractError("the cursor must be bound to the schedule content hash")
    if str(protocol["pilot_sampler"]["status"]) != "PILOT_ONLY":
        raise ScheduleContractError("the superseded random flat-stream sampler must stay labeled PILOT_ONLY")
    if bool(protocol["pilot_sampler"]["eligible_for_final_training"]):
        raise ScheduleContractError("the random flat-stream sampler may not be eligible for final training")
    return protocol


def assert_ready_for_real_schedules(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: a real-scale schedule needs real source-tagged shards first."""
    resolved = protocol or load_schedule_protocol()
    readiness = resolved["readiness"]
    blocked = [
        name
        for name in (
            "measured_real_shard_schedule",
            "measured_source_quota_reconciliation",
            "measured_dataloader_throughput",
        )
        if str(readiness.get(name)) != PASS
    ]
    if blocked:
        raise SchedulesNotReadyError(
            f"real-scale schedule construction is not ready: {blocked}. "
            f"blocker={readiness.get('blocker')} owner={readiness.get('owner')} "
            f"next_action={readiness.get('next_action')}"
        )


# --------------------------------------------------------------------------------------
# Immutable schedule records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleEntry:
    """One immutable ``(shard_id, token_offset, length)`` reference with its source tag."""

    shard_id: str
    token_offset: int
    length: int
    source_id: str
    namespace: str

    @property
    def reference(self) -> tuple[str, int, int]:
        """The exact reference tuple Plan Section 5.4 names."""
        return (self.shard_id, self.token_offset, self.length)

    @property
    def end_offset(self) -> int:
        return self.token_offset + self.length

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "token_offset": self.token_offset,
            "length": self.length,
            "source_id": self.source_id,
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScheduleEntry":
        return cls(
            shard_id=str(payload["shard_id"]),
            token_offset=int(payload["token_offset"]),
            length=int(payload["length"]),
            source_id=str(payload["source_id"]),
            namespace=str(payload["namespace"]),
        )


@dataclass(frozen=True)
class MaterializedSchedule:
    """An immutable materialized index schedule: references, source tags, content hash."""

    schedule_id: str
    split_id: str
    boundary: str
    sequence_length: int
    label_shift: int
    seed: int
    local_shuffle_buffer_sequences: int
    entries: tuple[ScheduleEntry, ...]
    manifest_content_hash: str
    protocol_digest: str
    requested_source_quotas: Mapping[str, int] = field(default_factory=dict)
    schema_version: str = _SCHEDULE_SCHEMA_VERSION

    # -- derived views ------------------------------------------------------------------

    @property
    def sequence_count(self) -> int:
        return len(self.entries)

    @property
    def tokens_per_entry(self) -> int:
        return self.sequence_length + self.label_shift

    @property
    def loss_tokens(self) -> int:
        """Tokens that contribute a next-token target across the whole schedule."""
        return self.sequence_length * len(self.entries)

    @property
    def sequences_per_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.source_id] = counts.get(entry.source_id, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def tokens_per_source(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for entry in self.entries:
            totals[entry.source_id] = totals.get(entry.source_id, 0) + entry.length
        return dict(sorted(totals.items()))

    @property
    def shard_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for entry in self.entries:
            seen.setdefault(entry.shard_id, None)
        return tuple(seen)

    # -- serialization -----------------------------------------------------------------

    def payload(self) -> dict[str, Any]:
        """Canonical payload the content hash is computed over (no hash inside)."""
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "split_id": self.split_id,
            "boundary": self.boundary,
            "sequence_length": self.sequence_length,
            "label_shift": self.label_shift,
            "seed": self.seed,
            "local_shuffle_buffer_sequences": self.local_shuffle_buffer_sequences,
            "manifest_content_hash": self.manifest_content_hash,
            "protocol_digest": self.protocol_digest,
            "requested_source_quotas": dict(sorted(self.requested_source_quotas.items())),
            "sequence_count": self.sequence_count,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def content_hash(self) -> str:
        """SHA-256 over the canonical payload: the schedule's identity."""
        return hashlib.sha256(canonical_payload_bytes(self.payload())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload()
        payload["content_hash"] = self.content_hash()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MaterializedSchedule":
        return cls(
            schedule_id=str(payload["schedule_id"]),
            split_id=str(payload["split_id"]),
            boundary=str(payload["boundary"]),
            sequence_length=int(payload["sequence_length"]),
            label_shift=int(payload["label_shift"]),
            seed=int(payload["seed"]),
            local_shuffle_buffer_sequences=int(payload["local_shuffle_buffer_sequences"]),
            entries=tuple(ScheduleEntry.from_dict(item) for item in payload["entries"]),
            manifest_content_hash=str(payload["manifest_content_hash"]),
            protocol_digest=str(payload["protocol_digest"]),
            requested_source_quotas={
                str(key): int(value) for key, value in dict(payload.get("requested_source_quotas", {})).items()
            },
            schema_version=str(payload.get("schema_version", _SCHEDULE_SCHEMA_VERSION)),
        )

    def cursor(self, position: int = 0) -> "ScheduleCursor":
        """A cursor bound to this schedule's content hash."""
        return ScheduleCursor(self.content_hash(), position, self.sequence_count)


def canonical_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    """Byte-identical serialization for identical payloads, independent of dict order."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


# --------------------------------------------------------------------------------------
# Deterministic construction
# --------------------------------------------------------------------------------------


def schedule_identity_seed(
    *,
    protocol_digest: str,
    manifest_content_hash: str,
    split_id: str,
    sequence_length: int,
    seed: int,
) -> int:
    """Derive the ordering RNG seed from the schedule identity, never from global state."""
    material = "|".join(
        [
            _SCHEDULE_SCHEMA_VERSION,
            str(protocol_digest),
            str(manifest_content_hash),
            str(split_id),
            str(int(sequence_length)),
            str(int(seed)),
        ]
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _shard_entries(record: ShardRecord, stride: int) -> tuple[ScheduleEntry, ...]:
    """Non-overlapping, ascending in-shard references of exactly ``stride`` tokens."""
    entries: list[ScheduleEntry] = []
    offset = 0
    while offset + stride <= record.token_count:
        entries.append(
            ScheduleEntry(
                shard_id=record.shard_id,
                token_offset=offset,
                length=stride,
                source_id=record.source_id,
                namespace=record.namespace,
            )
        )
        offset += stride
    return tuple(entries)


def _bounded_local_shuffle(
    stream: Sequence[ScheduleEntry], buffer_sequences: int, rng: np.random.Generator
) -> list[ScheduleEntry]:
    """Shuffle inside a bounded window of ``B`` sequences.

    The stream is filled into consecutive windows of ``B`` entries and each window is
    permuted independently, so an entry is displaced by **at most** ``B - 1`` positions from
    its sequential stream position. That hard bound is what makes "sequential reads inside
    shards" auditable: a reservoir buffer that evicts a random slot mixes more, but an entry
    can linger in it arbitrarily long, so displacement would only be bounded in expectation
    and an unbounded reordering could not be told apart from a legitimate one.

    ``B == 1`` reproduces the sequential stream exactly.
    """
    if buffer_sequences <= 1:
        return list(stream)
    output: list[ScheduleEntry] = []
    for start in range(0, len(stream), buffer_sequences):
        window = list(stream[start : start + buffer_sequences])
        order = rng.permutation(len(window))
        output.extend(window[int(index)] for index in order)
    return output


def build_materialized_schedule(
    manifest: SplitManifest,
    *,
    sequence_length: int,
    seed: int,
    source_sequence_quotas: Mapping[str, int] | None = None,
    local_shuffle_buffer_sequences: int | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> MaterializedSchedule:
    """Materialize one deterministic index schedule over a split's source-tagged shards.

    The construction is fully determined by ``(protocol digest, manifest content hash,
    split ID, sequence length, seed, quotas, buffer size)``:

    1. Enumerate non-overlapping ascending references inside every shard.
    2. Shuffle each source's shard order, then truncate to that source's quota, so a quota
       is not silently biased toward the first shard.
    3. Shuffle the resulting shard groups globally, so sources interleave while reads stay
       sequential inside a shard.
    4. Pass the stream through a bounded local shuffle buffer.

    ``source_sequence_quotas``, when supplied, must name every source in the manifest and
    is honoured exactly; an unmet quota fails closed rather than quietly shrinking.
    """
    resolved = protocol or load_schedule_protocol()
    ordering = resolved["ordering"]
    label_shift = int(resolved["entry"]["label_shift"])
    if int(sequence_length) < 1:
        raise ScheduleContractError(f"sequence_length must be at least 1, got {sequence_length}")
    stride = int(sequence_length) + label_shift

    buffer_sequences = int(
        ordering["default_local_shuffle_buffer_sequences"]
        if local_shuffle_buffer_sequences is None
        else local_shuffle_buffer_sequences
    )
    minimum = int(ordering["minimum_local_shuffle_buffer_sequences"])
    maximum = int(ordering["maximum_local_shuffle_buffer_sequences"])
    if not minimum <= buffer_sequences <= maximum:
        raise ScheduleContractError(
            f"local shuffle buffer {buffer_sequences} is outside the frozen bound [{minimum}, {maximum}]"
        )

    manifest_hash = manifest.content_hash()
    protocol_digest = str(resolved.get("_digest", ""))
    rng = np.random.default_rng(
        schedule_identity_seed(
            protocol_digest=protocol_digest,
            manifest_content_hash=manifest_hash,
            split_id=manifest.split_id,
            sequence_length=int(sequence_length),
            seed=int(seed),
        )
    )

    records = sorted(manifest.shards, key=lambda record: record.shard_id)
    if not records:
        raise ScheduleContractError(f"{SCHEDULE_EMPTY}: split {manifest.split_id!r} has no shards to schedule")
    for record in records:
        if record.boundary != manifest.boundary:
            raise ScheduleContractError(
                f"{SCHEDULE_BOUNDARY_MIXED}: shard {record.shard_id!r} declares boundary "
                f"{record.boundary!r} inside split {manifest.split_id!r} ({manifest.boundary!r})"
            )

    by_source: dict[str, list[ShardRecord]] = {}
    for record in records:
        by_source.setdefault(record.source_id, []).append(record)

    if source_sequence_quotas is not None:
        requested = {str(key): int(value) for key, value in source_sequence_quotas.items()}
        unknown = sorted(set(requested) - set(by_source))
        missing = sorted(set(by_source) - set(requested))
        if unknown or missing:
            raise ScheduleContractError(
                f"{SCHEDULE_SOURCE_QUOTA_MISMATCH}: quotas name unknown sources {unknown} "
                f"and omit manifest sources {missing}"
            )
        if any(value < 0 for value in requested.values()):
            raise ScheduleContractError(f"{SCHEDULE_SOURCE_QUOTA_MISMATCH}: quotas must be nonnegative")
    else:
        requested = {}

    # Steps 1-2: per-source shard-level shuffle, then exact quota truncation.
    groups: list[tuple[str, tuple[ScheduleEntry, ...]]] = []
    for source_id in sorted(by_source):
        shard_records = by_source[source_id]
        order = rng.permutation(len(shard_records)) if len(shard_records) > 1 else np.zeros(1, dtype=int)
        shuffled = [shard_records[int(index)] for index in order]
        supply = [(record.shard_id, _shard_entries(record, stride)) for record in shuffled]
        available = sum(len(entries) for _, entries in supply)
        if source_sequence_quotas is None:
            selected = supply
        else:
            quota = requested[source_id]
            if quota > available:
                raise ScheduleContractError(
                    f"{SCHEDULE_SOURCE_SUPPLY_EXHAUSTED}: source {source_id!r} can supply {available} "
                    f"sequences of {stride} tokens but {quota} were requested"
                )
            selected = []
            remaining = quota
            for shard_id, entries in supply:
                if remaining <= 0:
                    break
                take = min(remaining, len(entries))
                selected.append((shard_id, entries[:take]))
                remaining -= take
        groups.extend((shard_id, entries) for shard_id, entries in selected if entries)

    if not groups:
        raise ScheduleContractError(
            f"{SCHEDULE_EMPTY}: split {manifest.split_id!r} yielded no {stride}-token references"
        )

    # Step 3: global shard-level shuffle, sequential inside each shard.
    group_order = rng.permutation(len(groups)) if len(groups) > 1 else np.zeros(1, dtype=int)
    sequential: list[ScheduleEntry] = []
    for index in group_order:
        sequential.extend(groups[int(index)][1])

    # Step 4: bounded local shuffle.
    ordered = _bounded_local_shuffle(sequential, buffer_sequences, rng)

    schedule_id = str(resolved["identity"]["schedule_id_format"]).format(
        split_id=manifest.split_id, seed=int(seed), sequence_length=int(sequence_length)
    )
    return MaterializedSchedule(
        schedule_id=schedule_id,
        split_id=manifest.split_id,
        boundary=manifest.boundary,
        sequence_length=int(sequence_length),
        label_shift=label_shift,
        seed=int(seed),
        local_shuffle_buffer_sequences=buffer_sequences,
        entries=tuple(ordered),
        manifest_content_hash=manifest_hash,
        protocol_digest=protocol_digest,
        requested_source_quotas=requested,
    )


def available_sequences_per_source(
    manifest: SplitManifest, *, sequence_length: int, protocol: Mapping[str, Any] | None = None
) -> dict[str, int]:
    """How many ``sequence_length``-token references each source can supply."""
    resolved = protocol or load_schedule_protocol()
    stride = int(sequence_length) + int(resolved["entry"]["label_shift"])
    totals: dict[str, int] = {}
    for record in manifest.shards:
        totals[record.source_id] = totals.get(record.source_id, 0) + record.token_count // stride
    return dict(sorted(totals.items()))


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def write_schedule(path: Path, schedule: MaterializedSchedule) -> Path:
    """Write one schedule as canonical JSON, content hash included."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(canonical_payload_bytes(schedule.to_dict()))
        handle.write(b"\n")
    return target


def load_schedule(path: Path) -> MaterializedSchedule:
    """Load a schedule and fail closed if its recorded content hash does not match."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = payload.pop("content_hash", None)
    schedule = MaterializedSchedule.from_dict(payload)
    if int(payload.get("sequence_count", schedule.sequence_count)) != schedule.sequence_count:
        raise ScheduleContractError(
            f"{SCHEDULE_CONTENT_HASH_MISMATCH}: {path} declares sequence_count="
            f"{payload['sequence_count']} but carries {schedule.sequence_count} entries"
        )
    if recorded is None:
        raise ScheduleContractError(f"{SCHEDULE_CONTENT_HASH_MISMATCH}: {path} records no content hash")
    if recorded != schedule.content_hash():
        raise ScheduleContractError(f"{SCHEDULE_CONTENT_HASH_MISMATCH}: {path} does not match its payload")
    return schedule


# --------------------------------------------------------------------------------------
# The one integer cursor
# --------------------------------------------------------------------------------------


@dataclass
class ScheduleCursor:
    """One integer position into a schedule, bound to that schedule's content hash.

    The cursor is the entire consumption state. Together with the schedule content hash it
    completely determines which training input has been consumed and which sequence comes
    next, which is exactly what Plan Sections 5.4 and 7.2 require of a resume.
    """

    schedule_content_hash: str
    position: int = 0
    sequence_count: int | None = None

    def __post_init__(self) -> None:
        self.position = int(self.position)
        if self.position < 0:
            raise ScheduleContractError(f"{SCHEDULE_CURSOR_OUT_OF_RANGE}: position {self.position} is negative")
        if self.sequence_count is not None and self.position > int(self.sequence_count):
            raise ScheduleContractError(
                f"{SCHEDULE_CURSOR_OUT_OF_RANGE}: position {self.position} exceeds "
                f"{self.sequence_count} scheduled sequences"
            )

    def state_dict(self) -> dict[str, Any]:
        """The complete resume state: one integer plus the schedule it belongs to."""
        return {
            "format_version": _CURSOR_STATE_VERSION,
            CURSOR_STATE_KEY: int(self.position),
            "schedule_content_hash": self.schedule_content_hash,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the integer position, failing closed on a different schedule."""
        payload = dict(state)
        if CURSOR_STATE_KEY not in payload:
            raise ScheduleResumeError(
                f"resume state is missing {CURSOR_STATE_KEY!r}; a schedule cursor cannot be inferred"
            )
        recorded_hash = str(payload.get("schedule_content_hash", ""))
        if recorded_hash != self.schedule_content_hash:
            raise ScheduleResumeError(
                f"{SCHEDULE_CONTENT_HASH_MISMATCH}: resume state belongs to schedule {recorded_hash!r}, "
                f"not {self.schedule_content_hash!r}. Exposure order would change silently."
            )
        position = int(payload[CURSOR_STATE_KEY])
        if position < 0 or (self.sequence_count is not None and position > int(self.sequence_count)):
            raise ScheduleResumeError(
                f"{SCHEDULE_CURSOR_OUT_OF_RANGE}: resume position {position} is outside "
                f"[0, {self.sequence_count}]"
            )
        self.position = position

    def advance(self, count: int) -> None:
        if int(count) < 0:
            raise ScheduleContractError("a schedule cursor never moves backwards during training")
        target = self.position + int(count)
        if self.sequence_count is not None and target > int(self.sequence_count):
            raise ScheduleContractError(
                f"{SCHEDULE_CURSOR_OUT_OF_RANGE}: advancing to {target} exceeds "
                f"{self.sequence_count} scheduled sequences"
            )
        self.position = target

    @property
    def exhausted(self) -> bool:
        return self.sequence_count is not None and self.position >= int(self.sequence_count)


# --------------------------------------------------------------------------------------
# The final-training reader
# --------------------------------------------------------------------------------------


class ScheduledTokenStream:
    """Memory-mapped uint16 reader driven by a materialized schedule and one integer cursor.

    This replaces random flat-stream sampling for final training. Shards are memory-mapped
    once and read sequentially inside a shard, so the efficient ``np.memmap`` access the
    pilot path relies on is preserved while the consumed mixture becomes reproducible.

    The ``state_dict``/``load_state_dict``/``get_batch`` surface matches
    :class:`~tinybench_lm.data.PackedTokenDataset`, so existing checkpoint plumbing keeps
    working without learning a second protocol.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: SplitManifest,
        schedule: MaterializedSchedule,
        *,
        wrap: bool = False,
    ) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.schedule = schedule
        self.wrap = bool(wrap)
        self.content_hash = schedule.content_hash()
        if schedule.manifest_content_hash != manifest.content_hash():
            raise ScheduleContractError(
                f"{SCHEDULE_CONTENT_HASH_MISMATCH}: schedule was built over manifest "
                f"{schedule.manifest_content_hash!r}, not {manifest.content_hash()!r}"
            )
        self._records = {record.shard_id: record for record in manifest.shards}
        missing = sorted(set(schedule.shard_ids) - set(self._records))
        if missing:
            raise ScheduleContractError(f"{SCHEDULE_SHARD_UNKNOWN}: {missing} are absent from the split manifest")
        self._memmaps: dict[str, np.memmap] = {}
        self.cursor = schedule.cursor(0)

    # -- resume ------------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return self.cursor.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.cursor.load_state_dict(state)

    @property
    def position(self) -> int:
        return self.cursor.position

    # -- reading -----------------------------------------------------------------------

    def _tokens(self, shard_id: str) -> np.memmap:
        cached = self._memmaps.get(shard_id)
        if cached is None:
            record = self._records[shard_id]
            path = self.root / record.relative_path
            cached = np.memmap(path, dtype=STORAGE_DTYPE, mode="r")
            if cached.size != record.token_count:
                raise ScheduleContractError(
                    f"{SCHEDULE_REFERENCE_OUT_OF_BOUNDS}: {shard_id} holds {cached.size} tokens, "
                    f"the manifest records {record.token_count}"
                )
            self._memmaps[shard_id] = cached
        return cached

    def read_entry(self, entry: ScheduleEntry) -> np.ndarray:
        """Read one reference as int64 tokens, verifying it stays inside its shard."""
        record = self._records.get(entry.shard_id)
        if record is None:
            raise ScheduleContractError(f"{SCHEDULE_SHARD_UNKNOWN}: {entry.shard_id!r}")
        if entry.token_offset < 0 or entry.end_offset > record.token_count:
            raise ScheduleContractError(
                f"{SCHEDULE_REFERENCE_OUT_OF_BOUNDS}: {entry.reference} leaves shard "
                f"{entry.shard_id!r} of {record.token_count} tokens"
            )
        tokens = self._tokens(entry.shard_id)
        return np.asarray(tokens[entry.token_offset : entry.end_offset]).astype(np.int64, copy=False)

    def next_entries(self, count: int) -> tuple[ScheduleEntry, ...]:
        """Take the next ``count`` references and advance the integer cursor."""
        if int(count) < 1:
            raise ScheduleContractError("a batch needs at least one scheduled sequence")
        total = self.schedule.sequence_count
        taken: list[ScheduleEntry] = []
        for _ in range(int(count)):
            if self.cursor.position >= total:
                if not self.wrap:
                    raise ScheduleContractError(
                        f"{SCHEDULE_CURSOR_OUT_OF_RANGE}: the schedule holds {total} sequences and is exhausted. "
                        "Materialize a schedule that covers the planned horizon instead of resampling."
                    )
                self.cursor.position = 0
            taken.append(self.schedule.entries[self.cursor.position])
            self.cursor.position += 1
        return tuple(taken)

    def get_batch(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Next ``batch_size`` scheduled sequences as ``(inputs, targets)``."""
        if int(seq_len) != self.schedule.sequence_length:
            raise ScheduleContractError(
                f"the schedule was materialized for sequence_length={self.schedule.sequence_length}, "
                f"not {seq_len}. Rebuild the schedule instead of reshaping its references."
            )
        rows = np.stack([self.read_entry(entry) for entry in self.next_entries(batch_size)])
        batch = torch.from_numpy(rows)
        if device.type == "cuda":
            batch = batch.pin_memory().to(device, non_blocking=True)
        else:
            batch = batch.to(device)
        return batch[:, :-1], batch[:, 1:]

    def close(self) -> None:
        """Release the memory maps. Bounded tests use this to avoid Windows file locks."""
        for tokens in self._memmaps.values():
            mapping = getattr(tokens, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self._memmaps.clear()


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


def _verdict(check_id: str, requirement: str, observed: Any, ok: bool, reason: str) -> CheckResult:
    return CheckResult(check_id, requirement, str(observed), PASS if ok else FAIL, reason)


def reference_bound_violations(
    manifest: SplitManifest, schedule: MaterializedSchedule
) -> tuple[str, ...]:
    """Every reference that leaves its shard, names an unknown shard, or repeats."""
    records = {record.shard_id: record for record in manifest.shards}
    problems: list[str] = []
    seen: set[tuple[str, int]] = set()
    stride = schedule.tokens_per_entry
    for entry in schedule.entries:
        record = records.get(entry.shard_id)
        if record is None:
            problems.append(f"{SCHEDULE_SHARD_UNKNOWN}: {entry.shard_id!r}")
            continue
        if entry.length != stride:
            problems.append(
                f"{SCHEDULE_REFERENCE_OUT_OF_BOUNDS}: {entry.reference} is not {stride} tokens"
            )
        if entry.token_offset < 0 or entry.end_offset > record.token_count:
            problems.append(
                f"{SCHEDULE_REFERENCE_OUT_OF_BOUNDS}: {entry.reference} leaves a {record.token_count}-token shard"
            )
        if entry.token_offset % stride != 0:
            problems.append(
                f"{SCHEDULE_REFERENCE_OUT_OF_BOUNDS}: {entry.reference} overlaps a neighbouring reference"
            )
        if entry.source_id != record.source_id or entry.namespace != record.namespace:
            problems.append(
                f"{SCHEDULE_SHARD_UNKNOWN}: {entry.shard_id!r} is tagged {entry.source_id!r}/{entry.namespace!r} "
                f"but the manifest records {record.source_id!r}/{record.namespace!r}"
            )
        if record.boundary != schedule.boundary:
            problems.append(
                f"{SCHEDULE_BOUNDARY_MIXED}: {entry.shard_id!r} is {record.boundary!r}, schedule is {schedule.boundary!r}"
            )
        key = (entry.shard_id, entry.token_offset)
        if key in seen:
            problems.append(f"{SCHEDULE_DUPLICATE_REFERENCE}: {entry.reference} appears twice")
        seen.add(key)
    return tuple(problems)


def sequential_read_violations(schedule: MaterializedSchedule) -> tuple[str, ...]:
    """In-shard order inversions that exceed the bounded local shuffle window.

    Reads inside a shard are emitted in ascending ``token_offset`` order, and the bounded
    buffer can only displace an entry by fewer than ``B`` positions. Any larger inversion
    means the ordering was not produced by a bounded local shuffle of sequential reads.
    """
    bound = max(1, int(schedule.local_shuffle_buffer_sequences))
    positions: dict[str, list[tuple[int, int]]] = {}
    for position, entry in enumerate(schedule.entries):
        positions.setdefault(entry.shard_id, []).append((entry.token_offset, position))
    problems: list[str] = []
    for shard_id in sorted(positions):
        ordered = [position for _, position in sorted(positions[shard_id])]
        if not ordered:
            continue
        # Suffix minima make the worst inversion for each entry a single comparison.
        suffix_minimum = list(ordered)
        for index in range(len(ordered) - 2, -1, -1):
            suffix_minimum[index] = min(ordered[index], suffix_minimum[index + 1])
        for index, position in enumerate(ordered[:-1]):
            worst = suffix_minimum[index + 1]
            if worst < position and position - worst >= bound:
                problems.append(
                    f"{SCHEDULE_LOCAL_SHUFFLE_UNBOUNDED}: {shard_id} inverts positions "
                    f"{worst} and {position}, beyond the {bound}-sequence window"
                )
    return tuple(problems)


def quota_reconciliation(schedule: MaterializedSchedule) -> tuple[str, ...]:
    """Per-source sequence counts that do not equal the requested quota."""
    if not schedule.requested_source_quotas:
        return ()
    observed = schedule.sequences_per_source
    problems: list[str] = []
    for source_id, quota in sorted(schedule.requested_source_quotas.items()):
        actual = observed.get(source_id, 0)
        if actual != int(quota):
            problems.append(
                f"{SCHEDULE_SOURCE_QUOTA_MISMATCH}: source {source_id!r} scheduled {actual} sequences, "
                f"quota requested {quota}"
            )
    extra = sorted(set(observed) - set(schedule.requested_source_quotas))
    for source_id in extra:
        problems.append(f"{SCHEDULE_SOURCE_QUOTA_MISMATCH}: source {source_id!r} has no requested quota")
    return tuple(problems)


def verify_schedule(
    manifest: SplitManifest,
    schedule: MaterializedSchedule,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[CheckResult, ...]:
    """Reconcile a schedule against its manifest and the frozen contract.

    Determinism is proven by rebuilding the schedule from the recorded identity and
    comparing content hashes, so a hand-edited or drifted schedule cannot pass.
    """
    resolved = protocol or load_schedule_protocol()
    results: list[CheckResult] = []

    results.append(
        _verdict(
            "schedule.non_empty",
            "at least one scheduled sequence",
            schedule.sequence_count,
            schedule.sequence_count > 0,
            f"{SCHEDULE_EMPTY} unless the schedule materializes references",
        )
    )
    results.append(
        _verdict(
            "schedule.manifest_binding",
            manifest.content_hash(),
            schedule.manifest_content_hash,
            schedule.manifest_content_hash == manifest.content_hash(),
            f"{SCHEDULE_CONTENT_HASH_MISMATCH} unless the schedule names the manifest it indexes",
        )
    )
    results.append(
        _verdict(
            "schedule.protocol_binding",
            str(resolved.get("_digest", "")),
            schedule.protocol_digest,
            schedule.protocol_digest == str(resolved.get("_digest", "")),
            f"{SCHEDULE_CONTENT_HASH_MISMATCH} unless the schedule names the frozen contract it obeyed",
        )
    )

    bound_problems = reference_bound_violations(manifest, schedule)
    results.append(
        _verdict(
            "schedule.references_in_bounds",
            "every (shard_id, token_offset, length) inside its shard, exactly once",
            bound_problems[:3] or "in bounds",
            not bound_problems,
            f"{SCHEDULE_REFERENCE_OUT_OF_BOUNDS}/{SCHEDULE_DUPLICATE_REFERENCE}/"
            f"{SCHEDULE_SHARD_UNKNOWN}/{SCHEDULE_BOUNDARY_MIXED} fail closed",
        )
    )

    order_problems = sequential_read_violations(schedule)
    results.append(
        _verdict(
            "schedule.bounded_sequential_reads",
            f"in-shard inversions below {schedule.local_shuffle_buffer_sequences} positions",
            order_problems[:3] or "bounded",
            not order_problems,
            f"{SCHEDULE_LOCAL_SHUFFLE_UNBOUNDED} unless reads stay sequential inside a bounded window",
        )
    )

    quota_problems = quota_reconciliation(schedule)
    results.append(
        _verdict(
            "schedule.source_quotas_reconcile",
            dict(sorted(schedule.requested_source_quotas.items())) or "no quota requested",
            quota_problems[:3] or schedule.sequences_per_source,
            not quota_problems,
            f"{SCHEDULE_SOURCE_QUOTA_MISMATCH} unless every source contributes its requested sequences",
        )
    )

    rebuilt = build_materialized_schedule(
        manifest,
        sequence_length=schedule.sequence_length,
        seed=schedule.seed,
        source_sequence_quotas=dict(schedule.requested_source_quotas) or None,
        local_shuffle_buffer_sequences=schedule.local_shuffle_buffer_sequences,
        protocol=resolved,
    )
    results.append(
        _verdict(
            "schedule.deterministic_rebuild",
            rebuilt.content_hash(),
            schedule.content_hash(),
            rebuilt.content_hash() == schedule.content_hash(),
            "identical inputs and seed must rebuild a byte-identical schedule",
        )
    )
    results.append(
        _verdict(
            "schedule.cursor_state_is_one_integer",
            f"{CURSOR_STATE_KEY} plus the schedule content hash",
            sorted(schedule.cursor(0).state_dict()),
            sorted(schedule.cursor(0).state_dict()) == ["format_version", "schedule_content_hash", CURSOR_STATE_KEY],
            "resume restores one integer schedule cursor bound to the schedule hash",
        )
    )

    readiness = resolved["readiness"]
    results.append(
        CheckResult(
            "schedule.real_scale_construction",
            "measured schedule over real source-tagged shards",
            str(readiness["measured_real_shard_schedule"]),
            DEFERRED if str(readiness["measured_real_shard_schedule"]) == NOT_RUN else NOT_RUN,
            f"blocker={readiness['blocker']} owner={readiness['owner']} next_action={readiness['next_action']}",
        )
    )
    return tuple(results)


def assert_schedule_valid(
    manifest: SplitManifest,
    schedule: MaterializedSchedule,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed on any schedule check that is not deferred."""
    failures = [result for result in verify_schedule(manifest, schedule, protocol=protocol) if result.failed]
    if failures:
        raise ScheduleContractError("; ".join(f"{result.check_id}: {result.reason}" for result in failures))


def format_schedule_report(results: Sequence[CheckResult]) -> str:
    """Human-readable summary of schedule verification."""
    width = max((len(result.check_id) for result in results), default=0)
    lines = [
        f"{result.status:<9} {result.check_id:<{width}}  {result.requirement} -> {result.observed}"
        for result in results
    ]
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    failures = [result for result in results if result.failed]
    lines.append("")
    lines.append("Summary: " + ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))
    lines.append("RESULT: " + ("PASS" if not failures else "FAIL"))
    if failures:
        lines.append("Failures:")
        lines.extend(f"  {result.check_id}: {result.reason}" for result in failures)
    return "\n".join(lines)


def exposure_reference_hash(entries: Iterable[ScheduleEntry]) -> str:
    """Order-independent hash of an exposure multiset of ``(shard_id, token_offset, length)``.

    Plan Section 8.3 hashes sorted exposure references with multiplicity so two branch arms
    can be proven to consume the same data in a different order. Task 3.11 builds the arms;
    this is the shared primitive they hash with.
    """
    references = sorted(entry.reference for entry in entries)
    return hashlib.sha256(canonical_payload_bytes({"references": references})).hexdigest()


def training_order_hash(entries: Iterable[ScheduleEntry]) -> str:
    """Order-sensitive hash of an exposure sequence (Plan Section 8.3 ``training_order_hash``)."""
    references = [list(entry.reference) for entry in entries]
    return hashlib.sha256(canonical_payload_bytes({"order": references})).hexdigest()


def open_scheduled_stream(
    shard_root: str | Path,
    manifest_path: str | Path,
    schedule_path: str | Path,
    *,
    wrap: bool = False,
    verify: bool = True,
) -> ScheduledTokenStream:
    """Open a final-training reader from a split manifest and a materialized schedule.

    Both files are loaded through their content-hash checks, and the schedule is verified
    against the manifest before a single token is read, so a drifted pairing fails closed
    instead of silently training on a different mixture.
    """
    from .shards import load_split_manifest  # local import keeps the module import graph flat

    manifest = load_split_manifest(Path(manifest_path))
    schedule = load_schedule(Path(schedule_path))
    if verify:
        assert_schedule_valid(manifest, schedule)
    return ScheduledTokenStream(shard_root, manifest, schedule, wrap=wrap)


def clone_cursor(cursor: ScheduleCursor) -> ScheduleCursor:
    """A detached copy, so recording a resume point cannot alias the live cursor."""
    return copy.deepcopy(cursor)


__all__ = [
    "CURSOR_STATE_KEY",
    "FROZEN_SCHEDULE_PROTOCOL_SHA256",
    "MaterializedSchedule",
    "SCHEDULE_BOUNDARY_MIXED",
    "SCHEDULE_CONTENT_HASH_MISMATCH",
    "SCHEDULE_CURSOR_OUT_OF_RANGE",
    "SCHEDULE_DUPLICATE_REFERENCE",
    "SCHEDULE_EMPTY",
    "SCHEDULE_FAIL_CLOSED_REASON_CODES",
    "SCHEDULE_LOCAL_SHUFFLE_UNBOUNDED",
    "SCHEDULE_OK",
    "SCHEDULE_PROTOCOL_PATH",
    "SCHEDULE_REFERENCE_OUT_OF_BOUNDS",
    "SCHEDULE_SHARD_UNKNOWN",
    "SCHEDULE_SOURCE_QUOTA_MISMATCH",
    "SCHEDULE_SOURCE_SUPPLY_EXHAUSTED",
    "ScheduleContractError",
    "ScheduleCursor",
    "ScheduleEntry",
    "ScheduleResumeError",
    "ScheduledTokenStream",
    "SchedulesNotReadyError",
    "assert_ready_for_real_schedules",
    "assert_schedule_valid",
    "available_sequences_per_source",
    "build_materialized_schedule",
    "canonical_payload_bytes",
    "clone_cursor",
    "exposure_reference_hash",
    "format_schedule_report",
    "load_schedule",
    "load_schedule_protocol",
    "open_scheduled_stream",
    "quota_reconciliation",
    "reference_bound_violations",
    "schedule_identity_seed",
    "sequential_read_violations",
    "training_order_hash",
    "verify_schedule",
    "write_schedule",
]
