"""Source-tagged uint16 shard namespaces, independent split manifests, and G1 verification.

Plan Section 5.4 replaces the monolithic flat ``train.bin``/``validation.bin`` output
contract with **separate source-tagged uint16 shards**, one namespace per source, and
states that sources are never pre-mixed on disk. Plan Section 4.4 requires
``validation_dev`` and ``validation_final`` to be independent, source-stratified, and
near-duplicate-cluster isolated from stable training, reserved data, and each other. Plan
Sections 4.1-4.3 fix the post-filter mixture shares, the ``full_v1``/``degraded_v1``
profile thresholds, and the at-least-390.1M reserved margin. Section 13 G1 is the gate
those manifests have to satisfy.

This module is the mechanism, backed by one frozen config::

    configs/data/shards_v1.yaml

The guarantees mirror :mod:`tinybench_lm.data_protocols` and
:mod:`tinybench_lm.source_manifest`:

1. **Immutable.** The config is verified against a pinned SHA-256 digest
   (:data:`FROZEN_SHARD_PROTOCOL_SHA256`) on every load, so a namespace, tolerance, or
   profile threshold cannot drift after fixture calibration. Changing one means publishing
   ``shards_v2.yaml``.
2. **Source identity is never lost.** Every shard is written into exactly one registered
   namespace, carries its source ID and boundary, and fails closed if documents from more
   than one source or boundary reach the same shard.
3. **Deterministic.** Documents are packed in ``document_id`` order with an EOS at every
   document boundary. Each shard record carries per-document offsets and lengths plus the
   shard digest, so document boundaries are reproducible and auditable.
4. **Fail closed.** Boundary crossings, protected-slice crossings, undeclared validation
   slices, manifest overlap, digest mismatches, and profiles below their frozen thresholds
   all raise rather than degrade into a pass.
5. **Absence of evidence is never PASS.** Token totals and mixture shares are only
   evaluated at :data:`SCALE_FINAL`. A tiny fixture is audited at :data:`SCALE_FIXTURE`,
   where those checks report ``DEFERRED`` with their blocker, owner, and next action.

The legacy fixture builder does not acquire a corpus. ``build_split_manifests_streaming``
provides the real-scale bounded-memory packing path; global near-deduplication remains an
upstream operation and is never implied by merely setting a CLI attestation flag.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .data_protocols import (
    CLUSTER_CROSSES_BOUNDARY,
    DATA_PROTOCOL_DIR,
    REPOSITORY_ROOT,
    DedupReport,
    DocumentRecord,
    ProtocolError,
    ProtocolNotReadyError,
    deduplicate,
    load_dedup_protocol,
    load_protocol,
)
from .environment import CheckResult
from .source_manifest import (
    FINAL_TOKEN_COUNTER_ID,
    PROVISIONAL_TOKEN_COUNTER_ID,
    load_source_registry,
)
from .tokenizer import STORAGE_DTYPE, encode_document, load_tokenizer_protocol

SHARDS_PROTOCOL_PATH = DATA_PROTOCOL_DIR / "shards_v1.yaml"

#: SHA-256 of the frozen shard contract, over file bytes with CRLF normalized to LF.
FROZEN_SHARD_PROTOCOL_SHA256: Mapping[str, str] = {
    "shards_v1.yaml": "b93ba07f7ddf6844ce3522bd9568526563c1974b4e76a8bb34bb1b665760015d",
}

# --------------------------------------------------------------------------------------
# Boundaries, statuses, and the reason-code vocabulary
# --------------------------------------------------------------------------------------

STABLE_TRAIN = "stable_train"
RESERVED = "reserved"
VALIDATION_DEV = "validation_dev"
VALIDATION_FINAL = "validation_final"

#: Every boundary a duplicate cluster is forbidden to cross (Plan Sections 4.3-4.4).
ISOLATED_BOUNDARIES: tuple[str, ...] = (STABLE_TRAIN, RESERVED, VALIDATION_DEV, VALIDATION_FINAL)
VALIDATION_BOUNDARIES: tuple[str, ...] = (VALIDATION_DEV, VALIDATION_FINAL)

#: The nine source-tagged namespaces Plan Section 5.4 lists verbatim.
EXPECTED_SHARD_NAMESPACES: tuple[str, ...] = (
    "stable/fineweb_edu",
    "stable/dclm",
    "stable/openwebmath",
    "stable/narrative",
    "reserved/science",
    "reserved/textbook",
    "reserved/wikipedia",
    "reserved/edu_decile",
    "reserved/math_prose",
)

EXPECTED_PROTECTED_SLICES: tuple[str, ...] = (
    "broad_general",
    "educational_science",
    "narrative_coreference",
    "math_technical",
)

PASS = "PASS"
FAIL = "FAIL"
DEFERRED = "DEFERRED"
NOT_RUN = "NOT_RUN"

#: Fixture scope never evaluates a real-corpus token total; final scope does.
SCALE_FIXTURE = "FIXTURE"
SCALE_FINAL = "FINAL"

SHARD_OK = "SHARD_OK"
SHARD_SOURCE_MIXED = "SHARD_SOURCE_MIXED"
SHARD_BOUNDARY_MIXED = "SHARD_BOUNDARY_MIXED"
SHARD_MISSING_SOURCE_TAG = "SHARD_MISSING_SOURCE_TAG"
SHARD_NAMESPACE_UNREGISTERED = "SHARD_NAMESPACE_UNREGISTERED"
SHARD_DTYPE_NOT_UINT16 = "SHARD_DTYPE_NOT_UINT16"
SHARD_TOKEN_ID_OUT_OF_RANGE = "SHARD_TOKEN_ID_OUT_OF_RANGE"
SHARD_DIGEST_MISMATCH = "SHARD_DIGEST_MISMATCH"
SHARD_DOCUMENT_BOUNDARY_MISMATCH = "SHARD_DOCUMENT_BOUNDARY_MISMATCH"
SPLIT_MANIFESTS_NOT_INDEPENDENT = "SPLIT_MANIFESTS_NOT_INDEPENDENT"
CLUSTER_CROSSES_PROTECTED_SLICE = "CLUSTER_CROSSES_PROTECTED_SLICE"
VALIDATION_SLICE_NOT_DECLARED = "VALIDATION_SLICE_NOT_DECLARED"
MIXTURE_SHARE_OUT_OF_TOLERANCE = "MIXTURE_SHARE_OUT_OF_TOLERANCE"
VALIDATION_SPLIT_TOKENS_OUT_OF_RANGE = "VALIDATION_SPLIT_TOKENS_OUT_OF_RANGE"
RESERVED_MARGIN_NOT_MET = "RESERVED_MARGIN_NOT_MET"
PROFILE_BELOW_THRESHOLD = "PROFILE_BELOW_THRESHOLD"
DEGRADED_DECISION_RECORD_MISSING = "DEGRADED_DECISION_RECORD_MISSING"

SHARD_FAIL_CLOSED_REASON_CODES: frozenset[str] = frozenset(
    {
        SHARD_SOURCE_MIXED,
        SHARD_BOUNDARY_MIXED,
        SHARD_MISSING_SOURCE_TAG,
        SHARD_NAMESPACE_UNREGISTERED,
        SHARD_DTYPE_NOT_UINT16,
        SHARD_TOKEN_ID_OUT_OF_RANGE,
        SHARD_DIGEST_MISMATCH,
        SHARD_DOCUMENT_BOUNDARY_MISMATCH,
        SPLIT_MANIFESTS_NOT_INDEPENDENT,
        CLUSTER_CROSSES_BOUNDARY,
        CLUSTER_CROSSES_PROTECTED_SLICE,
        VALIDATION_SLICE_NOT_DECLARED,
        MIXTURE_SHARE_OUT_OF_TOLERANCE,
        VALIDATION_SPLIT_TOKENS_OUT_OF_RANGE,
        RESERVED_MARGIN_NOT_MET,
        PROFILE_BELOW_THRESHOLD,
        DEGRADED_DECISION_RECORD_MISSING,
    }
)

_MANIFEST_SCHEMA_VERSION = "shards_v1"


class ShardContractError(ProtocolError):
    """The frozen shard contract is malformed, or a shard/manifest violates it."""


class ShardIsolationError(ShardContractError):
    """A duplicate cluster crosses a split boundary or a protected reporting slice."""


class ProfileSelectionError(ShardContractError):
    """A corpus profile was claimed below its frozen threshold or without its decision record."""


class ShardsNotReadyError(ProtocolNotReadyError):
    """Real-scale shard production is gated behind corpus acquisition and the final tokenizer."""


class StreamingShardError(ShardContractError):
    """A streaming input or its restart/output preconditions are unsafe."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def load_shard_protocol(path: Path = SHARDS_PROTOCOL_PATH, *, verify: bool = True) -> dict[str, Any]:
    """Load the frozen shard contract, verifying its pinned digest by default."""
    protocol = load_protocol(path, verify=verify, registry=FROZEN_SHARD_PROTOCOL_SHA256)
    required = (
        "storage",
        "namespaces",
        "validation_namespace_prefixes",
        "splits",
        "manifest_independence",
        "isolation",
        "mixture_verification",
        "profiles",
        "profile_selection",
        "reserved_pool",
        "validation_split_tokens",
        "readiness",
    )
    for section in required:
        if section not in protocol:
            raise ShardContractError(f"shard protocol is missing required section {section!r}")
    if str(protocol["storage"]["dtype"]) != "uint16":
        raise ShardContractError("the frozen shard storage dtype must be uint16")
    if not bool(protocol["storage"]["never_pre_mix_sources_on_disk"]):
        raise ShardContractError("the frozen contract must forbid pre-mixing sources on disk")
    declared = namespace_index(protocol)
    if tuple(declared.values()) != EXPECTED_SHARD_NAMESPACES:
        raise ShardContractError(
            f"shard protocol declares namespaces {tuple(declared.values())}, expected {EXPECTED_SHARD_NAMESPACES}"
        )
    slices = tuple(str(name) for name in protocol["isolation"]["protected_slices"])
    if slices != EXPECTED_PROTECTED_SLICES:
        raise ShardContractError(f"shard protocol declares protected slices {slices}, expected {EXPECTED_PROTECTED_SLICES}")
    mixture = protocol["mixture_verification"]
    if str(mixture["final_token_counter_id"]) != FINAL_TOKEN_COUNTER_ID:
        raise ShardContractError(
            f"shard protocol names final counter {mixture['final_token_counter_id']!r}, "
            f"but the manifest schema uses {FINAL_TOKEN_COUNTER_ID!r}"
        )
    if str(mixture["provisional_token_counter_id"]) != PROVISIONAL_TOKEN_COUNTER_ID:
        raise ShardContractError(
            f"shard protocol names provisional counter {mixture['provisional_token_counter_id']!r}, "
            f"but the manifest schema uses {PROVISIONAL_TOKEN_COUNTER_ID!r}"
        )
    return protocol


def namespace_index(protocol: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Source ID -> its frozen stable/reserved namespace, in the declared Plan 5.4 order."""
    resolved = protocol or load_shard_protocol()
    index: dict[str, str] = {}
    for group in (STABLE_TRAIN, RESERVED):
        for source_id, namespace in resolved["namespaces"][group].items():
            key = str(source_id)
            if key in index:
                raise ShardContractError(f"source ID {key!r} is mapped to a namespace twice")
            index[key] = str(namespace)
    return index


def split_index(protocol: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Split ID -> its frozen declaration."""
    resolved = protocol or load_shard_protocol()
    return {str(entry["split_id"]): dict(entry) for entry in resolved["splits"]}


def namespace_for(
    source_id: str,
    boundary: str,
    *,
    protocol: Mapping[str, Any] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> str:
    """The one namespace this (source, boundary) pair may be packed into.

    Validation namespaces are the source leaf under the split's own root, so a validation
    shard can never be mistaken for a training shard.
    """
    resolved = protocol or load_shard_protocol()
    index = namespace_index(resolved)
    base = index.get(str(source_id))
    if base is None:
        raise ShardContractError(
            f"{SHARD_NAMESPACE_UNREGISTERED}: source ID {source_id!r} has no frozen shard namespace"
        )
    if registry is not None:
        registered = {
            str(entry["source_id"]): str(entry["boundary"])
            for entry in list(registry["stable_sources"]) + list(registry["reserved_sources"])
        }
        declared = registered.get(str(source_id))
        if declared is None:
            raise ShardContractError(
                f"{SHARD_NAMESPACE_UNREGISTERED}: source ID {source_id!r} is not in the frozen source registry"
            )
        if boundary not in VALIDATION_BOUNDARIES and declared != boundary:
            raise ShardContractError(
                f"{SHARD_BOUNDARY_MIXED}: source {source_id!r} is registered as {declared!r}, not {boundary!r}"
            )
    if boundary in (STABLE_TRAIN, RESERVED):
        expected_root = boundary if boundary == RESERVED else "stable"
        if not base.startswith(f"{expected_root}/"):
            raise ShardContractError(
                f"{SHARD_BOUNDARY_MIXED}: namespace {base!r} does not belong to boundary {boundary!r}"
            )
        return base
    if boundary in VALIDATION_BOUNDARIES:
        prefix = str(resolved["validation_namespace_prefixes"][boundary])
        return f"{prefix}/{base.split('/')[-1]}"
    raise ShardContractError(f"boundary {boundary!r} is not one of {ISOLATED_BOUNDARIES}")


def assert_ready_for_real_shard_production(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: real-scale shards need an acquired corpus and the final tokenizer."""
    resolved = protocol or load_shard_protocol()
    readiness = resolved["readiness"]
    blocked = [
        name
        for name in ("measured_shard_token_counts", "measured_mixture_shares", "measured_profile_selection")
        if str(readiness.get(name)) != PASS
    ]
    if blocked:
        raise ShardsNotReadyError(
            f"real-scale shard production is not ready: {blocked}. "
            f"blocker={readiness.get('blocker')} owner={readiness.get('owner')} "
            f"next_action={readiness.get('next_action')}"
        )


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardDocument:
    """One accepted document destined for exactly one source-tagged shard."""

    document_id: str
    source_id: str
    text: str
    boundary: str
    protected_slice: str | None = None
    # Optional upstream dedup cluster identity. It is not part of the frozen manifest,
    # but the streaming path checks it when supplied.
    cluster_id: str | None = None

    def as_dedup_record(self) -> DocumentRecord:
        return DocumentRecord(self.document_id, self.text, self.source_id, self.boundary)


@dataclass(frozen=True)
class ShardRecord:
    """One packed uint16 shard: its source tag, digest, and deterministic doc boundaries."""

    shard_id: str
    namespace: str
    source_id: str
    boundary: str
    relative_path: str
    dtype: str
    token_count: int
    document_count: int
    document_ids: tuple[str, ...]
    document_token_offsets: tuple[int, ...]
    document_token_lengths: tuple[int, ...]
    eos_id: int
    sha256: str
    token_counter_id: str
    protected_slices: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "namespace": self.namespace,
            "source_id": self.source_id,
            "boundary": self.boundary,
            "relative_path": self.relative_path,
            "dtype": self.dtype,
            "token_count": self.token_count,
            "document_count": self.document_count,
            "document_ids": list(self.document_ids),
            "document_token_offsets": list(self.document_token_offsets),
            "document_token_lengths": list(self.document_token_lengths),
            "eos_id": self.eos_id,
            "sha256": self.sha256,
            "token_counter_id": self.token_counter_id,
            "protected_slices": list(self.protected_slices),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShardRecord":
        return cls(
            shard_id=str(payload["shard_id"]),
            namespace=str(payload["namespace"]),
            source_id=str(payload["source_id"]),
            boundary=str(payload["boundary"]),
            relative_path=str(payload["relative_path"]),
            dtype=str(payload["dtype"]),
            token_count=int(payload["token_count"]),
            document_count=int(payload["document_count"]),
            document_ids=tuple(str(item) for item in payload["document_ids"]),
            document_token_offsets=tuple(int(item) for item in payload["document_token_offsets"]),
            document_token_lengths=tuple(int(item) for item in payload["document_token_lengths"]),
            eos_id=int(payload["eos_id"]),
            sha256=str(payload["sha256"]),
            token_counter_id=str(payload["token_counter_id"]),
            protected_slices=tuple(str(item) for item in payload.get("protected_slices", [])),
        )


@dataclass(frozen=True)
class NamespaceManifest:
    """Every shard belonging to one source-tagged namespace."""

    namespace: str
    source_id: str
    boundary: str
    shards: tuple[ShardRecord, ...]

    @property
    def token_count(self) -> int:
        return sum(shard.token_count for shard in self.shards)

    @property
    def document_count(self) -> int:
        return sum(shard.document_count for shard in self.shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "source_id": self.source_id,
            "boundary": self.boundary,
            "token_count": self.token_count,
            "document_count": self.document_count,
            "shards": [shard.to_dict() for shard in self.shards],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NamespaceManifest":
        return cls(
            namespace=str(payload["namespace"]),
            source_id=str(payload["source_id"]),
            boundary=str(payload["boundary"]),
            shards=tuple(ShardRecord.from_dict(item) for item in payload["shards"]),
        )


@dataclass(frozen=True)
class SplitManifest:
    """One independent split manifest. ``validation_dev`` and ``validation_final`` never share one."""

    split_id: str
    boundary: str
    namespaces: tuple[NamespaceManifest, ...]
    token_counter_id: str
    dtype: str = "uint16"
    schema_version: str = _MANIFEST_SCHEMA_VERSION
    shards_digest: str = ""
    sources_digest: str = ""
    tokenizer_digest: str = ""
    protected_slice_tokens: Mapping[str, int] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return sum(namespace.token_count for namespace in self.namespaces)

    @property
    def document_count(self) -> int:
        return sum(namespace.document_count for namespace in self.namespaces)

    @property
    def shards(self) -> tuple[ShardRecord, ...]:
        return tuple(shard for namespace in self.namespaces for shard in namespace.shards)

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(doc_id for shard in self.shards for doc_id in shard.document_ids)

    @property
    def tokens_per_source(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for namespace in self.namespaces:
            totals[namespace.source_id] = totals.get(namespace.source_id, 0) + namespace.token_count
        return totals

    def namespace(self, name: str) -> NamespaceManifest:
        for candidate in self.namespaces:
            if candidate.namespace == name:
                return candidate
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_id": self.split_id,
            "boundary": self.boundary,
            "dtype": self.dtype,
            "token_counter_id": self.token_counter_id,
            "token_count": self.token_count,
            "document_count": self.document_count,
            "shards_digest": self.shards_digest,
            "sources_digest": self.sources_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "protected_slice_tokens": dict(sorted(self.protected_slice_tokens.items())),
            "namespaces": [namespace.to_dict() for namespace in self.namespaces],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitManifest":
        return cls(
            split_id=str(payload["split_id"]),
            boundary=str(payload["boundary"]),
            namespaces=tuple(NamespaceManifest.from_dict(item) for item in payload["namespaces"]),
            token_counter_id=str(payload["token_counter_id"]),
            dtype=str(payload.get("dtype", "uint16")),
            schema_version=str(payload.get("schema_version", _MANIFEST_SCHEMA_VERSION)),
            shards_digest=str(payload.get("shards_digest", "")),
            sources_digest=str(payload.get("sources_digest", "")),
            tokenizer_digest=str(payload.get("tokenizer_digest", "")),
            protected_slice_tokens={str(key): int(value) for key, value in dict(payload.get("protected_slice_tokens", {})).items()},
        )

    def content_hash(self) -> str:
        """Digest of the manifest payload, so a split's shard set is identifiable."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Shard production
# --------------------------------------------------------------------------------------


def _packed_bytes_digest(tokens: np.ndarray) -> str:
    return hashlib.sha256(tokens.tobytes()).hexdigest()


def pack_shard_tokens(
    tokenizer,
    documents: Sequence[ShardDocument],
    *,
    tokenizer_protocol: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Pack documents in ``document_id`` order and return (uint16 stream, per-doc lengths).

    Every document contributes its content IDs followed by the frozen document-boundary
    EOS, so boundaries are deterministic and recoverable from the recorded lengths alone.
    """
    resolved = tokenizer_protocol or load_tokenizer_protocol()
    ceiling = int(resolved["vocabulary"]["maximum_representable_id"])
    stream: list[int] = []
    lengths: list[int] = []
    for document in documents:
        ids = encode_document(tokenizer, document.text, protocol=resolved)
        highest = max(ids, default=0)
        if highest > ceiling:
            raise ShardContractError(
                f"{SHARD_TOKEN_ID_OUT_OF_RANGE}: token ID {highest} does not fit the packed uint16 format"
            )
        stream.extend(ids)
        lengths.append(len(ids))
    return np.asarray(stream, dtype=STORAGE_DTYPE), tuple(lengths)


def write_shard(
    root: Path,
    tokenizer,
    documents: Sequence[ShardDocument],
    *,
    shard_index: int = 0,
    protocol: Mapping[str, Any] | None = None,
    registry: Mapping[str, Any] | None = None,
    tokenizer_protocol: Mapping[str, Any] | None = None,
    token_counter_id: str = PROVISIONAL_TOKEN_COUNTER_ID,
) -> ShardRecord:
    """Write one source-tagged uint16 shard and return its auditable record.

    Fails closed when the documents do not all share one source ID and one boundary; a
    shard that mixes sources would destroy the source identity the mixture depends on.
    """
    resolved = protocol or load_shard_protocol()
    ordered = sorted(documents, key=lambda document: document.document_id)
    if not ordered:
        raise ShardContractError("cannot write an empty shard")

    source_ids = {document.source_id for document in ordered}
    if len(source_ids) != 1:
        raise ShardContractError(f"{SHARD_SOURCE_MIXED}: one shard may not hold sources {sorted(source_ids)}")
    boundaries = {document.boundary for document in ordered}
    if len(boundaries) != 1:
        raise ShardContractError(f"{SHARD_BOUNDARY_MIXED}: one shard may not hold boundaries {sorted(boundaries)}")
    source_id = ordered[0].source_id
    boundary = ordered[0].boundary
    if not str(source_id).strip():
        raise ShardContractError(f"{SHARD_MISSING_SOURCE_TAG}: a shard must record its source ID")
    if len({document.document_id for document in ordered}) != len(ordered):
        raise ShardContractError("document IDs inside a shard must be unique")

    slice_rules = resolved["isolation"]
    if boundary in VALIDATION_BOUNDARIES and bool(slice_rules["validation_requires_declared_slice"]):
        undeclared = [document.document_id for document in ordered if not document.protected_slice]
        if undeclared:
            raise ShardContractError(
                f"{VALIDATION_SLICE_NOT_DECLARED}: {boundary} documents {undeclared} declare no protected slice"
            )
    known_slices = set(EXPECTED_PROTECTED_SLICES)
    unknown = sorted({document.protected_slice for document in ordered if document.protected_slice} - known_slices)
    if unknown:
        raise ShardContractError(f"protected slices {unknown} are not frozen in the shard contract")

    namespace = namespace_for(source_id, boundary, protocol=resolved, registry=registry)
    shard_id = str(resolved["storage"]["shard_id_format"]).format(namespace=namespace, index=shard_index)
    relative_path = f"{namespace}/shard_{shard_index:05d}{resolved['storage']['file_suffix']}"

    tokens, lengths = pack_shard_tokens(tokenizer, ordered, tokenizer_protocol=tokenizer_protocol)
    offsets: list[int] = []
    cursor = 0
    for length in lengths:
        offsets.append(cursor)
        cursor += length

    target = Path(root) / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(tokens.tobytes())

    resolved_tokenizer = tokenizer_protocol or load_tokenizer_protocol()
    return ShardRecord(
        shard_id=shard_id,
        namespace=namespace,
        source_id=source_id,
        boundary=boundary,
        relative_path=relative_path,
        dtype=str(resolved["storage"]["dtype"]),
        token_count=int(tokens.size),
        document_count=len(ordered),
        document_ids=tuple(document.document_id for document in ordered),
        document_token_offsets=tuple(offsets),
        document_token_lengths=tuple(lengths),
        eos_id=int(resolved_tokenizer["document_boundaries"]["eos_token_id"]),
        sha256=_packed_bytes_digest(tokens),
        token_counter_id=token_counter_id,
        protected_slices=tuple(sorted({document.protected_slice for document in ordered if document.protected_slice})),
    )


def build_split_manifest(
    root: Path,
    tokenizer,
    documents: Iterable[ShardDocument],
    *,
    split_id: str,
    shard_document_budget: int = 4096,
    protocol: Mapping[str, Any] | None = None,
    registry: Mapping[str, Any] | None = None,
    tokenizer_protocol: Mapping[str, Any] | None = None,
    token_counter_id: str = PROVISIONAL_TOKEN_COUNTER_ID,
) -> SplitManifest:
    """Pack one split into per-source shards and return its independent manifest."""
    resolved = protocol or load_shard_protocol()
    resolved_registry = registry if registry is not None else load_source_registry()
    resolved_tokenizer = tokenizer_protocol or load_tokenizer_protocol()
    splits = split_index(resolved)
    if split_id not in splits:
        raise ShardContractError(f"split {split_id!r} is not declared in the frozen shard contract")
    boundary = str(splits[split_id]["boundary"])

    grouped: dict[str, list[ShardDocument]] = {}
    for document in documents:
        if document.boundary != boundary:
            raise ShardContractError(
                f"{SHARD_BOUNDARY_MIXED}: document {document.document_id!r} declares boundary "
                f"{document.boundary!r} but is being packed into split {split_id!r}"
            )
        grouped.setdefault(document.source_id, []).append(document)
    if not grouped:
        raise ShardContractError(f"split {split_id!r} received no documents")

    namespaces: list[NamespaceManifest] = []
    slice_tokens: dict[str, int] = {}
    for source_id in sorted(grouped):
        ordered = sorted(grouped[source_id], key=lambda document: document.document_id)
        records: list[ShardRecord] = []
        for shard_index, start in enumerate(range(0, len(ordered), shard_document_budget)):
            chunk = ordered[start : start + shard_document_budget]
            record = write_shard(
                root,
                tokenizer,
                chunk,
                shard_index=shard_index,
                protocol=resolved,
                registry=resolved_registry,
                tokenizer_protocol=resolved_tokenizer,
                token_counter_id=token_counter_id,
            )
            records.append(record)
            for document, length in zip(chunk, record.document_token_lengths):
                if document.protected_slice:
                    slice_tokens[document.protected_slice] = slice_tokens.get(document.protected_slice, 0) + length
        namespaces.append(
            NamespaceManifest(records[0].namespace, source_id, boundary, tuple(records))
        )

    return SplitManifest(
        split_id=split_id,
        boundary=boundary,
        namespaces=tuple(sorted(namespaces, key=lambda item: item.namespace)),
        token_counter_id=token_counter_id,
        dtype=str(resolved["storage"]["dtype"]),
        shards_digest=str(resolved.get("_digest", "")),
        sources_digest=str(resolved_registry.get("_digest", "")),
        tokenizer_digest=str(resolved_tokenizer.get("_digest", "")),
        protected_slice_tokens=slice_tokens,
    )


@dataclass(frozen=True)
class StreamingBuildResult:
    """Result of a bounded-memory build and optional isolation report.

    The streaming producer deliberately does not run the frozen near-dedup algorithm:
    that algorithm requires a global candidate set and is quadratic by design.  Operators
    must therefore supply an isolation-clean accepted stream (and set
    ``isolation_verified``).  Explicit ``cluster_id`` fields, when present, are still
    checked while streaming so a bad upstream partition fails closed.
    """

    manifests: Mapping[str, SplitManifest]
    # None is intentional: global near-deduplication is not run by this stage.
    isolation: IsolationReport | None


class _LazyManifestMapping(Mapping[str, SplitManifest]):
    """Manifest handles backed by staged JSON, loading one only when requested."""

    def __init__(self, root: Path, paths: Mapping[str, Path]) -> None:
        self._root = Path(root)
        self._paths = dict(paths)

    def __getitem__(self, key: str) -> SplitManifest:
        try:
            return load_split_manifest(self._root / self._paths[key])
        except KeyError as exc:
            raise KeyError(key) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)


class _StreamingShardWriter:
    """Write one shard incrementally while retaining only its document index."""

    def __init__(
        self,
        root: Path,
        tokenizer,
        *,
        source_id: str,
        boundary: str,
        namespace: str,
        shard_index: int,
        protocol: Mapping[str, Any],
        tokenizer_protocol: Mapping[str, Any],
        token_counter_id: str,
    ) -> None:
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.source_id = source_id
        self.boundary = boundary
        self.namespace = namespace
        self.shard_index = shard_index
        self.protocol = protocol
        self.tokenizer_protocol = tokenizer_protocol
        self.token_counter_id = token_counter_id
        self.document_ids: list[str] = []
        self.offsets: list[int] = []
        self.lengths: list[int] = []
        self.protected_slices: set[str] = set()
        self.cursor = 0
        self._hasher = hashlib.sha256()
        relative_path = f"{namespace}/shard_{shard_index:05d}{protocol['storage']['file_suffix']}"
        self.relative_path = relative_path
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        self.target = target
        self.temporary = target.with_name(target.name + ".part")
        self._handle = self.temporary.open("wb")

    def append(self, document: ShardDocument) -> int:
        ids = encode_document(self.tokenizer, document.text, protocol=self.tokenizer_protocol)
        ceiling = int(self.tokenizer_protocol["vocabulary"]["maximum_representable_id"])
        highest = max(ids, default=0)
        if highest > ceiling or min(ids, default=0) < 0:
            self.abort()
            raise ShardContractError(
                f"{SHARD_TOKEN_ID_OUT_OF_RANGE}: token ID {highest} does not fit the packed uint16 format"
            )
        payload = np.asarray(ids, dtype=STORAGE_DTYPE).tobytes()
        self._handle.write(payload)
        self._hasher.update(payload)
        self.document_ids.append(document.document_id)
        self.offsets.append(self.cursor)
        self.lengths.append(len(ids))
        if document.protected_slice:
            self.protected_slices.add(document.protected_slice)
        self.cursor += len(ids)
        return len(ids)

    def finish(self) -> ShardRecord:
        if not self.document_ids:
            self.abort()
            raise StreamingShardError("cannot publish an empty streaming shard")
        self._handle.flush()
        self._handle.close()
        os.replace(self.temporary, self.target)
        return ShardRecord(
            shard_id=str(self.protocol["storage"]["shard_id_format"]).format(
                namespace=self.namespace, index=self.shard_index
            ),
            namespace=self.namespace,
            source_id=self.source_id,
            boundary=self.boundary,
            relative_path=self.relative_path,
            dtype=str(self.protocol["storage"]["dtype"]),
            token_count=self.cursor,
            document_count=len(self.document_ids),
            document_ids=tuple(self.document_ids),
            document_token_offsets=tuple(self.offsets),
            document_token_lengths=tuple(self.lengths),
            eos_id=int(self.tokenizer_protocol["document_boundaries"]["eos_token_id"]),
            sha256=self._hasher.hexdigest(),
            token_counter_id=self.token_counter_id,
            protected_slices=tuple(sorted(self.protected_slices)),
        )

    def abort(self) -> None:
        try:
            self._handle.close()
        finally:
            self.temporary.unlink(missing_ok=True)


def build_split_manifests_streaming(
    root: Path,
    tokenizer,
    documents: Iterable[ShardDocument],
    *,
    shard_document_budget: int = 4096,
    shard_token_budget: int | None = None,
    protocol: Mapping[str, Any] | None = None,
    registry: Mapping[str, Any] | None = None,
    tokenizer_protocol: Mapping[str, Any] | None = None,
    token_counter_id: str = PROVISIONAL_TOKEN_COUNTER_ID,
    isolation_verified: bool = False,
) -> StreamingBuildResult:
    """Build all split manifests from a sorted JSONL stream with bounded memory.

    Input must be globally ordered by ``(boundary, source_id, document_id)`` using the
    frozen boundary order.  This is a deliberate precondition: sorting an arbitrary
    corpus would require materializing it or an external-sort job, neither of which is
    hidden here.  A disk-backed SQLite key set detects duplicate IDs without retaining all
    IDs in RAM.  The output directory must be absent or empty; all files are staged in a
    sibling temporary directory and published with one atomic rename, so an interrupted
    or malformed run cannot leave a plausible final manifest behind.

    ``isolation_verified`` is required because full near-dedup clustering is not a
    streaming operation.  Upstream isolation evidence must cover all four boundaries and
    protected slices.  If rows include ``cluster_id`` (an optional accepted-stream field),
    this function independently rejects a cluster crossing either boundary or slice.
    """
    if shard_document_budget <= 0:
        raise StreamingShardError("shard_document_budget must be positive")
    if shard_token_budget is not None and shard_token_budget <= 0:
        raise StreamingShardError("shard_token_budget must be positive")
    if not isolation_verified:
        raise StreamingShardError(
            "streaming production requires upstream isolation evidence; pass isolation_verified=True"
        )
    resolved = protocol or load_shard_protocol()
    resolved_registry = registry if registry is not None else load_source_registry()
    resolved_tokenizer = tokenizer_protocol or load_tokenizer_protocol()
    splits = split_index(resolved)
    resolved_token_budget = int(
        shard_token_budget
        if shard_token_budget is not None
        else resolved["storage"]["default_shard_token_budget"]
    )
    boundary_order = {name: index for index, name in enumerate(ISOLATED_BOUNDARIES)}
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise StreamingShardError(
            f"streaming output {root} already exists and is non-empty; use a new output directory"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))
    ids_db = stage / ".document_ids.sqlite3"
    current_key: tuple[int, str, str] | None = None
    current_source: str | None = None
    current_boundary: str | None = None
    current_index = 0
    writer: _StreamingShardWriter | None = None
    slices: dict[str, dict[str, int]] = {}
    total_documents = 0
    connection: sqlite3.Connection | None = None

    def flush_writer() -> None:
        nonlocal writer, current_index
        if writer is None:
            return
        record = writer.finish()
        if connection is None:
            raise StreamingShardError("internal streaming state lost its index database")
        connection.execute(
            "INSERT INTO shard_records(boundary, source_id, shard_index, record_json) VALUES (?, ?, ?, ?)",
            (writer.boundary, writer.source_id, current_index, json.dumps(record.to_dict(), sort_keys=True)),
        )
        current_index += 1
        writer = None

    try:
        with sqlite3.connect(ids_db) as connection:
            connection.execute("CREATE TABLE document_ids (document_id TEXT PRIMARY KEY)")
            connection.execute(
                "CREATE TABLE cluster_scope (cluster_id TEXT PRIMARY KEY, boundary TEXT NOT NULL, protected_slice TEXT)"
            )
            connection.execute(
                "CREATE TABLE shard_records (boundary TEXT NOT NULL, source_id TEXT NOT NULL, "
                "shard_index INTEGER NOT NULL, record_json TEXT NOT NULL, "
                "PRIMARY KEY (boundary, source_id, shard_index))"
            )
            for document in documents:
                total_documents += 1
                if not isinstance(document, ShardDocument):
                    raise StreamingShardError("streaming documents must be ShardDocument instances")
                if not document.document_id.strip() or not document.source_id.strip() or not document.text:
                    raise StreamingShardError(f"malformed document row {document.document_id!r}")
                if document.boundary not in boundary_order or document.boundary not in splits:
                    raise StreamingShardError(f"{SHARD_BOUNDARY_MIXED}: unknown boundary {document.boundary!r}")
                if document.boundary in VALIDATION_BOUNDARIES and not document.protected_slice:
                    raise StreamingShardError(
                        f"{VALIDATION_SLICE_NOT_DECLARED}: {document.document_id!r} has no protected slice"
                    )
                if document.protected_slice and document.protected_slice not in EXPECTED_PROTECTED_SLICES:
                    raise StreamingShardError(f"unknown protected slice {document.protected_slice!r}")
                try:
                    connection.execute("INSERT INTO document_ids(document_id) VALUES (?)", (document.document_id,))
                except sqlite3.IntegrityError as exc:
                    raise StreamingShardError(
                        f"duplicate document ID {document.document_id!r}"
                    ) from exc
                key = (boundary_order[document.boundary], document.source_id, document.document_id)
                if current_key is not None and key < current_key:
                    raise StreamingShardError(
                        "streaming input is not sorted by (boundary, source_id, document_id); "
                        f"saw {document.document_id!r} after {current_key[2]!r}"
                    )
                current_key = key
                namespace = namespace_for(
                    document.source_id,
                    document.boundary,
                    protocol=resolved,
                    registry=resolved_registry,
                )
                cluster_id = getattr(document, "cluster_id", None)
                if cluster_id:
                    scope = (document.boundary, document.protected_slice)
                    prior_row = connection.execute(
                        "SELECT boundary, protected_slice FROM cluster_scope WHERE cluster_id = ?",
                        (cluster_id,),
                    ).fetchone()
                    if prior_row is not None:
                        prior = (str(prior_row[0]), str(prior_row[1]) if prior_row[1] is not None else None)
                        if prior[0] != scope[0] or (prior[1] and scope[1] and prior[1] != scope[1]):
                            reason = (
                                CLUSTER_CROSSES_BOUNDARY
                                if prior[0] != scope[0]
                                else CLUSTER_CROSSES_PROTECTED_SLICE
                            )
                            raise ShardIsolationError(
                                f"{reason}: cluster {cluster_id!r} crosses {prior!r} and {scope!r}"
                            )
                    else:
                        connection.execute(
                            "INSERT INTO cluster_scope(cluster_id, boundary, protected_slice) VALUES (?, ?, ?)",
                            (cluster_id, scope[0], scope[1]),
                        )

                source_changed = (document.source_id, document.boundary) != (current_source, current_boundary)
                if source_changed:
                    flush_writer()
                    current_index = 0
                    current_source, current_boundary = document.source_id, document.boundary
                if writer is None or len(writer.document_ids) >= shard_document_budget or (
                    writer.cursor >= resolved_token_budget and writer.document_ids
                ):
                    flush_writer()
                    writer = _StreamingShardWriter(
                        stage,
                        tokenizer,
                        source_id=document.source_id,
                        boundary=document.boundary,
                        namespace=namespace,
                        shard_index=current_index,
                        protocol=resolved,
                        tokenizer_protocol=resolved_tokenizer,
                        token_counter_id=token_counter_id,
                    )
                token_length = writer.append(document)
                if document.protected_slice:
                    bucket = slices.setdefault(document.boundary, {})
                    bucket[document.protected_slice] = bucket.get(document.protected_slice, 0) + token_length
            flush_writer()
            connection.commit()
        if total_documents == 0:
            raise StreamingShardError("streaming input contains no documents")

        manifest_paths: dict[str, Path] = {}

        def atom(value: Any) -> bytes:
            # Match SplitManifest.content_hash(), which uses json.dumps(..., sort_keys=True)
            # with its default comma/colon spacing.
            return json.dumps(value, sort_keys=True).encode("utf-8")

        def write_streamed_manifest(split_id: str) -> None:
            if connection is None:
                raise StreamingShardError("internal streaming state lost its index database")
            source_rows = connection.execute(
                "SELECT source_id, MIN(shard_index), SUM(json_extract(record_json, '$.token_count')), "
                "SUM(json_extract(record_json, '$.document_count')) FROM shard_records "
                "WHERE boundary = ? GROUP BY source_id ORDER BY source_id",
                (split_id,),
            ).fetchall()
            if not source_rows:
                return
            total_tokens = sum(int(row[2]) for row in source_rows)
            total_documents = sum(int(row[3]) for row in source_rows)
            split_slices = slices.get(split_id, {})
            payload_path = stage / f".{split_id}.manifest.payload"
            with payload_path.open("wb") as payload:
                def field(name: str, value: bytes, *, first: bool) -> bool:
                    if not first:
                        payload.write(b", ")
                    payload.write(atom(name) + b": " + value)
                    return False

                payload.write(b"{")
                first = True
                first = field("boundary", atom(str(splits[split_id]["boundary"])), first=first)
                first = field("document_count", atom(total_documents), first=first)
                first = field("dtype", atom(str(resolved["storage"]["dtype"])), first=first)
                first = field("namespaces", b"[", first=first)
                for namespace_index, (source_id, _, source_tokens, source_documents) in enumerate(source_rows):
                    if namespace_index:
                        payload.write(b", ")
                    first_json = connection.execute(
                        "SELECT record_json FROM shard_records WHERE boundary = ? AND source_id = ? "
                        "ORDER BY shard_index LIMIT 1",
                        (split_id, str(source_id)),
                    ).fetchone()[0]
                    namespace = str(json.loads(first_json)["namespace"])
                    payload.write(b"{")
                    namespace_first = True
                    namespace_first = field("boundary", atom(split_id), first=namespace_first)
                    namespace_first = field("document_count", atom(int(source_documents)), first=namespace_first)
                    namespace_first = field("namespace", atom(namespace), first=namespace_first)
                    namespace_first = field("shards", b"[", first=namespace_first)
                    rows = connection.execute(
                        "SELECT record_json FROM shard_records WHERE boundary = ? AND source_id = ? "
                        "ORDER BY shard_index",
                        (split_id, str(source_id)),
                    )
                    shard_first = True
                    for (record_json,) in rows:
                        if not shard_first:
                            payload.write(b", ")
                        payload.write(atom(json.loads(record_json)))
                        shard_first = False
                    payload.write(b"]")
                    namespace_first = False
                    namespace_first = field("source_id", atom(str(source_id)), first=namespace_first)
                    field("token_count", atom(int(source_tokens)), first=namespace_first)
                    payload.write(b"}")
                payload.write(b"]")
                first = False
                first = field("protected_slice_tokens", atom(dict(sorted(split_slices.items()))), first=first)
                first = field("schema_version", atom(_MANIFEST_SCHEMA_VERSION), first=first)
                first = field("shards_digest", atom(str(resolved.get("_digest", ""))), first=first)
                first = field("sources_digest", atom(str(resolved_registry.get("_digest", ""))), first=first)
                first = field("split_id", atom(split_id), first=first)
                first = field("token_count", atom(total_tokens), first=first)
                first = field("token_counter_id", atom(token_counter_id), first=first)
                field("tokenizer_digest", atom(str(resolved_tokenizer.get("_digest", ""))), first=first)
                payload.write(b"}")
            hasher = hashlib.sha256()
            with payload_path.open("rb") as payload:
                for chunk in iter(lambda: payload.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            final_path = split_manifest_path(stage, split_id, resolved)
            temporary = final_path.with_name(final_path.name + ".part")
            final_path.parent.mkdir(parents=True, exist_ok=True)
            with payload_path.open("rb") as source, temporary.open("wb+") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.seek(-1, os.SEEK_END)
                target.write(b', "content_hash": ' + atom(digest) + b"}\n")
            os.replace(temporary, final_path)
            payload_path.unlink(missing_ok=True)
            manifest_paths[split_id] = Path(final_path.relative_to(stage))

        for split_id in ISOLATED_BOUNDARIES:
            write_streamed_manifest(split_id)
        # Global near-deduplication was not run here. Keep this None so verification remains
        # NOT_RUN rather than converting a caller attestation into fabricated evidence.
        isolation = None
        if connection is not None:
            connection.close()
        ids_db.unlink(missing_ok=True)
        if root.exists():
            # The precondition above proved this is an empty caller-created directory;
            # remove only that exact directory so the staged tree can be renamed on Windows.
            root.rmdir()
        os.replace(stage, root)
        return StreamingBuildResult(_LazyManifestMapping(root, manifest_paths), isolation)
    except Exception:
        if connection is not None:
            connection.close()
        if writer is not None:
            writer.abort()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def split_manifest_path(root: Path, split_id: str, protocol: Mapping[str, Any] | None = None) -> Path:
    """The one file each split's manifest is written to (independence by construction)."""
    resolved = protocol or load_shard_protocol()
    splits = split_index(resolved)
    if split_id not in splits:
        raise ShardContractError(f"split {split_id!r} is not declared in the frozen shard contract")
    return Path(root) / str(splits[split_id]["manifest_file"])


def write_split_manifest(root: Path, manifest: SplitManifest, protocol: Mapping[str, Any] | None = None) -> Path:
    """Write one split manifest as JSON. Never writes two splits into one document."""
    path = split_manifest_path(root, manifest.split_id, protocol)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    payload["content_hash"] = manifest.content_hash()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_split_manifest(path: Path) -> SplitManifest:
    """Load a split manifest and reject a payload whose totals or content hash do not match.

    ``token_count`` and ``document_count`` are derived from the shard records, so they are
    reconciled explicitly: a tampered aggregate would otherwise be silently recomputed.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded_hash = payload.pop("content_hash", None)
    manifest = SplitManifest.from_dict(payload)
    for name, derived in (("token_count", manifest.token_count), ("document_count", manifest.document_count)):
        if name in payload and int(payload[name]) != derived:
            raise ShardContractError(
                f"{SHARD_DIGEST_MISMATCH}: {path} declares {name}={payload[name]} but its shards hold {derived}"
            )
    if recorded_hash is not None and recorded_hash != manifest.content_hash():
        raise ShardContractError(f"{SHARD_DIGEST_MISMATCH}: {path} content hash does not match its payload")
    return manifest


def read_shard_tokens(root: Path, record: ShardRecord) -> np.memmap:
    """Memory-map one shard's uint16 tokens, verifying the recorded length."""
    path = Path(root) / record.relative_path
    tokens = np.memmap(path, dtype=STORAGE_DTYPE, mode="r")
    if tokens.size != record.token_count:
        raise ShardContractError(
            f"{SHARD_DIGEST_MISMATCH}: {record.shard_id} holds {tokens.size} tokens, manifest says {record.token_count}"
        )
    return tokens


def iter_shard_documents(root: Path, record: ShardRecord) -> Iterable[tuple[str, np.ndarray]]:
    """Yield ``(document_id, token slice)`` using only the recorded deterministic boundaries."""
    tokens = read_shard_tokens(root, record)
    for document_id, offset, length in zip(
        record.document_ids, record.document_token_offsets, record.document_token_lengths
    ):
        yield document_id, np.asarray(tokens[offset : offset + length])


# --------------------------------------------------------------------------------------
# Cluster isolation across boundaries and protected slices
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IsolationViolation:
    cluster_id: str
    reason_code: str
    values: tuple[str, ...]
    document_ids: tuple[str, ...]


@dataclass(frozen=True)
class IsolationReport:
    """Cluster isolation evidence for one candidate set across all four boundaries."""

    dedup: DedupReport
    boundary_violations: tuple[IsolationViolation, ...]
    slice_violations: tuple[IsolationViolation, ...]
    undeclared_slice_document_ids: tuple[str, ...]
    reason_counts: Mapping[str, int]

    @property
    def violations(self) -> tuple[IsolationViolation, ...]:
        return self.boundary_violations + self.slice_violations

    @property
    def ok(self) -> bool:
        return not self.violations and not self.undeclared_slice_document_ids


def isolate_documents(
    documents: Iterable[ShardDocument],
    *,
    protocol: Mapping[str, Any] | None = None,
    dedup_protocol: Mapping[str, Any] | None = None,
) -> IsolationReport:
    """Cluster the candidates once, then report every boundary and protected-slice crossing.

    A single duplicate clustering answers both questions: a cluster whose members sit in
    more than one boundary leaks train/reserved/validation data, and a cluster whose members
    sit in more than one protected reporting slice makes slice regressions incomparable.
    """
    resolved = protocol or load_shard_protocol()
    resolved_dedup = dedup_protocol or load_dedup_protocol()
    materialized = list(documents)

    known = set(ISOLATED_BOUNDARIES)
    unexpected = sorted({document.boundary for document in materialized} - known)
    if unexpected:
        raise ShardContractError(f"boundaries {unexpected} are not isolated boundaries {ISOLATED_BOUNDARIES}")

    report = deduplicate((document.as_dedup_record() for document in materialized), resolved_dedup)
    by_id = {document.document_id: document for document in materialized}

    boundary_violations: list[IsolationViolation] = []
    slice_violations: list[IsolationViolation] = []
    for cluster_id in sorted(report.clusters):
        members = report.clusters[cluster_id]
        boundaries = sorted({by_id[doc_id].boundary for doc_id in members})
        if len(boundaries) > 1:
            boundary_violations.append(
                IsolationViolation(cluster_id, CLUSTER_CROSSES_BOUNDARY, tuple(boundaries), tuple(members))
            )
        if bool(resolved["isolation"]["protected_slice_isolation"]):
            slices = sorted({by_id[doc_id].protected_slice for doc_id in members if by_id[doc_id].protected_slice})
            if len(slices) > 1:
                slice_violations.append(
                    IsolationViolation(cluster_id, CLUSTER_CROSSES_PROTECTED_SLICE, tuple(slices), tuple(members))
                )

    undeclared: tuple[str, ...] = ()
    if bool(resolved["isolation"]["validation_requires_declared_slice"]):
        undeclared = tuple(
            sorted(
                document.document_id
                for document in materialized
                if document.boundary in VALIDATION_BOUNDARIES and not document.protected_slice
            )
        )

    counts = dict(report.reason_counts)
    if boundary_violations:
        counts[CLUSTER_CROSSES_BOUNDARY] = len(boundary_violations)
    if slice_violations:
        counts[CLUSTER_CROSSES_PROTECTED_SLICE] = len(slice_violations)
    if undeclared:
        counts[VALIDATION_SLICE_NOT_DECLARED] = len(undeclared)
    return IsolationReport(report, tuple(boundary_violations), tuple(slice_violations), undeclared, counts)


def enforce_shard_isolation(report: IsolationReport) -> None:
    """Fail closed on any boundary crossing, slice crossing, or undeclared validation slice."""
    problems: list[str] = []
    for violation in report.violations:
        problems.append(
            f"{violation.reason_code}: {violation.cluster_id} spans {list(violation.values)} "
            f"via {list(violation.document_ids)}"
        )
    if report.undeclared_slice_document_ids:
        problems.append(
            f"{VALIDATION_SLICE_NOT_DECLARED}: {list(report.undeclared_slice_document_ids)}"
        )
    if problems:
        raise ShardIsolationError("; ".join(problems))


# --------------------------------------------------------------------------------------
# Profile selection
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileDecisionRecord:
    """The dated scope-reduction record ``degraded_v1`` requires (Plan Section 4.2)."""

    profile_id: str
    date: str
    owner: str
    reason: str

    def missing_fields(self, required: Sequence[str]) -> tuple[str, ...]:
        payload = {"profile_id": self.profile_id, "date": self.date, "owner": self.owner, "reason": self.reason}
        return tuple(name for name in required if not str(payload.get(name) or "").strip())

    def parsed_date(self, date_format: str) -> date | None:
        try:
            return datetime.strptime(self.date, date_format).date()
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class ProfileSelection:
    """The profile a measured stable token total earns, or an explicit failure."""

    profile_id: str | None
    status: str
    reason_code: str
    observed_tokens: int
    threshold: int | None
    reason: str

    @property
    def ok(self) -> bool:
        return self.status == PASS


def select_profile(
    stable_tokens: int,
    *,
    decision_record: ProfileDecisionRecord | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> ProfileSelection:
    """Select ``full_v1``/``degraded_v1`` from a measured stable token total, or fail closed.

    ``full_v1`` needs at least 11B accepted stable tokens. ``degraded_v1`` needs at least 8B
    **and** a complete dated decision record: Plan Section 4.2 makes the reduction explicit,
    not a silent gate pass. Anything below 8B earns no profile.
    """
    resolved = protocol or load_shard_protocol()
    observed = int(stable_tokens)
    profiles = sorted(
        (dict(entry) for entry in resolved["profiles"]),
        key=lambda entry: int(entry["minimum_accepted_stable_tokens"]),
        reverse=True,
    )
    thresholds = {str(entry["profile_id"]): int(entry["minimum_accepted_stable_tokens"]) for entry in profiles}
    lowest = min(thresholds.values())

    for entry in profiles:
        profile_id = str(entry["profile_id"])
        threshold = int(entry["minimum_accepted_stable_tokens"])
        if observed < threshold:
            continue
        if not bool(entry.get("requires_dated_decision_record", False)):
            return ProfileSelection(
                profile_id,
                PASS,
                SHARD_OK,
                observed,
                threshold,
                f"{observed} accepted stable tokens meet the {profile_id} minimum of {threshold}",
            )
        required = [str(name) for name in entry.get("decision_record_required_fields", [])]
        date_format = str(entry.get("decision_record_date_format", "%Y-%m-%d"))
        if decision_record is None:
            return ProfileSelection(
                None,
                FAIL,
                DEGRADED_DECISION_RECORD_MISSING,
                observed,
                threshold,
                f"{profile_id} requires a dated scope-reduction decision record and none was supplied",
            )
        missing = decision_record.missing_fields(required)
        if missing:
            return ProfileSelection(
                None,
                FAIL,
                DEGRADED_DECISION_RECORD_MISSING,
                observed,
                threshold,
                f"{profile_id} decision record is missing {list(missing)}",
            )
        if decision_record.profile_id != profile_id:
            return ProfileSelection(
                None,
                FAIL,
                DEGRADED_DECISION_RECORD_MISSING,
                observed,
                threshold,
                f"decision record names profile {decision_record.profile_id!r}, not {profile_id!r}",
            )
        if decision_record.parsed_date(date_format) is None:
            return ProfileSelection(
                None,
                FAIL,
                DEGRADED_DECISION_RECORD_MISSING,
                observed,
                threshold,
                f"decision record date {decision_record.date!r} is not a {date_format} date",
            )
        return ProfileSelection(
            profile_id,
            PASS,
            SHARD_OK,
            observed,
            threshold,
            f"{observed} accepted stable tokens meet the {profile_id} minimum of {threshold} "
            f"under the dated decision record of {decision_record.date}",
        )

    return ProfileSelection(
        None,
        FAIL,
        PROFILE_BELOW_THRESHOLD,
        observed,
        lowest,
        f"{observed} accepted stable tokens are below every frozen profile minimum (lowest is {lowest})",
    )


def assert_profile_selected(selection: ProfileSelection) -> None:
    """Fail closed when a profile was claimed below its threshold or without its record."""
    if not selection.ok:
        raise ProfileSelectionError(f"{selection.reason_code}: {selection.reason}")


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


def _verdict(check_id: str, requirement: str, observed: Any, ok: bool, reason: str) -> CheckResult:
    return CheckResult(check_id, requirement, str(observed), PASS if ok else FAIL, reason)


def _deferred(check_id: str, requirement: str, observed: Any, reason: str) -> CheckResult:
    return CheckResult(check_id, requirement, str(observed), DEFERRED, reason)


def verify_shard_files(
    root: Path,
    manifest: SplitManifest,
    *,
    protocol: Mapping[str, Any] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> tuple[CheckResult, ...]:
    """Reconcile every packed shard against its record: dtype, digest, tags, boundaries."""
    resolved = protocol or load_shard_protocol()
    expected_dtype = str(resolved["storage"]["dtype"])
    results: list[CheckResult] = []

    for shard in manifest.shards:
        path = Path(root) / shard.relative_path
        prefix = f"shard.{shard.shard_id}"
        if not path.is_file():
            results.append(
                _verdict(f"{prefix}.present", "packed shard exists", "missing", False, "shard file is absent")
            )
            continue

        results.append(
            _verdict(
                f"{prefix}.dtype",
                f"{expected_dtype} storage",
                shard.dtype,
                shard.dtype == expected_dtype,
                f"{SHARD_DTYPE_NOT_UINT16} unless the shard is packed as {expected_dtype}",
            )
        )
        size = path.stat().st_size
        expected_size = shard.token_count * int(resolved["storage"]["bytes_per_token"])
        results.append(
            _verdict(
                f"{prefix}.byte_length",
                f"{expected_size} bytes",
                size,
                size == expected_size,
                "packed byte length reconciles with the recorded token count",
            )
        )
        tokens = np.memmap(path, dtype=STORAGE_DTYPE, mode="r")
        results.append(
            _verdict(
                f"{prefix}.digest",
                shard.sha256,
                hashlib.sha256(tokens.tobytes()).hexdigest(),
                hashlib.sha256(tokens.tobytes()).hexdigest() == shard.sha256,
                f"{SHARD_DIGEST_MISMATCH} unless the packed bytes match the recorded digest",
            )
        )
        try:
            expected_namespace = namespace_for(shard.source_id, shard.boundary, protocol=resolved, registry=registry)
            namespace_ok = expected_namespace == shard.namespace
            namespace_reason = "shard sits in its one frozen source-tagged namespace"
        except ShardContractError as error:
            expected_namespace, namespace_ok, namespace_reason = str(error), False, str(error)
        results.append(
            _verdict(f"{prefix}.namespace", expected_namespace, shard.namespace, namespace_ok, namespace_reason)
        )
        results.append(
            _verdict(
                f"{prefix}.source_tag",
                "non-empty source ID recorded",
                shard.source_id,
                bool(str(shard.source_id).strip()),
                f"{SHARD_MISSING_SOURCE_TAG} unless the shard records its source",
            )
        )
        results.append(
            _verdict(
                f"{prefix}.path_under_namespace",
                f"{shard.namespace}/...",
                shard.relative_path,
                shard.relative_path.startswith(f"{shard.namespace}/"),
                "shard path is inside its namespace, so sources are not pre-mixed on disk",
            )
        )

        offsets_ok = (
            len(shard.document_ids) == shard.document_count
            and len(shard.document_token_lengths) == shard.document_count
            and len(shard.document_token_offsets) == shard.document_count
            and sum(shard.document_token_lengths) == shard.token_count
            and list(shard.document_token_offsets)
            == list(np.cumsum([0, *shard.document_token_lengths[:-1]]).astype(int))
        )
        results.append(
            _verdict(
                f"{prefix}.document_offsets",
                "offsets are the cumulative document lengths and sum to the token count",
                {"documents": shard.document_count, "tokens": shard.token_count},
                offsets_ok,
                f"{SHARD_DOCUMENT_BOUNDARY_MISMATCH} unless recorded boundaries reconcile",
            )
        )
        boundary_ids = [
            int(tokens[offset + length - 1])
            for offset, length in zip(shard.document_token_offsets, shard.document_token_lengths)
            if length > 0
        ]
        results.append(
            _verdict(
                f"{prefix}.document_eos",
                f"EOS {shard.eos_id} closes every document",
                sorted(set(boundary_ids)),
                all(value == shard.eos_id for value in boundary_ids),
                f"{SHARD_DOCUMENT_BOUNDARY_MISMATCH} unless every document ends at the boundary EOS",
            )
        )
        highest = int(tokens.max()) if tokens.size else 0
        ceiling = int(resolved["storage"]["maximum_representable_id"])
        results.append(
            _verdict(
                f"{prefix}.token_id_range",
                f"<= {ceiling}",
                highest,
                highest <= ceiling,
                f"{SHARD_TOKEN_ID_OUT_OF_RANGE} unless every ID fits uint16",
            )
        )
    return tuple(results)


def manifest_independence_problems(manifests: Sequence[SplitManifest]) -> tuple[str, ...]:
    """Every overlap that would make two split manifests non-independent."""
    problems: list[str] = []
    seen_shards: dict[str, str] = {}
    seen_paths: dict[str, str] = {}
    seen_documents: dict[str, str] = {}
    for manifest in manifests:
        for shard in manifest.shards:
            if shard.shard_id in seen_shards:
                problems.append(
                    f"shard ID {shard.shard_id!r} appears in {seen_shards[shard.shard_id]!r} and {manifest.split_id!r}"
                )
            seen_shards[shard.shard_id] = manifest.split_id
            if shard.relative_path in seen_paths:
                problems.append(
                    f"path {shard.relative_path!r} appears in {seen_paths[shard.relative_path]!r} and {manifest.split_id!r}"
                )
            seen_paths[shard.relative_path] = manifest.split_id
            for document_id in shard.document_ids:
                if document_id in seen_documents:
                    problems.append(
                        f"document {document_id!r} appears in {seen_documents[document_id]!r} and {manifest.split_id!r}"
                    )
                seen_documents[document_id] = manifest.split_id
    return tuple(problems)


def assert_manifests_independent(manifests: Sequence[SplitManifest]) -> None:
    """Fail closed when two splits share a shard, a path, or a document."""
    problems = manifest_independence_problems(manifests)
    if problems:
        raise ShardContractError(f"{SPLIT_MANIFESTS_NOT_INDEPENDENT}: {'; '.join(problems)}")


def verify_mixture(
    manifests: Mapping[str, SplitManifest],
    *,
    scale: str = SCALE_FIXTURE,
    decision_record: ProfileDecisionRecord | None = None,
    isolation: IsolationReport | None = None,
    protocol: Mapping[str, Any] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> tuple[CheckResult, ...]:
    """Reconcile post-filter shares, token totals, isolation, and profile selection (G1).

    ``scale`` decides whether measured-quantity checks are evaluated. At
    :data:`SCALE_FIXTURE` the token totals, shares, and profile selection report
    ``DEFERRED`` with their blocker, owner, and next action, because a tiny local fixture is
    not a real-corpus measurement and must never be reported as a G1 pass.
    """
    if scale not in (SCALE_FIXTURE, SCALE_FINAL):
        raise ShardContractError(f"scale {scale!r} must be {SCALE_FIXTURE!r} or {SCALE_FINAL!r}")
    resolved = protocol or load_shard_protocol()
    resolved_registry = registry if registry is not None else load_source_registry()
    mixture = resolved["mixture_verification"]
    final_counter = str(mixture["final_token_counter_id"])
    measured = scale == SCALE_FINAL
    deferral_reason = (
        f"blocker={mixture['provisional_blocker']} owner={mixture['provisional_owner']} "
        f"next_action={mixture['provisional_next_action']}"
    )
    results: list[CheckResult] = []

    # 1. Namespaces: every Plan 5.4 namespace present exactly once, no pre-mixing.
    stable = manifests.get(STABLE_TRAIN)
    reserved = manifests.get(RESERVED)
    observed_namespaces = tuple(
        namespace.namespace
        for split_id in (STABLE_TRAIN, RESERVED)
        if (manifest := manifests.get(split_id)) is not None
        for namespace in manifest.namespaces
    )
    results.append(
        _verdict(
            "shards.namespaces_complete",
            str(EXPECTED_SHARD_NAMESPACES),
            observed_namespaces,
            set(observed_namespaces) == set(EXPECTED_SHARD_NAMESPACES)
            and len(observed_namespaces) == len(EXPECTED_SHARD_NAMESPACES),
            "every source-tagged namespace from Plan Section 5.4 is produced exactly once",
        )
    )
    all_shards = [shard for manifest in manifests.values() for shard in manifest.shards]
    results.append(
        _verdict(
            "shards.one_source_per_shard",
            "one source ID and one boundary per shard",
            len(all_shards),
            all(bool(str(shard.source_id).strip()) and bool(str(shard.boundary).strip()) for shard in all_shards),
            f"{SHARD_SOURCE_MIXED}/{SHARD_MISSING_SOURCE_TAG} unless every shard carries one source tag",
        )
    )
    results.append(
        _verdict(
            "shards.dtype_uint16",
            "uint16",
            sorted({shard.dtype for shard in all_shards}) or ["<none>"],
            bool(all_shards) and all(shard.dtype == str(resolved["storage"]["dtype"]) for shard in all_shards),
            f"{SHARD_DTYPE_NOT_UINT16} unless every shard is packed as uint16",
        )
    )

    # 2. Token counter identity. A provisional counter can never satisfy G1.
    counters = sorted({manifest.token_counter_id for manifest in manifests.values()})
    counter_ok = counters == [final_counter]
    results.append(
        CheckResult(
            "shards.token_counter_identity",
            f"post-filter counts measured with {final_counter}",
            str(counters),
            PASS if counter_ok else DEFERRED,
            "shares and totals are measured with the final tokenizer"
            if counter_ok
            else f"counts still use {PROVISIONAL_TOKEN_COUNTER_ID}; {deferral_reason}",
        )
    )
    evaluate_totals = measured and counter_ok

    # 3. Stable mixture shares and total (Plan Section 4.1).
    tolerance = float(mixture["stable_share_absolute_tolerance"])
    stable_total = stable.token_count if stable is not None else 0
    for entry in resolved_registry["stable_sources"]:
        source_id = str(entry["source_id"])
        expected_share = float(entry["stable_share"])
        observed_tokens = stable.tokens_per_source.get(source_id, 0) if stable is not None else 0
        observed_share = (observed_tokens / stable_total) if stable_total else 0.0
        check_id = f"shards.stable_share.{source_id}"
        requirement = f"{expected_share:.4f} +/- {tolerance:.4f} of post-filter stable tokens"
        observed = f"{observed_share:.6f} ({observed_tokens} tokens)"
        if evaluate_totals:
            results.append(
                _verdict(
                    check_id,
                    requirement,
                    observed,
                    abs(observed_share - expected_share) <= tolerance,
                    f"{MIXTURE_SHARE_OUT_OF_TOLERANCE} unless the measured share matches Plan Section 4.1",
                )
            )
        else:
            results.append(_deferred(check_id, requirement, observed, deferral_reason))

    reserved_tolerance = float(mixture["reserved_share_absolute_tolerance"])
    reserved_total = reserved.token_count if reserved is not None else 0
    for entry in resolved_registry["reserved_sources"]:
        source_id = str(entry["source_id"])
        expected_share = float(entry["reserved_share"])
        observed_tokens = reserved.tokens_per_source.get(source_id, 0) if reserved is not None else 0
        observed_share = (observed_tokens / reserved_total) if reserved_total else 0.0
        check_id = f"shards.reserved_share.{source_id}"
        requirement = f"{expected_share:.4f} +/- {reserved_tolerance:.4f} of post-filter reserved tokens"
        observed = f"{observed_share:.6f} ({observed_tokens} tokens)"
        if evaluate_totals:
            results.append(
                _verdict(
                    check_id,
                    requirement,
                    observed,
                    abs(observed_share - expected_share) <= reserved_tolerance,
                    f"{MIXTURE_SHARE_OUT_OF_TOLERANCE} unless the measured share matches Plan Section 4.3",
                )
            )
        else:
            results.append(_deferred(check_id, requirement, observed, deferral_reason))

    # 4. Reserved margin (Plan Section 4.3).
    reserved_minimum = int(resolved["reserved_pool"]["minimum_accepted_tokens"])
    if evaluate_totals:
        results.append(
            _verdict(
                "shards.reserved_pool_margin",
                f"at least {reserved_minimum} accepted reserved tokens",
                reserved_total,
                reserved_total >= reserved_minimum,
                f"{RESERVED_MARGIN_NOT_MET} unless the mandatory 30% margin is present",
            )
        )
    else:
        results.append(
            _deferred(
                "shards.reserved_pool_margin",
                f"at least {reserved_minimum} accepted reserved tokens",
                reserved_total,
                deferral_reason,
            )
        )

    # 5. Validation split sizes and independence (Plan Section 4.4).
    limits = resolved["validation_split_tokens"]
    minimum, maximum = int(limits["minimum"]), int(limits["maximum"])
    for split_id in VALIDATION_BOUNDARIES:
        manifest = manifests.get(split_id)
        observed = manifest.token_count if manifest is not None else 0
        check_id = f"shards.{split_id}_tokens"
        requirement = f"{minimum}-{maximum} tokens"
        if evaluate_totals:
            results.append(
                _verdict(
                    check_id,
                    requirement,
                    observed,
                    minimum <= observed <= maximum,
                    f"{VALIDATION_SPLIT_TOKENS_OUT_OF_RANGE} unless the split sits inside its frozen range",
                )
            )
        else:
            results.append(_deferred(check_id, requirement, observed, deferral_reason))

    present_validations = [manifests[split_id] for split_id in VALIDATION_BOUNDARIES if split_id in manifests]
    if len(present_validations) == 2:
        problems = manifest_independence_problems(present_validations)
        results.append(
            _verdict(
                "shards.validation_manifests_independent",
                "disjoint shard IDs, paths, and document IDs",
                problems or "disjoint",
                not problems,
                f"{SPLIT_MANIFESTS_NOT_INDEPENDENT} unless the two validation manifests are independent",
            )
        )
        for manifest in present_validations:
            observed_slices = tuple(sorted(manifest.protected_slice_tokens))
            results.append(
                _verdict(
                    f"shards.{manifest.split_id}_protected_slices",
                    str(EXPECTED_PROTECTED_SLICES),
                    observed_slices,
                    set(observed_slices) == set(EXPECTED_PROTECTED_SLICES),
                    "all four protected reporting slices are represented and frozen",
                )
            )
    else:
        results.append(
            _verdict(
                "shards.validation_manifests_independent",
                "both validation manifests supplied",
                sorted(manifest.split_id for manifest in present_validations),
                False,
                f"{SPLIT_MANIFESTS_NOT_INDEPENDENT} unless both validation splits are produced",
            )
        )

    # 6. Cluster isolation across all four boundaries and the protected slices.
    if isolation is None:
        results.append(
            CheckResult(
                "shards.cluster_isolation",
                "no cluster crosses a boundary or a protected slice",
                "<not supplied>",
                NOT_RUN,
                "no isolation report was supplied, so isolation is unproven rather than passing",
            )
        )
    else:
        results.append(
            _verdict(
                "shards.cluster_isolation",
                "no cluster crosses a boundary or a protected slice",
                {
                    "boundary_violations": len(isolation.boundary_violations),
                    "slice_violations": len(isolation.slice_violations),
                    "undeclared_slices": len(isolation.undeclared_slice_document_ids),
                },
                isolation.ok,
                f"{CLUSTER_CROSSES_BOUNDARY}/{CLUSTER_CROSSES_PROTECTED_SLICE} fail closed",
            )
        )

    # 7. Profile selection (Plan Section 4.2).
    if evaluate_totals:
        selection = select_profile(stable_total, decision_record=decision_record, protocol=resolved)
        results.append(
            CheckResult(
                "shards.profile_selection",
                "full_v1 at 11B, or degraded_v1 at 8B under a dated decision record",
                f"{selection.profile_id or 'none'} ({selection.observed_tokens} tokens)",
                selection.status,
                f"{selection.reason_code}: {selection.reason}",
            )
        )
    else:
        results.append(
            _deferred(
                "shards.profile_selection",
                "full_v1 at 11B, or degraded_v1 at 8B under a dated decision record",
                f"{stable_total} stable tokens",
                deferral_reason,
            )
        )

    # 8. Readiness for real-scale production stays explicit.
    readiness = resolved["readiness"]
    results.append(
        CheckResult(
            "shards.real_scale_production",
            "measured real-scale shards",
            str(readiness["billion_token_shard_production"]),
            str(readiness["billion_token_shard_production"]),
            f"blocker={readiness['blocker']} owner={readiness['owner']} next_action={readiness['next_action']}",
        )
    )
    return tuple(results)


def format_shard_report(results: Sequence[CheckResult]) -> str:
    """Human-readable summary of shard-file and mixture verification."""
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
    overall = "FAIL" if failures else ("DEFERRED" if DEFERRED in counts else "PASS")
    lines.append("RESULT: " + overall)
    if failures:
        lines.append("Failures:")
        lines.extend(f"  {result.check_id}: {result.reason}" for result in failures)
    return "\n".join(lines)


__all__ = [
    "CLUSTER_CROSSES_BOUNDARY",
    "CLUSTER_CROSSES_PROTECTED_SLICE",
    "DEGRADED_DECISION_RECORD_MISSING",
    "EXPECTED_PROTECTED_SLICES",
    "EXPECTED_SHARD_NAMESPACES",
    "FROZEN_SHARD_PROTOCOL_SHA256",
    "ISOLATED_BOUNDARIES",
    "IsolationReport",
    "IsolationViolation",
    "MIXTURE_SHARE_OUT_OF_TOLERANCE",
    "NamespaceManifest",
    "PROFILE_BELOW_THRESHOLD",
    "RESERVED",
    "RESERVED_MARGIN_NOT_MET",
    "REPOSITORY_ROOT",
    "SCALE_FINAL",
    "SCALE_FIXTURE",
    "SHARDS_PROTOCOL_PATH",
    "SHARD_BOUNDARY_MIXED",
    "SHARD_DIGEST_MISMATCH",
    "SHARD_DOCUMENT_BOUNDARY_MISMATCH",
    "SHARD_DTYPE_NOT_UINT16",
    "SHARD_FAIL_CLOSED_REASON_CODES",
    "SHARD_MISSING_SOURCE_TAG",
    "SHARD_NAMESPACE_UNREGISTERED",
    "SHARD_OK",
    "SHARD_SOURCE_MIXED",
    "SHARD_TOKEN_ID_OUT_OF_RANGE",
    "SPLIT_MANIFESTS_NOT_INDEPENDENT",
    "STABLE_TRAIN",
    "VALIDATION_BOUNDARIES",
    "VALIDATION_DEV",
    "VALIDATION_FINAL",
    "VALIDATION_SLICE_NOT_DECLARED",
    "VALIDATION_SPLIT_TOKENS_OUT_OF_RANGE",
    "ProfileDecisionRecord",
    "ProfileSelection",
    "ProfileSelectionError",
    "ShardContractError",
    "ShardDocument",
    "ShardIsolationError",
    "StreamingBuildResult",
    "StreamingShardError",
    "ShardRecord",
    "ShardsNotReadyError",
    "SplitManifest",
    "assert_manifests_independent",
    "assert_profile_selected",
    "assert_ready_for_real_shard_production",
    "build_split_manifest",
    "build_split_manifests_streaming",
    "enforce_shard_isolation",
    "format_shard_report",
    "isolate_documents",
    "iter_shard_documents",
    "load_shard_protocol",
    "load_split_manifest",
    "manifest_independence_problems",
    "namespace_for",
    "namespace_index",
    "pack_shard_tokens",
    "read_shard_tokens",
    "select_profile",
    "split_index",
    "split_manifest_path",
    "verify_mixture",
    "verify_shard_files",
    "write_shard",
    "write_split_manifest",
]
