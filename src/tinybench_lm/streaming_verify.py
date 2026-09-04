"""Bounded-memory verification of production shard manifests and aggregate evidence.

The normal :func:`tinybench_lm.shards.verify_mixture` API is intentionally convenient for
small fixtures, but ``SplitManifest`` contains every document boundary.  Materialising one
of the production manifests therefore defeats the bounded-memory property of G1.  This
module parses the frozen JSON manifest grammar incrementally, keeps only one shard record in
memory, and uses a temporary SQLite index for cross-split uniqueness.

The verifier is deliberately evidence-driven.  An operator boolean or a command-line
attestation is not isolation evidence; the optional isolation report must explicitly record
zero boundary/slice violations and coverage of all frozen boundaries and protected slices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

import numpy as np

from .environment import CheckResult
from .shards import (
    CLUSTER_CROSSES_BOUNDARY,
    CLUSTER_CROSSES_PROTECTED_SLICE,
    DEGRADED_DECISION_RECORD_MISSING,
    EXPECTED_PROTECTED_SLICES,
    EXPECTED_SHARD_NAMESPACES,
    FAIL,
    MIXTURE_SHARE_OUT_OF_TOLERANCE,
    NOT_RUN,
    PASS,
    PROFILE_BELOW_THRESHOLD,
    RESERVED,
    RESERVED_MARGIN_NOT_MET,
    SCALE_FINAL,
    SCALE_FIXTURE,
    SHARD_BOUNDARY_MIXED,
    SHARD_DIGEST_MISMATCH,
    SHARD_DOCUMENT_BOUNDARY_MISMATCH,
    SHARD_DTYPE_NOT_UINT16,
    SHARD_MISSING_SOURCE_TAG,
    SHARD_NAMESPACE_UNREGISTERED,
    SHARD_SOURCE_MIXED,
    SHARD_TOKEN_ID_OUT_OF_RANGE,
    SPLIT_MANIFESTS_NOT_INDEPENDENT,
    STABLE_TRAIN,
    VALIDATION_DEV,
    VALIDATION_FINAL,
    VALIDATION_SLICE_NOT_DECLARED,
    VALIDATION_SPLIT_TOKENS_OUT_OF_RANGE,
    ProfileDecisionRecord,
    ShardContractError,
    load_shard_protocol,
    namespace_for,
    split_index,
)
from .source_manifest import FINAL_TOKEN_COUNTER_ID, PROVISIONAL_TOKEN_COUNTER_ID, load_source_registry
from .tokenizer import load_tokenizer_protocol


class StreamingManifestError(ShardContractError):
    """A manifest is malformed or cannot be verified incrementally."""


@dataclass(frozen=True)
class StreamingShard:
    """One shard record; arrays are bounded by the shard's document ceiling."""

    values: Mapping[str, Any]
    document_ids: tuple[str, ...]
    offsets: tuple[int, ...]
    lengths: tuple[int, ...]
    protected_slices: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        value = dict(self.values)
        value["document_ids"] = list(self.document_ids)
        value["document_token_offsets"] = list(self.offsets)
        value["document_token_lengths"] = list(self.lengths)
        value["protected_slices"] = list(self.protected_slices)
        return value


@dataclass
class StreamingManifestSummary:
    """Aggregates from one manifest, with no retained complete document list."""

    path: Path
    fields: dict[str, Any] = field(default_factory=dict)
    namespace_count: int = 0
    shard_count: int = 0
    document_count: int = 0
    token_count: int = 0
    source_tokens: dict[str, int] = field(default_factory=dict)
    namespace_names: list[str] = field(default_factory=list)
    protected_slice_tokens: dict[str, int] = field(default_factory=dict)
    observed_protected_slices: set[str] = field(default_factory=set)
    content_hash: str | None = None
    parse_error: str | None = None
    namespace_spool: Path | None = None
    checks: list[CheckResult] = field(default_factory=list)
    shard_counter_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class StreamingVerificationReport:
    """Machine-readable result of one bounded-memory G1 verification."""

    results: tuple[CheckResult, ...]
    facts: Mapping[str, Any]

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(item for item in self.results if item.status == FAIL)

    @property
    def status(self) -> str:
        if self.failures:
            return FAIL
        if any(item.status in {NOT_RUN, "DEFERRED", "BLOCKED"} for item in self.results):
            return NOT_RUN
        return PASS

    @property
    def ok(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "results": [item.__dict__ for item in self.results],
            "facts": dict(self.facts),
        }


class _JsonStream:
    """Small incremental JSON reader used instead of a third-party parser dependency."""

    def __init__(self, handle, *, chunk_size: int = 1024 * 1024):
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False

    def _fill(self) -> None:
        if self.eof:
            return
        chunk = self.handle.read(self.chunk_size)
        if chunk:
            self.buffer += chunk
        else:
            self.eof = True

    def _compact(self) -> None:
        if self.position > self.chunk_size and self.position > len(self.buffer) // 2:
            self.buffer = self.buffer[self.position :]
            self.position = 0

    def peek(self) -> str:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            self._fill()
            if self.position >= len(self.buffer) and self.eof:
                raise StreamingManifestError("unexpected end of JSON")

    def take(self, expected: str | None = None) -> str:
        value = self.peek()
        if expected is not None and value != expected:
            raise StreamingManifestError(f"expected {expected!r}, got {value!r}")
        self.position += 1
        self._compact()
        return value

    def string(self) -> str:
        self.take('"')
        start = self.position - 1
        escaped = False
        while True:
            while self.position < len(self.buffer):
                char = self.buffer[self.position]
                self.position += 1
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    raw = self.buffer[start : self.position]
                    try:
                        value = json.loads(raw)
                        self._compact()
                        return value
                    except json.JSONDecodeError as exc:
                        raise StreamingManifestError("invalid JSON string") from exc
            if self.eof:
                raise StreamingManifestError("unterminated JSON string")
            self._fill()

    def scalar(self) -> Any:
        self.peek()
        start = self.position
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position] not in ",]} \t\r\n":
                self.position += 1
            if self.position < len(self.buffer):
                token = self.buffer[start : self.position]
                break
            if self.eof:
                token = self.buffer[start : self.position]
                break
            self._fill()
        try:
            value = json.loads(token)
            self._compact()
            return value
        except json.JSONDecodeError as exc:
            raise StreamingManifestError(f"invalid JSON scalar {token!r}") from exc

    def skip_value(self) -> None:
        opening = self.peek()
        if opening == '"':
            self.string()
            return
        if opening == "{":
            self.take("{")
            if self.peek() == "}":
                self.take("}")
                return
            while True:
                self.string()
                self.take(":")
                self.skip_value()
                if self.peek() == "}":
                    self.take("}")
                    return
                self.take(",")
        elif opening == "[":
            self.take("[")
            if self.peek() == "]":
                self.take("]")
                return
            while True:
                self.skip_value()
                if self.peek() == "]":
                    self.take("]")
                    return
                self.take(",")
        else:
            self.scalar()

    def array(self, item: Callable[[], Any]) -> list[Any]:
        self.take("[")
        values: list[Any] = []
        if self.peek() == "]":
            self.take("]")
            return values
        while True:
            values.append(item())
            if self.peek() == "]":
                self.take("]")
                return values
            self.take(",")


def _read_value(stream: _JsonStream) -> Any:
    opening = stream.peek()
    if opening == '"':
        return stream.string()
    if opening in "[{":
        # Used only for small metadata fields. Large namespaces/shards are handled below.
        decoder = json.JSONDecoder()
        while True:
            try:
                value, end = decoder.raw_decode(stream.buffer, stream.position)
                stream.position = end
                stream._compact()
                return value
            except json.JSONDecodeError:
                if stream.eof:
                    raise StreamingManifestError("invalid JSON value")
                stream._fill()
    return stream.scalar()


def _read_shard(stream: _JsonStream, *, on_document: Callable[[str], None]) -> StreamingShard:
    stream.take("{")
    values: dict[str, Any] = {}
    document_ids: list[str] = []
    offsets: list[int] = []
    lengths: list[int] = []
    protected: list[str] = []
    if stream.peek() == "}":
        stream.take("}")
        return StreamingShard(values, (), (), (), ())
    while True:
        key = stream.string()
        stream.take(":")
        if key == "document_ids":
            document_ids = stream.array(stream.string)
            for value in document_ids:
                if not isinstance(value, str) or not value:
                    raise StreamingManifestError("document_ids must contain non-blank strings")
                on_document(value)
        elif key == "document_token_offsets":
            offsets = stream.array(stream.scalar)
        elif key == "document_token_lengths":
            lengths = stream.array(stream.scalar)
        elif key == "protected_slices":
            protected = stream.array(stream.string)
        else:
            # Unknown extension fields are deliberately skipped.  A malformed extension
            # must not turn into an unbounded in-memory value.
            if key in {"boundary", "document_count", "dtype", "eos_id", "namespace", "relative_path", "sha256", "shard_id", "source_id", "token_count", "token_counter_id"}:
                values[key] = _read_value(stream)
            else:
                stream.skip_value()
        if stream.peek() == "}":
            stream.take("}")
            return StreamingShard(values, tuple(document_ids), tuple(int(v) for v in offsets), tuple(int(v) for v in lengths), tuple(protected))
        stream.take(",")


def _read_namespace(
    stream: _JsonStream,
    *,
    on_shard: Callable[[StreamingShard], None],
    namespace_sink: Callable[[dict[str, Any], int, int, int, Path], None],
    spool_directory: Path,
) -> None:
    stream.take("{")
    values: dict[str, Any] = {}
    shard_spool = tempfile.NamedTemporaryFile("wb", prefix="namespace-", suffix=".json", dir=spool_directory, delete=False)
    shard_count = token_count = document_count = 0
    first_shard = True
    if stream.peek() != "}":
        while True:
            key = stream.string()
            stream.take(":")
            if key == "shards":
                stream.take("[")
                if stream.peek() != "]":
                    while True:
                        shard = _read_shard(stream, on_document=lambda _doc: None)
                        on_shard(shard)
                        if not first_shard:
                            shard_spool.write(b", ")
                        shard_spool.write(_canonical(shard.payload()))
                        first_shard = False
                        shard_count += 1
                        token_count += int(shard.values.get("token_count", 0))
                        document_count += len(shard.document_ids)
                        if stream.peek() == "]":
                            break
                        stream.take(",")
                stream.take("]")
            else:
                if key in {"boundary", "document_count", "namespace", "protected_slice_tokens", "source_id", "token_count"}:
                    values[key] = _read_value(stream)
                else:
                    stream.skip_value()
            if stream.peek() == "}":
                break
            stream.take(",")
    stream.take("}")
    shard_spool.close()
    namespace_sink(values, shard_count, token_count, document_count, Path(shard_spool.name))


def _canonical(value: Any) -> bytes:
    # SplitManifest.content_hash uses json.dumps(..., sort_keys=True) with its default
    # separators and ensure_ascii=True; reproduce that exact byte representation.
    return json.dumps(value, sort_keys=True).encode("utf-8")


def _write_namespace_canonical(target, values: Mapping[str, Any], shard_path: Path) -> None:
    """Write one canonical namespace object while keeping its shard array on disk."""
    keys = sorted(values)
    target.write(b"{")
    for index, key in enumerate(keys):
        if index:
            target.write(b", ")
        target.write(_canonical(key))
        target.write(b": ")
        if key == "shards":
            target.write(b"[")
            with shard_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    target.write(chunk)
            target.write(b"]")
        else:
            target.write(_canonical(values[key]))
    target.write(b"}")


def _verify_content_hash(summary: StreamingManifestSummary) -> bool:
    """Recreate SplitManifest.content_hash from disk-bounded namespace payloads."""
    if summary.content_hash is None:
        return True
    if summary.namespace_spool is None:
        return False
    digest = hashlib.sha256()
    digest.update(b"{")
    keys = sorted(summary.fields)
    for index, key in enumerate(keys):
        if index:
            digest.update(b", ")
        digest.update(_canonical(key))
        digest.update(b": ")
        if key == "namespaces":
            digest.update(b"[")
            with summary.namespace_spool.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"]")
        else:
            digest.update(_canonical(summary.fields[key]))
    digest.update(b"}")
    return digest.hexdigest() == summary.content_hash


def _parse_manifest(
    path: Path,
    connection: sqlite3.Connection,
    *,
    root: Path,
    protocol: Mapping[str, Any],
    registry: Mapping[str, Any],
    spool_directory: Path,
) -> StreamingManifestSummary:
    summary = StreamingManifestSummary(Path(path))
    namespace_spool = tempfile.NamedTemporaryFile("wb", prefix="namespaces-", suffix=".json", dir=spool_directory, delete=False)
    first_namespace = True
    summary.namespace_spool = Path(namespace_spool.name)
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            stream = _JsonStream(handle)
            stream.take("{")
            if stream.peek() != "}":
                while True:
                    key = stream.string()
                    stream.take(":")
                    if key == "namespaces":
                        stream.take("[")
                        if stream.peek() != "]":
                            while True:
                                def consume_shard(shard: StreamingShard) -> None:
                                    summary.shard_count += 1
                                    summary.document_count += len(shard.document_ids)
                                    summary.token_count += int(shard.values.get("token_count", 0))
                                    source = str(shard.values.get("source_id", ""))
                                    summary.source_tokens[source] = summary.source_tokens.get(source, 0) + int(shard.values.get("token_count", 0))
                                    summary.observed_protected_slices.update(shard.protected_slices)
                                    for document_id in shard.document_ids:
                                        try:
                                            connection.execute("INSERT INTO documents(document_id, split_id) VALUES (?, ?)", (document_id, str(summary.fields.get("split_id", path.stem))))
                                        except sqlite3.IntegrityError:
                                            summary.parse_error = summary.parse_error or f"{SPLIT_MANIFESTS_NOT_INDEPENDENT}: duplicate document ID {document_id!r}"
                                    payload = shard.payload()
                                    summary.shard_counter_ids.add(str(payload.get("token_counter_id", "")))
                                    results = _verify_shard(root, payload, protocol=protocol, registry=registry)
                                    summary.checks.extend(results)
                                    shard_id = str(payload.get("shard_id", ""))
                                    relative = str(payload.get("relative_path", ""))
                                    try:
                                        connection.execute("INSERT INTO shards(shard_id, relative_path, split_id) VALUES (?, ?, ?)", (shard_id, relative, str(summary.fields.get("split_id", path.stem))))
                                    except sqlite3.IntegrityError:
                                        summary.parse_error = summary.parse_error or f"{SPLIT_MANIFESTS_NOT_INDEPENDENT}: duplicate shard ID {shard_id!r}"
                                    if any(item.status == FAIL for item in results):
                                        summary.parse_error = summary.parse_error or "; ".join(item.reason for item in results if item.status == FAIL)

                                def consume_namespace(values: dict[str, Any], shard_count: int, token_count: int, document_count: int, shard_path: Path) -> None:
                                    nonlocal first_namespace
                                    summary.namespace_count += 1
                                    namespace = str(values.get("namespace", ""))
                                    summary.namespace_names.append(namespace)
                                    namespace_payload = dict(values)
                                    namespace_payload["shards"] = None
                                    namespace_tokens = token_count
                                    namespace_documents = document_count
                                    if not first_namespace:
                                        namespace_spool.write(b", ")
                                    _write_namespace_canonical(namespace_spool, namespace_payload, shard_path)
                                    first_namespace = False
                                    expected_tokens = int(values.get("token_count", namespace_tokens))
                                    expected_documents = int(values.get("document_count", namespace_documents))
                                    if expected_tokens != namespace_tokens or expected_documents != namespace_documents:
                                        summary.parse_error = summary.parse_error or f"{SHARD_DIGEST_MISMATCH}: namespace {namespace!r} aggregate does not reconcile"
                                    source_id = str(values.get("source_id", ""))
                                    boundary = str(values.get("boundary", ""))
                                    try:
                                        expected_namespace = namespace_for(source_id, boundary, protocol=protocol, registry=registry)
                                        namespace_ok = expected_namespace == namespace
                                    except ShardContractError:
                                        expected_namespace, namespace_ok = "frozen namespace resolution failed", False
                                    summary.checks.append(_verdict(f"namespace.{namespace}.identity", expected_namespace, namespace, namespace_ok, SHARD_NAMESPACE_UNREGISTERED))
                                    for slice_name in values.get("protected_slice_tokens", {}):
                                        name = str(slice_name)
                                        summary.protected_slice_tokens[name] = summary.protected_slice_tokens.get(name, 0) + int(values["protected_slice_tokens"][slice_name])
                                    try:
                                        shard_path.unlink()
                                    except OSError:
                                        pass

                                _read_namespace(stream, on_shard=consume_shard, namespace_sink=consume_namespace, spool_directory=spool_directory)
                                if stream.peek() == "]":
                                    break
                                stream.take(",")
                        stream.take("]")
                    elif key == "content_hash":
                        summary.content_hash = str(_read_value(stream))
                    else:
                        if key in {"boundary", "document_count", "dtype", "protected_slice_tokens", "schema_version", "shards_digest", "sources_digest", "split_id", "token_count", "token_counter_id", "tokenizer_digest"}:
                            summary.fields[key] = _read_value(stream)
                        else:
                            stream.skip_value()
                    if stream.peek() == "}":
                        break
                    stream.take(",")
            stream.take("}")
            summary.fields["namespaces"] = "__streamed__"
            connection.commit()
            declared_tokens = summary.fields.get("token_count")
            declared_documents = summary.fields.get("document_count")
            if declared_tokens is not None and int(declared_tokens) != summary.token_count:
                summary.parse_error = summary.parse_error or f"{SHARD_DIGEST_MISMATCH}: manifest token_count does not reconcile"
            if declared_documents is not None and int(declared_documents) != summary.document_count:
                summary.parse_error = summary.parse_error or f"{SHARD_DIGEST_MISMATCH}: manifest document_count does not reconcile"
            declared_counter = str(summary.fields.get("token_counter_id", ""))
            if declared_counter and summary.shard_counter_ids != {declared_counter}:
                summary.parse_error = summary.parse_error or f"{SHARD_DIGEST_MISMATCH}: shard token counters do not match manifest"
            declared_slices = summary.fields.get("protected_slice_tokens", {})
            if isinstance(declared_slices, Mapping):
                summary.protected_slice_tokens = {
                    str(name): int(tokens) for name, tokens in declared_slices.items()
                }
            else:
                summary.parse_error = summary.parse_error or "protected_slice_tokens must be a mapping"
            namespace_spool.flush()
            if summary.content_hash is None:
                summary.parse_error = summary.parse_error or f"{SHARD_DIGEST_MISMATCH}: content hash is absent"
            elif not _verify_content_hash(summary):
                summary.parse_error = summary.parse_error or f"{SHARD_DIGEST_MISMATCH}: content hash does not match manifest payload"
            # The fields entry is a sentinel for the canonical hash writer only.
            summary.fields["namespaces"] = []
    except (OSError, UnicodeError, ValueError, StreamingManifestError, json.JSONDecodeError) as exc:
        summary.parse_error = str(exc)
    finally:
        namespace_spool.close()
    return summary


def _result(check_id: str, requirement: str, observed: Any, status: str, reason: str) -> CheckResult:
    return CheckResult(check_id, requirement, str(observed), status, reason)


def _verdict(check_id: str, requirement: str, observed: Any, ok: bool, reason: str) -> CheckResult:
    return _result(check_id, requirement, observed, PASS if ok else FAIL, reason)


def _not_run(check_id: str, requirement: str, observed: Any, reason: str) -> CheckResult:
    return _result(check_id, requirement, observed, NOT_RUN, reason)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_shard(root: Path, shard: Mapping[str, Any], *, protocol: Mapping[str, Any], registry: Mapping[str, Any]) -> tuple[CheckResult, ...]:
    shard_id = str(shard.get("shard_id", "<missing>"))
    prefix = f"shard.{shard_id}"
    source_id = str(shard.get("source_id", ""))
    boundary = str(shard.get("boundary", ""))
    relative = str(shard.get("relative_path", ""))
    namespace = str(shard.get("namespace", ""))
    relative_posix = PurePosixPath(relative.replace("\\", "/"))
    namespace_parts = PurePosixPath(namespace).parts
    path_ok = (not relative_posix.is_absolute() and ".." not in relative_posix.parts and relative_posix.parts[: len(namespace_parts)] == namespace_parts)
    path = root / Path(*relative_posix.parts) if path_ok else root / "__invalid_manifest_path__"
    expected_dtype = str(protocol["storage"]["dtype"])
    results: list[CheckResult] = []
    if not path_ok:
        return (_verdict(f"{prefix}.path_under_namespace", f"{namespace}/...", relative, False, SHARD_SOURCE_MIXED),)
    if not path.is_file():
        return (_result(f"{prefix}.present", "packed shard exists", "missing", FAIL, SHARD_DIGEST_MISMATCH),)
    results.append(_verdict(f"{prefix}.dtype", expected_dtype, shard.get("dtype"), shard.get("dtype") == expected_dtype, SHARD_DTYPE_NOT_UINT16))
    token_count = int(shard.get("token_count", 0))
    expected_size = token_count * int(protocol["storage"]["bytes_per_token"])
    size = path.stat().st_size
    results.append(_verdict(f"{prefix}.byte_length", f"{expected_size} bytes", size, size == expected_size, SHARD_DIGEST_MISMATCH))
    actual_digest = _hash_file(path)
    results.append(_verdict(f"{prefix}.digest", shard.get("sha256"), actual_digest, actual_digest == shard.get("sha256"), SHARD_DIGEST_MISMATCH))
    try:
        expected_namespace = namespace_for(source_id, boundary, protocol=protocol, registry=registry)
        namespace_ok = expected_namespace == str(shard.get("namespace", ""))
        namespace_reason = "source and boundary resolve to the frozen namespace"
    except ShardContractError as exc:
        expected_namespace, namespace_ok, namespace_reason = str(exc), False, str(exc)
    results.append(_verdict(f"{prefix}.namespace", expected_namespace, shard.get("namespace"), namespace_ok, SHARD_NAMESPACE_UNREGISTERED))
    results.append(_verdict(f"{prefix}.source_tag", "non-empty source ID", source_id, bool(source_id), SHARD_MISSING_SOURCE_TAG))
    results.append(_verdict(f"{prefix}.path_under_namespace", f"{namespace}/...", relative, path_ok, SHARD_SOURCE_MIXED))
    ids = list(shard.get("document_ids", []))
    offsets = [int(v) for v in shard.get("document_token_offsets", [])]
    lengths = [int(v) for v in shard.get("document_token_lengths", [])]
    boundaries_ok = len(ids) == len(offsets) == len(lengths) == int(shard.get("document_count", -1)) and sum(lengths) == token_count and offsets == list(np.cumsum([0, *lengths[:-1]]).astype(int))
    results.append(_verdict(f"{prefix}.document_offsets", "document offsets and lengths reconcile", {"documents": len(ids), "tokens": token_count}, boundaries_ok, SHARD_DOCUMENT_BOUNDARY_MISMATCH))
    if size == expected_size and expected_size % 2 == 0:
        tokens = np.memmap(path, dtype="<u2", mode="r")
        eos = int(shard.get("eos_id", -1))
        ends = [int(tokens[offset + length - 1]) for offset, length in zip(offsets, lengths) if length > 0 and offset + length <= tokens.size]
        results.append(_verdict(f"{prefix}.document_eos", f"EOS {eos} closes every document", sorted(set(ends)), all(value == eos for value in ends) and len(ends) == len(lengths), SHARD_DOCUMENT_BOUNDARY_MISMATCH))
        highest = int(tokens.max()) if tokens.size else 0
        ceiling = int(protocol["storage"]["maximum_representable_id"])
        results.append(_verdict(f"{prefix}.token_id_range", f"<= {ceiling}", highest, highest <= ceiling, SHARD_TOKEN_ID_OUT_OF_RANGE))
    else:
        results.append(_not_run(f"{prefix}.document_eos", "document EOS can be read", "not checked", "packed byte length failed"))
        results.append(_not_run(f"{prefix}.token_id_range", "token IDs can be read", "not checked", "packed byte length failed"))
    return tuple(results)


def _read_isolation_evidence(path: Path, *, protocol: Mapping[str, Any]) -> tuple[str, str]:
    if not path.is_file():
        return NOT_RUN, "isolation evidence file is absent"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return FAIL, f"invalid isolation evidence: {exc}"
    if not isinstance(payload, dict):
        return FAIL, "isolation evidence must be a JSON object"
    status = str(payload.get("status", ""))
    violations = payload.get("violations", payload.get("isolation", payload))
    if not isinstance(violations, dict):
        return FAIL, "isolation evidence has no structured counts"
    try:
        boundary_count = int(violations.get("boundary_violations", violations.get("boundary_violation_count", -1)))
        slice_count = int(violations.get("slice_violations", violations.get("slice_violation_count", -1)))
        undeclared = int(violations.get("undeclared_slices", violations.get("undeclared_slice_count", -1)))
    except (TypeError, ValueError) as exc:
        return FAIL, f"invalid isolation violation counts: {exc}"
    boundaries = set(str(v) for v in payload.get("boundaries", violations.get("boundaries", [])))
    slices = set(str(v) for v in payload.get("protected_slices", violations.get("protected_slices", [])))
    required_boundaries = {STABLE_TRAIN, RESERVED, VALIDATION_DEV, VALIDATION_FINAL}
    if status != PASS or min(boundary_count, slice_count, undeclared) != 0:
        return FAIL, f"{CLUSTER_CROSSES_BOUNDARY}/{CLUSTER_CROSSES_PROTECTED_SLICE}: non-zero or non-PASS isolation evidence"
    if not required_boundaries.issubset(boundaries) or not set(EXPECTED_PROTECTED_SLICES).issubset(slices):
        return FAIL, f"{VALIDATION_SLICE_NOT_DECLARED}: evidence does not cover all frozen boundaries and protected slices"
    return PASS, "isolation evidence explicitly reports zero violations and complete frozen coverage"


def verify_shard_outputs_streaming(
    root: Path,
    *,
    manifest_paths: Mapping[str, Path] | None = None,
    scale: str = SCALE_FINAL,
    decision_record: ProfileDecisionRecord | None = None,
    isolation_evidence: Path | None = None,
    protocol: Mapping[str, Any] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> StreamingVerificationReport:
    """Verify shard files and G1 aggregates without loading complete manifests.

    ``manifest_paths`` may provide a subset for a fixture; production verification defaults
    to the four frozen manifest filenames.  Final-scale token totals are only evaluated when
    every present manifest uses ``final_tokenizer_v1`` and all required split manifests exist.
    """
    if scale not in (SCALE_FIXTURE, SCALE_FINAL):
        raise ValueError(f"scale must be {SCALE_FIXTURE} or {SCALE_FINAL}")
    resolved = protocol or load_shard_protocol()
    resolved_registry = registry or load_source_registry()
    tokenizer_protocol = load_tokenizer_protocol()
    paths = dict(manifest_paths or {split_id: root / str(spec["manifest_file"]) for split_id, spec in split_index(resolved).items()})
    results: list[CheckResult] = []
    summaries: dict[str, StreamingManifestSummary] = {}
    with tempfile.TemporaryDirectory(prefix="tinybench-stream-verify-") as temp:
        with closing(sqlite3.connect(Path(temp) / "indexes.sqlite")) as db:
            db.execute("CREATE TABLE documents(document_id TEXT PRIMARY KEY, split_id TEXT NOT NULL)")
            db.execute("CREATE TABLE shards(shard_id TEXT PRIMARY KEY, relative_path TEXT NOT NULL, split_id TEXT NOT NULL)")
            for split_id, manifest_path in paths.items():
                if not Path(manifest_path).is_file():
                    continue
                summary = _parse_manifest(Path(manifest_path), db, root=Path(root), protocol=resolved, registry=resolved_registry, spool_directory=Path(temp))
                summaries[split_id] = summary
                if summary.parse_error:
                    results.append(_result(f"shards.{split_id}.manifest", "manifest parses and reconciles", summary.parse_error, FAIL, summary.parse_error))
                results.extend(summary.checks)
                declared_split = str(summary.fields.get("split_id", ""))
                declared_boundary = str(summary.fields.get("boundary", ""))
                expected_boundary = str(split_index(resolved).get(split_id, {}).get("boundary", split_id))
                results.append(_verdict(f"shards.{split_id}.identity", f"split_id={split_id}, boundary={expected_boundary}", f"split_id={declared_split}, boundary={declared_boundary}", declared_split == split_id and declared_boundary == expected_boundary, SHARD_BOUNDARY_MIXED))
                expected_fields = {
                    "schema_version": f"{resolved['protocol']}_{resolved['version']}",
                    "dtype": str(resolved["storage"]["dtype"]),
                    "shards_digest": str(resolved.get("_digest", "")),
                    "sources_digest": str(resolved_registry.get("_digest", "")),
                    "tokenizer_digest": str(tokenizer_protocol.get("_digest", "")),
                }
                for field_name, expected_value in expected_fields.items():
                    observed_value = str(summary.fields.get(field_name, ""))
                    results.append(
                        _verdict(
                            f"shards.{split_id}.{field_name}",
                            expected_value,
                            observed_value,
                            observed_value == expected_value,
                            SHARD_DIGEST_MISMATCH,
                        )
                    )
                db.commit()
            if not summaries:
                results.append(_not_run("shards.streaming_manifests", "production manifests exist", "none", "no shard manifest evidence was supplied"))
                return StreamingVerificationReport(tuple(results), {"manifests": 0})
            duplicate_paths = db.execute("SELECT relative_path, COUNT(*) FROM shards GROUP BY relative_path HAVING COUNT(*) > 1").fetchall()
        if duplicate_paths:
            results.append(_result("shards.manifest_paths_independent", "disjoint relative paths", duplicate_paths, FAIL, SPLIT_MANIFESTS_NOT_INDEPENDENT))
        else:
            results.append(_verdict("shards.manifest_paths_independent", "disjoint relative paths", "disjoint", True, "SQLite index found no path overlap"))
        expected_namespaces = set(EXPECTED_SHARD_NAMESPACES)
        observed_namespaces = [name for split_id in (STABLE_TRAIN, RESERVED) if split_id in summaries for name in summaries[split_id].namespace_names]
        results.append(_verdict("shards.namespaces_complete", str(EXPECTED_SHARD_NAMESPACES), observed_namespaces, set(observed_namespaces) == expected_namespaces and len(observed_namespaces) == len(expected_namespaces), SHARD_NAMESPACE_UNREGISTERED))
        counters = sorted({str(summary.fields.get("token_counter_id", "")) for summary in summaries.values()})
        counter_ok = counters == [FINAL_TOKEN_COUNTER_ID]
        counter_status = PASS if counter_ok else (NOT_RUN if counters == [PROVISIONAL_TOKEN_COUNTER_ID] else FAIL)
        results.append(_result("shards.token_counter_identity", FINAL_TOKEN_COUNTER_ID, counters, counter_status, "all manifest counters use the final tokenizer" if counter_ok else "final-tokenizer evidence is unavailable"))
        evaluate = scale == SCALE_FINAL and counter_ok and all(key in summaries for key in (STABLE_TRAIN, RESERVED, VALIDATION_DEV, VALIDATION_FINAL))
        if not evaluate:
            reason = "final-scale aggregates require all four manifests and final_tokenizer_v1"
            results.append(_not_run("shards.aggregate_measurements", "final-scale aggregates are measured", "not evaluated", reason))
        stable = summaries.get(STABLE_TRAIN)
        reserved = summaries.get(RESERVED)
        stable_total = sum(stable.source_tokens.values()) if stable else 0
        reserved_total = sum(reserved.source_tokens.values()) if reserved else 0
        mixture = resolved["mixture_verification"]
        tolerance = float(mixture["stable_share_absolute_tolerance"])
        for entry in resolved_registry["stable_sources"]:
            source_id = str(entry["source_id"])
            observed = stable.source_tokens.get(source_id, 0) if stable else 0
            share = observed / stable_total if stable_total else 0.0
            expected = float(entry["stable_share"])
            results.append(_verdict(f"shards.stable_share.{source_id}", f"{expected:.4f} +/- {tolerance:.4f}", f"{share:.6f} ({observed})", abs(share - expected) <= tolerance, MIXTURE_SHARE_OUT_OF_TOLERANCE) if evaluate else _not_run(f"shards.stable_share.{source_id}", "measured stable source share", observed, "final-scale aggregate measurement is not available"))
        reserved_tolerance = float(mixture["reserved_share_absolute_tolerance"])
        for entry in resolved_registry["reserved_sources"]:
            source_id = str(entry["source_id"])
            observed = reserved.source_tokens.get(source_id, 0) if reserved else 0
            share = observed / reserved_total if reserved_total else 0.0
            expected = float(entry["reserved_share"])
            results.append(_verdict(f"shards.reserved_share.{source_id}", f"{expected:.4f} +/- {reserved_tolerance:.4f}", f"{share:.6f} ({observed})", abs(share - expected) <= reserved_tolerance, MIXTURE_SHARE_OUT_OF_TOLERANCE) if evaluate else _not_run(f"shards.reserved_share.{source_id}", "measured reserved source share", observed, "final-scale aggregate measurement is not available"))
        minimum_reserved = int(resolved["reserved_pool"]["minimum_accepted_tokens"])
        results.append(_verdict("shards.reserved_pool_margin", f"at least {minimum_reserved}", reserved_total, reserved_total >= minimum_reserved, RESERVED_MARGIN_NOT_MET) if evaluate else _not_run("shards.reserved_pool_margin", f"at least {minimum_reserved}", reserved_total, "final-scale aggregate measurement is not available"))
        limits = resolved["validation_split_tokens"]
        for split_id in (VALIDATION_DEV, VALIDATION_FINAL):
            observed = sum(summaries[split_id].source_tokens.values()) if split_id in summaries else 0
            good = int(limits["minimum"]) <= observed <= int(limits["maximum"])
            results.append(_verdict(f"shards.{split_id}_tokens", f"{limits['minimum']}-{limits['maximum']}", observed, good, VALIDATION_SPLIT_TOKENS_OUT_OF_RANGE) if evaluate else _not_run(f"shards.{split_id}_tokens", "measured validation split size", observed, "final-scale aggregate measurement is not available"))
            if split_id in summaries:
                observed_slices = set(summaries[split_id].protected_slice_tokens)
                results.append(
                    _verdict(
                        f"shards.{split_id}_protected_slices",
                        str(EXPECTED_PROTECTED_SLICES),
                        sorted(observed_slices),
                        observed_slices == set(EXPECTED_PROTECTED_SLICES),
                        VALIDATION_SLICE_NOT_DECLARED,
                    )
                )
        if isolation_evidence is None:
            results.append(_not_run("shards.cluster_isolation", "zero cluster boundary/slice violations with complete evidence", "not supplied", "isolation evidence is unproven; an operator flag is not accepted"))
        else:
            status, reason = _read_isolation_evidence(Path(isolation_evidence), protocol=resolved)
            results.append(_result("shards.cluster_isolation", "zero cluster boundary/slice violations with complete evidence", status, status, reason))
        if evaluate:
            from .shards import select_profile
            selection = select_profile(stable_total, decision_record=decision_record, protocol=resolved)
            results.append(_result("shards.profile_selection", "full_v1 at 11B or degraded_v1 with dated record", selection.profile_id or "none", selection.status, f"{selection.reason_code}: {selection.reason}"))
        else:
            results.append(_not_run("shards.profile_selection", "full_v1 at 11B or degraded_v1 with dated record", stable_total, "final-scale profile selection is not available"))
        facts = {"manifests": len(summaries), "stable_tokens": stable_total, "reserved_tokens": reserved_total, "source_tokens": {split_id: dict(summary.source_tokens) for split_id, summary in summaries.items()}, "token_counter_ids": counters}
        return StreamingVerificationReport(tuple(results), facts)


def format_streaming_report(report: StreamingVerificationReport) -> str:
    width = max((len(item.check_id) for item in report.results), default=0)
    return "\n".join(f"{item.status:<7} {item.check_id:<{width}}  {item.observed}  {item.reason}" for item in report.results)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Bounded-memory verification of source-tagged production shards")
    parser.add_argument("root", type=Path, help="shard output root containing the four manifests")
    parser.add_argument("--scale", choices=(SCALE_FIXTURE, SCALE_FINAL), default=SCALE_FINAL)
    parser.add_argument("--isolation-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_shard_outputs_streaming(args.root, scale=args.scale, isolation_evidence=args.isolation_evidence)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report.status == PASS else 1


if __name__ == "__main__":
    raise SystemExit(_cli())


verify_mixture_streaming = verify_shard_outputs_streaming
verify_production_shards_streaming = verify_shard_outputs_streaming

__all__ = ["StreamingManifestError", "StreamingManifestSummary", "StreamingVerificationReport", "format_streaming_report", "verify_mixture_streaming", "verify_production_shards_streaming", "verify_shard_outputs_streaming"]
