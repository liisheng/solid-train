"""Disk-backed production corpus preparation.

The fixture implementations in :mod:`tinybench_lm.data_protocols` deliberately favour
clarity.  This module preserves their frozen decisions while moving corpus-sized state to
SQLite.  It never downloads a source implicitly; callers provide streamed candidates and
an explicit cache/output location.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sqlite3
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import yaml

from .data_protocols import (
    EXACT_DUPLICATE,
    KEEP,
    MIRROR_DUPLICATE,
    NEAR_DUPLICATE,
    UNIQUE,
    document_sha256,
    estimated_jaccard,
    load_dedup_protocol,
    minhash_signature,
    mirror_sha256,
    protocol_digest,
    word_shingles,
)
from .source_manifest import (
    CandidateDocument,
    build_record,
    load_filter_protocol,
    load_source_registry,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ACQUISITION_PROTOCOL_PATH = REPOSITORY_ROOT / "configs" / "data" / "acquisition_v1.yaml"
FROZEN_ACQUISITION_SHA256 = "f7a689193e898b499987581e76caca40aa36628a1386f0c47a7eb900ec5da45c"

BOUNDARY_ORDER = {"reserved": 0, "stable_train": 1, "validation_dev": 2, "validation_final": 3}
SCIENCE_WORD_RE = re.compile(r"[A-Za-z]+")


class CorpusPipelineError(RuntimeError):
    """Fail-closed production pipeline error."""


class AcquisitionProtocolError(CorpusPipelineError):
    """The frozen acquisition contract is absent, mutated, or inconsistent."""


class ResumeMismatchError(CorpusPipelineError):
    """Existing state belongs to different protocols or inputs."""


@dataclass(frozen=True)
class SourceCursor:
    source_id: str
    next_row_index: int
    complete: bool
    consumed_tokens: int = 0


@dataclass(frozen=True)
class StreamedSourceRow:
    row_index: int
    candidate: CandidateDocument
    score: float | None


@dataclass(frozen=True)
class PipelineSummary:
    documents_seen: int
    filter_accepted: int
    dedup_kept: int
    decontamination_clean: int
    assigned_documents: int
    assigned_tokens: int
    reason_counts: Mapping[str, int]
    boundary_tokens: Mapping[str, int]
    source_tokens: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents_seen": self.documents_seen,
            "filter_accepted": self.filter_accepted,
            "dedup_kept": self.dedup_kept,
            "decontamination_clean": self.decontamination_clean,
            "assigned_documents": self.assigned_documents,
            "assigned_tokens": self.assigned_tokens,
            "reason_counts": dict(self.reason_counts),
            "boundary_tokens": dict(self.boundary_tokens),
            "source_tokens": dict(self.source_tokens),
        }


@dataclass(frozen=True)
class SelectionResult:
    selection_id: str
    target_tokens: int
    selected_tokens: int
    selected_documents: int

    @property
    def complete(self) -> bool:
        return self.selected_tokens >= self.target_tokens


def load_acquisition_protocol(path: Path = ACQUISITION_PROTOCOL_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise AcquisitionProtocolError(f"acquisition protocol is absent: {path}")
    observed = protocol_digest(path)
    if path.resolve() == ACQUISITION_PROTOCOL_PATH.resolve() and observed != FROZEN_ACQUISITION_SHA256:
        raise AcquisitionProtocolError(
            f"{path.name} does not match its frozen digest "
            f"(expected {FROZEN_ACQUISITION_SHA256}, observed {observed})"
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("frozen") is not True or payload.get("version") != "v1":
        raise AcquisitionProtocolError("acquisition protocol must be a frozen v1 mapping")
    bands = payload["deduplication"]["minhash_candidate_index"]
    if int(bands["bands"]) * int(bands["rows_per_band"]) != 128:
        raise AcquisitionProtocolError("MinHash band layout must cover all 128 signature rows")
    payload["_digest"] = observed
    payload["_source"] = str(path)
    return payload


def selection_key(
    selection_id: str, origin_source_id: str, document_id: str, *, salt: str
) -> str:
    value = f"{salt}|{selection_id}|{origin_source_id}|{document_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def natural_document_id(row: Mapping[str, Any], origin_source_id: str, row_index: int) -> str:
    """Resolve the frozen source identity preference without relying on Python hashes."""
    for field in ("id", "url"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = row.get("METADATA")
    if isinstance(metadata, str):
        try:
            text_id = json.loads(metadata).get("text_id")
        except (TypeError, ValueError):
            text_id = None
        if text_id is not None and str(text_id).strip():
            return f"gutenberg:{text_id}"
    return f"{origin_source_id}:{row_index:012d}"


def assert_write_space(path: Path, protocol: Mapping[str, Any] | None = None) -> float:
    """Fail before new writes at the frozen free-space floor; return free fraction."""
    resolved = protocol or load_acquisition_protocol()
    target = Path(path).resolve()
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    fraction = usage.free / usage.total
    floor = float(resolved["storage"]["stop_writes_at_or_below_free_fraction"])
    if fraction <= floor:
        raise CorpusPipelineError(
            f"free space at {target} is {fraction:.2%}, at or below the frozen {floor:.0%} write-stop floor"
        )
    warning_floor = float(resolved["storage"]["warn_below_free_fraction"])
    if fraction < warning_floor:
        warnings.warn(
            f"free space at {target} is {fraction:.2%}, below the frozen {warning_floor:.0%} warning floor",
            RuntimeWarning,
            stacklevel=2,
        )
    return fraction


def source_registration(origin_source_id: str, registry: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    resolved = registry or load_source_registry()
    matches = [
        item
        for item in list(resolved["stable_sources"]) + list(resolved["reserved_sources"])
        if str(item["source_id"]) == origin_source_id
    ]
    if len(matches) != 1:
        raise CorpusPipelineError(
            f"physical source {origin_source_id!r} must have exactly one source registration, found {len(matches)}"
        )
    return matches[0]


def iter_huggingface_source(
    origin_source_id: str,
    *,
    cache_dir: Path,
    start_row_index: int = 0,
    registry: Mapping[str, Any] | None = None,
) -> Iterator[StreamedSourceRow]:
    """Stream one pinned physical source from an explicit cache directory.

    Importing ``datasets`` is deliberately lazy so fixture tests and state inspection never
    trigger network/cache initialization.
    """
    if start_row_index < 0:
        raise ValueError("start_row_index must be nonnegative")
    resolved = registry or load_source_registry()
    spec = source_registration(origin_source_id, resolved)
    cache = Path(cache_dir).resolve()
    assert_write_space(cache)
    from datasets import load_dataset

    options: dict[str, Any] = {
        "split": "train",
        "streaming": True,
        "revision": str(spec["intended_revision"]),
        "cache_dir": str(cache),
    }
    if spec.get("huggingface_config"):
        options["name"] = str(spec["huggingface_config"])
    stream = load_dataset(str(spec["huggingface_repo"]), **options)
    if start_row_index:
        stream = stream.skip(start_row_index)
    text_column = str(spec.get("text_column", "text"))
    for row_index, row in enumerate(stream, start=start_row_index):
        if not isinstance(row, Mapping):
            raise CorpusPipelineError(f"{origin_source_id} row {row_index} is not a mapping")
        text = row.get(text_column)
        text = text if isinstance(text, str) else ""
        document_id = natural_document_id(row, origin_source_id, row_index)
        url = row.get("url") or row.get("SOURCE")
        score = row.get("score")
        yield StreamedSourceRow(
            row_index,
            CandidateDocument(
                source_id=origin_source_id,
                document_id=document_id,
                text=text,
                revision=str(spec["intended_revision"]),
                license=str(spec["declared_license"]),
                url=str(url) if url else None,
            ),
            float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else None,
        )


def _signature_blob(signature: Sequence[int]) -> bytes:
    if len(signature) != 128:
        raise CorpusPipelineError(f"expected 128 MinHash rows, got {len(signature)}")
    return b"".join(int(value).to_bytes(8, "big", signed=False) for value in signature)


def _signature_from_blob(blob: bytes) -> tuple[int, ...]:
    if len(blob) != 1024:
        raise CorpusPipelineError(f"stored MinHash signature has {len(blob)} bytes, expected 1024")
    return tuple(int.from_bytes(blob[index : index + 8], "big") for index in range(0, len(blob), 8))


def _band_keys(signature: Sequence[int], bands: int, rows_per_band: int) -> Iterator[tuple[int, bytes]]:
    blob = _signature_blob(signature)
    width = rows_per_band * 8
    for band in range(bands):
        start = band * width
        yield band, hashlib.sha256(blob[start : start + width]).digest()


class CorpusState:
    """Restartable SQLite corpus state.

    Every mutating stage is idempotent.  Existing state is accepted only when all frozen
    protocol digests and the tokenizer identity match the caller's values.
    """

    def __init__(
        self,
        path: Path,
        *,
        token_counter: Callable[[str], int],
        token_counter_id: str,
        acquisition: Mapping[str, Any] | None = None,
        registry: Mapping[str, Any] | None = None,
        filters: Mapping[str, Any] | None = None,
        dedup: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.token_counter = token_counter
        self.token_counter_id = token_counter_id
        self.acquisition = dict(acquisition or load_acquisition_protocol())
        self.registry = dict(registry or load_source_registry())
        self.filters = dict(filters or load_filter_protocol())
        self.dedup = dict(dedup or load_dedup_protocol())
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self._bind_contract()

    def __enter__(self) -> "CorpusState":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS source_cursors (
                source_id TEXT PRIMARY KEY,
                next_row_index INTEGER NOT NULL CHECK(next_row_index >= 0),
                complete INTEGER NOT NULL CHECK(complete IN (0, 1)),
                consumed_tokens INTEGER NOT NULL DEFAULT 0 CHECK(consumed_tokens >= 0)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS documents (
                doc_key TEXT PRIMARY KEY,
                origin_source_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                score REAL,
                raw_sha256 TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                filter_action TEXT NOT NULL,
                filter_reason TEXT NOT NULL,
                token_count INTEGER,
                science_match INTEGER NOT NULL CHECK(science_match IN (0, 1)),
                UNIQUE(origin_source_id, document_id)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS documents_filter_order
                ON documents(filter_action, doc_key);
            CREATE TABLE IF NOT EXISTS ingest_events (
                origin_source_id TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                reason_code TEXT NOT NULL,
                matched_doc_key TEXT NOT NULL,
                PRIMARY KEY(origin_source_id, row_index)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS dedup_decisions (
                doc_key TEXT PRIMARY KEY REFERENCES documents(doc_key),
                action TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                cluster_id TEXT NOT NULL,
                matched_doc_key TEXT,
                estimated_jaccard REAL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS representatives (
                doc_key TEXT PRIMARY KEY REFERENCES documents(doc_key),
                exact_hash TEXT NOT NULL,
                mirror_hash TEXT NOT NULL,
                signature BLOB NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS representatives_exact ON representatives(exact_hash);
            CREATE INDEX IF NOT EXISTS representatives_mirror ON representatives(mirror_hash);
            CREATE TABLE IF NOT EXISTS minhash_bands (
                band INTEGER NOT NULL,
                band_key BLOB NOT NULL,
                doc_key TEXT NOT NULL REFERENCES representatives(doc_key),
                PRIMARY KEY(band, band_key, doc_key)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS boundary_review_pairs (
                left_doc_key TEXT NOT NULL,
                right_doc_key TEXT NOT NULL,
                estimated_jaccard REAL NOT NULL,
                PRIMARY KEY(left_doc_key, right_doc_key)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS decontamination (
                doc_key TEXT PRIMARY KEY REFERENCES representatives(doc_key),
                action TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                evidence_json TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS assignments (
                doc_key TEXT PRIMARY KEY REFERENCES representatives(doc_key),
                source_id TEXT NOT NULL,
                boundary TEXT NOT NULL,
                protected_slice TEXT,
                selection_id TEXT NOT NULL,
                selection_key TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS assignment_output_order
                ON assignments(boundary, source_id, doc_key);
            CREATE TABLE IF NOT EXISTS selection_keys (
                selection_id TEXT NOT NULL,
                doc_key TEXT NOT NULL REFERENCES representatives(doc_key),
                ranking_key TEXT NOT NULL,
                PRIMARY KEY(selection_id, doc_key)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS selection_ranking
                ON selection_keys(selection_id, ranking_key, doc_key);
            """
        )
        cursor_columns = {
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(source_cursors)")
        }
        if "consumed_tokens" not in cursor_columns:
            self.connection.execute(
                "ALTER TABLE source_cursors ADD COLUMN consumed_tokens INTEGER NOT NULL DEFAULT 0"
            )
        self.connection.commit()

    def _bind_contract(self) -> None:
        expected = {
            "acquisition_digest": str(self.acquisition.get("_digest", "")),
            "sources_digest": str(self.registry.get("_digest", "")),
            "filters_digest": str(self.filters.get("_digest", "")),
            "dedup_digest": str(self.dedup.get("_digest", "")),
            "token_counter_id": self.token_counter_id,
        }
        present = dict(self.connection.execute("SELECT key, value FROM metadata"))
        mismatches = {
            key: (present.get(key), value)
            for key, value in expected.items()
            if key in present and present[key] != value
        }
        if mismatches:
            raise ResumeMismatchError(f"pipeline state contract mismatch: {mismatches}")
        with self.connection:
            self.connection.executemany(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", expected.items()
            )

    def cursor(self, source_id: str) -> SourceCursor:
        row = self.connection.execute(
            "SELECT next_row_index, complete, consumed_tokens FROM source_cursors WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        return (
            SourceCursor(source_id, int(row[0]), bool(row[1]), int(row[2]))
            if row
            else SourceCursor(source_id, 0, False, 0)
        )

    def ingest(
        self,
        origin_source_id: str,
        candidates: Iterable[CandidateDocument],
        *,
        start_row_index: int = 0,
        commit_every: int | None = None,
        scores: Iterable[float | None] | None = None,
        mark_complete: bool = True,
        allow_empty_completion: bool = False,
    ) -> int:
        """Filter and persist one deterministic source stream, resuming by row index."""
        cursor = self.cursor(origin_source_id)
        if cursor.complete:
            return 0
        if cursor.next_row_index != start_row_index:
            raise ResumeMismatchError(
                f"{origin_source_id} must resume at row {cursor.next_row_index}, got {start_row_index}"
            )
        interval = int(
            self.acquisition["restart"]["commit_every_documents"]
            if commit_every is None
            else commit_every
        )
        if interval <= 0:
            raise ValueError("commit_every must be positive")
        score_iter = iter(scores) if scores is not None else None
        science_words = {
            word.casefold()
            for word in self.acquisition["reserved_selectors"]["reserved_science"]["topic_vocabulary"]
        }
        written = 0
        rows_seen = 0
        next_index = start_row_index
        consumed_tokens = cursor.consumed_tokens
        for row_index, candidate in enumerate(candidates, start=start_row_index):
            if candidate.source_id != origin_source_id:
                raise CorpusPipelineError(
                    f"stream {origin_source_id!r} yielded candidate from {candidate.source_id!r}"
                )
            score = next(score_iter) if score_iter is not None else None
            consumed_tokens += int(self.token_counter(candidate.text))
            record = build_record(
                candidate,
                registry=self.registry,
                filters=self.filters,
                token_counter=self.token_counter,
                token_counter_id=self.token_counter_id,
            )
            original_id = candidate.document_id
            doc_key = f"{origin_source_id}/{original_id}"
            existing = self.connection.execute(
                "SELECT raw_sha256 FROM documents WHERE doc_key = ?", (doc_key,)
            ).fetchone()
            if existing is not None and str(existing[0]) == record.raw_sha256:
                self.connection.execute(
                    "INSERT OR IGNORE INTO ingest_events VALUES (?, ?, ?, ?)",
                    (origin_source_id, row_index, EXACT_DUPLICATE, doc_key),
                )
                rows_seen += 1
                next_index = row_index + 1
                if rows_seen % interval == 0:
                    self._save_cursor(origin_source_id, next_index, False, consumed_tokens)
                    self.connection.commit()
                continue
            if existing is not None:
                resolved_id = f"{original_id}#{record.raw_sha256[:16]}"
                candidate = replace(candidate, document_id=resolved_id)
                record = build_record(
                    candidate,
                    registry=self.registry,
                    filters=self.filters,
                    token_counter=self.token_counter,
                    token_counter_id=self.token_counter_id,
                )
                doc_key = f"{origin_source_id}/{resolved_id}"
                resolved_existing = self.connection.execute(
                    "SELECT raw_sha256 FROM documents WHERE doc_key = ?", (doc_key,)
                ).fetchone()
                if resolved_existing is not None:
                    if str(resolved_existing[0]) != record.raw_sha256:
                        raise CorpusPipelineError(f"document ID hash suffix collision for {doc_key!r}")
                    self.connection.execute(
                        "INSERT OR IGNORE INTO ingest_events VALUES (?, ?, ?, ?)",
                        (origin_source_id, row_index, EXACT_DUPLICATE, doc_key),
                    )
                    rows_seen += 1
                    next_index = row_index + 1
                    if rows_seen % interval == 0:
                        self._save_cursor(origin_source_id, next_index, False, consumed_tokens)
                        self.connection.commit()
                    continue
            words = {word.casefold() for word in SCIENCE_WORD_RE.findall(candidate.text)}
            try:
                self.connection.execute(
                    """
                    INSERT INTO documents(
                        doc_key, origin_source_id, document_id, text, score, raw_sha256,
                        manifest_json, filter_action, filter_reason, token_count, science_match
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_key,
                        origin_source_id,
                        candidate.document_id,
                        candidate.text,
                        score,
                        record.raw_sha256,
                        record.to_json(),
                        record.action,
                        record.reason_code,
                        record.accepted_token_count,
                        int(bool(words & science_words)),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise CorpusPipelineError(f"duplicate document identity {doc_key!r}") from exc
            written += 1
            rows_seen += 1
            next_index = row_index + 1
            if rows_seen % interval == 0:
                self._save_cursor(origin_source_id, next_index, False, consumed_tokens)
                self.connection.commit()
        if mark_complete and rows_seen == 0 and not allow_empty_completion:
            raise CorpusPipelineError(
                f"refusing to mark {origin_source_id!r} complete from an empty replay without explicit EOF proof"
            )
        self._save_cursor(origin_source_id, next_index, mark_complete, consumed_tokens)
        self.connection.commit()
        return written

    def _save_cursor(
        self,
        source_id: str,
        next_row_index: int,
        complete: bool,
        consumed_tokens: int = 0,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO source_cursors(source_id, next_row_index, complete, consumed_tokens)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                next_row_index = excluded.next_row_index,
                complete = excluded.complete,
                consumed_tokens = excluded.consumed_tokens
            """,
            (source_id, next_row_index, int(complete), consumed_tokens),
        )

    def run_deduplication(self, *, limit: int | None = None) -> int:
        """Classify filtered documents using a disk-backed, complete MinHash candidate index."""
        pending = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM documents d
                LEFT JOIN dedup_decisions x USING(doc_key)
                WHERE d.filter_action = 'KEEP' AND x.doc_key IS NULL
                """
            ).fetchone()[0]
        )
        if pending == 0:
            return 0
        downstream = int(self.connection.execute("SELECT COUNT(*) FROM decontamination").fetchone()[0]) + int(
            self.connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
        )
        if downstream:
            raise CorpusPipelineError("deduplication cannot continue after decontamination or assignment has started")
        settings = self.acquisition["deduplication"]["minhash_candidate_index"]
        bands = int(settings["bands"])
        rows_per_band = int(settings["rows_per_band"])
        shingle_size = int(self.dedup["near_dedup"]["shingle"]["size"])
        threshold = float(self.dedup["near_dedup"]["estimated_jaccard_threshold"])
        review_band = float(self.dedup.get("boundary_review", {}).get("estimated_jaccard_band", 0.0))
        sql = """
            SELECT d.doc_key, d.text
            FROM documents d LEFT JOIN dedup_decisions x ON x.doc_key = d.doc_key
            WHERE d.filter_action = ? AND x.doc_key IS NULL
            ORDER BY d.doc_key
        """
        parameters: list[Any] = [KEEP]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(int(limit))
        processed = 0
        for row in self.connection.execute(sql, parameters):
            doc_key, text = str(row[0]), str(row[1])
            exact_hash = document_sha256(text, self.dedup)
            mirror_hash = mirror_sha256(text, self.dedup)
            signature = minhash_signature(word_shingles(text, shingle_size, self.dedup), self.dedup)
            exact_keys = {
                str(item[0])
                for item in self.connection.execute(
                "SELECT doc_key FROM representatives WHERE exact_hash = ? ORDER BY doc_key",
                (exact_hash,),
                )
            }
            mirror_keys = {
                str(item[0])
                for item in self.connection.execute(
                    "SELECT doc_key FROM representatives WHERE mirror_hash = ? ORDER BY doc_key",
                    (mirror_hash,),
                )
            }
            candidates: set[str] = set()
            band_values = tuple(_band_keys(signature, bands, rows_per_band))
            for band, band_key in band_values:
                candidates.update(
                    str(item[0])
                    for item in self.connection.execute(
                        "SELECT doc_key FROM minhash_bands WHERE band = ? AND band_key = ?",
                        (band, band_key),
                    )
                )
            estimates: dict[str, float] = {}
            near_keys: set[str] = set()
            for candidate_key in sorted(candidates | mirror_keys):
                candidate_blob = self.connection.execute(
                    "SELECT signature FROM representatives WHERE doc_key = ?", (candidate_key,)
                ).fetchone()[0]
                candidate_estimate = estimated_jaccard(_signature_from_blob(candidate_blob), signature)
                estimates[candidate_key] = candidate_estimate
                if abs(candidate_estimate - threshold) <= review_band:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO boundary_review_pairs VALUES (?, ?, ?)",
                        (candidate_key, doc_key, candidate_estimate),
                    )
                if candidate_estimate >= threshold:
                    near_keys.add(candidate_key)
            match_keys = exact_keys | mirror_keys | near_keys
            if exact_keys:
                matched_key = min(exact_keys)
                reason, estimate = EXACT_DUPLICATE, 1.0
            elif mirror_keys:
                matched_key = min(mirror_keys)
                reason, estimate = MIRROR_DUPLICATE, estimates[matched_key]
            elif near_keys:
                matched_key = min(near_keys, key=lambda key: (-estimates[key], key))
                reason, estimate = NEAR_DUPLICATE, estimates[matched_key]
            else:
                matched_key = None
                reason, estimate = UNIQUE, None
            blob = _signature_blob(signature)
            if matched_key is None:
                cluster_id = f"cluster:{doc_key}"
                with self.connection:
                    self.connection.execute(
                        "INSERT INTO representatives(doc_key, exact_hash, mirror_hash, signature) VALUES (?, ?, ?, ?)",
                        (doc_key, exact_hash, mirror_hash, blob),
                    )
                    self.connection.executemany(
                        "INSERT INTO minhash_bands(band, band_key, doc_key) VALUES (?, ?, ?)",
                        ((band, key, doc_key) for band, key in _band_keys(signature, bands, rows_per_band)),
                    )
                    self.connection.execute(
                        "INSERT INTO dedup_decisions VALUES (?, ?, ?, ?, ?, ?)",
                        (doc_key, KEEP, UNIQUE, cluster_id, None, None),
                    )
            else:
                cluster_rows = self.connection.execute(
                    f"SELECT DISTINCT cluster_id FROM dedup_decisions WHERE doc_key IN ({','.join('?' for _ in match_keys)})",
                    tuple(sorted(match_keys)),
                ).fetchall()
                cluster_ids = {str(item[0]) for item in cluster_rows}
                cluster_id = min(cluster_ids)
                with self.connection:
                    self.connection.execute(
                        "INSERT INTO representatives(doc_key, exact_hash, mirror_hash, signature) VALUES (?, ?, ?, ?)",
                        (doc_key, exact_hash, mirror_hash, blob),
                    )
                    self.connection.executemany(
                        "INSERT INTO minhash_bands(band, band_key, doc_key) VALUES (?, ?, ?)",
                        ((band, key, doc_key) for band, key in band_values),
                    )
                    for losing_cluster in sorted(cluster_ids - {cluster_id}):
                        losing_representative = self.connection.execute(
                            "SELECT doc_key FROM dedup_decisions WHERE cluster_id = ? AND action = ? ORDER BY doc_key LIMIT 1",
                            (losing_cluster, KEEP),
                        ).fetchone()
                        if losing_representative is not None:
                            losing_key = str(losing_representative[0])
                            losing_reason = (
                                EXACT_DUPLICATE
                                if losing_key in exact_keys
                                else MIRROR_DUPLICATE
                                if losing_key in mirror_keys
                                else NEAR_DUPLICATE
                            )
                            self.connection.execute(
                                "UPDATE dedup_decisions SET action = 'DROP', reason_code = ?, matched_doc_key = ?, estimated_jaccard = ? WHERE doc_key = ?",
                                (losing_reason, doc_key, estimates.get(losing_key, 1.0), losing_key),
                            )
                        self.connection.execute(
                            "UPDATE dedup_decisions SET cluster_id = ? WHERE cluster_id = ?",
                            (cluster_id, losing_cluster),
                        )
                    self.connection.execute(
                        "INSERT INTO dedup_decisions VALUES (?, ?, ?, ?, ?, ?)",
                        (doc_key, "DROP", reason, cluster_id, matched_key, estimate),
                    )
            processed += 1
        return processed

    def mark_all_clean_for_fixture(self) -> int:
        """Fixture-only helper; production callers must run benchmark decontamination."""
        rows = self.connection.execute(
            """
            SELECT r.doc_key FROM representatives r
            LEFT JOIN decontamination d ON d.doc_key = r.doc_key
            WHERE d.doc_key IS NULL ORDER BY r.doc_key
            """
        ).fetchall()
        with self.connection:
            self.connection.executemany(
                "INSERT INTO decontamination VALUES (?, 'KEEP', 'CLEAN', ?)",
                ((str(row[0]), json.dumps({"scale": "FIXTURE"}, sort_keys=True)) for row in rows),
            )
        return len(rows)

    def run_decontamination(
        self,
        benchmark_index: Any,
        *,
        limit: int | None = None,
        commit_every: int | None = None,
    ) -> int:
        """Classify every cluster member with a complete :class:`BenchmarkIndex`."""
        interval = int(
            self.acquisition["restart"]["commit_every_documents"]
            if commit_every is None
            else commit_every
        )
        if interval <= 0:
            raise ValueError("commit_every must be positive")
        sql = """
            SELECT r.doc_key, d.text FROM representatives r JOIN documents d USING(doc_key)
            LEFT JOIN decontamination c USING(doc_key)
            WHERE c.doc_key IS NULL ORDER BY r.doc_key
        """
        parameters: list[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(int(limit))
        processed = 0
        try:
            for row in self.connection.execute(sql, parameters):
                doc_key, text = str(row[0]), str(row[1])
                decision = benchmark_index.classify(doc_key, text)
                evidence = {
                    "rule_id": decision.rule_id,
                    "task_id": decision.task_id,
                    "item_id": decision.item_id,
                    "measurement": decision.measurement,
                    "matched_rules": [
                        {
                            "rule_id": match.rule_id,
                            "reason_code": match.reason_code,
                            "task_id": match.task_id,
                            "item_id": match.item_id,
                            "measurement": match.measurement,
                        }
                        for match in decision.matched_rules
                    ],
                }
                self.connection.execute(
                    "INSERT INTO decontamination VALUES (?, ?, ?, ?)",
                    (doc_key, decision.action, decision.reason_code, json.dumps(evidence, sort_keys=True)),
                )
                processed += 1
                if processed % interval == 0:
                    self.connection.commit()
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        return processed

    def _ensure_selection_keys(self, selection_id: str, where_sql: str, parameters: Sequence[Any]) -> None:
        salt = str(self.acquisition["selection"]["salt"])
        rows = self.connection.execute(
            f"""
            SELECT r.doc_key, d.origin_source_id, d.document_id
            FROM representatives r JOIN documents d USING(doc_key)
            JOIN dedup_decisions x USING(doc_key)
            JOIN decontamination c USING(doc_key)
            WHERE x.action = 'KEEP' AND c.action = 'KEEP' AND ({where_sql})
              AND NOT EXISTS (
                  SELECT 1 FROM dedup_decisions member
                  JOIN decontamination member_scan ON member_scan.doc_key = member.doc_key
                  WHERE member.cluster_id = x.cluster_id AND member_scan.action != 'KEEP'
              )
            ORDER BY r.doc_key
            """,
            parameters,
        )
        with self.connection:
            self.connection.executemany(
                "INSERT OR IGNORE INTO selection_keys VALUES (?, ?, ?)",
                (
                    (
                        selection_id,
                        str(row[0]),
                        selection_key(selection_id, str(row[1]), str(row[2]), salt=salt),
                    )
                    for row in rows
                ),
            )

    def _fill_selection(
        self,
        selection_id: str,
        *,
        source_id: str,
        boundary: str,
        target_tokens: int,
        where_sql: str,
        parameters: Sequence[Any],
        score_first: bool = False,
        protected_slice: str | None = None,
    ) -> SelectionResult:
        if boundary not in BOUNDARY_ORDER:
            raise CorpusPipelineError(f"unknown boundary {boundary!r}")
        self._ensure_selection_keys(selection_id, where_sql, parameters)
        existing = self.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(d.token_count), 0)
            FROM assignments a JOIN documents d USING(doc_key)
            WHERE a.selection_id = ?
            """,
            (selection_id,),
        ).fetchone()
        selected_documents, selected_tokens = int(existing[0]), int(existing[1])
        if selected_tokens >= target_tokens:
            return SelectionResult(selection_id, target_tokens, selected_tokens, selected_documents)
        quality_order = "d.score IS NULL, d.score DESC, " if score_first else ""
        rows = self.connection.execute(
            f"""
            SELECT k.doc_key, k.ranking_key, d.token_count
            FROM selection_keys k JOIN documents d USING(doc_key)
            LEFT JOIN assignments a USING(doc_key)
            WHERE k.selection_id = ? AND a.doc_key IS NULL
            ORDER BY {quality_order} k.ranking_key, k.doc_key
            """,
            (selection_id,),
        )
        pending = 0
        for row in rows:
            if selected_tokens >= target_tokens:
                break
            self.connection.execute(
                "INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?)",
                (str(row[0]), source_id, boundary, protected_slice, selection_id, str(row[1])),
            )
            selected_tokens += int(row[2])
            selected_documents += 1
            pending += 1
            if pending >= 1000:
                self.connection.commit()
                pending = 0
        self.connection.commit()
        return SelectionResult(selection_id, target_tokens, selected_tokens, selected_documents)

    def assign_production(self, *, target_fraction: float = 1.0) -> tuple[SelectionResult, ...]:
        """Fill frozen reserved, validation, then stable targets from clean clusters."""
        if not 0 < target_fraction <= 1:
            raise ValueError("target_fraction must be in (0, 1]")
        results: list[SelectionResult] = []
        reserved_targets = {
            str(item["source_id"]): int(math.ceil(int(item["target_tokens_at_minimum"]) * target_fraction))
            for item in self.registry["reserved_sources"]
        }
        reserved_specs = (
            ("reserved_science", "d.origin_source_id = 'fineweb_edu' AND d.science_match = 1", True),
            ("reserved_textbook", "d.origin_source_id = 'reserved_textbook'", False),
            ("reserved_wikipedia", "d.origin_source_id = 'reserved_wikipedia'", False),
            ("reserved_edu_decile", "d.origin_source_id = 'fineweb_edu' AND d.science_match = 0", True),
            ("reserved_math_prose", "d.origin_source_id = 'openwebmath'", False),
        )
        for source_id, where_sql, score_first in reserved_specs:
            results.append(
                self._fill_selection(
                    source_id,
                    source_id=source_id,
                    boundary="reserved",
                    target_tokens=reserved_targets[source_id],
                    where_sql=where_sql,
                    parameters=(),
                    score_first=score_first,
                )
            )
        validation = self.acquisition["validation"]
        shares = validation["stable_source_shares"]
        slice_map = validation["protected_slice_assignment"]["mapping"]
        for boundary in ("validation_dev", "validation_final"):
            split_target = int(math.ceil(int(validation["targets"][boundary]) * target_fraction))
            allocated = 0
            source_ids = list(shares)
            for index, origin in enumerate(source_ids):
                if index == len(source_ids) - 1:
                    quota = split_target - allocated
                else:
                    quota = int(math.floor(split_target * float(shares[origin]) + 0.5))
                    allocated += quota
                selection_id = f"{boundary}:{origin}"
                results.append(
                    self._fill_selection(
                        selection_id,
                        source_id=str(origin),
                        boundary=boundary,
                        target_tokens=quota,
                        where_sql="d.origin_source_id = ?",
                        parameters=(str(origin),),
                        protected_slice=str(slice_map[origin]),
                    )
                )
        for source in self.registry["stable_sources"]:
            source_id = str(source["source_id"])
            target = int(math.ceil(int(source["target_tokens_at_11b"]) * target_fraction))
            results.append(
                self._fill_selection(
                    f"stable_train:{source_id}",
                    source_id=source_id,
                    boundary="stable_train",
                    target_tokens=target,
                    where_sql="d.origin_source_id = ?",
                    parameters=(source_id,),
                )
            )
        return tuple(results)

    def assign_fixture(self) -> int:
        """Assign clean fixture representatives to their registered boundary.

        This is intentionally unavailable as a production shortcut.  It lets bounded tests
        prove sorting, restart, and shard hand-off while real selection uses explicit target
        filling in :meth:`assign_production`.
        """
        source_boundaries = {
            str(item["source_id"]): str(item["boundary"])
            for item in self.registry["stable_sources"] + self.registry["reserved_sources"]
        }
        rows = self.connection.execute(
            """
            SELECT r.doc_key, d.origin_source_id
            FROM representatives r JOIN documents d USING(doc_key)
            JOIN dedup_decisions x USING(doc_key)
            JOIN decontamination c USING(doc_key)
            LEFT JOIN assignments a USING(doc_key)
            WHERE x.action = 'KEEP' AND c.action = 'KEEP' AND a.doc_key IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM dedup_decisions member
                  JOIN decontamination member_scan ON member_scan.doc_key = member.doc_key
                  WHERE member.cluster_id = x.cluster_id AND member_scan.action != 'KEEP'
              )
            ORDER BY r.doc_key
            """
        ).fetchall()
        values = []
        salt = str(self.acquisition["selection"]["salt"])
        for row in rows:
            doc_key, source_id = str(row[0]), str(row[1])
            if source_id not in source_boundaries:
                raise CorpusPipelineError(f"source {source_id!r} has no registered boundary")
            boundary = source_boundaries[source_id]
            values.append(
                (doc_key, source_id, boundary, None, "fixture", selection_key("fixture", source_id, doc_key, salt=salt))
            )
        with self.connection:
            self.connection.executemany("INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?)", values)
        return len(values)

    def write_accepted_jsonl(self, path: Path) -> tuple[int, str]:
        """Write deterministic shard input atomically and return row count and SHA-256."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.staging")
        if path.exists() or temporary.exists():
            raise CorpusPipelineError(f"refusing to overwrite existing output or staging file for {path}")
        boundary_case = "CASE a.boundary WHEN 'reserved' THEN 0 WHEN 'stable_train' THEN 1 WHEN 'validation_dev' THEN 2 WHEN 'validation_final' THEN 3 ELSE 99 END"
        rows = self.connection.execute(
            f"""
            SELECT d.document_id, a.source_id, d.text, a.boundary, a.protected_slice, x.cluster_id
            FROM assignments a JOIN documents d USING(doc_key)
            JOIN dedup_decisions x USING(doc_key)
            ORDER BY {boundary_case}, a.source_id, d.document_id
            """
        )
        digest = hashlib.sha256()
        count = 0
        try:
            with temporary.open("xb") as handle:
                for row in rows:
                    payload = {
                        "document_id": str(row[0]),
                        "source_id": str(row[1]),
                        "text": str(row[2]),
                        "boundary": str(row[3]),
                        "protected_slice": row[4],
                        "cluster_id": str(row[5]),
                    }
                    line = (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
                    handle.write(line)
                    digest.update(line)
                    count += 1
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return count, digest.hexdigest()

    def write_decisions_jsonl(self, path: Path) -> tuple[int, str]:
        """Publish the complete reason-coded ledger without duplicating stored text."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.staging")
        if path.exists() or temporary.exists():
            raise CorpusPipelineError(f"refusing to overwrite existing output or staging file for {path}")
        rows = self.connection.execute(
            """
            SELECT d.doc_key, d.manifest_json,
                   x.action, x.reason_code, x.cluster_id, x.matched_doc_key, x.estimated_jaccard,
                   c.action, c.reason_code, c.evidence_json,
                   a.source_id, a.boundary, a.protected_slice, a.selection_id, a.selection_key
            FROM documents d
            LEFT JOIN dedup_decisions x USING(doc_key)
            LEFT JOIN decontamination c USING(doc_key)
            LEFT JOIN assignments a USING(doc_key)
            ORDER BY d.doc_key
            """
        )
        digest = hashlib.sha256()
        count = 0
        try:
            with temporary.open("xb") as handle:
                for row in rows:
                    payload = {
                        "event_type": "document",
                        "doc_key": str(row[0]),
                        "source_manifest": json.loads(str(row[1])),
                        "dedup": {
                            "action": row[2],
                            "reason_code": row[3],
                            "cluster_id": row[4],
                            "matched_doc_key": row[5],
                            "estimated_jaccard": row[6],
                        },
                        "decontamination": {
                            "action": row[7],
                            "reason_code": row[8],
                            "evidence": json.loads(str(row[9])) if row[9] is not None else None,
                        },
                        "assignment": {
                            "source_id": row[10],
                            "boundary": row[11],
                            "protected_slice": row[12],
                            "selection_id": row[13],
                            "selection_key": row[14],
                        },
                    }
                    line = (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
                    handle.write(line)
                    digest.update(line)
                    count += 1
                events = self.connection.execute(
                    """
                    SELECT origin_source_id, row_index, reason_code, matched_doc_key
                    FROM ingest_events ORDER BY origin_source_id, row_index
                    """
                )
                for event in events:
                    payload = {
                        "event_type": "ingest_event",
                        "origin_source_id": str(event[0]),
                        "row_index": int(event[1]),
                        "reason_code": str(event[2]),
                        "matched_doc_key": str(event[3]),
                    }
                    line = (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
                    handle.write(line)
                    digest.update(line)
                    count += 1
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return count, digest.hexdigest()

    def publish_jsonl_bundle(
        self, accepted_path: Path, decisions_path: Path
    ) -> tuple[int, str, int, str]:
        """Publish accepted text and its complete ledger with one directory rename."""
        accepted_path = Path(accepted_path)
        decisions_path = Path(decisions_path)
        if accepted_path.parent.resolve() != decisions_path.parent.resolve():
            raise CorpusPipelineError("accepted and decisions outputs must share one bundle directory")
        if accepted_path.name == decisions_path.name:
            raise CorpusPipelineError("accepted and decisions outputs must have different names")
        bundle = accepted_path.parent
        staging = bundle.with_name(f".{bundle.name}.staging")
        if bundle.exists() or staging.exists():
            raise CorpusPipelineError(f"refusing to overwrite existing output bundle or staging directory for {bundle}")
        bundle.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        try:
            accepted_count, accepted_hash = self.write_accepted_jsonl(staging / accepted_path.name)
            decision_count, decision_hash = self.write_decisions_jsonl(staging / decisions_path.name)
            staging.replace(bundle)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return accepted_count, accepted_hash, decision_count, decision_hash

    def summary(self) -> PipelineSummary:
        scalar = lambda sql: int(self.connection.execute(sql).fetchone()[0])
        reason_counts = {
            str(row[0]): int(row[1])
            for row in self.connection.execute(
                "SELECT filter_reason, COUNT(*) FROM documents GROUP BY filter_reason"
            )
        }
        for row in self.connection.execute(
            "SELECT reason_code, COUNT(*) FROM dedup_decisions GROUP BY reason_code"
        ):
            reason = str(row[0])
            reason_counts[reason] = reason_counts.get(reason, 0) + int(row[1])
        for row in self.connection.execute(
            "SELECT reason_code, COUNT(*) FROM ingest_events GROUP BY reason_code"
        ):
            reason = str(row[0])
            reason_counts[reason] = reason_counts.get(reason, 0) + int(row[1])
        boundary_tokens = {
            str(row[0]): int(row[1] or 0)
            for row in self.connection.execute(
                "SELECT a.boundary, SUM(d.token_count) FROM assignments a JOIN documents d USING(doc_key) GROUP BY a.boundary"
            )
        }
        source_tokens = {
            str(row[0]): int(row[1] or 0)
            for row in self.connection.execute(
                "SELECT a.source_id, SUM(d.token_count) FROM assignments a JOIN documents d USING(doc_key) GROUP BY a.source_id"
            )
        }
        return PipelineSummary(
            documents_seen=scalar("SELECT COUNT(*) FROM documents"),
            filter_accepted=scalar("SELECT COUNT(*) FROM documents WHERE filter_action = 'KEEP'"),
            dedup_kept=scalar("SELECT COUNT(*) FROM dedup_decisions WHERE action = 'KEEP'"),
            decontamination_clean=scalar("SELECT COUNT(*) FROM decontamination WHERE action = 'KEEP'"),
            assigned_documents=scalar("SELECT COUNT(*) FROM assignments"),
            assigned_tokens=sum(boundary_tokens.values()),
            reason_counts=reason_counts,
            boundary_tokens=boundary_tokens,
            source_tokens=source_tokens,
        )

    def isolation_evidence(self) -> dict[str, Any]:
        """Return structured cluster/boundary evidence consumed by the shard verifier."""
        allowed_boundaries = set(BOUNDARY_ORDER)
        allowed_slices = set(self.registry["protected_slices"])
        observed_boundaries = {
            str(row[0]) for row in self.connection.execute("SELECT DISTINCT boundary FROM assignments")
        }
        observed_slices = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT DISTINCT protected_slice FROM assignments WHERE protected_slice IS NOT NULL"
            )
        }
        boundary_violations = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT x.cluster_id FROM assignments a JOIN dedup_decisions x USING(doc_key)
                    GROUP BY x.cluster_id HAVING COUNT(DISTINCT a.boundary) > 1
                )
                """
            ).fetchone()[0]
        )
        slice_violations = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT x.cluster_id FROM assignments a JOIN dedup_decisions x USING(doc_key)
                    WHERE a.protected_slice IS NOT NULL
                    GROUP BY x.cluster_id HAVING COUNT(DISTINCT a.protected_slice) > 1
                )
                """
            ).fetchone()[0]
        )
        undeclared = len(observed_slices - allowed_slices)
        coverage_complete = allowed_boundaries.issubset(observed_boundaries) and allowed_slices.issubset(observed_slices)
        status = "PASS" if boundary_violations == slice_violations == undeclared == 0 and coverage_complete else "NOT_RUN"
        return {
            "status": status,
            "boundary_violations": boundary_violations,
            "slice_violations": slice_violations,
            "undeclared_slices": undeclared,
            "boundaries": sorted(observed_boundaries),
            "protected_slices": sorted(observed_slices),
            "cluster_count": int(
                self.connection.execute("SELECT COUNT(DISTINCT cluster_id) FROM dedup_decisions").fetchone()[0]
            ),
            "boundary_review_pairs": int(
                self.connection.execute("SELECT COUNT(*) FROM boundary_review_pairs").fetchone()[0]
            ),
            "boundary_review_sample": [
                {
                    "left_doc_key": str(row[0]),
                    "right_doc_key": str(row[1]),
                    "estimated_jaccard": float(row[2]),
                }
                for row in self.connection.execute(
                    "SELECT left_doc_key, right_doc_key, estimated_jaccard FROM boundary_review_pairs ORDER BY left_doc_key, right_doc_key LIMIT ?",
                    (int(self.dedup.get("boundary_review", {}).get("sample_per_band", 0)),),
                )
            ],
            "assignment_rule": "one representative assignment per duplicate cluster",
        }


__all__ = [
    "ACQUISITION_PROTOCOL_PATH",
    "AcquisitionProtocolError",
    "CorpusPipelineError",
    "CorpusState",
    "FROZEN_ACQUISITION_SHA256",
    "PipelineSummary",
    "ResumeMismatchError",
    "SelectionResult",
    "SourceCursor",
    "StreamedSourceRow",
    "assert_write_space",
    "iter_huggingface_source",
    "load_acquisition_protocol",
    "natural_document_id",
    "selection_key",
    "source_registration",
]
