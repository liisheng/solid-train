"""Disk-backed exact index for the frozen benchmark quarantine rules."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data_protocols import (
    CLEAN,
    KEEP,
    QUARANTINE,
    DecontaminationDecision,
    RuleMatch,
    load_decontamination_protocol,
    normalize_for_matching,
    rule_by_id,
)


_SHORT_CANDIDATE_SQL = """
    SELECT DISTINCT t.text_key, t.item_key, t.text_index, i.task_id, i.item_id,
                    t.normalized, t.word_count
    FROM doc_ngrams g CROSS JOIN short_texts s
      ON s.word_count = g.size AND s.digest = g.digest
    JOIN texts t USING(text_key) JOIN items i USING(item_key)
"""


class BenchmarkIndexError(RuntimeError):
    """A benchmark index or its source evidence is invalid."""


@dataclass(frozen=True)
class BenchmarkIndexSummary:
    items: int
    texts: int
    short_texts: int
    shingles_13: int
    complete: bool


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _word_digest(words: Sequence[str]) -> bytes:
    return hashlib.sha256(" ".join(words).encode("utf-8")).digest()


def _ngrams(words: Sequence[str], size: int) -> Iterable[tuple[int, bytes]]:
    if len(words) < size:
        return ()
    return ((index, _word_digest(words[index : index + size])) for index in range(len(words) - size + 1))


def _longest_contiguous_overlap(left: Sequence[str], right: Sequence[str]) -> int:
    """Exact longest common word substring, sparse in matching word positions."""
    positions: dict[str, list[int]] = {}
    for index, word in enumerate(right):
        positions.setdefault(word, []).append(index)
    previous: dict[int, int] = {}
    best = 0
    for word in left:
        current: dict[int, int] = {}
        for right_index in positions.get(word, ()):
            length = previous.get(right_index - 1, 0) + 1
            current[right_index] = length
            best = max(best, length)
        previous = current
    return best


class BenchmarkIndex:
    """Restartable SQLite representation of benchmark texts and 13/50-word shingles."""

    def __init__(
        self,
        path: Path,
        *,
        protocol: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.protocol = dict(protocol or load_decontamination_protocol())
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self._bind_protocol()

    def __enter__(self) -> "BenchmarkIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS items(
                item_key TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                UNIQUE(task_id, item_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS texts(
                text_key TEXT PRIMARY KEY,
                item_key TEXT NOT NULL REFERENCES items(item_key),
                text_index INTEGER NOT NULL,
                normalized TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                UNIQUE(item_key, text_index)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS short_texts(
                word_count INTEGER NOT NULL,
                digest BLOB NOT NULL,
                text_key TEXT NOT NULL REFERENCES texts(text_key),
                PRIMARY KEY(word_count, digest, text_key)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS shingles_13(
                digest BLOB NOT NULL,
                text_key TEXT NOT NULL REFERENCES texts(text_key),
                PRIMARY KEY(digest, text_key)
            ) WITHOUT ROWID;
            """
        )
        self.connection.commit()

    def _bind_protocol(self) -> None:
        expected = str(self.protocol.get("_digest", ""))
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'protocol_digest'"
        ).fetchone()
        if row and str(row[0]) != expected:
            raise BenchmarkIndexError("benchmark index was built under a different decontamination protocol")
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('protocol_digest', ?)", (expected,)
            )

    def build(
        self,
        benchmark_items: Path,
        *,
        expected_sha256: str,
        commit_every: int = 1000,
    ) -> BenchmarkIndexSummary:
        """Index a pinned JSONL file, resuming from its last committed byte offset."""
        source = Path(benchmark_items).resolve()
        if not source.is_file():
            raise BenchmarkIndexError(f"benchmark item file is absent: {source}")
        observed = file_sha256(source)
        if observed != expected_sha256:
            raise BenchmarkIndexError(
                f"benchmark item hash mismatch (expected {expected_sha256}, observed {observed})"
            )
        if commit_every <= 0:
            raise ValueError("commit_every must be positive")
        metadata = dict(self.connection.execute("SELECT key, value FROM metadata"))
        recorded_source = metadata.get("source_sha256")
        if recorded_source and recorded_source != observed:
            raise BenchmarkIndexError("benchmark index source hash changed across resume")
        if metadata.get("complete") == "1":
            return self.summary()
        offset = int(metadata.get("next_byte_offset", "0"))
        rows_since_commit = 0
        with source.open("rb") as handle:
            handle.seek(offset)
            while raw_line := handle.readline():
                next_offset = handle.tell()
                if not raw_line.strip():
                    offset = next_offset
                    continue
                try:
                    payload = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise BenchmarkIndexError(f"invalid benchmark JSONL at byte {offset}") from exc
                self._insert_item(payload)
                offset = next_offset
                rows_since_commit += 1
                if rows_since_commit >= commit_every:
                    self._save_progress(observed, offset, False)
                    self.connection.commit()
                    rows_since_commit = 0
        self._save_progress(observed, offset, True)
        self.connection.commit()
        return self.summary()

    def _save_progress(self, source_sha256: str, next_offset: int, complete: bool) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (
                ("source_sha256", source_sha256),
                ("next_byte_offset", str(next_offset)),
                ("complete", "1" if complete else "0"),
            ),
        )

    def _insert_item(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise BenchmarkIndexError("benchmark row must be a JSON object")
        task_id = str(payload.get("task_id", "")).strip()
        item_id = str(payload.get("item_id", "")).strip()
        texts = payload.get("texts")
        if not task_id or not item_id or not isinstance(texts, list):
            raise BenchmarkIndexError("benchmark row requires task_id, item_id, and a texts list")
        item_key = f"{task_id}/{item_id}"
        try:
            self.connection.execute(
                "INSERT INTO items(item_key, task_id, item_id) VALUES (?, ?, ?)",
                (item_key, task_id, item_id),
            )
        except sqlite3.IntegrityError as exc:
            raise BenchmarkIndexError(f"duplicate benchmark item {item_key!r}") from exc
        for text_index, value in enumerate(texts):
            normalized = normalize_for_matching(str(value), self.protocol)
            if not normalized:
                continue
            words = normalized.split()
            text_key = f"{item_key}/{text_index}"
            self.connection.execute(
                "INSERT INTO texts VALUES (?, ?, ?, ?, ?)",
                (text_key, item_key, text_index, normalized, len(words)),
            )
            if len(words) < 13:
                self.connection.execute(
                    "INSERT INTO short_texts VALUES (?, ?, ?)",
                    (len(words), _word_digest(words), text_key),
                )
            else:
                self.connection.executemany(
                    "INSERT OR IGNORE INTO shingles_13 VALUES (?, ?)",
                    ((digest, text_key) for _, digest in _ngrams(words, 13)),
                )

    def _require_complete(self) -> None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key = 'complete'").fetchone()
        if not row or str(row[0]) != "1":
            raise BenchmarkIndexError("benchmark index is incomplete")

    def classify(self, doc_id: str, text: str) -> DecontaminationDecision:
        """Apply all frozen rules using indexed candidates, preserving exact semantics."""
        self._require_complete()
        normalized = normalize_for_matching(text, self.protocol)
        words = normalized.split()
        self.connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS doc_ngrams(size INTEGER, position INTEGER, digest BLOB, PRIMARY KEY(size, position)) WITHOUT ROWID"
        )
        self.connection.execute("DELETE FROM doc_ngrams")
        values: list[tuple[int, int, bytes]] = []
        for size in range(1, min(12, len(words)) + 1):
            values.extend((size, position, digest) for position, digest in _ngrams(words, size))
        values.extend((13, position, digest) for position, digest in _ngrams(words, 13))
        self.connection.executemany("INSERT INTO doc_ngrams VALUES (?, ?, ?)", values)

        item_rows: dict[str, sqlite3.Row] = {}
        substring_keys: set[str] = set()
        # CROSS JOIN intentionally fixes doc_ngrams as the small outer loop. A normal JOIN
        # lets SQLite scan the corpus-sized short_texts table once per document.
        for row in self.connection.execute(_SHORT_CANDIDATE_SQL):
            item_rows[str(row["text_key"])] = row
            substring_keys.add(str(row["text_key"]))
        matched_13: dict[str, list[tuple[bytes, int]]] = {}
        for row in self.connection.execute(
            """
            SELECT s.text_key, s.digest, g.position, t.item_key, i.task_id, i.item_id,
                   t.normalized, t.word_count
            FROM shingles_13 s JOIN doc_ngrams g ON g.size = 13 AND g.digest = s.digest
            JOIN texts t USING(text_key) JOIN items i USING(item_key)
            ORDER BY s.text_key, g.position
            """
        ):
            key = str(row["text_key"])
            item_rows[key] = row
            matched_13.setdefault(key, []).append((bytes(row["digest"]), int(row["position"])))
            if f" {str(row['normalized'])} " in f" {normalized} ":
                substring_keys.add(key)

        # Any 50-word overlap necessarily shares at least one 13-word shingle, so the
        # same complete candidate set serves rules 2 and 3 without a second huge index.
        overlap_keys = set(matched_13)
        rule1 = rule_by_id(self.protocol, "RULE_1_COMPLETE_ITEM_SUBSTRING")
        rule2 = rule_by_id(self.protocol, "RULE_2_LONGEST_CONTIGUOUS_OVERLAP")
        rule3 = rule_by_id(self.protocol, "RULE_3_SHINGLE_COVERAGE")
        rule1_by_item: dict[str, RuleMatch] = {}
        for key in sorted(substring_keys, key=lambda value: (str(item_rows[value]["item_key"]), int(item_rows[value]["text_index"]))):
            row = item_rows[key]
            # Hashes only propose candidates; verify the complete word-boundary-aligned text.
            if f" {str(row['normalized'])} " in f" {normalized} ":
                rule1_by_item.setdefault(
                    str(row["item_key"]),
                    RuleMatch(
                        "RULE_1_COMPLETE_ITEM_SUBSTRING",
                        str(rule1["reason_code"]),
                        str(row["task_id"]),
                        str(row["item_id"]),
                        float(row["word_count"]),
                    ),
                )
        rule2_measurements: dict[str, tuple[float, sqlite3.Row]] = {}
        for key in sorted(overlap_keys):
            row = item_rows.get(key)
            if row is None:
                row = self.connection.execute(
                    "SELECT t.*, i.task_id, i.item_id FROM texts t JOIN items i USING(item_key) WHERE text_key = ?",
                    (key,),
                ).fetchone()
                item_rows[key] = row
            overlap = _longest_contiguous_overlap(words, str(row["normalized"]).split())
            if overlap >= int(rule2["minimum_words"]):
                item_key = str(row["item_key"])
                prior = rule2_measurements.get(item_key)
                if prior is None or overlap > prior[0]:
                    rule2_measurements[item_key] = float(overlap), row
        minimum_shingles = int(rule3["minimum_distinct_shingles"])
        minimum_coverage = float(rule3["minimum_document_word_coverage"])
        rule3_measurements: dict[str, tuple[float, bool, sqlite3.Row]] = {}
        for key, hits in sorted(matched_13.items()):
            distinct = {digest for digest, _ in hits}
            covered = {position + offset for _, position in hits for offset in range(13)}
            coverage = len(covered) / max(len(words), 1)
            row = item_rows[key]
            item_key = str(row["item_key"])
            fired = len(distinct) >= minimum_shingles and coverage >= minimum_coverage
            prior = rule3_measurements.get(item_key)
            if prior is None:
                rule3_measurements[item_key] = coverage, fired, row
            else:
                best_coverage, prior_fired, prior_row = prior
                rule3_measurements[item_key] = (
                    max(best_coverage, coverage),
                    prior_fired or fired,
                    row if coverage > best_coverage else prior_row,
                )
        matches: list[RuleMatch] = list(rule1_by_item.values())
        matches.extend(
            RuleMatch(
                "RULE_2_LONGEST_CONTIGUOUS_OVERLAP",
                str(rule2["reason_code"]),
                str(row["task_id"]),
                str(row["item_id"]),
                measurement,
            )
            for measurement, row in rule2_measurements.values()
        )
        matches.extend(
            RuleMatch(
                "RULE_3_SHINGLE_COVERAGE",
                str(rule3["reason_code"]),
                str(row["task_id"]),
                str(row["item_id"]),
                measurement,
            )
            for measurement, fired, row in rule3_measurements.values()
            if fired
        )
        order = {rule_id: index for index, rule_id in enumerate(self.protocol["decision"]["rule_evaluation_order"])}
        matches.sort(key=lambda match: (order[match.rule_id], match.task_id, match.item_id))
        if not matches:
            return DecontaminationDecision(doc_id, KEEP, CLEAN)
        primary = matches[0]
        return DecontaminationDecision(
            doc_id,
            QUARANTINE,
            primary.reason_code,
            primary.rule_id,
            primary.task_id,
            primary.item_id,
            primary.measurement,
            tuple(matches),
        )

    def summary(self) -> BenchmarkIndexSummary:
        count = lambda table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        complete = self.connection.execute("SELECT value FROM metadata WHERE key = 'complete'").fetchone()
        return BenchmarkIndexSummary(
            items=count("items"),
            texts=count("texts"),
            short_texts=count("short_texts"),
            shingles_13=count("shingles_13"),
            complete=bool(complete and str(complete[0]) == "1"),
        )


__all__ = [
    "BenchmarkIndex",
    "BenchmarkIndexError",
    "BenchmarkIndexSummary",
    "file_sha256",
]
