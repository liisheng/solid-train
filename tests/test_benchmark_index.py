from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinybench_lm.benchmark_index import (
    _SHORT_CANDIDATE_SQL,
    BenchmarkIndex,
    BenchmarkIndexError,
    file_sha256,
)
from tinybench_lm.data_protocols import BenchmarkItem, DocumentRecord, decontaminate, load_decontamination_protocol


def write_items(path: Path) -> list[BenchmarkItem]:
    overlap_words = tuple(f"overlap{i}" for i in range(60))
    coverage_words = tuple(f"coverage{i}" for i in range(30))
    items = [
        BenchmarkItem("task", "short", ("rare cobalt falcon",)),
        BenchmarkItem("task", "overlap", (" ".join(overlap_words),)),
        BenchmarkItem("task", "coverage", (" ".join(coverage_words),)),
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps({"task_id": item.task_id, "item_id": item.item_id, "texts": list(item.texts)}) + "\n")
    return items


def reference(doc_id: str, text: str, items: list[BenchmarkItem]):
    report = decontaminate(
        [DocumentRecord(doc_id, text)],
        items,
        load_decontamination_protocol(Path("configs/data/decontam_v2.yaml")),
    )
    return report.decision(doc_id)


def test_index_matches_frozen_reference_for_all_three_rules_and_clean(tmp_path: Path) -> None:
    source = tmp_path / "items.jsonl"
    items = write_items(source)
    documents = {
        "rule1": "Before the rare cobalt falcon appears after the introduction.",
        "rule2": "prefix " + " ".join(f"overlap{i}" for i in range(5, 55)) + " suffix",
        "rule3": "prefix words here " + " ".join(f"coverage{i}" for i in range(15)) + " unrelated ending words",
        "clean": "A completely unrelated document has no planted benchmark sequence in its words.",
    }
    with BenchmarkIndex(tmp_path / "index.sqlite") as index:
        summary = index.build(source, expected_sha256=file_sha256(source), commit_every=1)
        assert summary.complete
        assert summary.items == 3
        assert index.build(source, expected_sha256=file_sha256(source)).items == 3
        for doc_id, text in documents.items():
            observed = index.classify(doc_id, text)
            expected = reference(doc_id, text, items)
            assert observed.action == expected.action
            assert observed.reason_code == expected.reason_code
            assert observed.rule_id == expected.rule_id
            assert observed.task_id == expected.task_id
            assert observed.item_id == expected.item_id
            assert observed.measurement == pytest.approx(expected.measurement)
            assert observed == expected


def test_index_rejects_wrong_source_hash_and_incomplete_use(tmp_path: Path) -> None:
    source = tmp_path / "items.jsonl"
    write_items(source)
    with BenchmarkIndex(tmp_path / "index.sqlite") as index:
        with pytest.raises(BenchmarkIndexError, match="hash mismatch"):
            index.build(source, expected_sha256="0" * 64)
        with pytest.raises(BenchmarkIndexError, match="incomplete"):
            index.classify("doc", "some text")


def test_short_candidate_lookup_probes_index_from_document_ngrams(tmp_path: Path) -> None:
    source = tmp_path / "items.jsonl"
    write_items(source)
    with BenchmarkIndex(tmp_path / "index.sqlite") as index:
        index.build(source, expected_sha256=file_sha256(source))
        index.connection.execute(
            """
            CREATE TEMP TABLE doc_ngrams(
                size INTEGER,
                position INTEGER,
                digest BLOB,
                PRIMARY KEY(size, position)
            ) WITHOUT ROWID
            """
        )
        plan = [
            str(row[3])
            for row in index.connection.execute("EXPLAIN QUERY PLAN " + _SHORT_CANDIDATE_SQL)
        ]
        assert any(detail.startswith("SCAN g") for detail in plan)
        assert any(
            detail.startswith("SEARCH s USING PRIMARY KEY (word_count=? AND digest=?)")
            for detail in plan
        )


@pytest.mark.parametrize("payload", [[], None, 7, "text"])
def test_index_rejects_non_object_json_rows(tmp_path: Path, payload: object) -> None:
    source = tmp_path / "items.jsonl"
    source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with BenchmarkIndex(tmp_path / "index.sqlite") as index:
        with pytest.raises(BenchmarkIndexError, match="JSON object"):
            index.build(source, expected_sha256=file_sha256(source))
