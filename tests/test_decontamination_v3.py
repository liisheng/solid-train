from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.fork_decontamination_v3 import fork_state
from tinybench_lm.benchmark_index import BenchmarkIndex, BenchmarkIndexError, file_sha256
from tinybench_lm.corpus_pipeline import CorpusState, ResumeMismatchError
from tinybench_lm.data_protocols import (
    BenchmarkItem, CLEAN, KEEP, QUARANTINE, DecontaminationDecision,
    DocumentRecord, FROZEN_PROTOCOL_SHA256, PRODUCTION_DECONTAM_PROTOCOL_PATH,
    decontaminate, load_decontamination_protocol, rule_by_id,
)
from test_corpus_pipeline import counter, source_candidate


def protocol(version: int) -> dict:
    return load_decontamination_protocol(PRODUCTION_DECONTAM_PROTOCOL_PATH.with_name(f"decontam_v{version}.yaml"))


def write_items(path: Path, items: list[BenchmarkItem]) -> None:
    path.write_text("".join(json.dumps({"task_id": i.task_id, "item_id": i.item_id, "texts": i.texts}) + "\n" for i in items), encoding="utf-8")


def test_v3_changes_only_rule1_resolution_and_preserves_pinned_inputs() -> None:
    old, new = protocol(2), protocol(3)
    for key in ("matching_normalization", "benchmark_scope", "decision", "boundary_review"):
        assert old[key] == new[key]
    assert old["quarantine_rules"][1:] == new["quarantine_rules"][1:]
    assert rule_by_id(new, "RULE_1_COMPLETE_ITEM_SUBSTRING")["minimum_item_words"] == 13
    assert new["calibrated_before_real_corpus_scan"] is False


@pytest.mark.parametrize("length", [1, 2, 12, 13, 14])
def test_rule1_boundary_and_normalization_match_reference(tmp_path: Path, length: int) -> None:
    phrase = " ".join(f"fieldword{i}" for i in range(length))
    items = [BenchmarkItem("task", "threshold", ("2", phrase))]
    source = tmp_path / "items.jsonl"
    write_items(source, items)
    text = "prefix 2 " + phrase.upper().replace(" ", "\t  ") + " suffix"
    with BenchmarkIndex(tmp_path / "index.sqlite", protocol=protocol(3)) as index:
        index.build(source, expected_sha256=file_sha256(source))
        observed = index.classify("doc", text)
        expected = decontaminate([DocumentRecord("doc", text)], items, protocol(3)).decision("doc")
        assert observed == expected
        assert observed.action == (QUARANTINE if length >= 13 else KEEP)
        if length >= 13:
            assert observed.measurement == length
        assert index.classify("boundary", "prefix" + phrase + "suffix").action == KEEP


def test_short_answers_are_clean_but_full_prompts_and_overlap_still_quarantine(tmp_path: Path) -> None:
    question = "a young ocean animal has two parents. each of the parents has eight arms. how many arms does the young animal most likely have?"
    items = [
        BenchmarkItem("arc_challenge", "test:234", (question, "2", "4", "8", "16")),
        BenchmarkItem("task", "long", (" ".join(f"longword{i}" for i in range(60)),)),
        BenchmarkItem("task", "coverage", (" ".join(f"coverword{i}" for i in range(30)),)),
    ]
    source = tmp_path / "items.jsonl"
    write_items(source, items)
    documents = {
        "clean": "The table has 2 columns and 8 rows; we ordered 4 books for 16 students.",
        "full": "Introduction " + question + " conclusion",
        "overlap": "prefix " + " ".join(f"longword{i}" for i in range(5, 55)) + " suffix",
        "coverage": "prefix " + " ".join(f"coverword{i}" for i in range(15)) + " suffix",
    }
    with BenchmarkIndex(tmp_path / "index.sqlite", protocol=protocol(3)) as index:
        index.build(source, expected_sha256=file_sha256(source))
        for doc_id, text in documents.items():
            observed = index.classify(doc_id, text)
            assert observed == decontaminate([DocumentRecord(doc_id, text)], items, protocol(3)).decision(doc_id)
            assert observed.action == (KEEP if doc_id == "clean" else QUARANTINE)
        assert index.classify("full", documents["full"]).measurement == 24
        assert any(m.rule_id == "RULE_2_LONGEST_CONTIGUOUS_OVERLAP" for m in index.classify("overlap", documents["overlap"]).matched_rules)
        assert index.classify("coverage", documents["coverage"]).rule_id == "RULE_3_SHINGLE_COVERAGE"
    assert decontaminate([DocumentRecord("clean", documents["clean"])], items, protocol(2)).decision("clean").action == QUARANTINE


def test_complete_v2_index_is_reused_without_relabeling(tmp_path: Path) -> None:
    source, path = tmp_path / "items.jsonl", tmp_path / "index.sqlite"
    write_items(source, [BenchmarkItem("task", "answer", ("2",))])
    with BenchmarkIndex(path, protocol=protocol(2)) as index:
        index.build(source, expected_sha256=file_sha256(source))
        assert index.classify("doc", "There are 2 cats").action == QUARANTINE
    with BenchmarkIndex(path, protocol=protocol(3)) as index:
        assert index.classify("doc", "There are 2 cats").action == KEEP
        assert index.source_sha256 == file_sha256(source)
        assert dict(index.connection.execute("SELECT key,value FROM metadata"))["protocol_digest"] == protocol(2)["_digest"]
    with pytest.raises(BenchmarkIndexError, match="different"):
        BenchmarkIndex(path, protocol=protocol(1))


def populated(path: Path) -> CorpusState:
    state = CorpusState(path, token_counter=counter, token_counter_id="fixture")
    state.ingest("fineweb_edu", [source_candidate("fineweb_edu", str(i), f"unique{i}") for i in range(4)])
    state.run_deduplication()
    return state


class CleanIndex:
    protocol = {"_digest": "fixture_protocol"}
    source_sha256 = "fixture_source"

    def classify(self, doc_id: str, _text: str) -> DecontaminationDecision:
        return DecontaminationDecision(doc_id, KEEP, CLEAN)


def test_resume_binding_limit_and_wal_checkpoint_between_batches(tmp_path: Path) -> None:
    with populated(tmp_path / "state.sqlite") as state:
        class CheckingIndex(CleanIndex):
            calls = 0

            def classify(self, doc_id: str, text: str) -> DecontaminationDecision:
                self.calls += 1
                if self.calls == 3:
                    assert tuple(state.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()) == (0, 0, 0)
                return super().classify(doc_id, text)

        assert state.run_decontamination(CheckingIndex(), limit=3, commit_every=2) == 3
        assert state.run_decontamination(CleanIndex(), limit=0) == 0
        assert state.run_decontamination(CleanIndex(), commit_every=2) == 1
        assert state.run_decontamination(CleanIndex()) == 0
        with pytest.raises(ResumeMismatchError, match="changed"):
            state.bind_decontamination("different", "fixture_source")
        with pytest.raises(ResumeMismatchError, match="changed"):
            state.bind_decontamination("fixture_protocol", "different")


def test_unbound_legacy_decisions_cannot_be_silently_resumed(tmp_path: Path) -> None:
    with populated(tmp_path / "state.sqlite") as state:
        state.mark_all_clean_for_fixture()
        with pytest.raises(ResumeMismatchError, match="legacy unbound"):
            state.run_decontamination(CleanIndex())


def test_migration_preserves_original_and_upstream_rows_and_refuses_overwrite(tmp_path: Path) -> None:
    source, destination = tmp_path / "old.sqlite", tmp_path / "v3"
    with populated(source) as state:
        state.mark_all_clean_for_fixture()
        before = {t: [tuple(r) for r in state.connection.execute(f"SELECT * FROM {t}")] for t in ("documents", "dedup_decisions", "source_cursors", "decontamination")}
    digest = file_sha256(source)
    report = fork_state(source, destination, legacy_protocol_digest=protocol(2)["_digest"])
    assert file_sha256(source) == digest
    assert report["previous_decision_counts"] == [(KEEP, CLEAN, 4)]
    with sqlite3.connect(destination / "state.sqlite") as copied:
        for table in ("documents", "dedup_decisions", "source_cursors"):
            assert copied.execute(f"SELECT * FROM {table}").fetchall() == before[table]
        assert copied.execute("SELECT COUNT(*) FROM decontamination").fetchone()[0] == 0
        assert dict(copied.execute("SELECT key,value FROM metadata"))["decontamination_protocol_digest"] == FROZEN_PROTOCOL_SHA256["decontam_v3.yaml"]
    assert (destination / "migration.json").is_file()
    with pytest.raises(ValueError, match="already exists"):
        fork_state(source, destination, legacy_protocol_digest=protocol(2)["_digest"])


def test_migration_refuses_downstream_selection(tmp_path: Path) -> None:
    source = tmp_path / "old.sqlite"
    with populated(source) as state:
        state.mark_all_clean_for_fixture()
        state.assign_fixture()
    destination = tmp_path / "v3"
    with pytest.raises(ValueError, match="downstream"):
        fork_state(source, destination, legacy_protocol_digest=protocol(2)["_digest"])
    assert not destination.exists()
