from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.prepare_corpus import bounded_rows, remaining_slice_budget

from tinybench_lm.benchmark_index import BenchmarkIndex, file_sha256
from tinybench_lm.data_protocols import CLEAN, KEEP, DecontaminationDecision
from tinybench_lm.corpus_pipeline import (
    CorpusPipelineError,
    CorpusState,
    ResumeMismatchError,
    StreamedSourceRow,
    assert_write_space,
    load_acquisition_protocol,
    natural_document_id,
    selection_key,
)
from tinybench_lm.source_manifest import CandidateDocument
from tinybench_lm.source_manifest import load_source_registry


REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
LICENSE = "ODC-By 1.0"
BASE = (
    "This educational article explains how the physical system works and why the result "
    "is useful for students who want to understand the ideas in a careful way. "
) * 8


def candidate(document_id: str, text: str = BASE) -> CandidateDocument:
    return CandidateDocument(
        source_id="fineweb_edu",
        document_id=document_id,
        text=text,
        revision=REVISION,
        license=LICENSE,
        url=f"https://example.test/{document_id}",
    )


def counter(text: str) -> int:
    return len(text.split())


def test_frozen_acquisition_protocol_loads_and_selection_keys_are_stable() -> None:
    protocol = load_acquisition_protocol()
    assert protocol["_digest"]
    assert selection_key("stable", "fineweb_edu", "doc:1", salt="salt") == selection_key(
        "stable", "fineweb_edu", "doc:1", salt="salt"
    )
    assert selection_key("stable", "fineweb_edu", "doc:1", salt="salt") != selection_key(
        "stable", "fineweb_edu", "doc:2", salt="salt"
    )


def test_natural_document_id_uses_frozen_preference() -> None:
    assert natural_document_id({"id": "id-1", "url": "https://ignored"}, "source", 3) == "id-1"
    assert natural_document_id({"url": "https://used"}, "source", 3) == "https://used"
    assert natural_document_id({"METADATA": json.dumps({"text_id": 42})}, "source", 3) == "gutenberg:42"
    assert natural_document_id({}, "source", 3) == "source:000000000003"


def test_disk_floors_warn_then_fail_closed(tmp_path: Path, monkeypatch) -> None:
    usage = type("Usage", (), {"total": 100, "free": 12})()
    monkeypatch.setattr("tinybench_lm.corpus_pipeline.shutil.disk_usage", lambda _path: usage)
    with pytest.warns(RuntimeWarning, match="warning floor"):
        assert assert_write_space(tmp_path) == pytest.approx(0.12)
    usage.free = 10
    with pytest.raises(CorpusPipelineError, match="write-stop floor"):
        assert_write_space(tmp_path)


def test_pipeline_is_restartable_deduplicates_and_writes_sorted_shard_input(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    prefix_changed = "Different opening words appear here. " + BASE
    near_changed = "Another opening phrase is deliberately distinct. " + BASE
    with CorpusState(state_path, token_counter=counter, token_counter_id="test_counter") as state:
        assert state.ingest(
            "fineweb_edu",
            [candidate("doc:001"), candidate("doc:002"), candidate("doc:003", prefix_changed), candidate("doc:004", near_changed)],
            scores=[4.0, 4.0, 3.0, 2.0],
            commit_every=1,
        ) == 4
        assert state.cursor("fineweb_edu").complete
        assert state.run_deduplication() == 4
        assert state.run_deduplication() == 0
        assert state.mark_all_clean_for_fixture() >= 1
        assert state.assign_fixture() >= 1
        output = tmp_path / "accepted.jsonl"
        count, digest = state.write_accepted_jsonl(output)
        assert count == state.summary().assigned_documents
        assert len(digest) == 64
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert rows == sorted(rows, key=lambda row: (row["source_id"], row["document_id"]))
        assert all(row["boundary"] == "stable_train" for row in rows)
        assert all(row["cluster_id"].startswith("cluster:") for row in rows)
        ledger = tmp_path / "decisions.jsonl"
        ledger_count, ledger_hash = state.write_decisions_jsonl(ledger)
        assert ledger_count == 4
        assert len(ledger_hash) == 64
        decisions = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        assert all("source_manifest" in row and "dedup" in row and "decontamination" in row for row in decisions)

    with CorpusState(state_path, token_counter=counter, token_counter_id="test_counter") as resumed:
        assert resumed.ingest("fineweb_edu", []) == 0
        reasons = resumed.summary().reason_counts
        assert reasons["UNIQUE"] >= 1
        assert reasons["EXACT_DUPLICATE"] == 1
        assert reasons["NEAR_DUPLICATE"] >= 1


def test_resume_rejects_wrong_cursor_and_contract(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite"
    with CorpusState(state_path, token_counter=counter, token_counter_id="counter-a") as state:
        state._save_cursor("fineweb_edu", 1, False)
        state.connection.commit()
        with pytest.raises(ResumeMismatchError, match="resume at row 1"):
            state.ingest("fineweb_edu", [], start_row_index=0)
    with pytest.raises(ResumeMismatchError, match="token_counter_id"):
        CorpusState(state_path, token_counter=counter, token_counter_id="counter-b")


def test_repeated_identity_is_resolved_or_reason_coded(tmp_path: Path) -> None:
    with CorpusState(tmp_path / "state.sqlite", token_counter=counter, token_counter_id="test") as state:
        changed = BASE.replace("physical system", "historical process")
        assert state.ingest("fineweb_edu", [candidate("same"), candidate("same"), candidate("same", changed)]) == 2
        identities = [row[0] for row in state.connection.execute("SELECT document_id FROM documents ORDER BY document_id")]
        assert identities[0] == "same"
        assert identities[1].startswith("same#")
        event = state.connection.execute("SELECT reason_code FROM ingest_events").fetchone()
        assert event[0] == "EXACT_DUPLICATE"
        ledger = tmp_path / "decisions.jsonl"
        count, _ = state.write_decisions_jsonl(ledger)
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        assert count == 3
        assert rows[-1] == {
            "event_type": "ingest_event",
            "matched_doc_key": "fineweb_edu/same",
            "origin_source_id": "fineweb_edu",
            "reason_code": "EXACT_DUPLICATE",
            "row_index": 1,
        }


def test_incomplete_empty_replay_cannot_claim_source_exhaustion(tmp_path: Path) -> None:
    with CorpusState(tmp_path / "state.sqlite", token_counter=counter, token_counter_id="test") as state:
        state.ingest("fineweb_edu", [candidate("first")], mark_complete=False)
        cursor = state.cursor("fineweb_edu")
        assert cursor.consumed_tokens == counter(BASE)
        with pytest.raises(CorpusPipelineError, match="empty replay"):
            state.ingest(
                "fineweb_edu",
                [],
                start_row_index=cursor.next_row_index,
                mark_complete=True,
            )
        assert not state.cursor("fineweb_edu").complete


def test_resumed_slice_uses_only_its_persisted_remaining_budget(tmp_path: Path) -> None:
    rows = [
        StreamedSourceRow(index, candidate(f"doc:{index}"), None)
        for index in range(3)
    ]
    budget = counter(BASE)
    with CorpusState(tmp_path / "state.sqlite", token_counter=counter, token_counter_id="test") as state:
        first = list(bounded_rows(rows, token_counter=counter, token_budget=budget))
        state.ingest(
            "fineweb_edu",
            (row.candidate for row in first),
            mark_complete=False,
        )
        cursor = state.cursor("fineweb_edu")
        assert cursor.next_row_index == 1
        assert remaining_slice_budget(budget, cursor.consumed_tokens) == 0
        assert state.summary().documents_seen == 1


def test_completed_slice_stages_are_idempotent_without_new_ingestion(tmp_path: Path) -> None:
    with CorpusState(tmp_path / "state.sqlite", token_counter=counter, token_counter_id="test") as state:
        state.ingest("fineweb_edu", [candidate("only")], mark_complete=False)
        state.run_deduplication()
        state.mark_all_clean_for_fixture()
        state.assign_fixture()
        assert state.run_deduplication() == 0


def test_decontamination_batches_are_restart_safe(tmp_path: Path) -> None:
    class FailingIndex:
        calls = 0

        def classify(self, doc_id: str, _text: str) -> DecontaminationDecision:
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError("injected classifier failure")
            return DecontaminationDecision(doc_id, KEEP, CLEAN)

    state_path = tmp_path / "state.sqlite"
    with CorpusState(state_path, token_counter=counter, token_counter_id="test") as state:
        state.ingest(
            "fineweb_edu",
            [source_candidate("fineweb_edu", f"doc:{index}", f"batch{index}") for index in range(4)],
        )
        state.run_deduplication()
        with pytest.raises(RuntimeError, match="injected"):
            state.run_decontamination(FailingIndex(), commit_every=2)
        assert not state.connection.in_transaction
        assert state.connection.execute("SELECT COUNT(1) FROM decontamination").fetchone()[0] == 2
    with CorpusState(state_path, token_counter=counter, token_counter_id="test") as resumed:
        assert resumed.connection.execute("SELECT COUNT(1) FROM decontamination").fetchone()[0] == 2


def test_decontamination_rejects_zero_commit_interval(tmp_path: Path) -> None:
    with CorpusState(tmp_path / "state.sqlite", token_counter=counter, token_counter_id="test") as state:
        with pytest.raises(ValueError, match="commit_every"):
            state.run_decontamination(object(), commit_every=0)


def test_output_bundle_is_atomic_on_failure(tmp_path: Path, monkeypatch) -> None:
    with CorpusState(tmp_path / "state.sqlite", token_counter=counter, token_counter_id="test") as state:
        state.ingest("fineweb_edu", [candidate("only")])
        state.run_deduplication()
        state.mark_all_clean_for_fixture()
        state.assign_fixture()
        bundle = tmp_path / "published"

        def fail_ledger(_path: Path) -> tuple[int, str]:
            raise OSError("injected ledger failure")

        monkeypatch.setattr(state, "write_decisions_jsonl", fail_ledger)
        with pytest.raises(OSError, match="injected"):
            state.publish_jsonl_bundle(bundle / "accepted.jsonl", bundle / "decisions.jsonl")
        assert not bundle.exists()
        assert not (tmp_path / ".published.staging").exists()


def test_pipeline_records_indexed_decontamination_decisions(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text(
        json.dumps({"task_id": "task", "item_id": "item", "texts": ["rare benchmark phrase"]}) + "\n",
        encoding="utf-8",
    )
    with BenchmarkIndex(tmp_path / "benchmark.sqlite") as index:
        index.build(items, expected_sha256=file_sha256(items))
        with CorpusState(tmp_path / "state.sqlite", token_counter=counter, token_counter_id="test") as state:
            state.ingest("fineweb_edu", [candidate("doc", BASE + " rare benchmark phrase")])
            state.run_deduplication()
            assert state.run_decontamination(index) == 1
            assert state.run_decontamination(index) == 0
            row = state.connection.execute(
                "SELECT action, reason_code FROM decontamination"
            ).fetchone()
            assert tuple(row) == ("QUARANTINE", "BENCHMARK_ITEM_SUBSTRING")


def test_contaminated_duplicate_member_quarantines_complete_cluster(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text(
        json.dumps({"task_id": "task", "item_id": "item", "texts": ["rare benchmark phrase"]}) + "\n",
        encoding="utf-8",
    )
    clean = BASE + " clean ending"
    contaminated = BASE + " rare benchmark phrase"
    with BenchmarkIndex(tmp_path / "benchmark.sqlite") as index:
        index.build(items, expected_sha256=file_sha256(items))
        with CorpusState(tmp_path / "state.sqlite", token_counter=counter, token_counter_id="test") as state:
            state.ingest("fineweb_edu", [candidate("a", clean), candidate("b", contaminated)])
            state.run_deduplication()
            assert state.connection.execute(
                "SELECT COUNT(DISTINCT cluster_id) FROM dedup_decisions"
            ).fetchone()[0] == 1
            assert state.run_decontamination(index) == 2
            assert state.assign_fixture() == 0
            assert state.connection.execute(
                "SELECT COUNT(*) FROM decontamination WHERE action = 'QUARANTINE'"
            ).fetchone()[0] == 1


def test_near_duplicate_bridge_merges_connected_components(tmp_path: Path, monkeypatch) -> None:
    import tinybench_lm.corpus_pipeline as pipeline

    signatures = {
        "alpha": (0,) * 128,
        "charlie": (1,) * 20 + (0,) * 108,
        "zulu": (1,) * 10 + (0,) * 118,
    }
    monkeypatch.setattr(pipeline, "document_sha256", lambda text, _protocol: f"exact:{text.split()[-1]}")
    monkeypatch.setattr(pipeline, "mirror_sha256", lambda text, _protocol: f"mirror:{text.split()[-1]}")
    monkeypatch.setattr(pipeline, "word_shingles", lambda text, _size, _protocol: (text.split()[-1],))
    monkeypatch.setattr(pipeline, "minhash_signature", lambda shingles, _protocol: signatures[shingles[0]])
    with CorpusState(tmp_path / "state.sqlite", token_counter=counter, token_counter_id="test") as state:
        state.ingest(
            "fineweb_edu",
            [
                candidate("a", BASE + " alpha"),
                candidate("c", BASE + " charlie"),
                candidate("z", BASE + " zulu"),
            ],
        )
        state.run_deduplication()
        rows = state.connection.execute(
            "SELECT action, cluster_id FROM dedup_decisions ORDER BY doc_key"
        ).fetchall()
        assert [row[0] for row in rows] == ["KEEP", "DROP", "DROP"]
        assert len({row[1] for row in rows}) == 1


def source_candidate(source_id: str, document_id: str, unique: str) -> CandidateDocument:
    registry = load_source_registry()
    spec = next(
        item
        for item in registry["stable_sources"] + registry["reserved_sources"]
        if item["source_id"] == source_id
    )
    translation = str.maketrans("0123456789", "abcdefghij")
    sentences = []
    for index in range(30):
        words = [
            hashlib.sha256(f"{unique}:{index}:{suffix}".encode()).hexdigest()[:8].translate(translation)
            for suffix in range(4)
        ]
        sentences.append(f"The {' '.join(words)} is explained.")
    text = ("Physics is scientific. " if unique == "physics" else "") + " ".join(sentences)
    return CandidateDocument(
        source_id=source_id,
        document_id=document_id,
        text=text,
        revision=spec["intended_revision"],
        license=spec["declared_license"],
    )


def test_production_assignment_is_disjoint_cluster_atomic_and_target_driven(tmp_path: Path) -> None:
    registry = deepcopy(load_source_registry())
    for source in registry["stable_sources"]:
        source["target_tokens_at_11b"] = 30
    for source in registry["reserved_sources"]:
        source["target_tokens_at_minimum"] = 10
    counts = {
        "fineweb_edu": 6,
        "dclm": 4,
        "openwebmath": 5,
        "narrative": 4,
        "reserved_textbook": 1,
        "reserved_wikipedia": 1,
    }
    with CorpusState(
        tmp_path / "state.sqlite",
        token_counter=counter,
        token_counter_id="test",
        registry=registry,
    ) as state:
        for source_id, count in counts.items():
            candidates = [
                source_candidate(
                    source_id,
                    f"{source_id}:{index}",
                    ("physics" if source_id == "fineweb_edu" and index == 0 else f"unique{source_id}{index}"),
                )
                for index in range(count)
            ]
            state.ingest(source_id, candidates, scores=range(count), commit_every=2)
        state.run_deduplication()
        state.mark_all_clean_for_fixture()
        results = state.assign_production(target_fraction=0.00001)
        assert all(result.complete for result in results), [result for result in results if not result.complete]
        duplicate_clusters = state.connection.execute(
            "SELECT cluster_id FROM assignments JOIN dedup_decisions USING(doc_key) GROUP BY cluster_id HAVING COUNT(*) > 1"
        ).fetchall()
        assert duplicate_clusters == []
        boundaries = {row[0] for row in state.connection.execute("SELECT DISTINCT boundary FROM assignments")}
        assert boundaries == {"reserved", "stable_train", "validation_dev", "validation_final"}
        assert state.isolation_evidence()["status"] == "PASS"
        science = state.connection.execute(
            "SELECT d.science_match FROM assignments a JOIN documents d USING(doc_key) WHERE a.source_id = 'reserved_science'"
        ).fetchone()
        assert science[0] == 1
