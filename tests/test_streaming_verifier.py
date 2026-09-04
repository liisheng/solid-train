"""Focused bounded-memory manifest verifier checks."""

from __future__ import annotations

import json
from pathlib import Path

from tinybench_lm.shards import (
    RESERVED,
    SCALE_FIXTURE,
    STABLE_TRAIN,
    VALIDATION_DEV,
    VALIDATION_FINAL,
    ShardDocument,
    build_split_manifest,
    load_shard_protocol,
    write_split_manifest,
)
from tinybench_lm.source_manifest import load_source_registry
from tinybench_lm.streaming_verify import verify_shard_outputs_streaming
from tinybench_lm.tokenizer import load_tokenizer_artifact, load_tokenizer_protocol


def _docs(sources: list[str], boundary: str, *, prefix: str) -> list[ShardDocument]:
    slices = {
        "fineweb_edu": "educational_science",
        "dclm": "broad_general",
        "openwebmath": "math_technical",
        "narrative": "narrative_coreference",
    }
    return [
        ShardDocument(
            f"{prefix}-{source}",
            source,
            f"A bounded verification document for {source} with enough prose.",
            boundary,
            protected_slice=(slices[source] if boundary in (VALIDATION_DEV, VALIDATION_FINAL) else None),
        )
        for source in sources
    ]


def _build_fixture(root: Path) -> None:
    protocol = load_shard_protocol()
    registry = load_source_registry()
    tokenizer_protocol = load_tokenizer_protocol()
    tokenizer, _ = load_tokenizer_artifact(Path("data/tokenizer_final"))
    groups = {
        STABLE_TRAIN: _docs(["fineweb_edu", "dclm", "openwebmath", "narrative"], STABLE_TRAIN, prefix="stable"),
        RESERVED: _docs(["reserved_science", "reserved_textbook", "reserved_wikipedia", "reserved_edu_decile", "reserved_math_prose"], RESERVED, prefix="reserved"),
        VALIDATION_DEV: _docs(["fineweb_edu", "dclm", "openwebmath", "narrative"], VALIDATION_DEV, prefix="dev"),
        VALIDATION_FINAL: _docs(["fineweb_edu", "dclm", "openwebmath", "narrative"], VALIDATION_FINAL, prefix="final"),
    }
    for split_id, docs in groups.items():
        manifest = build_split_manifest(root, tokenizer, docs, split_id=split_id, protocol=protocol, registry=registry, tokenizer_protocol=tokenizer_protocol)
        write_split_manifest(root, manifest, protocol)


def test_streaming_verifier_parses_one_shard_at_a_time(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    report = verify_shard_outputs_streaming(tmp_path, scale=SCALE_FIXTURE)
    assert report.status == "NOT_RUN"
    assert not any(item.status == "FAIL" for item in report.results), [item.__dict__ for item in report.results if item.status == "FAIL"]
    assert report.facts["manifests"] == 4
    assert all(item.status != "PASS" for item in report.results if item.check_id.startswith("shards.stable_share."))


def test_streaming_verifier_rejects_manifest_content_hash_tampering(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    path = tmp_path / "stable_train.manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["content_hash"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = verify_shard_outputs_streaming(tmp_path, scale=SCALE_FIXTURE)
    assert any(item.status == "FAIL" and "manifest" in item.check_id for item in report.results)


def test_streaming_verifier_requires_structured_isolation_evidence(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    evidence = tmp_path / "isolation.json"
    evidence.write_text(json.dumps({"status": "PASS", "isolation_verified": True}), encoding="utf-8")
    report = verify_shard_outputs_streaming(tmp_path, scale=SCALE_FIXTURE, isolation_evidence=evidence)
    result = next(item for item in report.results if item.check_id == "shards.cluster_isolation")
    assert result.status == "FAIL"
