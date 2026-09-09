from pathlib import Path

import pytest

import copy

from tinybench_lm.evaluation_binding import (
    EvaluationBindingError,
    resolve_effective_binding,
    verify_smoke_bundle,
    verify_smoke_results,
)
from tinybench_lm.evaluation_protocol import load_evaluation_protocol, write_run_bundle


@pytest.fixture(scope="module")
def protocol():
    return load_evaluation_protocol(Path("configs/evaluation/evaluation_provisional_v2.yaml"))


def test_full_binding_resolves_v3_revisions_and_records_wheel_identity(protocol):
    configs, facts = resolve_effective_binding(protocol, ["hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103"])
    assert [config["task"] for config in configs] == ["hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103"]
    assert facts["tasks"]["piqa"]["dataset_path"] == "ybisk/piqa"
    assert configs[2]["dataset_kwargs"]["trust_remote_code"] is True
    assert facts["harness_version"] == "0.4.12"
    assert facts["harness_content_sha256"]
    assert facts["harness_file_count"] > 0
    assert facts["harness_commit_status"] == "BLOCKED_UNOBSERVABLE_FROM_WHEEL"
    assert facts["binding_digest"]


def test_binding_rejects_task_definition_drift(protocol, monkeypatch):
    import tinybench_lm.evaluation_binding as binding
    monkeypatch.setitem(binding.EXPECTED_TASK_FILES, "piqa.yaml", "0" * 64)
    with pytest.raises(EvaluationBindingError, match="drifted"):
        resolve_effective_binding(protocol, ["piqa"])


def test_smoke_counts_are_required_and_bounded():
    assert verify_smoke_results({"n-samples": {"piqa": 3}}, ["piqa"], limit=100) == {"piqa": 3}
    with pytest.raises(EvaluationBindingError, match="missing"):
        verify_smoke_results({"n-samples": {}}, ["piqa"])
    with pytest.raises(EvaluationBindingError, match="limit"):
        verify_smoke_results({"n-samples": {"piqa": 101}}, ["piqa"])


def test_smoke_bundle_rejects_wrong_source_or_protocol_identity(protocol, tmp_path: Path):
    """A complete-looking rehearsal cannot be relabelled with another source or protocol."""
    task_ids = ["hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103"]
    _, facts = resolve_effective_binding(protocol, task_ids)
    counts = {task: {"effective": 1} for task in task_ids}
    bundle = write_run_bundle(
        tmp_path / "bundle",
        protocol=protocol,
        command=["python", "evaluate.py", "--smoke", "--limit", "1"],
        raw_results={"n-samples": counts},
        task_ids=task_ids,
        sample_counts={task: 1 for task in task_ids},
        runtime_seconds={"total": 0.5},
        device="cpu",
        precision="float32",
        model_identity={"checkpoint_sha256": "a" * 64, "tokenizer_sha256": "b" * 64},
        harness_facts=facts,
    )
    verified = verify_smoke_bundle(
        bundle.directory,
        protocol,
        expected_checkpoint_sha256="a" * 64,
        expected_tokenizer_sha256="b" * 64,
    )
    assert verified["sample_counts"] == {task: 1 for task in task_ids}
    with pytest.raises(EvaluationBindingError, match="checkpoint_sha256"):
        verify_smoke_bundle(bundle.directory, protocol, expected_checkpoint_sha256="0" * 64)
    with pytest.raises(EvaluationBindingError, match="tokenizer_sha256"):
        verify_smoke_bundle(bundle.directory, protocol, expected_tokenizer_sha256="0" * 64)
    drifted = copy.deepcopy(protocol)
    drifted["runtime"]["seed"] = 999
    with pytest.raises(EvaluationBindingError, match="integrity"):
        verify_smoke_bundle(bundle.directory, drifted)
