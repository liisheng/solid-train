"""Full-score evidence regressions; synthetic samples, no benchmark scoring."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

from tinybench_lm.evaluation_coverage import validate_execution, verify_full_results
from tinybench_lm.evaluation_protocol import load_evaluation_protocol, write_run_bundle, verify_run_bundle


TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103"]


@pytest.mark.parametrize("flags,expected", [(["--full"], True), (["--smoke", "--limit", "1"], False)])
def test_cli_retains_document_records_required_by_full_verifier(monkeypatch, flags, expected):
    import evaluate
    import tinybench_lm.evaluation_binding as binding
    import tinybench_lm.evaluation_coverage as coverage

    class Model:
        def policy_identity(self):
            return {"device": "cpu"}

    class ScoringReached(Exception):
        pass

    def score(**kwargs):
        assert kwargs["log_samples"] is expected
        raise ScoringReached

    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", "fixture.pt", "--tokenizer", "fixture.json", *flags])
    monkeypatch.setattr(evaluate, "TinyBenchHarnessLM", lambda **kwargs: Model())
    monkeypatch.setattr(binding, "resolve_effective_binding", lambda *args: (TASKS, {}))
    monkeypatch.setattr(coverage, "prepare_full_tasks", lambda *args: (TASKS, coverage.FULL_SPLITS_V2))
    monkeypatch.setattr(evaluate.lm_eval, "simple_evaluate", score)
    with pytest.raises(ScoringReached):
        evaluate.main()


@pytest.mark.parametrize("flags", [
    ["--full", "--smoke", "--limit", "1"],
    ["--full", "--limit", "1"],
    ["--full", "--tasks", "piqa"],
    ["--full", "--secondary"],
])
def test_cli_rejects_partial_full_before_model_loading(monkeypatch, flags):
    import evaluate
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", "absent.pt", "--tokenizer", "absent.json", *flags])
    monkeypatch.setattr(evaluate, "TinyBenchHarnessLM", lambda **kwargs: pytest.fail("loaded model before rejecting invalid full request"))
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2


def test_full_accepts_reordered_complete_set_but_rejects_duplicates():
    validate_execution("full", None, TASKS[::-1], TASKS)
    with pytest.raises(ValueError):
        validate_execution("full", None, TASKS + [TASKS[0]], TASKS)


def fixture_results():
    coverage = {name: {"count": 3, "split": "test", "dataset_path": "fixture", "dataset_name": None, "revision": "fixture-revision"} for name in TASKS}
    raw = {
        "results": {name: {"fixture_score": 0} for name in TASKS},
        "n-samples": {name: {"original": 3, "effective": 3} for name in TASKS},
        "samples": {name: [{"doc_id": i} for i in range(3)] for name in TASKS},
        "configs": {name: {"test_split": "test", "dataset_path": "fixture", "dataset_name": None, "dataset_kwargs": {"revision": "fixture-revision"}} for name in TASKS},
        "config": {"limit": None},
    }
    return raw, coverage


@pytest.mark.parametrize("mutation", ["count", "duplicate", "missing_task", "split", "limit", "missing_sample"])
def test_full_rejects_incomplete_actual_coverage(mutation):
    raw, coverage = fixture_results()
    verify_full_results(raw, TASKS, coverage)
    if mutation == "count":
        raw["n-samples"]["piqa"]["effective"] = 1
    elif mutation == "duplicate":
        raw["samples"]["piqa"][2]["doc_id"] = 1
    elif mutation == "missing_task":
        del raw["results"]["piqa"]
    elif mutation == "split":
        raw["configs"]["piqa"]["test_split"] = "validation"
    elif mutation == "limit":
        raw["config"]["limit"] = 1
    else:
        raw["samples"]["piqa"].pop()
    with pytest.raises(ValueError):
        verify_full_results(raw, TASKS, coverage)


def test_bundle_rechecks_full_results_even_after_manifest_rehash(tmp_path, monkeypatch):
    protocol = load_evaluation_protocol(Path("configs/evaluation/evaluation_provisional_v2.yaml"))
    raw, coverage = fixture_results()
    monkeypatch.setattr("tinybench_lm.evaluation_coverage.FULL_SPLITS_V2", coverage)
    bundle = write_run_bundle(
        tmp_path / "bundle", protocol=protocol, command=["python", "evaluate.py", "--full"],
        task_ids=TASKS, raw_results=raw, sample_counts=raw["n-samples"], runtime_seconds=1,
        device="cpu", precision="float32", execution={"mode": "full", "limit": None, "full_coverage": coverage},
    )
    assert verify_run_bundle(bundle.directory, protocol).ok
    names = protocol["evidence_bundle"]["required_artifacts"]
    raw["samples"]["piqa"].pop()
    raw_path = bundle.directory / names["raw_results"]
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    manifest_path = bundle.directory / names["manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][names["raw_results"]] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = verify_run_bundle(bundle.directory, protocol)
    assert report.result("bundle.manifest_hashes_match").status == "PASS"
    assert report.result("bundle.execution_coverage").status == "FAIL"


def test_full_command_cannot_claim_partial_bundle(tmp_path):
    protocol = load_evaluation_protocol()
    bundle = write_run_bundle(tmp_path, protocol=protocol, command=["python", "evaluate.py", "--full", "--limit=1"],
                              task_ids=["piqa"], raw_results={}, sample_counts={}, runtime_seconds=0,
                              device="cpu", precision="float32")
    assert not verify_run_bundle(bundle.directory, protocol).ok


def test_self_consistent_small_full_bundle_cannot_redefine_split_sizes(tmp_path):
    protocol = load_evaluation_protocol(Path("configs/evaluation/evaluation_provisional_v2.yaml"))
    raw, coverage = fixture_results()
    bundle = write_run_bundle(
        tmp_path, protocol=protocol, command=["python", "evaluate.py", "--full"],
        task_ids=TASKS, raw_results=raw, sample_counts=raw["n-samples"], runtime_seconds=1,
        device="cpu", precision="float32", execution={"mode": "full", "limit": None, "full_coverage": coverage},
    )
    report = verify_run_bundle(bundle.directory, protocol)
    assert report.result("bundle.manifest_hashes_match").status == "PASS"
    assert report.result("bundle.execution_coverage").status == "FAIL"
    assert "trusted" in report.result("bundle.execution_coverage").reason


def test_trusted_full_split_fixture_passes_without_scoring(tmp_path):
    from tinybench_lm.evaluation_coverage import FULL_SPLITS_V2
    protocol = load_evaluation_protocol(Path("configs/evaluation/evaluation_provisional_v2.yaml"))
    raw = {
        "results": {name: {"fixture_score": 0} for name in TASKS},
        "n-samples": {name: {"original": values["count"], "effective": values["count"]} for name, values in FULL_SPLITS_V2.items()},
        "samples": {name: [{"doc_id": i} for i in range(values["count"])] for name, values in FULL_SPLITS_V2.items()},
        "configs": {name: {"test_split" if values["split"] == "test" else "validation_split": values["split"],
                            "dataset_path": values["dataset_path"], "dataset_name": values["dataset_name"],
                            "dataset_kwargs": {"revision": values["revision"]}} for name, values in FULL_SPLITS_V2.items()},
        "config": {"limit": None},
    }
    bundle = write_run_bundle(tmp_path, protocol=protocol, command=["python", "evaluate.py", "--full"],
                             task_ids=TASKS[::-1], raw_results=raw, sample_counts=raw["n-samples"], runtime_seconds=1,
                             device="cpu", precision="float32", execution={"mode": "full", "limit": None, "full_coverage": FULL_SPLITS_V2})
    assert verify_run_bundle(bundle.directory, protocol).ok
