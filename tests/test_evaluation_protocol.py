"""Frozen evaluation-protocol tests: identity, labelling, promotion, and run bundles.

Stubs and fixtures only (Plan Sections 2.2, 10.1-10.4). Nothing here downloads a benchmark,
runs a real harness suite, or scores WikiText-103. The claims under test are structural:

* a score can cite exactly one immutable protocol identity,
* a provisional protocol can never back a number labelled official,
* organizer answers create a new protocol instead of rewriting provisional history,
* a bundle carries the exact command, raw JSON, stderr, samples, runtime, and hashes,
* benchmark results cannot reach a training decision.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tinybench_lm.data_protocols import ProtocolError, frozen_benchmark_task_ids, protocol_digest
from tinybench_lm.evaluation_protocol import (
    BLOCKED,
    EVALUATION_PROTOCOL_DIR,
    FAIL,
    FROZEN_EVALUATION_PROTOCOL_SHA256,
    NOT_RUN,
    ORGANIZER_FINAL_PROTOCOL_ID,
    PASS,
    PROVISIONAL_PROTOCOL_ID,
    PROVISIONAL_PROTOCOL_PATH,
    REQUIRED_TASK_IDS,
    SECONDARY_TASK_IDS,
    EvaluationProtocolError,
    EvaluationProtocolNotReadyError,
    OrganizerAnswers,
    ProvisionalResultMisrepresentedError,
    SecondaryInfluenceError,
    UndeclaredTaskError,
    assert_no_training_influence,
    assert_provisional_is_labelled,
    assert_ready_for_official_results,
    assert_task_identities_match_decontamination,
    assert_tasks_declared,
    build_run_metadata,
    classify_tasks,
    compute_protocol_hash,
    declared_task_ids,
    format_report,
    harness_task_names,
    is_official,
    load_evaluation_protocol,
    num_fewshot_for,
    outstanding_organizer_questions,
    promote_to_organizer_final,
    protocol_identity,
    required_task_ids,
    resolved_num_fewshot,
    secondary_task_ids,
    sidecar_path,
    unpinned_identity_fields,
    verify_evaluation_protocol,
    verify_run_bundle,
    write_run_bundle,
)

REQUIRED_COMMONSENSE = ["hellaswag", "arc_easy", "piqa", "winogrande"]


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_evaluation_protocol()


def _answers(protocol: dict, **overrides) -> OrganizerAnswers:
    """A complete, obviously-synthetic organizer answer set for promotion fixtures."""
    payload = {
        "num_fewshot": {task_id: 0 for task_id in required_task_ids(protocol)},
        "metric_keys": {task_id: ["acc_norm"] for task_id in required_task_ids(protocol)},
        "wikitext_103": {
            "split": "test",
            "slice": "full_split",
            "normalization": "fixture_normalization",
            "bos_eos_handling": "fixture_bos_eos",
            "context_length": 1024,
            "stride": 512,
            "denominator": "fixture_denominator",
        },
        "judges_rerun_policy": "fixture: judges use submitted raw outputs",
        "own_weight_upload_policy": "fixture: uploading self-trained weights is permitted",
        "answered_on": "fixture-date",
        "source": "fixture answer set, not a real organizer reply",
    }
    payload.update(overrides)
    return OrganizerAnswers(**payload)


# --------------------------------------------------------------------------------------
# Freeze and identity
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_frozen_provisional_protocol_matches_its_pinned_digest() -> None:
    observed = protocol_digest(PROVISIONAL_PROTOCOL_PATH)
    assert observed == FROZEN_EVALUATION_PROTOCOL_SHA256[PROVISIONAL_PROTOCOL_PATH.name], observed


# **Validates: Requirements 1.2, 2.2, 2.4**
def test_editing_the_frozen_protocol_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / PROVISIONAL_PROTOCOL_PATH.name
    copied.write_bytes(PROVISIONAL_PROTOCOL_PATH.read_bytes() + b"\n# silent edit\n")
    # ProtocolMutatedError for a registry-pinned name, EvaluationProtocolError for a
    # sidecar-pinned promoted protocol. Both are ProtocolError: both fail closed.
    with pytest.raises(ProtocolError):
        load_evaluation_protocol(copied)


# **Validates: Requirements 1.1, 2.1, 2.5**
def test_protocol_identity_is_versioned_and_hashed(protocol: dict) -> None:
    identity = protocol_identity(protocol)
    assert identity.protocol_id == PROVISIONAL_PROTOCOL_ID
    assert identity.version == "v1"
    assert identity.official is False
    assert identity.label == "PROVISIONAL_NOT_OFFICIAL"
    assert identity.config_digest == FROZEN_EVALUATION_PROTOCOL_SHA256[PROVISIONAL_PROTOCOL_PATH.name]
    assert len(identity.protocol_hash) == 64
    assert identity.protocol_hash != identity.config_digest


# **Validates: Requirements 2.1, 2.4**
def test_protocol_hash_ignores_bookkeeping_but_tracks_semantics(protocol: dict) -> None:
    baseline = compute_protocol_hash(protocol)

    bookkeeping = copy.deepcopy(protocol)
    bookkeeping["notes"] = ["rewritten note"]
    bookkeeping["verification"]["required_checks"] = list(bookkeeping["verification"]["required_checks"])[:1]
    assert compute_protocol_hash(bookkeeping) == baseline

    for mutation in (
        lambda payload: payload["tasks"]["required"][0].__setitem__("num_fewshot", 5),
        lambda payload: payload["prompt_rendering"].__setitem__("bos_policy_id", "SOMETHING_ELSE"),
        lambda payload: payload["runtime"].__setitem__("seed", 4321),
        lambda payload: payload["runtime"]["batch_policy"].__setitem__("batch_size", 1),
        lambda payload: payload["wikitext_103"]["provisional_defaults"].__setitem__("stride", 256),
        lambda payload: payload["adapter_policies"].__setitem__("scoring", "OTHER_V1"),
    ):
        semantic = copy.deepcopy(protocol)
        mutation(semantic)
        assert compute_protocol_hash(semantic) != baseline


# --------------------------------------------------------------------------------------
# Task declarations and labels
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1**
def test_required_and_secondary_tables_match_the_plan(protocol: dict) -> None:
    assert required_task_ids(protocol) == REQUIRED_TASK_IDS
    assert secondary_task_ids(protocol) == SECONDARY_TASK_IDS
    assert harness_task_names(protocol, tier="required_official")[:4] == tuple(REQUIRED_COMMONSENSE)
    assert harness_task_names(protocol, tier="required_official")[4] == "wikitext"


# **Validates: Requirements 1.1, 2.1, 3.3**
def test_current_task_names_remain_provisional_defaults(protocol: dict) -> None:
    """Preservation: the names evaluate.py already used stay valid provisional defaults."""
    labels = classify_tasks(protocol, REQUIRED_COMMONSENSE)
    assert set(labels.values()) == {"REQUIRED_OFFICIAL_TASK_PROVISIONAL_SETTINGS"}
    assert resolved_num_fewshot(protocol, REQUIRED_COMMONSENSE) == 0
    for task_id in REQUIRED_COMMONSENSE:
        value, status = num_fewshot_for(protocol, task_id)
        assert (value, status) == (0, "PROVISIONAL")


# **Validates: Requirements 1.1, 2.1, 2.5**
def test_secondary_tasks_are_labelled_non_official(protocol: dict) -> None:
    labels = classify_tasks(protocol, list(SECONDARY_TASK_IDS))
    assert set(labels.values()) == {"SECONDARY_NON_OFFICIAL"}


# **Validates: Requirements 1.2, 2.2**
def test_undeclared_task_fails_closed_then_labels_on_opt_in(protocol: dict) -> None:
    with pytest.raises(UndeclaredTaskError):
        assert_tasks_declared(protocol, ["hellaswag", "lambada_openai"])
    assert assert_tasks_declared(protocol, ["lambada_openai"], allow_undeclared=True) == ("lambada_openai",)
    assert classify_tasks(protocol, ["lambada_openai"]) == {"lambada_openai": "UNDECLARED_NOT_OFFICIAL"}


# **Validates: Requirements 1.1, 2.1**
def test_task_identities_agree_with_decontamination(protocol: dict) -> None:
    assert_task_identities_match_decontamination(protocol)
    assert set(declared_task_ids(protocol)) == set(frozen_benchmark_task_ids())


# **Validates: Requirements 1.2, 2.2**
def test_task_identity_drift_from_decontamination_is_detected(protocol: dict) -> None:
    drifted = copy.deepcopy(protocol)
    drifted["tasks"]["secondary"] = list(drifted["tasks"]["secondary"])[:-1]
    with pytest.raises(EvaluationProtocolError):
        assert_task_identities_match_decontamination(drifted)


# --------------------------------------------------------------------------------------
# Provisional-versus-official refusals
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.3**
def test_official_results_are_blocked_while_provisional(protocol: dict) -> None:
    assert is_official(protocol) is False
    with pytest.raises(EvaluationProtocolNotReadyError) as error:
        assert_ready_for_official_results(protocol)
    message = str(error.value)
    assert "PROTOCOL_PROVISIONAL" in message
    for field in ("blocker=", "owner=", "next_action="):
        assert field in message
    assert len(outstanding_organizer_questions(protocol)) == 5


# **Validates: Requirements 1.1, 2.1, 2.3**
def test_unpinned_identity_fields_are_reported_not_invented(protocol: dict) -> None:
    unpinned = unpinned_identity_fields(protocol)
    assert "harness.commit" in unpinned
    assert "model_identity.tokenizer_revision" in unpinned
    assert "model_identity.model_revision" in unpinned
    for name in ("split", "slice", "normalization", "bos_eos_handling", "context_length", "stride", "denominator"):
        assert f"wikitext_103.{name}" in unpinned
    for task_id in REQUIRED_TASK_IDS:
        assert f"tasks.{task_id}.dataset_revision" in unpinned
        assert f"tasks.{task_id}.num_fewshot" in unpinned
    assert protocol["harness"]["commit"] == "PENDING_PIN"


# **Validates: Requirements 1.1, 2.1, 2.5**
def test_provisional_result_cannot_claim_official_status(protocol: dict) -> None:
    assert_provisional_is_labelled(protocol, claimed_official=False)
    with pytest.raises(ProvisionalResultMisrepresentedError):
        assert_provisional_is_labelled(protocol, claimed_official=True)


# **Validates: Requirements 1.1, 2.1**
def test_benchmark_results_cannot_influence_training(protocol: dict) -> None:
    assert_no_training_influence(protocol, REQUIRED_COMMONSENSE, purpose="reporting")
    for purpose in ("mixture_selection", "branch_selection", "checkpoint_selection"):
        with pytest.raises(SecondaryInfluenceError):
            assert_no_training_influence(protocol, list(SECONDARY_TASK_IDS), purpose=purpose)
        with pytest.raises(SecondaryInfluenceError):
            assert_no_training_influence(protocol, REQUIRED_COMMONSENSE, purpose=purpose)


# --------------------------------------------------------------------------------------
# Verification report
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_verification_passes_structure_and_blocks_organizer_fields(protocol: dict) -> None:
    report = verify_evaluation_protocol(protocol, path=PROVISIONAL_PROTOCOL_PATH)
    assert report.ok, [result.__dict__ for result in report.failures]
    assert report.check_ids == tuple(str(item) for item in protocol["verification"]["required_checks"])

    for check_id in (
        "protocol.frozen_digest",
        "protocol.provisional_label",
        "protocol.identity_hash_deterministic",
        "tasks.required_table_complete",
        "tasks.identities_match_decontamination",
        "tasks.secondary_labeled_non_official",
        "tasks.secondary_excluded_from_training",
        "prompt_rendering.bos_policy_shared",
        "adapter.policy_identity_matches",
        "runtime.seed_and_batch_policy_declared",
        "evidence.bundle_contract_declared",
    ):
        assert report.result(check_id).status == PASS, report.result(check_id).__dict__

    blocked = {result.check_id for result in report.blocked}
    assert blocked == {"harness.commit_pinned", "tasks.dataset_revisions_pinned", "wikitext.official_definition_pinned"}
    for result in report.blocked:
        for field in ("blocker=", "owner=", "next_action="):
            assert field in result.reason

    assert {result.check_id for result in report.not_run} == {"runtime.budget_measured"}
    assert "PROVISIONAL_NOT_OFFICIAL" in format_report(report)


# **Validates: Requirements 1.2, 2.2, 2.4**
def test_a_deferral_without_an_owner_becomes_a_failure(protocol: dict) -> None:
    """An unexplained gap must not masquerade as an explicit deferral."""
    stripped = copy.deepcopy(protocol)
    for field in ("blocker", "owner", "next_action"):
        stripped["harness"].pop(field, None)
    report = verify_evaluation_protocol(stripped, path=PROVISIONAL_PROTOCOL_PATH)
    assert report.result("harness.commit_pinned").status == FAIL
    assert not report.ok


# **Validates: Requirements 1.2, 2.2**
def test_marking_provisional_wikitext_reportable_fails(protocol: dict) -> None:
    dishonest = copy.deepcopy(protocol)
    dishonest["wikitext_103"]["may_be_reported_as_official"] = True
    report = verify_evaluation_protocol(dishonest, path=PROVISIONAL_PROTOCOL_PATH)
    assert report.result("wikitext.official_definition_pinned").status == FAIL


# **Validates: Requirements 1.2, 2.2**
def test_permitting_training_influence_fails_verification(protocol: dict) -> None:
    leaky = copy.deepcopy(protocol)
    leaky["training_influence"]["secondary_results_may_influence_training"] = True
    report = verify_evaluation_protocol(leaky, path=PROVISIONAL_PROTOCOL_PATH)
    assert report.result("tasks.secondary_excluded_from_training").status == FAIL


# --------------------------------------------------------------------------------------
# Organizer-answer promotion
# --------------------------------------------------------------------------------------


# **Validates: Requirements 2.1, 2.2, 2.4, 3.3**
def test_promotion_creates_a_new_protocol_and_preserves_provisional_history(
    protocol: dict, tmp_path: Path
) -> None:
    before = protocol_digest(PROVISIONAL_PROTOCOL_PATH)
    target = tmp_path / "evaluation_organizer_final_v1.yaml"
    result = promote_to_organizer_final(_answers(protocol), protocol=protocol, output_path=target)

    # The provisional protocol is untouched: history is retained, not rewritten.
    assert protocol_digest(PROVISIONAL_PROTOCOL_PATH) == before
    assert result.superseded_digest == before
    assert result.supersedes == PROVISIONAL_PROTOCOL_ID
    assert target.is_file()
    assert sidecar_path(target).is_file()

    promoted = load_evaluation_protocol(target)
    assert promoted["protocol_id"] == ORGANIZER_FINAL_PROTOCOL_ID
    assert is_official(promoted) is True
    assert promoted["status"]["supersedes"] == PROVISIONAL_PROTOCOL_ID
    assert promoted["status"]["superseded_config_digest"] == before
    assert outstanding_organizer_questions(promoted) == ()
    assert result.protocol_hash != compute_protocol_hash(protocol)

    fields = promoted["wikitext_103"]["organizer_specified_fields"]
    assert fields["split"] == "test"
    assert fields["stride"] == 512
    assert fields["denominator"] == "fixture_denominator"
    assert promoted["wikitext_103"]["official_scoring_blocked"] is False
    assert promoted["wikitext_103"]["provisional_defaults"]["superseded"] is True

    for entry in promoted["tasks"]["required"]:
        assert entry["num_fewshot_status"] == "OFFICIAL"
        assert entry["metric_keys_status"] == "OFFICIAL"
        assert entry["official_metric_keys"] == ["acc_norm"]

    # Answers cannot manufacture pins nobody supplied.
    assert promoted["harness"]["commit"] == "PENDING_PIN"
    assert "harness.commit" in result.still_blocked


# **Validates: Requirements 1.2, 2.2, 2.4**
def test_promoted_protocol_is_immutable(protocol: dict, tmp_path: Path) -> None:
    target = tmp_path / "evaluation_organizer_final_v1.yaml"
    promote_to_organizer_final(_answers(protocol), protocol=protocol, output_path=target)

    with pytest.raises(EvaluationProtocolError):
        promote_to_organizer_final(_answers(protocol), protocol=protocol, output_path=target)

    target.write_text(target.read_text(encoding="utf-8") + "\n# silent edit\n", encoding="utf-8")
    with pytest.raises(EvaluationProtocolError):
        load_evaluation_protocol(target)


# **Validates: Requirements 1.2, 2.2, 2.3**
@pytest.mark.parametrize(
    "override",
    [
        {"num_fewshot": {"hellaswag": 0}},
        {"metric_keys": {"hellaswag": ["acc"]}},
        {"wikitext_103": {"split": "test", "slice": "full_split"}},
        {"judges_rerun_policy": "   "},
        {"own_weight_upload_policy": ""},
    ],
)
def test_partial_organizer_answers_refuse_promotion(protocol: dict, tmp_path: Path, override: dict) -> None:
    target = tmp_path / "evaluation_organizer_final_v1.yaml"
    with pytest.raises(EvaluationProtocolNotReadyError) as error:
        promote_to_organizer_final(_answers(protocol, **override), protocol=protocol, output_path=target)
    assert "ORGANIZER_ANSWER_MISSING" in str(error.value)
    assert not target.exists()


# **Validates: Requirements 2.1, 2.4, 2.5**
def test_promoted_protocol_still_blocks_official_results_without_operator_pins(
    protocol: dict, tmp_path: Path
) -> None:
    target = tmp_path / "evaluation_organizer_final_v1.yaml"
    promote_to_organizer_final(_answers(protocol), protocol=protocol, output_path=target)
    promoted = load_evaluation_protocol(target)
    with pytest.raises(EvaluationProtocolNotReadyError):
        assert_ready_for_official_results(promoted)

    report = verify_evaluation_protocol(promoted, path=target)
    assert report.ok, [result.__dict__ for result in report.failures]
    assert report.result("protocol.frozen_digest").status == PASS
    assert report.result("wikitext.official_definition_pinned").status == PASS


# **Validates: Requirements 1.1, 2.1**
def test_no_promoted_protocol_is_committed_yet() -> None:
    """The repository must not ship an official protocol before organizer answers exist."""
    assert not (EVALUATION_PROTOCOL_DIR / "evaluation_organizer_final_v1.yaml").exists()


# --------------------------------------------------------------------------------------
# Run bundles
# --------------------------------------------------------------------------------------


def _stub_results(task_ids: list[str]) -> dict:
    """A stubbed harness result payload. Values are obviously synthetic placeholders."""
    return {
        "results": {task_id: {"acc,none": 0.25, "acc_stderr,none": 0.01} for task_id in task_ids},
        "n-samples": {task_id: {"original": 8, "effective": 8} for task_id in task_ids},
        "config": {"model": "TinyBenchHarnessLM", "batch_size": 2, "note": "fixture stub, not a measured result"},
    }


def _write_stub_bundle(protocol: dict, directory: Path, task_ids: list[str], **overrides):
    payload = {
        "command": ["python", "evaluate.py", "--tasks", ",".join(task_ids)],
        "raw_results": _stub_results(task_ids),
        "task_ids": task_ids,
        "sample_counts": _stub_results(task_ids)["n-samples"],
        "runtime_seconds": {"total": 1.5},
        "device": "cpu",
        "precision": "float32",
        "stderr_text": "fixture stderr line\n",
        "model_identity": {"checkpoint_sha256": "0" * 64, "tokenizer_sha256": "1" * 64},
        "harness_facts": {"installed_version": "fixture"},
    }
    payload.update(overrides)
    return write_run_bundle(directory, protocol=protocol, **payload)


# **Validates: Requirements 2.1, 2.4, 2.5**
def test_run_bundle_records_command_raw_json_stderr_samples_and_hashes(
    protocol: dict, tmp_path: Path
) -> None:
    bundle = _write_stub_bundle(protocol, tmp_path / "bundle", REQUIRED_COMMONSENSE)

    assert bundle.protocol_hash == compute_protocol_hash(protocol)
    assert bundle.official is False
    assert bundle.label == "PROVISIONAL_NOT_OFFICIAL"
    for name in ("command", "raw_results", "stderr_log", "run_metadata", "protocol_hash", "manifest"):
        assert bundle.artifacts[name].is_file()

    assert "--tasks" in bundle.artifacts["command"].read_text(encoding="utf-8")
    assert bundle.artifacts["stderr_log"].read_text(encoding="utf-8") == "fixture stderr line\n"
    assert bundle.protocol_hash in bundle.artifacts["protocol_hash"].read_text(encoding="utf-8")

    metadata = json.loads(bundle.artifacts["run_metadata"].read_text(encoding="utf-8"))
    for key in protocol["evidence_bundle"]["required_metadata_keys"]:
        assert str(key) in metadata
    assert metadata["sample_counts"]["piqa"]["effective"] == 8
    assert metadata["runtime_seconds"]["total"] == pytest.approx(1.5)
    assert metadata["seed"] == protocol["runtime"]["seed"]
    assert metadata["adapter_policies"] == dict(protocol["adapter_policies"])
    assert metadata["unpinned_identity_fields"]
    assert len(metadata["outstanding_organizer_questions"]) == 5
    assert all(entry["label"] == "REQUIRED_OFFICIAL_TASK_PROVISIONAL_SETTINGS" for entry in metadata["tasks"])

    report = verify_run_bundle(bundle.directory, protocol)
    assert report.ok, [result.__dict__ for result in report.failures]
    assert report.result("bundle.official_claim_honest").status == PASS


# **Validates: Requirements 1.2, 2.2, 2.4**
def test_bundle_verification_detects_mutation_and_deletion(protocol: dict, tmp_path: Path) -> None:
    bundle = _write_stub_bundle(protocol, tmp_path / "bundle", ["hellaswag"])

    bundle.artifacts["raw_results"].write_text('{"results": {"hellaswag": {"acc,none": 0.99}}}\n', encoding="utf-8")
    mutated = verify_run_bundle(bundle.directory, protocol)
    assert mutated.result("bundle.manifest_hashes_match").status == FAIL
    assert "BUNDLE_ARTIFACT_MUTATED" in mutated.result("bundle.manifest_hashes_match").reason

    bundle.artifacts["stderr_log"].unlink()
    missing = verify_run_bundle(bundle.directory, protocol)
    assert missing.result("bundle.artifact.stderr_log").status == FAIL
    assert not missing.ok


# **Validates: Requirements 1.1, 1.2, 2.2, 2.5**
def test_bundle_claiming_official_status_fails_closed(protocol: dict, tmp_path: Path) -> None:
    bundle = _write_stub_bundle(protocol, tmp_path / "bundle", ["hellaswag"])
    metadata_path = bundle.artifacts["run_metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["official"] = True
    metadata["label"] = "OFFICIAL_ORGANIZER_SPECIFIED"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_run_bundle(bundle.directory, protocol)
    assert report.result("bundle.official_claim_honest").status == FAIL
    assert "PROVISIONAL_PRESENTED_AS_OFFICIAL" in report.result("bundle.official_claim_honest").reason


# **Validates: Requirements 1.2, 2.2**
def test_bundle_cannot_cite_a_different_protocol(protocol: dict, tmp_path: Path) -> None:
    drifted = copy.deepcopy(protocol)
    drifted["runtime"]["seed"] = 999
    bundle = _write_stub_bundle(drifted, tmp_path / "bundle", ["hellaswag"])
    report = verify_run_bundle(bundle.directory, protocol)
    assert report.result("bundle.protocol_identity_matches").status == FAIL


# **Validates: Requirements 1.2, 2.2**
def test_bundle_refuses_undeclared_tasks_without_opt_in(protocol: dict, tmp_path: Path) -> None:
    with pytest.raises(UndeclaredTaskError):
        _write_stub_bundle(protocol, tmp_path / "bundle", ["lambada_openai"])
    bundle = _write_stub_bundle(
        protocol, tmp_path / "opted-in", ["lambada_openai"], allow_undeclared=True
    )
    metadata = json.loads(bundle.artifacts["run_metadata"].read_text(encoding="utf-8"))
    assert metadata["tasks"] == [
        {"task_id": "lambada_openai", "tier": "undeclared", "label": "UNDECLARED_NOT_OFFICIAL"}
    ]


# **Validates: Requirements 2.1, 2.5**
def test_secondary_bundle_stays_separable_from_the_required_table(protocol: dict, tmp_path: Path) -> None:
    bundle = _write_stub_bundle(protocol, tmp_path / "secondary", list(SECONDARY_TASK_IDS))
    metadata = json.loads(bundle.artifacts["run_metadata"].read_text(encoding="utf-8"))
    assert {entry["label"] for entry in metadata["tasks"]} == {"SECONDARY_NON_OFFICIAL"}
    assert metadata["training_influence"]["secondary_results_may_influence_training"] is False
    assert verify_run_bundle(bundle.directory, protocol).ok


# --------------------------------------------------------------------------------------
# Property tests over synthetic task selections
# --------------------------------------------------------------------------------------


_declared = list(REQUIRED_TASK_IDS) + list(SECONDARY_TASK_IDS)


# **Validates: Requirements 2.1, 2.4, 2.5**
@settings(max_examples=40, deadline=None)
@given(st.lists(st.sampled_from(_declared), min_size=1, max_size=6, unique=True))
def test_every_declared_task_selection_is_labelled_and_traceable(task_ids: list[str]) -> None:
    """Any selection of declared tasks yields one protocol hash and a label per task."""
    protocol = load_evaluation_protocol()
    metadata = build_run_metadata(
        protocol,
        command=["python", "evaluate.py"],
        task_ids=task_ids,
        sample_counts={task_id: 4 for task_id in task_ids},
        runtime_seconds=0.5,
        device="cpu",
        precision="float32",
    )
    assert metadata["protocol_hash"] == compute_protocol_hash(protocol)
    assert metadata["official"] is False
    assert metadata["label"] == "PROVISIONAL_NOT_OFFICIAL"
    assert len(metadata["tasks"]) == len(task_ids)
    for entry in metadata["tasks"]:
        expected = (
            "REQUIRED_OFFICIAL_TASK_PROVISIONAL_SETTINGS"
            if entry["task_id"] in REQUIRED_TASK_IDS
            else "SECONDARY_NON_OFFICIAL"
        )
        assert entry["label"] == expected
        assert entry["num_fewshot_status"] == "PROVISIONAL"
        assert entry["dataset_revision"] == "PENDING_PIN"


# **Validates: Requirements 1.2, 2.2**
@settings(max_examples=25, deadline=None)
@given(st.text(min_size=1, max_size=20).filter(lambda name: name.strip() not in set(_declared)))
def test_no_undeclared_task_name_is_silently_accepted(name: str) -> None:
    protocol = load_evaluation_protocol()
    with pytest.raises(UndeclaredTaskError):
        assert_tasks_declared(protocol, [name])
    assert classify_tasks(protocol, [name]) == {name: "UNDECLARED_NOT_OFFICIAL"}
