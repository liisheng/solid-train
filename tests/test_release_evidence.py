"""Release evidence matrix, templates, and stop-condition checks (Plan Sections 1-2, 11-17).

Nothing here publishes anything, uploads weights, creates Devpost content, or manufactures a
final asset. What the tests prove is that the submission surface cannot be quietly overstated:

- every Plan Section 16 checklist item and every G0-G6 gate has a row, so nothing goes missing,
- a ``PASS`` must name a verifier and a path that exist, and an external action -- teammate
  eligibility, public access, manual review -- can never be a ``PASS``,
- a ``BLOCKED`` row must name its owner and next action,
- personal eligibility and organizer answers are ``BLOCKED``; future artifacts are ``NOT_RUN``
  or ``TBD``, never ticked off,
- superseded documents keep their measured evidence but must carry a historical label,
- Plan Section 17's forbidden additions need measured need *and* schedule headroom, both,
- an undecayed peak-LR mainline checkpoint can never be released as the fallback,
- the README states the final 49,658,368-parameter architecture, not the superseded pilot one.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tinybench_lm.data_protocols import REPOSITORY_ROOT, ProtocolMutatedError, protocol_digest
from tinybench_lm.operations import GATE_IDS
from tinybench_lm.release import (
    BLOCKED,
    FROZEN_RELEASE_PROTOCOL_SHA256,
    RELEASE_BLOCKER_DETAIL_INCOMPLETE,
    RELEASE_FAIL_CLOSED_REASON_CODES,
    RELEASE_PATH_MISSING,
    RELEASE_PROTOCOL_PATH,
    RELEASE_SCOPE_CREEP,
    RELEASE_STATUSES,
    RELEASE_SUPERSEDED_UNLABELED,
    RELEASE_TEMPLATE_MISSING,
    RELEASE_UNDECAYED_FALLBACK,
    RELEASE_UNKNOWN_STATUS,
    RELEASE_UNSUPPORTED_PASS,
    RELEASE_VERIFIER_MISSING,
    TBD,
    ReleaseContractError,
    ReleaseNotReadyError,
    assert_ready_for_release,
    assert_release_matrix_valid,
    entry_index,
    evidence_entries,
    fallback_release_violations,
    format_release_matrix,
    load_release_protocol,
    matrix_violations,
    readiness_results,
    scope_creep_violations,
    superseded_document_violations,
    template_violations,
    verify_release_matrix,
)
from tinybench_lm.shards import FAIL, NOT_RUN, PASS

#: Plan Section 16 has eighteen checklist lines.
CONTRACT_ITEM_COUNT = 18
FINAL_PARAMETERS = "49,658,368"
SUPERSEDED_PARAMETERS = "49,295,872"


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_release_protocol()


# --------------------------------------------------------------------------------------
# The frozen contract
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_frozen_release_protocol_is_pinned_and_immutable(tmp_path: Path) -> None:
    assert (
        protocol_digest(RELEASE_PROTOCOL_PATH)
        == FROZEN_RELEASE_PROTOCOL_SHA256["evidence_matrix_v1.yaml"]
    )
    text = RELEASE_PROTOCOL_PATH.read_text(encoding="utf-8")
    mutated = tmp_path / "evidence_matrix_v1.yaml"
    mutated.write_text(text.replace("owner: operator", "owner: nobody", 1), encoding="utf-8")
    with pytest.raises(ProtocolMutatedError):
        load_release_protocol(mutated)


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
@pytest.mark.parametrize(
    "old,new",
    [
        ("  absence_of_evidence_is_never_pass: true", "  absence_of_evidence_is_never_pass: false"),
        ("  pass_requires_verifier_and_path: true", "  pass_requires_verifier_and_path: false"),
        ("  blocked_requires_owner_and_next_action: true", "  blocked_requires_owner_and_next_action: false"),
        ("  personal_eligibility_status: BLOCKED", "  personal_eligibility_status: PASS"),
        ("  organizer_answer_status: BLOCKED", "  organizer_answer_status: PASS"),
        ("  future_campaign_artifact_status: NOT_RUN", "  future_campaign_artifact_status: PASS"),
        ("  never_release_undecayed_peak_lr_mainline: true", "  never_release_undecayed_peak_lr_mainline: false"),
        ("  report_null_or_incomplete_honestly: true", "  report_null_or_incomplete_honestly: false"),
        # A gate quietly renamed is a gate quietly dropped.
        ("    - item_id: G5", "    - item_id: G9"),
    ],
)
def test_weakening_the_release_contract_fails_closed(tmp_path: Path, old: str, new: str) -> None:
    text = RELEASE_PROTOCOL_PATH.read_text(encoding="utf-8")
    assert old in text, f"replacement would be a no-op: {old!r}"
    weakened = tmp_path / "weakened_v1.yaml"
    weakened.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ReleaseContractError):
        load_release_protocol(weakened, verify=False)


# --------------------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5**
def test_the_matrix_covers_every_contract_item_and_every_gate(protocol: dict) -> None:
    entries = evidence_entries(protocol)
    contract = [entry for entry in entries if entry.group == "contract"]
    gates = [entry for entry in entries if entry.group == "gate"]

    assert len(contract) == CONTRACT_ITEM_COUNT
    assert tuple(entry.item_id for entry in gates) == GATE_IDS
    # No item is listed twice, and every one declares a failure policy.
    assert len({entry.item_id for entry in entries}) == len(entries)
    for entry in entries:
        assert entry.requirement.strip()
        assert entry.failure_policy.strip()
        assert entry.status in RELEASE_STATUSES


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5**
def test_the_matrix_validates_clean_against_the_working_tree(protocol: dict) -> None:
    assert matrix_violations(protocol) == ()
    assert template_violations(protocol) == ()
    assert superseded_document_violations(protocol) == ()
    report = verify_release_matrix(protocol)
    assert report.ok, [result.reason for result in report.failures]
    assert_release_matrix_valid(protocol)
    assert format_release_matrix(report.entries)


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_almost_nothing_is_passing_and_every_pass_is_earned(protocol: dict) -> None:
    """A matrix full of ticks before a run exists would be the whole failure mode."""
    entries = evidence_entries(protocol)
    passing = [entry for entry in entries if entry.status == PASS]

    # Only repository facts can pass today: the parameter cap and the eligibility scan.
    assert {entry.item_id for entry in passing} == {"parameter_cap", "no_pretrained_weights"}
    for entry in passing:
        assert not entry.is_external
        assert (REPOSITORY_ROOT / entry.path).exists()
        assert (REPOSITORY_ROOT / entry.verifier).exists()

    # No gate has passed, because no gate has been run.
    assert all(entry.status != PASS for entry in entries if entry.group == "gate")


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_personal_and_organizer_items_are_blocked_with_an_owner(protocol: dict) -> None:
    index = entry_index(protocol)
    for item_id in ("teammates_eligible_registered", "monitoring_window"):
        entry = index[item_id]
        assert entry.status == BLOCKED
        assert entry.owner.strip() and entry.next_action.strip()
        # A personal attestation is never something repository code can settle.
        assert entry.is_external

    for entry in evidence_entries(protocol):
        if entry.status == BLOCKED:
            assert entry.owner.strip(), entry.item_id
            assert entry.next_action.strip(), entry.item_id


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_future_artifacts_are_not_run_or_tbd(protocol: dict) -> None:
    index = entry_index(protocol)
    assert index["demo_video"].status == NOT_RUN
    assert index["screenshots"].status == NOT_RUN
    assert index["final_hashes_archived"].status == NOT_RUN
    assert index["step_zero_random_init_evidence"].status == NOT_RUN
    # A measurement that does not exist is TBD, not an estimate.
    assert index["readme_efficiency_figures"].status == TBD


# --------------------------------------------------------------------------------------
# The status rules are enforced, not merely declared
# --------------------------------------------------------------------------------------


def _with_entry(protocol: dict, item: dict) -> dict:
    matrix = protocol["release_evidence_matrix"]
    return {
        **protocol,
        "release_evidence_matrix": {
            **matrix,
            "contract_items": list(matrix["contract_items"]) + [item],
        },
    }


_BASE_ITEM = {
    "item_id": "injected",
    "requirement": "an injected row",
    "path": "README.md",
    "verifier": "scripts/count_params.py",
    "status": NOT_RUN,
    "failure_policy": "reject",
}


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5**
def test_a_pass_naming_a_missing_path_is_rejected(protocol: dict) -> None:
    injected = _with_entry(protocol, {**_BASE_ITEM, "status": PASS, "path": "docs/does_not_exist.md"})
    problems = matrix_violations(injected)
    assert any(RELEASE_PATH_MISSING in problem for problem in problems)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5**
def test_a_pass_naming_a_missing_verifier_is_rejected(protocol: dict) -> None:
    injected = _with_entry(
        protocol, {**_BASE_ITEM, "status": PASS, "verifier": "scripts/no_such_verifier.py"}
    )
    problems = matrix_violations(injected)
    assert any(RELEASE_VERIFIER_MISSING in problem for problem in problems)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5**
@pytest.mark.parametrize(
    "verifier", ["none-personal-attestation", "none-external-access-check", "none-manual-review"]
)
def test_an_external_action_can_never_be_a_pass(protocol: dict, verifier: str) -> None:
    """Publishing, approving, and eyeballing a video are not things a test can assert."""
    injected = _with_entry(protocol, {**_BASE_ITEM, "status": PASS, "verifier": verifier})
    problems = matrix_violations(injected)
    assert any(RELEASE_UNSUPPORTED_PASS in problem for problem in problems)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5**
def test_a_blocked_row_without_an_owner_is_rejected(protocol: dict) -> None:
    injected = _with_entry(protocol, {**_BASE_ITEM, "status": BLOCKED})
    problems = matrix_violations(injected)
    assert any(RELEASE_BLOCKER_DETAIL_INCOMPLETE in problem for problem in problems)

    # An owner without a next action is still an excuse.
    half = _with_entry(protocol, {**_BASE_ITEM, "status": BLOCKED, "owner": "operator"})
    assert any(RELEASE_BLOCKER_DETAIL_INCOMPLETE in problem for problem in matrix_violations(half))


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_an_unknown_status_is_rejected(protocol: dict) -> None:
    injected = _with_entry(protocol, {**_BASE_ITEM, "status": "PROBABLY_FINE"})
    assert any(RELEASE_UNKNOWN_STATUS in problem for problem in matrix_violations(injected))


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_a_row_pointing_at_nothing_is_rejected(protocol: dict) -> None:
    injected = _with_entry(protocol, {**_BASE_ITEM, "path": "docs/absent.md"})
    assert any(RELEASE_PATH_MISSING in problem for problem in matrix_violations(injected))


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_a_missing_template_is_reported(protocol: dict, tmp_path: Path) -> None:
    """Point the checker at an empty tree so a present template cannot mask the check."""
    problems = template_violations(protocol, root=tmp_path)
    assert any(RELEASE_TEMPLATE_MISSING in problem for problem in problems)


# --------------------------------------------------------------------------------------
# Superseded documents keep their evidence and gain a label
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1, 3.2, 3.3**
def test_superseded_documents_are_labeled_but_still_present(protocol: dict) -> None:
    assert superseded_document_violations(protocol) == ()

    pilot = (REPOSITORY_ROOT / "docs/PILOT_REPORT.md").read_text(encoding="utf-8")
    assert "superseded" in pilot[:1200].lower()
    # The measured pilot evidence survives; only the recommendation is superseded.
    assert "66,592 tok/s" in pilot
    assert "76,457 tok/s" in pilot
    assert "5.21 GiB" in pilot
    assert SUPERSEDED_PARAMETERS in pilot
    assert FINAL_PARAMETERS in pilot, "the banner must name the architecture that replaced it"

    research = (REPOSITORY_ROOT / "docs/RESEARCH_PLAN.md").read_text(encoding="utf-8")
    assert "superseded" in research[:1200].lower()
    assert "arxiv.org" in research, "the literature it collected stays useful"


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_an_unlabeled_superseded_document_is_reported(protocol: dict, tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/PILOT_REPORT.md").write_text("# Pilot report\n\nNumbers.\n", encoding="utf-8")
    (tmp_path / "docs/RESEARCH_PLAN.md").write_text("# Research plan\n\nAdvice.\n", encoding="utf-8")
    problems = superseded_document_violations(protocol, root=tmp_path)
    assert len(problems) == 2
    assert all(RELEASE_SUPERSEDED_UNLABELED in problem for problem in problems)


# --------------------------------------------------------------------------------------
# README reconciliation
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 3.1, 3.2, 3.3**
def test_readme_states_the_final_architecture() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    assert FINAL_PARAMETERS in readme
    assert "341,632" in readme
    for value in ("14 layers", "12,288", "1,504", "4 key/value heads"):
        assert value in readme, value

    # The superseded count may appear only where it is explicitly labeled superseded.
    for line in readme.splitlines():
        if SUPERSEDED_PARAMETERS in line:
            assert "superseded" in line.lower(), line

    # Random initialization and the evidence matrix are both stated on the first screen.
    assert "random initialization" in readme.lower()
    assert "configs/release/evidence_matrix_v1.yaml" in readme
    # No fabricated result table.
    assert "NOT_RUN" in readme or "not reported here" in readme


# --------------------------------------------------------------------------------------
# Stop conditions and fallback (Plan Sections 15 and 17)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5, 3.3**
def test_forbidden_additions_need_measured_need_and_schedule_headroom(protocol: dict) -> None:
    assert scope_creep_violations([], protocol=protocol) == ()
    assert scope_creep_violations(["a_bounded_test"], protocol=protocol) == ()

    for addition in protocol["stop_conditions"]["forbidden_additions"]:
        problems = scope_creep_violations([addition], protocol=protocol)
        assert any(RELEASE_SCOPE_CREEP in problem for problem in problems), addition

    # The exception needs BOTH conditions, not either.
    assert scope_creep_violations(["dashboard"], measured_need=True, protocol=protocol) != ()
    assert scope_creep_violations(["dashboard"], schedule_ahead=True, protocol=protocol) != ()
    assert (
        scope_creep_violations(
            ["dashboard"], measured_need=True, schedule_ahead=True, protocol=protocol
        )
        == ()
    )


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5, 3.3**
def test_an_undecayed_peak_lr_mainline_is_never_released(protocol: dict) -> None:
    assert fallback_release_violations(is_undecayed_peak_lr_mainline=False, protocol=protocol) == ()
    problems = fallback_release_violations(is_undecayed_peak_lr_mainline=True, protocol=protocol)
    assert any(RELEASE_UNDECAYED_FALLBACK in problem for problem in problems)
    policy = protocol["fallback_policy"]
    assert str(policy["prefer_ordinary_decay_arm_when_no_effect_supported"]) == "A"
    assert bool(policy["report_null_or_incomplete_honestly"])


# --------------------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_release_readiness_is_not_run_or_blocked_with_an_owner(protocol: dict) -> None:
    results = readiness_results(protocol)
    assert results
    statuses = {result.status for result in results}
    assert PASS not in statuses
    assert statuses <= {NOT_RUN, BLOCKED}
    for result in results:
        assert "blocker=" in result.reason
        assert "owner=" in result.reason
        assert "next_action=" in result.reason


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_release_refuses_to_proceed_without_artifacts_and_approval(protocol: dict) -> None:
    with pytest.raises(ReleaseNotReadyError) as excinfo:
        assert_ready_for_release(protocol)
    message = str(excinfo.value)
    assert "teammate_approval_recorded" in message
    assert "next_action=" in message


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_reason_code_vocabulary_matches_the_frozen_contract(protocol: dict) -> None:
    assert set(protocol["reason_codes"]["fail_closed"]) == set(RELEASE_FAIL_CLOSED_REASON_CODES)
