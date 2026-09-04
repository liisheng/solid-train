"""The authoritative alignment auditor and its no-op regression check (Plan Sections 2.1, 13, 16-17).

This complements ``tests/test_alignment_audit.py`` rather than replacing it: that file holds the
task-1 bug-condition property and its deterministic counterexamples, which task 3.21 re-runs
unchanged. What is tested here is the broader checklist auditor:

- it covers required paths, frozen architecture values, the independent parameter
  reconciliation, every pinned protocol digest, named verifier outcomes, integrity scans, and
  the release matrix's evidence links,
- an unexplained difference always fails: a missing path, a changed value, a drifted digest, a
  failing verifier, and a prohibited pattern each produce a FAIL,
- a DEFERRED entry is permitted only for the five categories a repository cannot settle itself,
  and only when it names an owner and a next action,
- the audit is read-only, so auditing an already conforming repository twice returns the
  identical report and does not rewrite a single byte.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tinybench_lm.alignment import (
    ALIGNMENT_PROTOCOL_PATH,
    AUDIT_DEFERRAL_CATEGORY_INVALID,
    AUDIT_DEFERRAL_DETAIL_INCOMPLETE,
    AUDIT_DIGEST_DRIFT,
    AUDIT_FAIL_CLOSED_REASON_CODES,
    AUDIT_INTEGRITY_VIOLATION,
    AUDIT_STATUSES,
    AUDIT_UNEXPLAINED_DIFFERENCE,
    AUDIT_VERIFIER_FAILED,
    DEFERRED,
    FROZEN_ALIGNMENT_PROTOCOL_SHA256,
    AlignmentContractError,
    assert_no_unexplained_differences,
    audit_is_idempotent,
    audit_repository,
    audit_results,
    format_audit_report,
    load_alignment_checklist,
    reconcile_parameter_count,
    tree_fingerprint,
)
from tinybench_lm.data_protocols import REPOSITORY_ROOT, ProtocolMutatedError, protocol_digest
from tinybench_lm.shards import FAIL, PASS

FINAL_PARAMETERS = 49_658_368
CAP = 50_000_000


@pytest.fixture(scope="module")
def checklist() -> dict:
    return load_alignment_checklist()


# --------------------------------------------------------------------------------------
# The frozen checklist
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_frozen_checklist_is_pinned_and_immutable(tmp_path: Path) -> None:
    assert (
        protocol_digest(ALIGNMENT_PROTOCOL_PATH)
        == FROZEN_ALIGNMENT_PROTOCOL_SHA256["alignment_v1.yaml"]
    )
    text = ALIGNMENT_PROTOCOL_PATH.read_text(encoding="utf-8")
    mutated = tmp_path / "alignment_v1.yaml"
    mutated.write_text(text.replace("owner: operator", "owner: nobody", 1), encoding="utf-8")
    with pytest.raises(ProtocolMutatedError):
        load_alignment_checklist(mutated)


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
@pytest.mark.parametrize(
    "old,new",
    [
        ("  read_only: true", "  read_only: false"),
        ("  idempotent: true", "  idempotent: false"),
        ("  unexplained_difference_fails: true", "  unexplained_difference_fails: false"),
        ("  requires_owner: true", "  requires_owner: false"),
        ("  requires_next_action: true", "  requires_next_action: false"),
        ("    - measurement", "    - whenever_we_feel_like_it"),
        ("  headroom: 341632", "  headroom: 999999"),
    ],
)
def test_weakening_the_checklist_fails_closed(tmp_path: Path, old: str, new: str) -> None:
    text = ALIGNMENT_PROTOCOL_PATH.read_text(encoding="utf-8")
    assert old in text, f"replacement would be a no-op: {old!r}"
    weakened = tmp_path / "weakened_v1.yaml"
    weakened.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(AlignmentContractError):
        load_alignment_checklist(weakened, verify=False)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_the_parameter_contract_reconciles(checklist: dict) -> None:
    contract = checklist["parameter_contract"]
    assert int(contract["unique_trainable_parameters"]) == FINAL_PARAMETERS
    assert int(contract["cap"]) == CAP
    assert int(contract["unique_trainable_parameters"]) + int(contract["headroom"]) == CAP

    payload = json.loads((REPOSITORY_ROOT / "configs/final_49m.json").read_text(encoding="utf-8"))
    # The audit recomputes the count from the config's own fields, independently of the model.
    assert reconcile_parameter_count(payload) == FINAL_PARAMETERS
    assert set(checklist["reason_codes"]["fail_closed"]) == set(AUDIT_FAIL_CLOSED_REASON_CODES)


# --------------------------------------------------------------------------------------
# The repository audits clean
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4**
def test_the_repository_has_no_unexplained_differences(checklist: dict) -> None:
    report = audit_repository(checklist)
    assert report.ok, [f"{entry.entry_id}: {entry.detail}" for entry in report.failures]
    assert report.failures == ()
    assert_no_unexplained_differences(checklist)

    # Deferrals are explained, not failures — and every one names its own next step.
    assert report.deferrals
    for entry in report.deferrals:
        assert entry.status == DEFERRED
        assert "owner=" in entry.detail
        assert "next_action=" in entry.detail
        assert "category=" in entry.detail

    assert {entry.status for entry in report.entries} <= set(AUDIT_STATUSES)
    assert format_audit_report(report)
    assert audit_results(checklist)


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_every_deferral_names_a_permitted_category(checklist: dict) -> None:
    allowed = set(checklist["deferral_policy"]["allowed_categories"])
    assert allowed == {
        "measurement",
        "organizer_answer",
        "personal_attestation",
        "public_release",
        "long_running_campaign",
    }
    for item in checklist["deferrals"]:
        assert str(item["category"]) in allowed, item["entry_id"]
        assert str(item["owner"]).strip(), item["entry_id"]
        assert str(item["next_action"]).strip(), item["entry_id"]


# --------------------------------------------------------------------------------------
# Unexplained differences fail
# --------------------------------------------------------------------------------------


def _entry(report, entry_id: str):
    return {entry.entry_id: entry for entry in report.entries}[entry_id]


#: Skipped everywhere: large or generated, and never part of the audited surface.
_SKIP_ANYWHERE = frozenset(
    {
        ".cache",
        ".git",
        ".hypothesis",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
    }
)

#: Skipped only at the repository root. `data` must NOT be matched by name, or the copy
#: would silently lose `configs/data/` and the mirror would fail the audit for the wrong reason.
_SKIP_AT_ROOT = frozenset({"data", "runs"})


@pytest.fixture
def mirror(tmp_path: Path) -> Path:
    """A working copy of the repository the tests may damage."""
    destination = tmp_path / "repo"

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = {name for name in names if name in _SKIP_ANYWHERE}
        if Path(directory).resolve() == REPOSITORY_ROOT.resolve():
            skipped |= {name for name in names if name in _SKIP_AT_ROOT}
        return skipped

    shutil.copytree(REPOSITORY_ROOT, destination, ignore=ignore)
    return destination


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_a_missing_required_path_fails(checklist: dict, mirror: Path) -> None:
    assert audit_repository(checklist, root=mirror).ok
    (mirror / "docs/templates/MODEL_CARD.md").unlink()
    report = audit_repository(checklist, root=mirror)
    assert not report.ok
    entry = _entry(report, "paths.model_card_template")
    assert entry.status == FAIL
    assert AUDIT_UNEXPLAINED_DIFFERENCE in entry.detail


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_a_changed_architecture_value_fails(checklist: dict, mirror: Path) -> None:
    config_path = mirror / "configs/final_49m.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["n_layers"] = 12
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = audit_repository(checklist, root=mirror)
    assert not report.ok
    assert _entry(report, "architecture.n_layers").status == FAIL
    # The independent reconciliation catches it too, so one edit trips two entries.
    assert _entry(report, "architecture.unique_parameter_count").status == FAIL


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_an_over_cap_model_fails(checklist: dict, mirror: Path) -> None:
    config_path = mirror / "configs/final_49m.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["n_layers"] = 40  # far past the cap
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = audit_repository(checklist, root=mirror)
    assert not report.ok
    assert reconcile_parameter_count(payload) > CAP
    # The cap breach is reported in its own right, not merely as a mismatch with the
    # contract's expected count. The cap decides eligibility, so it stands alone.
    entry = _entry(report, "architecture.unique_parameter_count")
    assert entry.status == FAIL
    assert AUDIT_INTEGRITY_VIOLATION in entry.detail
    assert "exceeds" in entry.detail


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_a_drifted_frozen_digest_fails(checklist: dict, mirror: Path) -> None:
    target = mirror / "configs/branches/exposure_v1.yaml"
    target.write_text(
        target.read_text(encoding="utf-8").replace("owner: operator", "owner: nobody", 1),
        encoding="utf-8",
    )
    report = audit_repository(checklist, root=mirror)
    assert not report.ok
    entry = _entry(report, "digest.branch_protocol")
    assert entry.status == FAIL
    assert AUDIT_DIGEST_DRIFT in entry.detail


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_a_prohibited_pretrained_call_fails(checklist: dict, mirror: Path) -> None:
    """The single most important integrity scan: no pretrained initialization."""
    smuggled = mirror / "src/tinybench_lm/loader.py"
    smuggled.write_text(
        "from transformers import AutoModel\n\n\ndef build():\n    return AutoModel.from_pretrained('gpt2')\n",
        encoding="utf-8",
    )
    report = audit_repository(checklist, root=mirror)
    assert not report.ok
    entry = _entry(report, "integrity.no_pretrained_initialization")
    assert entry.status == FAIL
    assert AUDIT_INTEGRITY_VIOLATION in entry.detail
    assert "loader.py" in entry.detail
    # The eligibility verifier reports it independently of the scan.
    assert _entry(report, "verifier.eligibility_scan").status == FAIL


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_a_cosine_final_decay_fails(checklist: dict, mirror: Path) -> None:
    train = mirror / "train.py"
    train.write_text(train.read_text(encoding="utf-8") + "\n_ = math.cos(0.0)\n", encoding="utf-8")
    report = audit_repository(checklist, root=mirror)
    assert not report.ok
    assert _entry(report, "integrity.no_cosine_final_decay").status == FAIL


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_a_dangling_evidence_path_fails(checklist: dict, mirror: Path) -> None:
    (mirror / "docs/templates/BUILT_WITH.md").unlink()
    report = audit_repository(checklist, root=mirror)
    assert not report.ok
    entry = _entry(report, "evidence.release_matrix_paths")
    assert entry.status == FAIL
    assert "built_with" in entry.detail


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_assert_no_unexplained_differences_raises(checklist: dict, mirror: Path) -> None:
    (mirror / "configs/final_49m.json").unlink()
    with pytest.raises(AlignmentContractError):
        assert_no_unexplained_differences(checklist, root=mirror)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_a_damaged_repository_is_reported_not_crashed(checklist: dict, mirror: Path) -> None:
    """An auditor that raises on damage tells you nothing about the damage."""
    (mirror / "configs/final_49m.json").unlink()

    report = audit_repository(checklist, root=mirror)  # must not raise
    assert not report.ok
    # The value checks, the reconciliation, and the verifier all report rather than explode.
    assert _entry(report, "architecture.n_layers").status == FAIL
    assert _entry(report, "architecture.unique_parameter_count").status == FAIL
    assert _entry(report, "verifier.parameter_count").status == FAIL
    for entry in report.failures:
        assert entry.detail.strip()


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_an_unparseable_config_is_reported_not_crashed(checklist: dict, mirror: Path) -> None:
    (mirror / "configs/final_49m.json").write_text("{ not json", encoding="utf-8")

    report = audit_repository(checklist, root=mirror)  # must not raise
    assert not report.ok
    entry = _entry(report, "architecture.unique_parameter_count")
    assert entry.status == FAIL
    assert "JSONDecodeError" in entry.detail or AUDIT_UNEXPLAINED_DIFFERENCE in entry.detail


# --------------------------------------------------------------------------------------
# Deferral rules are enforced, not merely declared
# --------------------------------------------------------------------------------------


def _with_deferral(checklist: dict, item: dict) -> dict:
    return {**checklist, "deferrals": list(checklist["deferrals"]) + [item]}


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5**
def test_a_deferral_outside_the_permitted_categories_fails(checklist: dict) -> None:
    """"We'll get to it" is not a category."""
    injected = _with_deferral(
        checklist,
        {
            "entry_id": "deferred.injected",
            "category": "not_got_round_to_it",
            "requirement": "something",
            "owner": "operator",
            "next_action": "do it",
        },
    )
    report = audit_repository(injected)
    assert not report.ok
    entry = _entry(report, "deferred.injected")
    assert entry.status == FAIL
    assert AUDIT_DEFERRAL_CATEGORY_INVALID in entry.detail


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5**
@pytest.mark.parametrize("field", ["owner", "next_action"])
def test_a_deferral_without_an_owner_or_next_action_fails(checklist: dict, field: str) -> None:
    item = {
        "entry_id": "deferred.injected",
        "category": "measurement",
        "requirement": "something",
        "owner": "operator",
        "next_action": "measure it",
    }
    item[field] = "   "
    report = audit_repository(_with_deferral(checklist, item))
    assert not report.ok
    entry = _entry(report, "deferred.injected")
    assert entry.status == FAIL
    assert AUDIT_DEFERRAL_DETAIL_INCOMPLETE in entry.detail


# --------------------------------------------------------------------------------------
# Idempotence: the no-op regression check
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4**
def test_auditing_a_conforming_repository_twice_changes_nothing(checklist: dict, mirror: Path) -> None:
    """An auditor that rewrites files turns a verification step into a change."""
    before = tree_fingerprint(mirror)
    first = audit_repository(checklist, root=mirror)
    after_first = tree_fingerprint(mirror)
    second = audit_repository(checklist, root=mirror)
    after_second = tree_fingerprint(mirror)

    assert first.to_dict() == second.to_dict()
    assert first.ok and second.ok
    # Byte-for-byte: content is hashed, so a rewrite preserving mtime is still caught.
    assert before == after_first == after_second

    idempotent, problems = audit_is_idempotent(checklist, root=mirror)
    assert idempotent, problems
    assert problems == ()


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_idempotence_check_detects_a_repository_mutation(
    checklist: dict, mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audit really is idempotent, so break it deliberately to prove the check works."""
    from tinybench_lm import alignment

    real = alignment.audit_repository
    counter = {"runs": 0}

    def writes_a_file(protocol=None, *, root=mirror):
        counter["runs"] += 1
        (root / f"side_effect_{counter['runs']}.txt").write_text("oops", encoding="utf-8")
        return real(protocol, root=root)

    monkeypatch.setattr(alignment, "audit_repository", writes_a_file)
    idempotent, problems = alignment.audit_is_idempotent(checklist, root=mirror)
    assert not idempotent
    assert any(alignment.AUDIT_MUTATED_REPOSITORY in problem for problem in problems)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_idempotence_check_detects_an_unstable_report(
    checklist: dict, mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tinybench_lm import alignment

    real = alignment.audit_repository
    counter = {"runs": 0}

    def drifts(protocol=None, *, root=mirror):
        counter["runs"] += 1
        report = real(protocol, root=root)
        if counter["runs"] > 1:
            # A second run that disagrees with the first is not a verification step.
            return alignment.AuditReport(report.entries[:-1])
        return report

    monkeypatch.setattr(alignment, "audit_repository", drifts)
    idempotent, problems = alignment.audit_is_idempotent(checklist, root=mirror)
    assert not idempotent
    assert any(alignment.AUDIT_UNEXPLAINED_DIFFERENCE in problem for problem in problems)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_the_fingerprint_notices_any_edit(mirror: Path) -> None:
    """The idempotence proof is only as good as the fingerprint behind it."""
    before = tree_fingerprint(mirror)
    target = mirror / "README.md"
    original = target.read_bytes()

    target.write_bytes(original + b"\n")
    assert tree_fingerprint(mirror) != before

    target.write_bytes(original)
    assert tree_fingerprint(mirror) == before

    # A new file changes it too, so a stray artifact cannot slip past.
    (mirror / "stray.txt").write_text("x", encoding="utf-8")
    assert tree_fingerprint(mirror) != before


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_the_audit_report_is_deterministic(checklist: dict) -> None:
    first = audit_repository(checklist)
    second = audit_repository(checklist)
    assert first.to_dict() == second.to_dict()
    assert [entry.entry_id for entry in first.entries] == [entry.entry_id for entry in second.entries]
    assert format_audit_report(first) == format_audit_report(second)
