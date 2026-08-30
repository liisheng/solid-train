"""Release evidence matrix, templates, and stop-condition checks (Plan Sections 1-2, 11-17).

Every other frozen protocol in this package answers "is this step correct?". This one answers
a different question: *where is the evidence for each promise the submission makes, and what is
its status right now?*

The rule that makes the matrix worth having is that a status can never be optimistic. A `PASS`
must name a verifier and a path that actually exist on disk; a `BLOCKED` must name its owner
and next action; anything unmeasured is `NOT_RUN` or `TBD`. :func:`matrix_violations` enforces
all of that, so the easiest way to ship an unsupported claim -- ticking a box early -- fails a
test instead.

The matrix also carries the two things a deadline erodes first:

*Superseded documents.* ``docs/PILOT_REPORT.md`` measured a real 49,295,872-parameter
architecture that the final plan replaced. Those measurements stay factual and stay in the
repository; what changes is the label. :func:`superseded_document_violations` fails if a
superseded document is not marked as historical, because a stale recommendation read as current
is worse than a deleted one.

*Stop conditions.* Plan Section 17 forbids adding another architecture, data family,
optimization method, dashboard, frontend, or hosted-model workflow without measured need.
:func:`scope_creep_violations` makes that checkable rather than aspirational.

The contract is backed by one frozen config::

    configs/release/evidence_matrix_v1.yaml

Nothing here publishes anything, uploads weights, creates Devpost content, or manufactures a
final asset. It validates paths, statuses, and labels in the working tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data_protocols import (
    REPOSITORY_ROOT,
    ProtocolError,
    ProtocolNotReadyError,
    load_protocol,
)
from .environment import CheckResult
from .operations import GATE_IDS
from .shards import FAIL, NOT_RUN, PASS

RELEASE_PROTOCOL_DIR = REPOSITORY_ROOT / "configs" / "release"
RELEASE_PROTOCOL_PATH = RELEASE_PROTOCOL_DIR / "evidence_matrix_v1.yaml"

#: SHA-256 of the frozen release contract, over file bytes with CRLF normalized to LF.
FROZEN_RELEASE_PROTOCOL_SHA256: Mapping[str, str] = {
    "evidence_matrix_v1.yaml": "60ee19b60927dc2731aa21ff1253379c94fedbb18740f3349eff28e24ca07b9d",
}

BLOCKED = "BLOCKED"
TBD = "TBD"

#: Every status the matrix may carry. Only FAIL is a failure; the rest are honest absences.
RELEASE_STATUSES: tuple[str, ...] = (PASS, FAIL, BLOCKED, NOT_RUN, TBD)

#: Statuses that assert something has been verified.
ASSERTIVE_STATUSES: frozenset[str] = frozenset({PASS})

#: A verifier value naming a check no repository code can perform.
_EXTERNAL_VERIFIERS: frozenset[str] = frozenset(
    {"none-personal-attestation", "none-external-access-check", "none-manual-review"}
)

# --------------------------------------------------------------------------------------
# Reason-code vocabulary
# --------------------------------------------------------------------------------------

RELEASE_OK = "RELEASE_OK"
RELEASE_BLOCKER_DETAIL_INCOMPLETE = "RELEASE_BLOCKER_DETAIL_INCOMPLETE"
RELEASE_PATH_MISSING = "RELEASE_PATH_MISSING"
RELEASE_SCOPE_CREEP = "RELEASE_SCOPE_CREEP"
RELEASE_SUPERSEDED_UNLABELED = "RELEASE_SUPERSEDED_UNLABELED"
RELEASE_TEMPLATE_MISSING = "RELEASE_TEMPLATE_MISSING"
RELEASE_UNDECAYED_FALLBACK = "RELEASE_UNDECAYED_FALLBACK"
RELEASE_UNKNOWN_STATUS = "RELEASE_UNKNOWN_STATUS"
RELEASE_UNSUPPORTED_PASS = "RELEASE_UNSUPPORTED_PASS"
RELEASE_VERIFIER_MISSING = "RELEASE_VERIFIER_MISSING"

RELEASE_FAIL_CLOSED_REASON_CODES: frozenset[str] = frozenset(
    {
        RELEASE_BLOCKER_DETAIL_INCOMPLETE,
        RELEASE_PATH_MISSING,
        RELEASE_SCOPE_CREEP,
        RELEASE_SUPERSEDED_UNLABELED,
        RELEASE_TEMPLATE_MISSING,
        RELEASE_UNDECAYED_FALLBACK,
        RELEASE_UNKNOWN_STATUS,
        RELEASE_UNSUPPORTED_PASS,
        RELEASE_VERIFIER_MISSING,
    }
)

#: The label a superseded document must carry so a reader cannot mistake it for current.
SUPERSEDED_LABEL = "historical"


class ReleaseContractError(ProtocolError):
    """The frozen release contract is malformed, or an evidence entry violates it."""


class ReleaseNotReadyError(ProtocolNotReadyError):
    """A release action needs artifacts and approvals that do not exist yet."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def load_release_protocol(
    path: Path = RELEASE_PROTOCOL_PATH, *, verify: bool = True
) -> dict[str, Any]:
    """Load the frozen release contract, verifying its pinned digest by default."""
    protocol = load_protocol(path, verify=verify, registry=FROZEN_RELEASE_PROTOCOL_SHA256)
    required = (
        "status_policy",
        "release_evidence_matrix",
        "stop_conditions",
        "fallback_policy",
        "superseded_documents",
        "required_templates",
        "readiness",
    )
    for section in required:
        if section not in protocol:
            raise ReleaseContractError(f"release protocol is missing required section {section!r}")

    policy = protocol["status_policy"]
    if tuple(str(status) for status in policy["statuses"]) != RELEASE_STATUSES:
        raise ReleaseContractError(f"release statuses must be {RELEASE_STATUSES}")
    if not bool(policy["absence_of_evidence_is_never_pass"]):
        raise ReleaseContractError(
            f"{RELEASE_UNSUPPORTED_PASS}: absence of evidence may never become PASS"
        )
    for flag in ("pass_requires_verifier_and_path", "blocked_requires_owner_and_next_action"):
        if not bool(policy[flag]):
            raise ReleaseContractError(f"the release contract must assert {flag}")
    if str(policy["personal_eligibility_status"]) != BLOCKED:
        raise ReleaseContractError(
            "personal eligibility is an attestation no repository check can supply"
        )
    if str(policy["organizer_answer_status"]) != BLOCKED:
        raise ReleaseContractError("an organizer answer is BLOCKED until it arrives")
    if str(policy["future_campaign_artifact_status"]) != NOT_RUN:
        raise ReleaseContractError("a future campaign artifact is NOT_RUN, never checked off")

    matrix = protocol["release_evidence_matrix"]
    for group in ("contract_items", "gate_items"):
        if not matrix.get(group):
            raise ReleaseContractError(f"release evidence matrix declares no {group}")

    declared_gates = tuple(str(item["item_id"]) for item in matrix["gate_items"])
    if declared_gates != GATE_IDS:
        raise ReleaseContractError(
            f"the matrix must cover gates {GATE_IDS}, found {declared_gates}"
        )

    if not bool(protocol["fallback_policy"]["never_release_undecayed_peak_lr_mainline"]):
        raise ReleaseContractError(
            f"{RELEASE_UNDECAYED_FALLBACK}: Plan Section 15 forbids an undecayed peak-LR release"
        )
    if not bool(protocol["fallback_policy"]["report_null_or_incomplete_honestly"]):
        raise ReleaseContractError("null and incomplete outcomes are reported, never omitted")
    if not protocol["stop_conditions"]["forbidden_additions"]:
        raise ReleaseContractError("Plan Section 17 names the additions this project will not make")
    return protocol


def _resolved(protocol: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return protocol if protocol is not None else load_release_protocol()


# --------------------------------------------------------------------------------------
# Evidence entries
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceEntry:
    """One promise, its evidence path, its verifier, and its honest status."""

    item_id: str
    group: str
    requirement: str
    path: str
    verifier: str
    status: str
    failure_policy: str
    owner: str = ""
    next_action: str = ""
    note: str = ""

    @property
    def is_external(self) -> bool:
        """True when no repository code can settle this item."""
        return self.verifier in _EXTERNAL_VERIFIERS

    @property
    def claims_verified(self) -> bool:
        return self.status in ASSERTIVE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "group": self.group,
            "requirement": self.requirement,
            "path": self.path,
            "verifier": self.verifier,
            "status": self.status,
            "failure_policy": self.failure_policy,
            "owner": self.owner,
            "next_action": self.next_action,
            "note": self.note,
        }


def _entry(payload: Mapping[str, Any], group: str) -> EvidenceEntry:
    return EvidenceEntry(
        item_id=str(payload["item_id"]),
        group=group,
        requirement=str(payload["requirement"]),
        path=str(payload["path"]),
        verifier=str(payload["verifier"]),
        status=str(payload["status"]),
        failure_policy=str(payload["failure_policy"]),
        owner=str(payload.get("owner", "")),
        next_action=str(payload.get("next_action", "")),
        note=str(payload.get("note", "")),
    )


def evidence_entries(protocol: Mapping[str, Any] | None = None) -> tuple[EvidenceEntry, ...]:
    """Every contract item and gate item, in declaration order."""
    matrix = _resolved(protocol)["release_evidence_matrix"]
    entries = [_entry(item, "contract") for item in matrix["contract_items"]]
    entries.extend(_entry(item, "gate") for item in matrix["gate_items"])
    return tuple(entries)


def entry_index(protocol: Mapping[str, Any] | None = None) -> dict[str, EvidenceEntry]:
    return {entry.item_id: entry for entry in evidence_entries(protocol)}


def matrix_violations(
    protocol: Mapping[str, Any] | None = None, *, root: Path = REPOSITORY_ROOT
) -> tuple[str, ...]:
    """Every status must be earned: a PASS needs real evidence, a BLOCKED needs an owner."""
    resolved = _resolved(protocol)
    problems: list[str] = []
    seen: set[str] = set()

    for entry in evidence_entries(resolved):
        if entry.item_id in seen:
            problems.append(f"{RELEASE_UNKNOWN_STATUS}: duplicate item {entry.item_id!r}")
        seen.add(entry.item_id)

        if entry.status not in RELEASE_STATUSES:
            problems.append(
                f"{RELEASE_UNKNOWN_STATUS}: {entry.item_id} has status {entry.status!r}, "
                f"expected one of {RELEASE_STATUSES}"
            )
            continue

        if not entry.failure_policy.strip():
            problems.append(f"{RELEASE_UNKNOWN_STATUS}: {entry.item_id} declares no failure policy")

        # A PASS is a claim, so it must point at evidence that exists.
        if entry.claims_verified:
            if entry.is_external:
                problems.append(
                    f"{RELEASE_UNSUPPORTED_PASS}: {entry.item_id} claims PASS but its verifier "
                    f"{entry.verifier!r} is an external action no repository check performs"
                )
            if not entry.verifier.strip():
                problems.append(f"{RELEASE_VERIFIER_MISSING}: {entry.item_id} claims PASS without a verifier")
            elif not entry.is_external and not (root / entry.verifier).exists():
                problems.append(
                    f"{RELEASE_VERIFIER_MISSING}: {entry.item_id} names verifier "
                    f"{entry.verifier!r}, which does not exist"
                )
            if not (root / entry.path).exists():
                problems.append(
                    f"{RELEASE_PATH_MISSING}: {entry.item_id} claims PASS but "
                    f"{entry.path!r} does not exist"
                )

        # A BLOCKED without an owner and a next action is an excuse, not a status.
        if entry.status == BLOCKED and not (entry.owner.strip() and entry.next_action.strip()):
            problems.append(
                f"{RELEASE_BLOCKER_DETAIL_INCOMPLETE}: {entry.item_id} is BLOCKED without "
                "an owner and a next action"
            )

        # Every named path should exist even when the status is not a claim; a matrix that
        # points at nothing cannot be followed.
        if entry.path.strip() and not (root / entry.path).exists():
            if not entry.claims_verified:
                problems.append(
                    f"{RELEASE_PATH_MISSING}: {entry.item_id} names {entry.path!r}, which does not exist"
                )
    return tuple(problems)


def template_violations(
    protocol: Mapping[str, Any] | None = None, *, root: Path = REPOSITORY_ROOT
) -> tuple[str, ...]:
    """Every required submission template must exist."""
    resolved = _resolved(protocol)
    missing = [
        str(path)
        for path in resolved["required_templates"]["paths"]
        if not (root / str(path)).exists()
    ]
    if missing:
        return (f"{RELEASE_TEMPLATE_MISSING}: {sorted(missing)}",)
    return ()


def superseded_document_violations(
    protocol: Mapping[str, Any] | None = None, *, root: Path = REPOSITORY_ROOT
) -> tuple[str, ...]:
    """A superseded document must say so, in its own text, near the top.

    Deleting ``docs/PILOT_REPORT.md`` would destroy real measurements; leaving it unlabeled
    would let a superseded recommendation read as current. Labeling is the only option that
    preserves the evidence and prevents the misreading.
    """
    resolved = _resolved(protocol)
    problems: list[str] = []
    for document in resolved["superseded_documents"]["documents"]:
        path = root / str(document["path"])
        label = str(document["label_required"]).lower()
        if not path.exists():
            problems.append(
                f"{RELEASE_SUPERSEDED_UNLABELED}: {document['path']} is declared superseded "
                "but does not exist"
            )
            continue
        head = path.read_text(encoding="utf-8")[:1200].lower()
        if label not in head and "superseded" not in head:
            problems.append(
                f"{RELEASE_SUPERSEDED_UNLABELED}: {document['path']} is not labeled "
                f"{label!r} or 'superseded' near the top"
            )
    return tuple(problems)


def scope_creep_violations(
    proposed_additions: Iterable[str],
    *,
    measured_need: bool = False,
    schedule_ahead: bool = False,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Plan Section 17: no new architecture, data family, dashboard, or hosted workflow.

    The exception is narrow on purpose -- measured evidence of a specific need *and* a
    critical schedule that is ahead. Both, not either.
    """
    resolved = _resolved(protocol)
    forbidden = {str(name) for name in resolved["stop_conditions"]["forbidden_additions"]}
    requested = {str(name) for name in proposed_additions}
    offending = sorted(requested & forbidden)
    if not offending:
        return ()
    if measured_need and schedule_ahead:
        return ()
    return (
        f"{RELEASE_SCOPE_CREEP}: {offending} requires measured evidence of a specific need "
        f"and a critical schedule that is ahead (measured_need={measured_need}, "
        f"schedule_ahead={schedule_ahead})",
    )


def fallback_release_violations(
    *, is_undecayed_peak_lr_mainline: bool, protocol: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Plan Section 15: never release an undecayed peak-LR mainline checkpoint."""
    resolved = _resolved(protocol)
    if (
        bool(resolved["fallback_policy"]["never_release_undecayed_peak_lr_mainline"])
        and is_undecayed_peak_lr_mainline
    ):
        return (
            f"{RELEASE_UNDECAYED_FALLBACK}: an undecayed peak-LR mainline checkpoint may "
            "never be released as the fallback",
        )
    return ()


# --------------------------------------------------------------------------------------
# Readiness and the whole-matrix report
# --------------------------------------------------------------------------------------

_BLOCKER_FIELDS = ("blocker", "owner", "next_action")

_READINESS_GATED = (
    "final_run_completed",
    "required_evaluation_completed",
    "public_artifacts_published",
    "teammate_approval_recorded",
)


def assert_ready_for_release(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: a release needs artifacts and approvals that do not exist yet."""
    readiness = _resolved(protocol)["readiness"]
    blocked = [name for name in _READINESS_GATED if str(readiness.get(name)) != PASS]
    if blocked:
        raise ReleaseNotReadyError(
            f"the release is not ready: {blocked}. "
            f"blocker={readiness.get('blocker')} owner={readiness.get('owner')} "
            f"next_action={readiness.get('next_action')}"
        )


def readiness_results(protocol: Mapping[str, Any] | None = None) -> tuple[CheckResult, ...]:
    """Report each unmet release prerequisite with its own owner and next action."""
    readiness = _resolved(protocol)["readiness"]
    detail = {name: readiness.get(name) for name in _BLOCKER_FIELDS}
    named = all(str(value or "").strip() for value in detail.values())
    results: list[CheckResult] = []
    for name, value in readiness.items():
        if name in _BLOCKER_FIELDS or not isinstance(value, str):
            continue
        status = str(value)
        if status == PASS:
            results.append(
                CheckResult(f"release.readiness.{name}", "evidence exists", status, PASS, RELEASE_OK)
            )
            continue
        if not named:
            results.append(
                CheckResult(
                    f"release.readiness.{name}",
                    "an unmet prerequisite must name its blocker, owner, and next action",
                    status,
                    FAIL,
                    f"{RELEASE_BLOCKER_DETAIL_INCOMPLETE}: readiness detail is incomplete",
                )
            )
            continue
        results.append(
            CheckResult(
                f"release.readiness.{name}",
                "evidence exists",
                status,
                status,
                f"blocker={detail['blocker']} owner={detail['owner']} next_action={detail['next_action']}",
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class ReleaseReport:
    """The whole submission surface, with nothing ticked off early."""

    results: tuple[CheckResult, ...]
    entries: tuple[EvidenceEntry, ...]

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status == FAIL)

    @property
    def ok(self) -> bool:
        return not self.failures

    def by_status(self) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {status: [] for status in RELEASE_STATUSES}
        for entry in self.entries:
            grouped.setdefault(entry.status, []).append(entry.item_id)
        return {status: tuple(items) for status, items in grouped.items()}


def _verdict(check_id: str, requirement: str, observed: Any, ok: bool, reason: str) -> CheckResult:
    return CheckResult(check_id, requirement, str(observed), PASS if ok else FAIL, reason)


def verify_release_matrix(
    protocol: Mapping[str, Any] | None = None, *, root: Path = REPOSITORY_ROOT
) -> ReleaseReport:
    """Validate the matrix, the templates, and the superseded labels."""
    resolved = _resolved(protocol)
    entries = evidence_entries(resolved)
    results: list[CheckResult] = []

    matrix_problems = matrix_violations(resolved, root=root)
    results.append(
        _verdict(
            "release.matrix_statuses",
            "every status names evidence that exists; no unsupported PASS",
            matrix_problems or f"{len(entries)} entries",
            not matrix_problems,
            RELEASE_OK if not matrix_problems else "; ".join(matrix_problems),
        )
    )

    template_problems = template_violations(resolved, root=root)
    results.append(
        _verdict(
            "release.templates",
            "every required submission template exists",
            template_problems or "all present",
            not template_problems,
            RELEASE_OK if not template_problems else "; ".join(template_problems),
        )
    )

    superseded_problems = superseded_document_violations(resolved, root=root)
    results.append(
        _verdict(
            "release.superseded_labels",
            "every superseded document is labeled historical",
            superseded_problems or "all labeled",
            not superseded_problems,
            RELEASE_OK if not superseded_problems else "; ".join(superseded_problems),
        )
    )

    covered = {entry.item_id for entry in entries if entry.group == "gate"}
    missing_gates = [gate for gate in GATE_IDS if gate not in covered]
    results.append(
        _verdict(
            "release.gate_coverage",
            f"the matrix covers every gate {GATE_IDS}",
            missing_gates or "all covered",
            not missing_gates,
            RELEASE_OK if not missing_gates else f"{RELEASE_UNKNOWN_STATUS}: {missing_gates}",
        )
    )

    results.extend(readiness_results(resolved))
    return ReleaseReport(tuple(results), entries)


def assert_release_matrix_valid(
    protocol: Mapping[str, Any] | None = None, *, root: Path = REPOSITORY_ROOT
) -> None:
    """Raise on any FAIL. NOT_RUN, BLOCKED, and TBD are honest absences."""
    report = verify_release_matrix(protocol, root=root)
    if not report.ok:
        raise ReleaseContractError(
            "; ".join(f"{result.check_id}: {result.reason}" for result in report.failures)
        )


def format_release_matrix(entries: Sequence[EvidenceEntry]) -> str:
    """Render the matrix as one aligned, greppable status block."""
    width = max((len(entry.item_id) for entry in entries), default=0)
    lines: list[str] = []
    for entry in entries:
        detail = entry.path
        if entry.status == BLOCKED and entry.owner:
            detail = f"{entry.path}  owner={entry.owner} next_action={entry.next_action}"
        lines.append(f"{entry.status:<8} {entry.item_id:<{width}}  {detail}")
    return "\n".join(lines)


__all__ = [
    "ASSERTIVE_STATUSES",
    "BLOCKED",
    "FROZEN_RELEASE_PROTOCOL_SHA256",
    "RELEASE_BLOCKER_DETAIL_INCOMPLETE",
    "RELEASE_FAIL_CLOSED_REASON_CODES",
    "RELEASE_OK",
    "RELEASE_PATH_MISSING",
    "RELEASE_PROTOCOL_PATH",
    "RELEASE_SCOPE_CREEP",
    "RELEASE_STATUSES",
    "RELEASE_SUPERSEDED_UNLABELED",
    "RELEASE_TEMPLATE_MISSING",
    "RELEASE_UNDECAYED_FALLBACK",
    "RELEASE_UNKNOWN_STATUS",
    "RELEASE_UNSUPPORTED_PASS",
    "RELEASE_VERIFIER_MISSING",
    "SUPERSEDED_LABEL",
    "TBD",
    "EvidenceEntry",
    "ReleaseContractError",
    "ReleaseNotReadyError",
    "ReleaseReport",
    "assert_ready_for_release",
    "assert_release_matrix_valid",
    "entry_index",
    "evidence_entries",
    "fallback_release_violations",
    "format_release_matrix",
    "load_release_protocol",
    "matrix_violations",
    "readiness_results",
    "scope_creep_violations",
    "superseded_document_violations",
    "template_violations",
    "verify_release_matrix",
]
