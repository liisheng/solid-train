"""The authoritative alignment auditor (Plan Sections 2.1, 13, 16-17).

This module answers one question about the whole repository: *does it match the authoritative
plan, and where it does not, is the difference explained?*

Two properties make the answer worth trusting.

**An unexplained difference always fails.** There is exactly one way for an entry not to pass:
it is a `DEFERRED` entry whose category is one of the five things a repository genuinely cannot
settle by itself -- a measurement nobody has taken, an organizer answer nobody has sent, a
personal attestation only a person can make, a public release nobody has published, or a
long-running campaign nobody has run -- *and* it names an owner and a next action. Every other
mismatch is a `FAIL`. There is no "known issue" state and no way to silence one.

**The audit is read-only, and therefore idempotent.** It opens files, hashes bytes, and calls
verifiers. It never writes, formats, repairs, or normalizes anything. So running it twice
returns the identical report and leaves every byte untouched -- which is exactly what
:func:`audit_is_idempotent` checks, by fingerprinting the tree before and after. An auditor that
quietly fixes what it finds cannot tell you what was wrong, and an auditor that rewrites files
turns a verification step into a change.

The contract is backed by one frozen config::

    configs/audit/alignment_v1.yaml

This complements ``tests/test_alignment_audit.py`` rather than replacing it: that file holds the
task-1 bug-condition property and its deterministic counterexamples, and is re-run unchanged.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .data_protocols import (
    REPOSITORY_ROOT,
    ProtocolError,
    load_protocol,
    protocol_digest,
)
from .environment import CheckResult
from .shards import FAIL, PASS

ALIGNMENT_PROTOCOL_DIR = REPOSITORY_ROOT / "configs" / "audit"
ALIGNMENT_PROTOCOL_PATH = ALIGNMENT_PROTOCOL_DIR / "alignment_v1.yaml"

#: SHA-256 of the frozen alignment checklist, over file bytes with CRLF normalized to LF.
FROZEN_ALIGNMENT_PROTOCOL_SHA256: Mapping[str, str] = {
    "alignment_v1.yaml": "34ab4029bc2d5620e6d54d78be9296c6742ea70860eb9ea17c2db709632af5df",
}

DEFERRED = "DEFERRED"

#: The three statuses an audit entry may hold. Only FAIL is a failure.
AUDIT_STATUSES: tuple[str, ...] = (PASS, FAIL, DEFERRED)

# --------------------------------------------------------------------------------------
# Reason-code vocabulary
# --------------------------------------------------------------------------------------

AUDIT_OK = "AUDIT_OK"
AUDIT_DEFERRAL_CATEGORY_INVALID = "AUDIT_DEFERRAL_CATEGORY_INVALID"
AUDIT_DEFERRAL_DETAIL_INCOMPLETE = "AUDIT_DEFERRAL_DETAIL_INCOMPLETE"
AUDIT_DIGEST_DRIFT = "AUDIT_DIGEST_DRIFT"
AUDIT_INTEGRITY_VIOLATION = "AUDIT_INTEGRITY_VIOLATION"
AUDIT_MUTATED_REPOSITORY = "AUDIT_MUTATED_REPOSITORY"
AUDIT_UNEXPLAINED_DIFFERENCE = "AUDIT_UNEXPLAINED_DIFFERENCE"
AUDIT_VERIFIER_FAILED = "AUDIT_VERIFIER_FAILED"

AUDIT_FAIL_CLOSED_REASON_CODES: frozenset[str] = frozenset(
    {
        AUDIT_DEFERRAL_CATEGORY_INVALID,
        AUDIT_DEFERRAL_DETAIL_INCOMPLETE,
        AUDIT_DIGEST_DRIFT,
        AUDIT_INTEGRITY_VIOLATION,
        AUDIT_MUTATED_REPOSITORY,
        AUDIT_UNEXPLAINED_DIFFERENCE,
        AUDIT_VERIFIER_FAILED,
    }
)


class AlignmentContractError(ProtocolError):
    """The frozen alignment checklist is malformed."""


# --------------------------------------------------------------------------------------
# Frozen checklist loading
# --------------------------------------------------------------------------------------


def load_alignment_checklist(
    path: Path = ALIGNMENT_PROTOCOL_PATH, *, verify: bool = True
) -> dict[str, Any]:
    """Load the frozen checklist, verifying its pinned digest by default."""
    protocol = load_protocol(path, verify=verify, registry=FROZEN_ALIGNMENT_PROTOCOL_SHA256)
    required = (
        "audit_policy",
        "deferral_policy",
        "required_paths",
        "required_values",
        "parameter_contract",
        "frozen_digests",
        "verifier_outcomes",
        "integrity_scans",
        "evidence_links",
        "deferrals",
    )
    for section in required:
        if section not in protocol:
            raise AlignmentContractError(f"alignment checklist is missing required section {section!r}")

    policy = protocol["audit_policy"]
    for flag in ("read_only", "idempotent", "unexplained_difference_fails"):
        if not bool(policy[flag]):
            raise AlignmentContractError(f"the alignment audit must assert {flag}")
    if tuple(str(status) for status in policy["statuses"]) != AUDIT_STATUSES:
        raise AlignmentContractError(f"audit statuses must be {AUDIT_STATUSES}")

    deferral = protocol["deferral_policy"]
    categories = tuple(str(name) for name in deferral["allowed_categories"])
    if categories != (
        "measurement",
        "organizer_answer",
        "personal_attestation",
        "public_release",
        "long_running_campaign",
    ):
        raise AlignmentContractError(
            "only measurement, organizer, personal-attestation, public-release, and "
            "long-running-campaign prerequisites may be deferred"
        )
    for flag in ("requires_owner", "requires_next_action"):
        if not bool(deferral[flag]):
            raise AlignmentContractError(f"a deferral must satisfy {flag}")

    contract = protocol["parameter_contract"]
    if int(contract["unique_trainable_parameters"]) + int(contract["headroom"]) != int(contract["cap"]):
        raise AlignmentContractError(
            "the parameter contract must reconcile: count + headroom must equal the cap"
        )
    return protocol


def _resolved(protocol: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return protocol if protocol is not None else load_alignment_checklist()


# --------------------------------------------------------------------------------------
# Independent parameter reconciliation
# --------------------------------------------------------------------------------------


def reconcile_parameter_count(config: Mapping[str, Any]) -> int:
    """Recompute unique trainable parameters from the config's own fields.

    Deliberately independent of the model code: if both the model and the audit derived the
    count the same way, a shared mistake would agree with itself.
    """
    width = int(config["d_model"])
    heads = int(config["n_heads"])
    kv_heads = int(config["n_kv_heads"])
    head_dim = width // heads
    kv_width = kv_heads * head_dim
    embedding = int(config["vocab_size"]) * width
    attention = width * width + 2 * width * kv_width + width * width
    swiglu = 3 * width * int(config["d_ff"])
    block_norms = 2 * width
    return embedding + int(config["n_layers"]) * (attention + swiglu + block_norms) + width


# --------------------------------------------------------------------------------------
# Integrity scans
# --------------------------------------------------------------------------------------

_SCAN_EXCLUDED_DIRECTORIES = frozenset(
    {".git", ".kiro", ".pytest_cache", ".venv", "__pycache__", "tests", "node_modules"}
)


def _production_python_files(root: Path) -> list[Path]:
    """Every production Python file. Tests are excluded: fixtures may name what they forbid."""
    found: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            if path.is_dir():
                if path.name not in _SCAN_EXCLUDED_DIRECTORIES:
                    pending.append(path)
                continue
            if path.suffix == ".py":
                found.append(path)
    return found


def _scan_production_from_pretrained(root: Path) -> tuple[str, ...]:
    """No production model-weight path may call ``from_pretrained()``."""
    problems: list[str] = []
    for path in _production_python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pretrained"
            ):
                problems.append(
                    f"{AUDIT_INTEGRITY_VIOLATION}: {path.relative_to(root).as_posix()} "
                    f"line {node.lineno} calls from_pretrained()"
                )
    return tuple(problems)


def _scan_cosine_final_decay(root: Path) -> tuple[str, ...]:
    """The final campaign uses WSD linear decay; a cosine schedule would be a silent change."""
    train = root / "train.py"
    if not train.is_file():
        return (f"{AUDIT_INTEGRITY_VIOLATION}: train.py is absent",)
    source = train.read_text(encoding="utf-8")
    if "math.cos" in source:
        return (f"{AUDIT_INTEGRITY_VIOLATION}: train.py still uses cosine decay",)
    return ()


INTEGRITY_SCANS: Mapping[str, Callable[[Path], tuple[str, ...]]] = {
    "production_from_pretrained": _scan_production_from_pretrained,
    "cosine_final_decay": _scan_cosine_final_decay,
}


# --------------------------------------------------------------------------------------
# Verifier registry
# --------------------------------------------------------------------------------------


def _verify_parameter_count(root: Path) -> tuple[str, ...]:
    from .config import ModelConfig
    from .model import TinyBenchLM
    from .parameters import count_unique_trainable_parameters

    checklist = load_alignment_checklist()
    contract = checklist["parameter_contract"]
    config_path = root / str(contract["path"])
    if not config_path.is_file():
        return (f"{AUDIT_UNEXPLAINED_DIFFERENCE}: {contract['path']} is absent",)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    expected = int(contract["unique_trainable_parameters"])

    reconciled = reconcile_parameter_count(payload)
    counted = count_unique_trainable_parameters(TinyBenchLM(ModelConfig(**payload)))
    problems: list[str] = []
    if reconciled != expected:
        problems.append(
            f"{AUDIT_UNEXPLAINED_DIFFERENCE}: reconciled {reconciled}, contract says {expected}"
        )
    if counted != expected:
        problems.append(
            f"{AUDIT_UNEXPLAINED_DIFFERENCE}: enumerated {counted}, contract says {expected}"
        )
    if counted > int(contract["cap"]):
        problems.append(f"{AUDIT_INTEGRITY_VIOLATION}: {counted} exceeds the {contract['cap']} cap")
    return tuple(problems)


def _verify_eligibility_scan(root: Path) -> tuple[str, ...]:
    return _scan_production_from_pretrained(root)


def _verify_campaign_contract(root: Path) -> tuple[str, ...]:
    from .campaign import verify_campaign

    del root
    report = verify_campaign()
    return tuple(f"{AUDIT_VERIFIER_FAILED}: {item.check_id}: {item.reason}" for item in report.failures)


def _verify_release_matrix(root: Path) -> tuple[str, ...]:
    from .release import verify_release_matrix

    report = verify_release_matrix(root=root)
    return tuple(f"{AUDIT_VERIFIER_FAILED}: {item.check_id}: {item.reason}" for item in report.failures)


VERIFIERS: Mapping[str, Callable[[Path], tuple[str, ...]]] = {
    "parameter_count": _verify_parameter_count,
    "eligibility_scan": _verify_eligibility_scan,
    "campaign_contract": _verify_campaign_contract,
    "release_matrix": _verify_release_matrix,
}


# --------------------------------------------------------------------------------------
# The audit
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEntry:
    """One checklist outcome."""

    entry_id: str
    status: str
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == FAIL


@dataclass(frozen=True)
class AuditReport:
    """The complete alignment result. Deterministic, so two runs compare equal."""

    entries: tuple[AuditEntry, ...]

    @property
    def failures(self) -> tuple[AuditEntry, ...]:
        return tuple(entry for entry in self.entries if entry.failed)

    @property
    def deferrals(self) -> tuple[AuditEntry, ...]:
        return tuple(entry for entry in self.entries if entry.status == DEFERRED)

    @property
    def ok(self) -> bool:
        """True when there is no unexplained difference. Deferrals are explained."""
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [
                {"entry_id": entry.entry_id, "status": entry.status, "detail": entry.detail}
                for entry in self.entries
            ],
            "failures": len(self.failures),
            "deferrals": len(self.deferrals),
            "ok": self.ok,
        }


def _passed(entry_id: str) -> AuditEntry:
    return AuditEntry(entry_id, PASS, AUDIT_OK)


def _failed(entry_id: str, detail: str) -> AuditEntry:
    return AuditEntry(entry_id, FAIL, detail)


def _guarded(
    entry_id: str, check: Callable[[], tuple[str, ...]], reason_code: str
) -> AuditEntry:
    """Run one check, converting a crash into a reported FAIL.

    An auditor that raises on a damaged repository tells you nothing about the damage. A
    missing config, an unparseable file, or an import that fails is itself a difference, so
    it belongs in the report rather than in a traceback.
    """
    try:
        problems = check()
    except Exception as error:  # noqa: BLE001 - any failure is a reportable difference
        return _failed(entry_id, f"{reason_code}: {type(error).__name__}: {error}")
    return _passed(entry_id) if not problems else _failed(entry_id, "; ".join(problems))


def audit_repository(
    protocol: Mapping[str, Any] | None = None, *, root: Path = REPOSITORY_ROOT
) -> AuditReport:
    """Run every checklist entry. Read-only: nothing here writes to the repository."""
    resolved = _resolved(protocol)
    entries: list[AuditEntry] = []

    for item in resolved["required_paths"]:
        entry_id = str(item["entry_id"])
        target = root / str(item["path"])
        entries.append(
            _passed(entry_id)
            if target.exists()
            else _failed(entry_id, f"{AUDIT_UNEXPLAINED_DIFFERENCE}: {item['path']} is absent")
        )

    def check_value(item: Mapping[str, Any]) -> tuple[str, ...]:
        source = root / str(item["path"])
        if not source.is_file():
            return (f"{AUDIT_UNEXPLAINED_DIFFERENCE}: {item['path']} is absent",)
        payload = json.loads(source.read_text(encoding="utf-8"))
        observed = payload.get(str(item["key"]))
        expected = item["expected"]
        if observed == expected:
            return ()
        return (
            f"{AUDIT_UNEXPLAINED_DIFFERENCE}: {item['key']} is {observed!r}, expected {expected!r}",
        )

    for item in resolved["required_values"]:
        entries.append(
            _guarded(str(item["entry_id"]), lambda i=item: check_value(i), AUDIT_UNEXPLAINED_DIFFERENCE)
        )

    contract = resolved["parameter_contract"]

    def check_parameter_contract() -> tuple[str, ...]:
        config_path = root / str(contract["path"])
        if not config_path.is_file():
            return (f"{AUDIT_UNEXPLAINED_DIFFERENCE}: {contract['path']} is absent",)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        reconciled = reconcile_parameter_count(payload)
        expected = int(contract["unique_trainable_parameters"])
        cap = int(contract["cap"])
        # Both checks always run. Chaining them would make the cap check unreachable, since
        # a count that equals the contract is by construction under the cap -- and the cap is
        # the one limit that decides eligibility, so it must never depend on another check.
        problems: list[str] = []
        if reconciled != expected:
            problems.append(
                f"{AUDIT_UNEXPLAINED_DIFFERENCE}: reconciled {reconciled}, expected {expected}"
            )
        if reconciled > cap:
            problems.append(f"{AUDIT_INTEGRITY_VIOLATION}: {reconciled} exceeds the {cap} cap")
        return tuple(problems)

    entries.append(
        _guarded(str(contract["entry_id"]), check_parameter_contract, AUDIT_UNEXPLAINED_DIFFERENCE)
    )

    for item in resolved["frozen_digests"]:
        entry_id = str(item["entry_id"])
        try:
            module = importlib.import_module(str(item["module"]))
            registry: Mapping[str, str] = getattr(module, str(item["registry"]))
        except (ImportError, AttributeError) as error:
            entries.append(_failed(entry_id, f"{AUDIT_DIGEST_DRIFT}: {error}"))
            continue
        drift: list[str] = []
        for name, pinned in registry.items():
            target = root / str(item["directory"]) / name
            if not target.is_file():
                drift.append(f"{name} is absent")
                continue
            observed = protocol_digest(target)
            if observed != pinned:
                drift.append(f"{name} is {observed}, pinned {pinned}")
        entries.append(
            _passed(entry_id) if not drift else _failed(entry_id, f"{AUDIT_DIGEST_DRIFT}: {drift}")
        )

    for item in resolved["verifier_outcomes"]:
        entry_id = str(item["entry_id"])
        name = str(item["verifier"])
        verifier = VERIFIERS.get(name)
        if verifier is None:
            entries.append(_failed(entry_id, f"{AUDIT_VERIFIER_FAILED}: unknown verifier {name!r}"))
            continue
        entries.append(_guarded(entry_id, lambda v=verifier: v(root), AUDIT_VERIFIER_FAILED))

    for item in resolved["integrity_scans"]:
        entry_id = str(item["entry_id"])
        name = str(item["scan"])
        scan = INTEGRITY_SCANS.get(name)
        if scan is None:
            entries.append(_failed(entry_id, f"{AUDIT_INTEGRITY_VIOLATION}: unknown scan {name!r}"))
            continue
        entries.append(_guarded(entry_id, lambda s=scan: s(root), AUDIT_INTEGRITY_VIOLATION))

    def check_evidence_links() -> tuple[str, ...]:
        from .release import evidence_entries, load_release_protocol

        dangling = [
            evidence.item_id
            for evidence in evidence_entries(load_release_protocol())
            if evidence.path.strip() and not (root / evidence.path).exists()
        ]
        if dangling:
            return (f"{AUDIT_UNEXPLAINED_DIFFERENCE}: dangling evidence paths {dangling}",)
        return ()

    for item in resolved["evidence_links"]:
        entries.append(
            _guarded(str(item["entry_id"]), check_evidence_links, AUDIT_UNEXPLAINED_DIFFERENCE)
        )

    allowed = {str(name) for name in resolved["deferral_policy"]["allowed_categories"]}
    for item in resolved["deferrals"]:
        entry_id = str(item["entry_id"])
        category = str(item.get("category", ""))
        owner = str(item.get("owner", "")).strip()
        next_action = str(item.get("next_action", "")).strip()
        if category not in allowed:
            entries.append(
                _failed(entry_id, f"{AUDIT_DEFERRAL_CATEGORY_INVALID}: {category!r} is not a permitted category")
            )
            continue
        if not (owner and next_action):
            entries.append(
                _failed(entry_id, f"{AUDIT_DEFERRAL_DETAIL_INCOMPLETE}: a deferral needs an owner and a next action")
            )
            continue
        entries.append(
            AuditEntry(entry_id, DEFERRED, f"category={category} owner={owner} next_action={next_action}")
        )

    return AuditReport(tuple(entries))


def assert_no_unexplained_differences(
    protocol: Mapping[str, Any] | None = None, *, root: Path = REPOSITORY_ROOT
) -> None:
    """Raise on any FAIL. A DEFERRED entry is explained, not a failure."""
    report = audit_repository(protocol, root=root)
    if not report.ok:
        raise AlignmentContractError(
            "; ".join(f"{entry.entry_id}: {entry.detail}" for entry in report.failures)
        )


# --------------------------------------------------------------------------------------
# Idempotence
# --------------------------------------------------------------------------------------

_FINGERPRINT_EXCLUDED = frozenset({".git", ".pytest_cache", ".venv", "__pycache__", "node_modules"})


def tree_fingerprint(root: Path = REPOSITORY_ROOT) -> str:
    """One hash over every tracked file's path, size, and bytes.

    Used to prove the audit changed nothing. Hashing content rather than mtime means a
    rewrite that happens to preserve the timestamp is still caught.
    """
    digest = hashlib.sha256()
    pending = [root]
    files: list[Path] = []
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            if path.is_dir():
                if path.name not in _FINGERPRINT_EXCLUDED:
                    pending.append(path)
                continue
            files.append(path)
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:  # pragma: no cover - unreadable file
            digest.update(b"<unreadable>")
    return digest.hexdigest()


def audit_is_idempotent(
    protocol: Mapping[str, Any] | None = None, *, root: Path = REPOSITORY_ROOT
) -> tuple[bool, tuple[str, ...]]:
    """Audit twice and prove both the report and the repository are unchanged."""
    before = tree_fingerprint(root)
    first = audit_repository(protocol, root=root)
    middle = tree_fingerprint(root)
    second = audit_repository(protocol, root=root)
    after = tree_fingerprint(root)

    problems: list[str] = []
    if first.to_dict() != second.to_dict():
        problems.append(f"{AUDIT_UNEXPLAINED_DIFFERENCE}: two audits returned different reports")
    if not (before == middle == after):
        problems.append(f"{AUDIT_MUTATED_REPOSITORY}: the audit rewrote files")
    return (not problems, tuple(problems))


def format_audit_report(report: AuditReport) -> str:
    """Render the checklist as one aligned, greppable status block."""
    width = max((len(entry.entry_id) for entry in report.entries), default=0)
    lines = [
        f"{entry.status:<8} {entry.entry_id:<{width}}  {entry.detail}" for entry in report.entries
    ]
    lines.append("")
    lines.append(
        f"{len(report.entries)} entries: "
        f"{len(report.entries) - len(report.failures) - len(report.deferrals)} PASS, "
        f"{len(report.failures)} FAIL, {len(report.deferrals)} DEFERRED"
    )
    return "\n".join(lines)


def audit_results(
    protocol: Mapping[str, Any] | None = None, *, root: Path = REPOSITORY_ROOT
) -> tuple[CheckResult, ...]:
    """The audit rendered as the shared CheckResult shape."""
    report = audit_repository(protocol, root=root)
    return tuple(
        CheckResult(entry.entry_id, "matches the authoritative plan", entry.status, entry.status, entry.detail)
        for entry in report.entries
    )


__all__ = [
    "ALIGNMENT_PROTOCOL_PATH",
    "AUDIT_DEFERRAL_CATEGORY_INVALID",
    "AUDIT_DEFERRAL_DETAIL_INCOMPLETE",
    "AUDIT_DIGEST_DRIFT",
    "AUDIT_FAIL_CLOSED_REASON_CODES",
    "AUDIT_INTEGRITY_VIOLATION",
    "AUDIT_MUTATED_REPOSITORY",
    "AUDIT_OK",
    "AUDIT_STATUSES",
    "AUDIT_UNEXPLAINED_DIFFERENCE",
    "AUDIT_VERIFIER_FAILED",
    "DEFERRED",
    "FROZEN_ALIGNMENT_PROTOCOL_SHA256",
    "INTEGRITY_SCANS",
    "VERIFIERS",
    "AlignmentContractError",
    "AuditEntry",
    "AuditReport",
    "assert_no_unexplained_differences",
    "audit_is_idempotent",
    "audit_repository",
    "audit_results",
    "format_audit_report",
    "load_alignment_checklist",
    "reconcile_parameter_count",
    "tree_fingerprint",
]
