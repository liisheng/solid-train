from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ContractEntry:
    entry_id: str
    kind: str
    path: str
    expected: Any = None
    key: str | None = None
    rule: str | None = None
    needle: str | tuple[str, ...] | None = None


@dataclass(frozen=True)
class AuditMismatch:
    entry_id: str
    path: str
    expected: Any
    observed: Any
    reason: str


def _parameter_count(config: dict[str, Any]) -> int:
    """Independently reconcile unique parameters for the implemented tied-head model."""
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


def _implementation_text(repository: Path) -> str:
    included_suffixes = {".json", ".md", ".py", ".toml", ".yaml", ".yml"}
    excluded_parts = {".git", ".kiro", ".pytest_cache", ".venv", "__pycache__", "tests"}
    chunks: list[str] = []
    pending = [repository]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            if path.is_dir():
                if path.name not in excluded_parts:
                    pending.append(path)
                continue
            if path.suffix.lower() not in included_suffixes:
                continue
            try:
                chunks.append(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                continue
    return "\n".join(chunks)


def _adamw_uses_distinct_decay_groups(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "AdamW"
            and node.args
        ):
            continue
        first_argument = node.args[0]
        is_undifferentiated_parameters_call = (
            isinstance(first_argument, ast.Call)
            and isinstance(first_argument.func, ast.Attribute)
            and first_argument.func.attr == "parameters"
        )
        return not is_undifferentiated_parameters_call
    return False


def _content_satisfies(repository: Path, entry: ContractEntry, content: str) -> bool:
    if entry.rule == "contains":
        return isinstance(entry.needle, str) and entry.needle in content
    if entry.rule == "not_contains":
        return isinstance(entry.needle, str) and entry.needle not in content
    if entry.rule == "adamw_decay_groups":
        return _adamw_uses_distinct_decay_groups(content)
    if entry.rule == "tree_contains_all":
        needles = entry.needle if isinstance(entry.needle, tuple) else (entry.needle,)
        implementation = _implementation_text(repository)
        return all(isinstance(needle, str) and needle in implementation for needle in needles)
    raise ValueError(f"Unknown content rule: {entry.rule}")


def audit_entry(repository: Path, entry: ContractEntry) -> AuditMismatch | None:
    path = repository / entry.path
    if entry.kind == "required_path":
        if path.exists():
            return None
        return AuditMismatch(entry.entry_id, entry.path, "present", "missing", "required path is absent")

    if entry.kind == "forbidden_path":
        if not path.exists():
            return None
        return AuditMismatch(entry.entry_id, entry.path, "absent", "present", "prohibited path exists")

    if entry.kind == "required_value":
        if not path.is_file():
            return AuditMismatch(entry.entry_id, entry.path, entry.expected, "missing", "value source is absent")
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed = _parameter_count(payload) if entry.key == "__unique_parameter_count__" else payload.get(entry.key)
        if observed == entry.expected:
            return None
        return AuditMismatch(entry.entry_id, entry.path, entry.expected, observed, "required value differs")

    if entry.kind == "content_predicate":
        if entry.rule == "tree_contains_all":
            content = ""
        elif path.is_file():
            content = path.read_text(encoding="utf-8")
        else:
            return AuditMismatch(entry.entry_id, entry.path, entry.expected, "missing", "predicate source is absent")
        if _content_satisfies(repository, entry, content):
            return None
        return AuditMismatch(entry.entry_id, entry.path, entry.expected, "predicate failed", "required content predicate is false")

    if entry.kind == "evidence_backed_deferral":
        if not path.is_file():
            return AuditMismatch(entry.entry_id, entry.path, "evidence-backed status", "missing", "status record is absent")
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = payload.get("status")
        if status in {"DEFERRED", "BLOCKED"}:
            missing = [field for field in ("blocker", "owner", "next_action") if not payload.get(field)]
            if not missing:
                return None
            return AuditMismatch(entry.entry_id, entry.path, "blocker/owner/next_action", missing, "deferral evidence is incomplete")
        if status == "PASS" and payload.get("evidence"):
            return None
        return AuditMismatch(entry.entry_id, entry.path, "evidence-backed status", status, "completion has no evidence")

    raise ValueError(f"Unknown contract-entry kind: {entry.kind}")


def audit_repository(repository: Path, target: Iterable[ContractEntry]) -> list[AuditMismatch]:
    return [mismatch for entry in target if (mismatch := audit_entry(repository, entry)) is not None]


def is_bug_condition(repository: Path, target: ContractEntry | Iterable[ContractEntry]) -> bool:
    """True exactly when at least one authoritative contract entry is not satisfied."""
    entries = [target] if isinstance(target, ContractEntry) else list(target)
    return bool(audit_repository(repository, entries))


def _generated_entries(seed: int) -> list[ContractEntry]:
    rng = random.Random(seed)
    suffix = f"{seed:02d}_{rng.randrange(1_000_000):06d}"
    return [
        ContractEntry(f"generated.value.{suffix}", "required_value", f"configs/{suffix}.json", rng.randrange(1, 10_000), "value"),
        ContractEntry(f"generated.required_path.{suffix}", "required_path", f"protocols/{suffix}.yaml"),
        ContractEntry(f"generated.forbidden_path.{suffix}", "forbidden_path", f"prohibited/{suffix}.bin"),
        ContractEntry(
            f"generated.predicate.{suffix}",
            "content_predicate",
            f"src/{suffix}.py",
            expected=f"contains contract_{suffix}",
            rule="contains",
            needle=f"contract_{suffix}",
        ),
        ContractEntry(f"generated.deferral.{suffix}", "evidence_backed_deferral", f"evidence/{suffix}.json"),
    ]


def _write(path: Path, text: str = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _materialize_fixture(repository: Path, entry: ContractEntry, *, conforming: bool) -> None:
    repository.mkdir(parents=True, exist_ok=True)
    path = repository / entry.path
    if entry.kind == "required_value":
        value = entry.expected if conforming else entry.expected + 1
        _write(path, json.dumps({entry.key: value}))
    elif entry.kind == "required_path":
        if conforming:
            _write(path)
    elif entry.kind == "forbidden_path":
        if not conforming:
            _write(path)
    elif entry.kind == "content_predicate":
        content = f"# {entry.needle}\n" if conforming else "# missing required predicate\n"
        _write(path, content)
    elif entry.kind == "evidence_backed_deferral":
        payload = (
            {"status": "DEFERRED", "blocker": "measurement not run", "owner": "operator", "next_action": "run bounded measurement"}
            if conforming
            else {"status": "PASS"}
        )
        _write(path, json.dumps(payload))
    else:
        raise ValueError(entry.kind)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_generated_contract_entry_property(tmp_path: Path) -> None:
    """Property 1: every generated bug condition is reported, while conformity is not."""
    covered_kinds: set[str] = set()
    for seed in range(16):
        for entry in _generated_entries(seed):
            covered_kinds.add(entry.kind)
            divergent_repository = tmp_path / "divergent" / entry.entry_id
            conforming_repository = tmp_path / "conforming" / entry.entry_id
            _materialize_fixture(divergent_repository, entry, conforming=False)
            _materialize_fixture(conforming_repository, entry, conforming=True)

            assert is_bug_condition(divergent_repository, entry)
            assert [mismatch.entry_id for mismatch in audit_repository(divergent_repository, [entry])] == [entry.entry_id]
            assert not is_bug_condition(conforming_repository, entry)
            assert audit_repository(conforming_repository, [entry]) == []

    assert covered_kinds == {
        "required_value",
        "required_path",
        "forbidden_path",
        "content_predicate",
        "evidence_backed_deferral",
    }


AUTHORITATIVE_TARGET = [
    ContractEntry("architecture.vocab_size", "required_value", "configs/baseline_49m.json", 12_288, "vocab_size"),
    ContractEntry("architecture.n_layers", "required_value", "configs/baseline_49m.json", 14, "n_layers"),
    ContractEntry("architecture.n_kv_heads", "required_value", "configs/baseline_49m.json", 4, "n_kv_heads"),
    ContractEntry("architecture.d_ff", "required_value", "configs/baseline_49m.json", 1_504, "d_ff"),
    ContractEntry("architecture.unique_parameter_count", "required_value", "configs/baseline_49m.json", 49_658_368, "__unique_parameter_count__"),
    ContractEntry(
        "training.wsd_not_cosine",
        "content_predicate",
        "train.py",
        expected="no cosine final-campaign decay",
        rule="not_contains",
        needle="math.cos",
    ),
    ContractEntry(
        "training.optimizer_decay_groups",
        "content_predicate",
        "train.py",
        expected="distinct AdamW groups excluding embeddings and normalization weights",
        rule="adamw_decay_groups",
    ),
    ContractEntry(
        "data.materialized_schedule",
        "content_predicate",
        ".",
        expected="materialized schedule cursor instead of final random flat-stream sampling",
        rule="tree_contains_all",
        needle=("schedule_cursor", "shard_id", "token_offset"),
    ),
    ContractEntry("data.dedup_protocol", "required_path", "configs/data/dedup_v1.yaml"),
    ContractEntry("data.decontamination_protocol", "required_path", "configs/data/decontam_v1.yaml"),
    ContractEntry(
        "branches.exposure_and_annealing_protocol",
        "content_predicate",
        ".",
        expected="frozen A/B/C exposure identity and annealing protocol",
        rule="tree_contains_all",
        needle=("stable_exposure_hash", "stable_control", "reserved_in_update"),
    ),
    ContractEntry(
        "evaluation.frozen_provisional_contract",
        "content_predicate",
        ".",
        expected="versioned provisional evaluation protocol with protocol hash",
        rule="tree_contains_all",
        needle=("evaluation_provisional_v1", "protocol_hash"),
    ),
    ContractEntry(
        "evidence.final_contract",
        "content_predicate",
        ".",
        expected="release evidence matrix with unresolved statuses preserved",
        rule="tree_contains_all",
        needle=("release_evidence_matrix", "NOT_RUN", "BLOCKED"),
    ),
]


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_repository_matches_authoritative_plan() -> None:
    """Exploration assertion: expected to fail until every implementable divergence is fixed."""
    mismatches = audit_repository(REPOSITORY_ROOT, AUTHORITATIVE_TARGET)
    minimized = mismatches[0] if mismatches else None
    report = {
        "minimized_counterexample": minimized.__dict__ if minimized else None,
        "mismatches": [mismatch.__dict__ for mismatch in mismatches],
    }
    assert not is_bug_condition(REPOSITORY_ROOT, AUTHORITATIVE_TARGET), json.dumps(report, indent=2, sort_keys=True)
