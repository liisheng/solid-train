"""Repository eligibility audit for the submitted model's weight sources.

The competition rules allow random initialization only: no pretrained initialization, no
fine-tuning of an existing model, no distillation, no teacher outputs, and no hosted-model
inference in the submitted model's path (Plan Sections 2, 3.3, 4.5, 11.2, 13 G0/G6).
Documentation alone cannot prove that, so this module scans the eligible production
sources and fails closed on the patterns that would make the artifact ineligible.

The scan is syntax-aware. Only real calls, imports, and identifiers are considered, so
prose, comments, and the rule names in this module cannot trigger a violation. Loading a
local checkpoint with ``torch.load`` plus ``load_state_dict`` stays explicitly allowed:
resume, evaluation, and generation all depend on it.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .environment import REPOSITORY_ROOT, CheckResult

#: Entry-point modules that ship as the eligible training/evaluation path.
PRODUCTION_ENTRY_POINTS: tuple[str, ...] = ("train.py", "generate.py", "evaluate.py")

#: Directories whose Python sources belong to the eligible production path.
PRODUCTION_SOURCE_DIRECTORIES: tuple[str, ...] = ("src/tinybench_lm", "scripts")

RULE_PRETRAINED_WEIGHTS = "pretrained_model_weights"
RULE_REMOTE_WEIGHT_FETCH = "remote_weight_fetch"
RULE_KNOWLEDGE_TRANSFER = "knowledge_transfer_from_another_model"
RULE_HOSTED_INFERENCE = "hosted_model_inference_dependency"

ELIGIBILITY_RULES: tuple[str, ...] = (
    RULE_PRETRAINED_WEIGHTS,
    RULE_REMOTE_WEIGHT_FETCH,
    RULE_KNOWLEDGE_TRANSFER,
    RULE_HOSTED_INFERENCE,
)

_RULE_REASONS = {
    RULE_PRETRAINED_WEIGHTS: "production code must not load pretrained model weights",
    RULE_REMOTE_WEIGHT_FETCH: "production code must not fetch model weights from a remote source",
    RULE_KNOWLEDGE_TRANSFER: "production code must not transfer knowledge from another model",
    RULE_HOSTED_INFERENCE: "production code must not depend on hosted-model inference",
}

#: Calls that load weights published by someone else.
_PRETRAINED_LOADERS = frozenset({"from_pretrained", "load_pretrained", "pretrained_model"})

#: Calls that pull weights over the network.
_REMOTE_WEIGHT_FUNCTIONS = frozenset(
    {
        "download_url_to_file",
        "get_file",
        "hf_hub_download",
        "load_state_dict_from_url",
        "load_url",
        "snapshot_download",
        "urlopen",
        "urlretrieve",
    }
)

#: Dotted call paths that pull weights over the network.
_REMOTE_WEIGHT_CALL_PATHS = frozenset({"torch.hub.load", "torch.hub.load_state_dict_from_url"})

#: Modules whose presence in the production path implies remote weight retrieval.
_REMOTE_WEIGHT_MODULES = frozenset(
    {"aiohttp", "httpx", "huggingface_hub", "requests", "timm", "urllib", "torchvision"}
)

#: Modules that are hosted-inference clients.
_HOSTED_INFERENCE_MODULES = frozenset(
    {
        "anthropic",
        "cohere",
        "featherless",
        "featherless_ai",
        "google",
        "groq",
        "mistralai",
        "openai",
        "replicate",
        "together",
        "vertexai",
    }
)

#: Callables that instantiate a hosted or third-party inference model.
_HOSTED_INFERENCE_FUNCTIONS = frozenset({"InferenceClient", "pipeline"})

#: Identifier fragments that indicate knowledge transfer from another model.
_PROHIBITED_IDENTIFIER_PATTERN = re.compile(
    "|".join(("teach", "distil", "soft_target", "kd_loss", "logit_match")), re.IGNORECASE
)


class EligibilityError(ValueError):
    """Raised when the eligible production path contains an ineligible construct."""


@dataclass(frozen=True)
class EligibilityViolation:
    """One ineligible construct located in the production path."""

    rule: str
    path: str
    line: int
    detail: str

    def location(self) -> str:
        return f"{self.path}:{self.line}"

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "path": self.path, "line": self.line, "detail": self.detail}


@dataclass(frozen=True)
class EligibilityReport:
    """Complete audit outcome for a set of scanned production sources."""

    violations: tuple[EligibilityViolation, ...]
    scanned_paths: tuple[str, ...]
    results: tuple[CheckResult, ...]

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.failed)

    @property
    def ok(self) -> bool:
        return not self.violations and not self.failures

    def violations_for(self, rule: str) -> tuple[EligibilityViolation, ...]:
        return tuple(violation for violation in self.violations if violation.rule == rule)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "scanned_paths": list(self.scanned_paths),
            "violations": [violation.to_dict() for violation in self.violations],
            "results": [result.__dict__ for result in self.results],
        }


def production_python_paths(repository_root: Path = REPOSITORY_ROOT) -> tuple[Path, ...]:
    """Every Python source that ships in the eligible production path."""
    paths: list[Path] = []
    for name in PRODUCTION_ENTRY_POINTS:
        candidate = repository_root / name
        if candidate.is_file():
            paths.append(candidate)
    for relative in PRODUCTION_SOURCE_DIRECTORIES:
        directory = repository_root / Path(relative)
        if directory.is_dir():
            paths.extend(sorted(directory.glob("*.py")))
    return tuple(paths)


def _dotted_name(node: ast.AST) -> str:
    """Render ``torch.hub.load`` style attribute chains, or "" when not a plain chain."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


def _called_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return ""


def _imported_roots(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".", 1)[0] for alias in node.names]
    if node.level:  # a relative import can only reach our own package
        return []
    module = node.module or ""
    return [module.split(".", 1)[0]] if module else []


def _identifier_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.keyword) and node.arg:
        return [node.arg]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = [alias.asname or alias.name for alias in node.names]
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        return names
    return []


def scan_source(source: str, *, path: str) -> tuple[EligibilityViolation, ...]:
    """Report every ineligible construct in one Python source string."""
    tree = ast.parse(source, filename=path)
    violations: list[EligibilityViolation] = []

    def add(rule: str, node: ast.AST, detail: str) -> None:
        violations.append(EligibilityViolation(rule, path, getattr(node, "lineno", 0), detail))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = _called_name(node)
            dotted = _dotted_name(node.func)
            if called in _PRETRAINED_LOADERS:
                add(RULE_PRETRAINED_WEIGHTS, node, f"call to {dotted or called}()")
            if called in _REMOTE_WEIGHT_FUNCTIONS or dotted in _REMOTE_WEIGHT_CALL_PATHS:
                add(RULE_REMOTE_WEIGHT_FETCH, node, f"call to {dotted or called}()")
            if called in _HOSTED_INFERENCE_FUNCTIONS:
                add(RULE_HOSTED_INFERENCE, node, f"call to {dotted or called}()")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for root in _imported_roots(node):
                if root in _REMOTE_WEIGHT_MODULES:
                    add(RULE_REMOTE_WEIGHT_FETCH, node, f"imports {root}")
                if root in _HOSTED_INFERENCE_MODULES:
                    add(RULE_HOSTED_INFERENCE, node, f"imports {root}")
        for name in _identifier_names(node):
            match = _PROHIBITED_IDENTIFIER_PATTERN.search(name)
            if match:
                add(RULE_KNOWLEDGE_TRANSFER, node, f"identifier {name!r} matches {match.group()!r}")

    unique = {
        (violation.rule, violation.path, violation.line, violation.detail): violation
        for violation in violations
    }
    return tuple(sorted(unique.values(), key=lambda item: (item.line, item.rule, item.detail)))


def audit_eligibility(
    repository_root: Path = REPOSITORY_ROOT,
    paths: Sequence[Path] | None = None,
) -> EligibilityReport:
    """Scan the eligible production path and return an auditable pass/fail report."""
    scanned = tuple(paths) if paths is not None else production_python_paths(repository_root)
    violations: list[EligibilityViolation] = []
    relative_paths: list[str] = []
    for path in scanned:
        try:
            relative = path.resolve().relative_to(repository_root.resolve()).as_posix()
        except ValueError:
            relative = path.as_posix()
        relative_paths.append(relative)
        violations.extend(scan_source(path.read_text(encoding="utf-8"), path=relative))

    results: list[CheckResult] = []
    for rule in ELIGIBILITY_RULES:
        found = [violation for violation in violations if violation.rule == rule]
        results.append(
            CheckResult(
                f"eligibility.{rule}",
                "absent from the production path",
                "; ".join(violation.location() for violation in found) or "absent",
                "FAIL" if found else "PASS",
                _RULE_REASONS[rule] if found else f"no {rule.replace('_', ' ')} construct found",
            )
        )
    results.append(
        CheckResult(
            "eligibility.scanned_paths",
            "at least one production source",
            str(len(relative_paths)),
            "PASS" if relative_paths else "FAIL",
            "production sources were scanned" if relative_paths else "no production source was found",
        )
    )
    return EligibilityReport(tuple(violations), tuple(relative_paths), tuple(results))


def assert_eligible(
    repository_root: Path = REPOSITORY_ROOT,
    paths: Sequence[Path] | None = None,
) -> EligibilityReport:
    """Fail closed when the production path is not eligible."""
    report = audit_eligibility(repository_root, paths)
    if not report.ok:
        detail = "; ".join(
            f"{violation.rule} at {violation.location()} ({violation.detail})"
            for violation in report.violations
        )
        raise EligibilityError(detail or "eligibility audit failed without a located violation")
    return report


def format_eligibility_report(report: EligibilityReport) -> str:
    """Human-readable summary of an eligibility audit."""
    width = max((len(result.check_id) for result in report.results), default=0)
    lines = [f"Scanned {len(report.scanned_paths)} production source(s)."]
    lines.extend(
        f"{result.status:<6} {result.check_id:<{width}}  {result.observed}"
        for result in report.results
    )
    if report.violations:
        lines.append("Violations:")
        lines.extend(
            f"  {violation.rule} {violation.location()}: {violation.detail}"
            for violation in report.violations
        )
    lines.append("RESULT: " + ("PASS" if report.ok else "FAIL"))
    return "\n".join(lines)


def iter_rules() -> Iterable[str]:
    return ELIGIBILITY_RULES
