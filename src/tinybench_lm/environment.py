"""Reproducible-environment contract: exact pins, verified constraints, and a check command.

The competition contract requires a reviewable, reproducible setup (Plan Sections 2.1,
9.1, 10.1, 13 G0/G4). Open-ended runtime ranges let a fresh install silently resolve a
different tokenizer, dataset, or evaluation-harness protocol than the one that produced
our recorded evidence, so every runtime distribution is pinned exactly and cross-checked
against a constraints file that was generated from a verified environment.

Optional GPU/backend choices are reported as informational facts only. Nothing in this
module selects a backend, changes model semantics, or promotes an unmeasured claim.
"""

from __future__ import annotations

import platform
import re
import sys
import tomllib
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
CONSTRAINTS_PATH = REPOSITORY_ROOT / "constraints" / "verified-py311-windows.txt"

#: Distributions that are environment scaffolding rather than project dependencies.
UNPINNED_TOOLING = frozenset({"pip", "setuptools", "wheel", "tinybench-lm"})

#: Optional dependency groups that are not required for the checker to report PASS.
OPTIONAL_GROUPS = frozenset({"test"})

_COMPARISON_OPERATORS = ("===", "==", "~=", "!=", ">=", "<=", ">", "<")
_LOCAL_LABEL = re.compile(r"\+.*$")


class EnvironmentContractError(ValueError):
    """Raised when the declared environment contract itself is malformed."""


@dataclass(frozen=True)
class Pin:
    """One declared dependency requirement."""

    name: str
    operator: str
    version: str
    group: str
    raw: str

    @property
    def is_exact(self) -> bool:
        return self.operator in {"==", "==="} and bool(self.version) and "*" not in self.version

    @property
    def normalized_name(self) -> str:
        return normalize_distribution_name(self.name)


@dataclass(frozen=True)
class CheckResult:
    """One auditable environment check outcome."""

    check_id: str
    requirement: str
    observed: str
    status: str
    reason: str

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


@dataclass(frozen=True)
class EnvironmentReport:
    """Complete dependency-check result plus informational, non-binding facts."""

    results: tuple[CheckResult, ...]
    facts: Mapping[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.failed)

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "results": [result.__dict__ for result in self.results],
            "facts": dict(self.facts),
        }


def normalize_distribution_name(name: str) -> str:
    """PEP 503 name normalization so `PyYAML`, `pyyaml`, and `py_yaml` compare equal."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def public_version(version: str) -> str:
    """Drop the local build label (`2.5.1+cu124` -> `2.5.1`)."""
    return _LOCAL_LABEL.sub("", version.strip())


def local_label(version: str) -> str:
    _, separator, label = version.strip().partition("+")
    return label if separator else ""


def parse_requirement(specifier: str) -> Pin:
    """Parse `name<op>version` requirement text, ignoring extras and markers."""
    text = specifier.split(";", 1)[0].split("#", 1)[0].strip()
    if not text:
        raise EnvironmentContractError("Empty requirement specifier")
    for operator in _COMPARISON_OPERATORS:
        head, found, tail = text.partition(operator)
        if found:
            name = head.split("[", 1)[0].strip()
            if not name:
                raise EnvironmentContractError(f"Requirement has no distribution name: {specifier!r}")
            return Pin(name, operator, tail.strip(), "", specifier.strip())
    return Pin(text.split("[", 1)[0].strip(), "", "", "", specifier.strip())


def version_satisfies_pin(pinned: str, installed: str) -> bool:
    """Exact-pin match. A pin without a local label accepts any local build of it."""
    pinned_text = pinned.strip()
    installed_text = installed.strip()
    if local_label(pinned_text):
        return pinned_text.lower() == installed_text.lower()
    return public_version(pinned_text).lower() == public_version(installed_text).lower()


def load_pyproject(path: Path = PYPROJECT_PATH) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def declared_pins(pyproject: Mapping[str, Any]) -> tuple[Pin, ...]:
    """Runtime and optional-group requirements declared by the project."""
    project = pyproject.get("project", {})
    pins: list[Pin] = []
    for specifier in project.get("dependencies", []):
        parsed = parse_requirement(specifier)
        pins.append(Pin(parsed.name, parsed.operator, parsed.version, "runtime", parsed.raw))
    for group, specifiers in (project.get("optional-dependencies", {}) or {}).items():
        for specifier in specifiers:
            parsed = parse_requirement(specifier)
            pins.append(Pin(parsed.name, parsed.operator, parsed.version, group, parsed.raw))
    return tuple(pins)


def parse_constraints(text: str) -> tuple[Pin, ...]:
    """Parse a constraints file into pins, skipping comments and blank lines."""
    pins: list[Pin] = []
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        parsed = parse_requirement(stripped)
        pins.append(Pin(parsed.name, parsed.operator, parsed.version, "constraints", parsed.raw))
    return tuple(pins)


def load_constraints(path: Path = CONSTRAINTS_PATH) -> tuple[Pin, ...]:
    return parse_constraints(path.read_text(encoding="utf-8"))


def installed_versions() -> dict[str, str]:
    """Normalized distribution name -> installed version for the running interpreter."""
    observed: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if not name:
            continue
        observed[normalize_distribution_name(name)] = distribution.version
    return observed


def required_python(pyproject: Mapping[str, Any]) -> str:
    return str(pyproject.get("project", {}).get("requires-python", "")).strip()


def _release_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in public_version(version).split("."):
        match = re.match(r"^\d+", chunk)
        if not match:
            break
        parts.append(int(match.group()))
    return tuple(parts)


def check_python_version(
    requires_python: str,
    version_info: Sequence[int] = tuple(sys.version_info[:3]),
) -> CheckResult:
    """Preserve declared Python support; fail only on a measured floor violation."""
    observed = ".".join(str(part) for part in version_info)
    pin = parse_requirement(f"python{requires_python}") if requires_python else None
    if pin is None or pin.operator != ">=":
        return CheckResult(
            "python.requires",
            requires_python or "<unset>",
            observed,
            "FAIL",
            "requires-python must declare a supported floor such as >=3.11",
        )
    floor = _release_tuple(pin.version)
    if tuple(version_info)[: len(floor)] >= floor:
        return CheckResult("python.requires", f"python{requires_python}", observed, "PASS", "interpreter meets the declared floor")
    return CheckResult("python.requires", f"python{requires_python}", observed, "FAIL", "interpreter is below the declared floor")


def check_runtime_pins_are_exact(pins: Iterable[Pin]) -> list[CheckResult]:
    """Fail closed on any open-ended runtime range: it permits silent protocol drift."""
    results: list[CheckResult] = []
    for pin in pins:
        if pin.group != "runtime":
            continue
        status = "PASS" if pin.is_exact else "FAIL"
        reason = (
            "runtime requirement is pinned exactly"
            if pin.is_exact
            else "open-ended range can silently resolve incompatible behavior"
        )
        results.append(CheckResult(f"pin.exact.{pin.normalized_name}", pin.raw, pin.operator or "<none>", status, reason))
    return results


def check_installed_pins(pins: Iterable[Pin], installed: Mapping[str, str]) -> list[CheckResult]:
    """Compare each declared pin with the interpreter's installed distribution."""
    results: list[CheckResult] = []
    for pin in pins:
        observed = installed.get(pin.normalized_name)
        optional = pin.group in OPTIONAL_GROUPS
        check_id = f"installed.{pin.group}.{pin.normalized_name}"
        if observed is None:
            status = "OPTIONAL_NOT_INSTALLED" if optional else "FAIL"
            reason = (
                "optional group is not installed in this environment"
                if optional
                else "required distribution is not installed"
            )
            results.append(CheckResult(check_id, pin.raw, "<not installed>", status, reason))
            continue
        if not pin.is_exact:
            results.append(
                CheckResult(check_id, pin.raw, observed, "FAIL", "requirement is not an exact pin, so the install is not reproducible")
            )
            continue
        matches = version_satisfies_pin(pin.version, observed)
        results.append(
            CheckResult(
                check_id,
                pin.raw,
                observed,
                "PASS" if matches else "FAIL",
                "installed version matches the pin" if matches else "installed version differs from the pin",
            )
        )
    return results


def check_constraints_agree(pins: Iterable[Pin], constraints: Iterable[Pin]) -> list[CheckResult]:
    """Every declared pin must appear in the verified constraints file with the same version."""
    constraint_versions = {pin.normalized_name: pin for pin in constraints}
    results: list[CheckResult] = []
    for pin in constraints:
        if not pin.is_exact:
            results.append(
                CheckResult(
                    f"constraints.exact.{pin.normalized_name}",
                    pin.raw,
                    pin.operator or "<none>",
                    "FAIL",
                    "constraints entries must be exact pins",
                )
            )
    for pin in pins:
        check_id = f"constraints.declared.{pin.group}.{pin.normalized_name}"
        constraint = constraint_versions.get(pin.normalized_name)
        if constraint is None:
            results.append(CheckResult(check_id, pin.raw, "<absent>", "FAIL", "declared dependency is missing from the constraints file"))
            continue
        if not pin.is_exact:
            results.append(CheckResult(check_id, pin.raw, constraint.raw, "FAIL", "declared dependency is not an exact pin"))
            continue
        agrees = version_satisfies_pin(pin.version, constraint.version)
        results.append(
            CheckResult(
                check_id,
                pin.raw,
                constraint.raw,
                "PASS" if agrees else "FAIL",
                "constraints file agrees with the declared pin" if agrees else "constraints file contradicts the declared pin",
            )
        )
    return results


def check_constraints_installed(constraints: Iterable[Pin], installed: Mapping[str, str]) -> list[CheckResult]:
    """Constraint entries that are installed must match; absent entries are not failures."""
    results: list[CheckResult] = []
    for pin in constraints:
        observed = installed.get(pin.normalized_name)
        if observed is None:
            continue
        if not pin.is_exact:
            continue
        matches = version_satisfies_pin(pin.version, observed)
        results.append(
            CheckResult(
                f"constraints.installed.{pin.normalized_name}",
                pin.raw,
                observed,
                "PASS" if matches else "FAIL",
                "installed transitive version matches the verified constraint"
                if matches
                else "installed transitive version differs from the verified constraint",
            )
        )
    return results


def optional_backend_facts() -> dict[str, Any]:
    """Informational GPU/backend facts. Never used to select a backend or gate a claim."""
    facts: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "backend_promotion": "NOT_RUN: sustained-throughput and backend promotion measurement is deferred",
    }
    try:
        import torch
    except Exception as error:  # pragma: no cover - torch is a pinned runtime dependency
        facts["torch"] = f"<import failed: {type(error).__name__}>"
        return facts
    facts["torch_version"] = torch.__version__
    facts["torch_cuda_build"] = torch.version.cuda or "cpu-only build"
    try:
        available = bool(torch.cuda.is_available())
    except Exception as error:  # pragma: no cover - driver-dependent
        facts["torch_cuda_available"] = f"<query failed: {type(error).__name__}>"
        return facts
    facts["torch_cuda_available"] = available
    if available:
        facts["cuda_devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return facts


def check_environment(
    pyproject_path: Path = PYPROJECT_PATH,
    constraints_path: Path = CONSTRAINTS_PATH,
    *,
    installed: Mapping[str, str] | None = None,
    include_facts: bool = True,
) -> EnvironmentReport:
    """Run the complete dependency check for a repository checkout."""
    pyproject = load_pyproject(pyproject_path)
    pins = declared_pins(pyproject)
    constraints = load_constraints(constraints_path)
    observed = dict(installed) if installed is not None else installed_versions()

    results: list[CheckResult] = [check_python_version(required_python(pyproject))]
    results.extend(check_runtime_pins_are_exact(pins))
    results.extend(check_installed_pins(pins, observed))
    results.extend(check_constraints_agree(pins, constraints))
    results.extend(check_constraints_installed(constraints, observed))
    facts = optional_backend_facts() if include_facts else {}
    return EnvironmentReport(tuple(results), facts)


def render_constraints(
    installed: Mapping[str, str],
    *,
    python_version: str | None = None,
    platform_name: str | None = None,
) -> str:
    """Render a constraints file from versions that are actually installed."""
    header = [
        "# Verified environment constraints for TinyBench-LM.",
        "# Generated by scripts/check_environment.py --write-constraints from an installed,",
        "# working environment. Every line is an observed fact, not an aspiration.",
        f"# Python: {python_version or platform.python_version()}",
        f"# Platform: {platform_name or platform.platform()}",
        "# Excluded: packaging scaffolding (pip, setuptools, wheel) and the editable project itself.",
        "# Use with: pip install -e . -c constraints/verified-py311-windows.txt",
        "",
    ]
    lines: list[str] = []
    for name in sorted(installed):
        if name in UNPINNED_TOOLING:
            continue
        version = installed[name]
        line = f"{name}=={public_version(version)}"
        label = local_label(version)
        if label:
            line = f"{line}  # verified local build: {name}=={version}"
        lines.append(line)
    return "\n".join(header + lines) + "\n"


def format_report(report: EnvironmentReport) -> str:
    """Human-readable summary of a dependency check."""
    width = max((len(result.check_id) for result in report.results), default=0)
    lines = [f"{result.status:<22} {result.check_id:<{width}}  {result.requirement} -> {result.observed}" for result in report.results]
    lines.append("")
    lines.append("Optional backend facts (informational only, no semantic effect):")
    for key, value in report.facts.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    counts: dict[str, int] = {}
    for result in report.results:
        counts[result.status] = counts.get(result.status, 0) + 1
    lines.append("Summary: " + ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))
    lines.append("RESULT: " + ("PASS" if report.ok else "FAIL"))
    if not report.ok:
        lines.append("Failures:")
        lines.extend(f"  {result.check_id}: {result.reason}" for result in report.failures)
    return "\n".join(lines)
