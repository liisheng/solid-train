from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from hypothesis import given, settings, strategies as st

from tinybench_lm.environment import (
    CONSTRAINTS_PATH,
    PYPROJECT_PATH,
    CheckResult,
    Pin,
    check_environment,
    check_installed_pins,
    check_python_version,
    check_runtime_pins_are_exact,
    declared_pins,
    installed_versions,
    load_constraints,
    load_pyproject,
    normalize_distribution_name,
    parse_constraints,
    parse_requirement,
    public_version,
    render_constraints,
    required_python,
    version_satisfies_pin,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPOSITORY_ROOT / "scripts" / "check_environment.py"
ENVIRONMENT_DOC = REPOSITORY_ROOT / "docs" / "ENVIRONMENT.md"


def _runtime_pins() -> tuple[Pin, ...]:
    return tuple(pin for pin in declared_pins(load_pyproject()) if pin.group == "runtime")


def _fixture_project(
    directory: Path,
    dependencies: list[str],
    optional: dict[str, list[str]] | None = None,
    *,
    constraints: list[str] | None = None,
    requires_python: str = ">=3.11",
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    optional_section = ""
    if optional:
        rendered = "\n".join(
            f"{group} = [{', '.join(json.dumps(item) for item in items)}]" for group, items in optional.items()
        )
        optional_section = f"\n[project.optional-dependencies]\n{rendered}\n"
    pyproject = directory / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'name = "fixture"\n'
        'version = "0.0.0"\n'
        f'requires-python = "{requires_python}"\n'
        f"dependencies = [{', '.join(json.dumps(item) for item in dependencies)}]\n"
        f"{optional_section}",
        encoding="utf-8",
    )
    constraint_path = directory / "constraints.txt"
    constraint_lines = constraints if constraints is not None else dependencies
    constraint_path.write_text("# fixture\n" + "\n".join(constraint_lines) + "\n", encoding="utf-8")
    return pyproject, constraint_path


def _statuses(results: tuple[CheckResult, ...] | list[CheckResult]) -> dict[str, str]:
    return {result.check_id: result.status for result in results}


# --- Declared contract -------------------------------------------------------------


def test_runtime_dependencies_are_pinned_exactly() -> None:
    """A fresh install must not be free to resolve a different protocol version."""
    pins = _runtime_pins()
    assert pins, "the project must still declare runtime dependencies"
    unpinned = [pin.raw for pin in pins if not pin.is_exact]
    assert unpinned == []
    results = check_runtime_pins_are_exact(pins)
    assert len(results) == len(pins)
    assert [result.status for result in results] == ["PASS"] * len(pins)


def test_python_support_floor_is_preserved() -> None:
    pyproject = load_pyproject()
    assert required_python(pyproject) == ">=3.11"
    assert check_python_version(">=3.11", (3, 11, 0)).status == "PASS"
    assert check_python_version(">=3.11", (3, 12, 5)).status == "PASS"
    assert check_python_version(">=3.11", (3, 10, 14)).status == "FAIL"


def test_test_tooling_is_a_separate_optional_group() -> None:
    """Property testing tooling is pinned but must not enter the runtime contract."""
    pins = declared_pins(load_pyproject())
    runtime_names = {pin.normalized_name for pin in pins if pin.group == "runtime"}
    test_pins = {pin.normalized_name: pin for pin in pins if pin.group == "test"}
    assert "hypothesis" in test_pins
    assert "pytest" in test_pins
    assert runtime_names.isdisjoint({"hypothesis", "pytest"})
    assert all(pin.is_exact for pin in test_pins.values())


def test_declared_pins_appear_in_the_verified_constraints_file() -> None:
    constraint_versions = {pin.normalized_name: pin.version for pin in load_constraints()}
    for pin in declared_pins(load_pyproject()):
        assert pin.normalized_name in constraint_versions, pin.raw
        assert version_satisfies_pin(pin.version, constraint_versions[pin.normalized_name]), pin.raw


# --- Checker behavior --------------------------------------------------------------


def test_open_ended_range_is_reported_as_a_failure(tmp_path: Path) -> None:
    pyproject, constraints = _fixture_project(
        tmp_path / "open_ended",
        ["numpy>=1.26"],
        constraints=["numpy==1.26.4"],
    )
    report = check_environment(pyproject, constraints, installed={"numpy": "1.26.4"}, include_facts=False)
    assert not report.ok
    reasons = " ".join(result.reason for result in report.failures)
    assert "silently" in reasons
    assert _statuses(report.results)["pin.exact.numpy"] == "FAIL"


def test_matching_exact_pins_pass(tmp_path: Path) -> None:
    pyproject, constraints = _fixture_project(tmp_path / "exact", ["numpy==1.26.4", "tqdm==4.67.1"])
    report = check_environment(
        pyproject,
        constraints,
        installed={"numpy": "1.26.4", "tqdm": "4.67.1"},
        include_facts=False,
    )
    assert report.ok, [result.__dict__ for result in report.failures]


def test_missing_required_distribution_fails_closed(tmp_path: Path) -> None:
    pyproject, constraints = _fixture_project(tmp_path / "missing", ["numpy==1.26.4"])
    report = check_environment(pyproject, constraints, installed={}, include_facts=False)
    assert _statuses(report.results)["installed.runtime.numpy"] == "FAIL"
    assert not report.ok


def test_divergent_installed_version_fails_closed(tmp_path: Path) -> None:
    pyproject, constraints = _fixture_project(tmp_path / "divergent", ["numpy==1.26.4"])
    report = check_environment(pyproject, constraints, installed={"numpy": "2.1.0"}, include_facts=False)
    assert _statuses(report.results)["installed.runtime.numpy"] == "FAIL"
    assert not report.ok


def test_constraints_file_contradiction_fails_closed(tmp_path: Path) -> None:
    pyproject, constraints = _fixture_project(
        tmp_path / "contradiction",
        ["numpy==1.26.4"],
        constraints=["numpy==2.1.0"],
    )
    report = check_environment(pyproject, constraints, installed={"numpy": "1.26.4"}, include_facts=False)
    statuses = _statuses(report.results)
    assert statuses["constraints.declared.runtime.numpy"] == "FAIL"
    assert statuses["constraints.installed.numpy"] == "FAIL"


def test_absent_optional_group_is_not_a_failure(tmp_path: Path) -> None:
    pyproject, constraints = _fixture_project(
        tmp_path / "optional",
        ["numpy==1.26.4"],
        {"test": ["hypothesis==6.130.5"]},
        constraints=["numpy==1.26.4", "hypothesis==6.130.5"],
    )
    report = check_environment(pyproject, constraints, installed={"numpy": "1.26.4"}, include_facts=False)
    assert _statuses(report.results)["installed.test.hypothesis"] == "OPTIONAL_NOT_INSTALLED"
    assert report.ok


def test_local_build_labels_are_accepted_only_when_compatible() -> None:
    assert version_satisfies_pin("2.5.1", "2.5.1+cu124")
    assert version_satisfies_pin("2.5.1+cu124", "2.5.1+cu124")
    assert not version_satisfies_pin("2.5.1+cu121", "2.5.1+cu124")
    assert not version_satisfies_pin("2.5.0", "2.5.1+cu124")


def test_requirement_parsing_handles_extras_markers_and_comments() -> None:
    assert parse_requirement("lm-eval==0.4.12").name == "lm-eval"
    assert parse_requirement("datasets[audio]==3.2.0").name == "datasets"
    assert parse_requirement("torch==2.5.1 ; python_version >= '3.11'").version == "2.5.1"
    assert parse_requirement("numpy>=1.26").is_exact is False
    assert parse_requirement("numpy==1.26.*").is_exact is False


def test_rendered_constraints_exclude_scaffolding_and_round_trip() -> None:
    installed = {"numpy": "1.26.4", "torch": "2.5.1+cu124", "pip": "24.0", "tinybench-lm": "0.1.0"}
    rendered = render_constraints(installed, python_version="3.11.4", platform_name="Windows-fixture")
    parsed = {pin.normalized_name: pin.version for pin in parse_constraints(rendered)}
    assert parsed == {"numpy": "1.26.4", "torch": "2.5.1"}
    assert "2.5.1+cu124" in rendered  # the observed local build stays recorded as a comment
    assert "3.11.4" in rendered


# --- Generated coverage of the comparison logic ------------------------------------


@st.composite
def _pin_cases(draw: st.DrawFn) -> tuple[Pin, str, bool]:
    name = draw(st.sampled_from(("numpy", "PyYAML", "lm_eval", "typing-extensions", "torch")))
    release = draw(st.lists(st.integers(min_value=0, max_value=99), min_size=1, max_size=3))
    pinned = ".".join(str(part) for part in release)
    local = draw(st.sampled_from(("", "+cu124", "+cpu")))
    drift = draw(st.booleans())
    installed_release = list(release)
    if drift:
        installed_release[-1] += 1
    installed = ".".join(str(part) for part in installed_release) + local
    pin = Pin(name, "==", pinned, "runtime", f"{name}=={pinned}")
    return pin, installed, not drift


@given(case=_pin_cases())
@settings(max_examples=50, deadline=None, derandomize=True)
def test_installed_pin_check_agrees_with_version_comparison(case: tuple[Pin, str, bool]) -> None:
    """Local invariant: a pin passes exactly when the installed release matches it."""
    pin, installed, should_pass = case
    results = check_installed_pins([pin], {normalize_distribution_name(pin.name): installed})
    assert len(results) == 1
    assert (results[0].status == "PASS") is should_pass
    assert public_version(installed).startswith(pin.version) or not should_pass


@given(
    name=st.sampled_from(("typing_extensions", "typing-extensions", "Typing.Extensions", "TYPING__EXTENSIONS")),
    version=st.sampled_from(("4.16.0", "4.16.0+local")),
)
@settings(max_examples=16, deadline=None, derandomize=True)
def test_distribution_name_normalization_is_stable(name: str, version: str) -> None:
    """Local invariant: PEP 503 equivalent spellings resolve to the same installed record."""
    results = check_installed_pins(
        [Pin(name, "==", "4.16.0", "runtime", f"{name}==4.16.0")],
        {"typing-extensions": version},
    )
    assert results[0].status == "PASS"


def test_case_only_spelling_differences_resolve_to_the_same_record() -> None:
    for spelling in ("PyYAML", "pyyaml", "PYYAML"):
        results = check_installed_pins(
            [Pin(spelling, "==", "6.0.2", "runtime", f"{spelling}==6.0.2")],
            {"pyyaml": "6.0.2"},
        )
        assert results[0].status == "PASS", spelling


# --- The real environment and the recorded facts -----------------------------------


def test_current_environment_satisfies_the_declared_contract() -> None:
    report = check_environment(include_facts=False)
    assert report.ok, json.dumps([result.__dict__ for result in report.failures], indent=2)


def test_check_command_runs_and_reports_pass() -> None:
    completed = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--json"],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["results"], "the report must contain checks"
    assert payload["facts"]["python_implementation"] == "CPython"
    assert "backend_promotion" in payload["facts"]


def test_recorded_environment_facts_match_the_installed_environment() -> None:
    """Documented versions must be observed facts, not transcription."""
    assert ENVIRONMENT_DOC.is_file()
    content = ENVIRONMENT_DOC.read_text(encoding="utf-8")
    assert "scripts\\check_environment.py" in content
    installed = installed_versions()
    documented = re.findall(r"^\| ([a-z0-9_.\-]+) \| ([0-9][^|]*) \| ([0-9][^|]*) \|$", content, flags=re.MULTILINE)
    assert documented, "the verified-facts table must record at least one dependency"
    for name, pin, observed in documented:
        normalized = normalize_distribution_name(name)
        assert normalized in installed, name
        assert version_satisfies_pin(pin.strip(), installed[normalized]), name
        assert observed.strip() == installed[normalized], name


def test_readme_documents_the_dependency_check_command() -> None:
    content = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    assert "scripts\\check_environment.py" in content
    assert CONSTRAINTS_PATH.name in content
    assert PYPROJECT_PATH.is_file()
