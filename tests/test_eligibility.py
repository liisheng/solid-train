from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tinybench_lm.eligibility import (
    ELIGIBILITY_RULES,
    RULE_HOSTED_INFERENCE,
    RULE_KNOWLEDGE_TRANSFER,
    RULE_PRETRAINED_WEIGHTS,
    RULE_REMOTE_WEIGHT_FETCH,
    EligibilityError,
    assert_eligible,
    audit_eligibility,
    format_eligibility_report,
    production_python_paths,
    scan_source,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Ineligible constructs, kept as fixture text so the audit never sees them in the
# production path. Each entry maps a rule to a source line that must be reported.
INELIGIBLE_FIXTURES: dict[str, tuple[str, ...]] = {
    RULE_PRETRAINED_WEIGHTS: (
        "import transformers\nmodel = transformers.AutoModel.from_pretrained('gpt2')\n",
        "def build():\n    return load_pretrained('checkpoint')\n",
    ),
    RULE_REMOTE_WEIGHT_FETCH: (
        "import torch\nstate = torch.hub.load_state_dict_from_url('https://example/w.pt')\n",
        "import torch\nmodel = torch.hub.load('repo', 'model')\n",
        "from huggingface_hub import hf_hub_download\npath = hf_hub_download('r', 'w.pt')\n",
        "import requests\nblob = requests.get('https://example/w.pt').content\n",
    ),
    RULE_KNOWLEDGE_TRANSFER: (
        "def loss(student_logits, teacher_logits):\n    return student_logits - teacher_logits\n",
        "def distillation_step(batch):\n    return batch\n",
        "kd_loss = 0.0\n",
    ),
    RULE_HOSTED_INFERENCE: (
        "import openai\nclient = openai.OpenAI()\n",
        "from transformers import pipeline\nscorer = pipeline('text-generation')\n",
    ),
}

# Constructs that must stay allowed: local checkpoint loading for resume, evaluation,
# and generation, plus the required evaluation harness and corpus tooling.
ELIGIBLE_FIXTURES: tuple[str, ...] = (
    "import torch\n"
    "checkpoint = torch.load('runs/pilot/best.pt', map_location='cpu', weights_only=False)\n"
    "model.load_state_dict(checkpoint['model'])\n",
    "import lm_eval\nresults = lm_eval.simple_evaluate(model=adapter, tasks=['piqa'])\n",
    "from datasets import load_dataset\ncorpus = load_dataset('wikipedia', split='train')\n",
    "from tokenizers import Tokenizer\ntokenizer = Tokenizer.from_file('tokenizer.json')\n",
)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.4, 2.5**
def test_repository_production_path_is_eligible() -> None:
    report = audit_eligibility(REPOSITORY_ROOT)

    scanned = set(report.scanned_paths)
    assert {"train.py", "generate.py", "evaluate.py"} <= scanned
    assert "src/tinybench_lm/model.py" in scanned
    assert "scripts/count_params.py" in scanned
    assert report.violations == (), [violation.to_dict() for violation in report.violations]
    assert report.ok, report.to_dict()
    assert {result.check_id for result in report.results} == {
        f"eligibility.{rule}" for rule in ELIGIBILITY_RULES
    } | {"eligibility.scanned_paths"}
    assert assert_eligible(REPOSITORY_ROOT) is not None
    assert "RESULT: PASS" in format_eligibility_report(report)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2**
@pytest.mark.parametrize(
    ("rule", "source"),
    [(rule, source) for rule, sources in INELIGIBLE_FIXTURES.items() for source in sources],
)
def test_audit_reports_every_ineligible_construct(rule: str, source: str) -> None:
    violations = scan_source(source, path="fixture.py")

    assert violations, source
    assert rule in {violation.rule for violation in violations}
    for violation in violations:
        assert violation.path == "fixture.py"
        assert 1 <= violation.line <= len(source.splitlines())


# **Validates: Requirements 3.3**
@pytest.mark.parametrize("source", ELIGIBLE_FIXTURES)
def test_audit_preserves_explicit_local_checkpoint_and_harness_usage(source: str) -> None:
    assert scan_source(source, path="fixture.py") == ()


# **Validates: Requirements 1.1, 2.1, 2.2**
def test_audit_fails_closed_on_an_ineligible_production_path(tmp_path: Path) -> None:
    module = tmp_path / "ineligible.py"
    module.write_text(
        "import transformers\n\n\ndef build():\n"
        "    return transformers.AutoModel.from_pretrained('gpt2')\n",
        encoding="utf-8",
    )

    report = audit_eligibility(tmp_path, paths=[module])

    assert not report.ok
    assert [violation.rule for violation in report.violations] == [RULE_PRETRAINED_WEIGHTS]
    assert report.violations[0].line == 5
    failed = {result.check_id for result in report.failures}
    assert failed == {f"eligibility.{RULE_PRETRAINED_WEIGHTS}"}
    with pytest.raises(EligibilityError, match=RULE_PRETRAINED_WEIGHTS):
        assert_eligible(tmp_path, paths=[module])


# **Validates: Requirements 2.4, 2.5**
def test_empty_production_path_is_a_failure_not_a_silent_pass(tmp_path: Path) -> None:
    report = audit_eligibility(tmp_path)

    assert report.scanned_paths == ()
    assert report.violations == ()
    assert not report.ok
    assert {result.check_id for result in report.failures} == {"eligibility.scanned_paths"}


# **Validates: Requirements 1.1, 2.1, 2.2**
@given(
    module_name=st.from_regex(r"\A[a-z]{3,8}\Z", fullmatch=True),
    padding=st.integers(min_value=0, max_value=4),
    rule_index=st.integers(min_value=0, max_value=len(ELIGIBILITY_RULES) - 1),
)
@settings(max_examples=12, deadline=None, derandomize=True)
def test_injecting_an_ineligible_construct_is_always_reported(
    module_name: str, padding: int, rule_index: int
) -> None:
    rule = ELIGIBILITY_RULES[rule_index]
    injected = INELIGIBLE_FIXTURES[rule][0]
    clean = "import torch\n\n\ndef build(path):\n    return torch.load(path, map_location='cpu')\n"
    prefix = "\n" * padding
    path = f"{module_name}.py"

    assert scan_source(prefix + clean, path=path) == ()
    violations = scan_source(prefix + clean + injected, path=path)
    assert rule in {violation.rule for violation in violations}
    assert all(violation.line > padding for violation in violations)


# **Validates: Requirements 3.2, 3.3**
def test_production_path_enumeration_covers_entry_points_library_and_scripts() -> None:
    paths = production_python_paths(REPOSITORY_ROOT)
    relative = {path.relative_to(REPOSITORY_ROOT).as_posix() for path in paths}

    assert {"train.py", "generate.py", "evaluate.py"} <= relative
    assert {"src/tinybench_lm/checkpoint.py", "src/tinybench_lm/provenance.py"} <= relative
    assert {"scripts/audit_eligibility.py", "scripts/verify_provenance.py"} <= relative
    assert all(path.suffix == ".py" for path in paths)
    assert not any("tests/" in path for path in relative)
