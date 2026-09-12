"""Selection custody and successor isolation without production inputs."""
import copy
import hashlib
import json

import pytest

from scripts.integrate_selected_baseline import verify_selection
from scripts import run_reduced_baseline as runner
from tinybench_lm.baseline_contract import SELECTED_BASELINE_SHA256
from tinybench_lm.exposure import CompositeExposure
from tinybench_lm.schedule import ScheduleResumeError
from test_exposure import _exposure


def test_selected_recipe_preserves_control_math_and_changes_contract():
    old = runner._verified_baseline_config()
    new = runner._verified_baseline_config(config_path=runner.SELECTED_CONFIG)
    for section in ("horizon", "batch", "optimizer", "seed", "precision", "data", "tokenizer"):
        assert new[section] == old[section]
    assert new["learning_rate"]["peak_lr"] == old["learning_rate"]["peak_lr"] == 0.0006
    assert new["selection"]["experiment_weights_permitted"] is False
    assert runner._norm_hash(runner.SELECTED_CONFIG) == SELECTED_BASELINE_SHA256
    changed = copy.deepcopy(new)
    changed["learning_rate"]["peak_lr"] = 0.001
    with pytest.raises(runner.RunnerError, match="trusted frozen semantics"):
        runner._verified_baseline_config(changed)


def test_successor_exposure_changes_training_identity_and_rejects_old_cursor():
    old = _exposure()
    new = CompositeExposure(old.components, SELECTED_BASELINE_SHA256, recipe_sha256_normalized_lf=SELECTED_BASELINE_SHA256)
    assert new.source_counts == old.source_counts
    assert new.content_hash != old.content_hash
    with pytest.raises(ScheduleResumeError):
        new.cursor().load_state_dict(old.cursor(2).state_dict())


def test_selection_requires_exact_reviewed_report():
    report = {"stage": "final", "status": "CONTROL", "selected_treatment": {"lr": 0.0006, "mixture": "base"}}
    settings = {"source_report_canonical_sha256": hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    verify_selection(report, settings)
    changed = copy.deepcopy(report)
    changed["selected_treatment"]["mixture"] = "edu"
    with pytest.raises(ValueError, match="reviewed selection"):
        verify_selection(changed, settings)


def test_selected_prepare_rejects_unregistered_contract(tmp_path):
    fake = tmp_path / "baseline_reduced_v2.yaml"
    fake.write_bytes(runner.SELECTED_CONFIG.read_bytes())
    with pytest.raises(runner.RunnerError, match="bound to"):
        runner.prepare(config_path=fake)
