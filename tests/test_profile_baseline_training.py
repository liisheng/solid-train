"""Synthetic measurements and real CPU subprocess checks; no train.py launch."""
import json
import math
import sys

import pytest

from scripts import profile_baseline_training as profile


def identity():
    return {
        "run_id": "baseline-fixture", "exposure_content_hash": "exposure-fixture",
        "validation_schedule_hash": "dev-fixture", "total_updates": 3815,
        "warmup_updates": 38, "validation_mode": "full-dev", "eval_interval": 100,
        "save_interval": 100, "micro_batch_size": 8, "sequence_length": 1024,
        "gradient_accumulation": 32, "peak_lr": 0.0006, "decay_updates": 382,
        "paths": {"model_config": "model.json", "dev_manifest": "dev.manifest.json",
                  "dev_schedule": "dev.json", "exposure_plan": "plan.json",
                  "component_1": "one.json", "component_2": "two.json"},
    }


def metric_rows(durations=None):
    rows, start = [], 10.0
    for index, seconds in enumerate(durations or [10.0] * 240):
        record = {"update_index": index, "step_seconds": seconds, "loss_tokens_per_update": 262144,
                  "tokens_per_second": 262144 / seconds, "peak_vram_gib": 5.0,
                  "update_started_seconds": start, "optimizer_finished_seconds": start + seconds,
                  "run_id": "run-fixture", "invocation_id": "invocation-fixture",
                  "schedule_content_hash": "exposure-fixture", "schedule_cursor": (index + 1) * 256,
                  "consumed_loss_tokens": (index + 1) * 262144}
        if index == 0 or (index + 1) % 100 == 0:
            record.update({"validation_loss": 1.0, "validation_mode": "full-dev",
                           "validation_sequence_count": 753, "validation_scored_token_count": 771072,
                           "validation_batch_count": 95, "validation_schedule_content_hash": "dev-fixture"})
        rows.append(record)
        start += seconds + 0.1
    return rows


def test_command_preserves_recipe_and_three_independent_bounds(tmp_path):
    command = profile.build_profile_command(identity(), tmp_path / "train")
    for flag, value in {"--steps": "3815", "--warmup-steps": "38", "--decay-updates": "382",
                        "--validation-mode": "full-dev", "--eval-interval": "100",
                        "--save-interval": "100", "--stop-after-updates": "800",
                        "--stop-after-training-seconds": "2400.0"}.items():
        assert command[command.index(flag) + 1] == value
    assert "--resume" not in command
    assert profile.OUTER_WATCHDOG_SECONDS == 3600


@pytest.mark.parametrize("field,value", [("total_updates", 2048), ("warmup_updates", 2),
                                       ("validation_mode", "sampled"), ("eval_interval", 99),
                                       ("micro_batch_size", 4), ("peak_lr", 0.001)])
def test_wrong_fixed_identity_is_rejected(field, value):
    with pytest.raises(profile.ProfileError, match=field):
        profile.build_profile_command({**identity(), field: value}, profile.ROOT / "ignored")


def test_warmup_exclusion_elapsed_window_and_weighted_nearest_rank_math():
    report = profile.analyze_rows(metric_rows([10.] * 138 + [20.] * 51))
    assert report["sample_count"] == 151
    assert report["window_seconds"] == 2020
    assert report["optimizer_seconds_all_updates"] == 2400
    assert report["first_included_update_index"] == 38
    assert report["throughput_p10"] == pytest.approx(13107.2)
    assert report["throughput_median"] == pytest.approx(26214.4)
    assert report["throughput_p90"] == pytest.approx(26214.4)
    assert report["weighted_tokens_per_second"] == pytest.approx(151 * 262144 / 2020)
    assert report["elapsed_training_window_seconds"] == pytest.approx(2035)
    assert report["full_dev_completed_updates"] == [1, 100]


@pytest.mark.parametrize("mutation", [
    lambda r: r[38].update(step_seconds=0), lambda r: r[38].update(tokens_per_second=math.nan),
    lambda r: r[38].update(tokens_per_second=1), lambda r: r[38].update(loss_tokens_per_update=123),
    lambda r: r[38].update(peak_vram_gib=-1), lambda r: r[38].update(update_started_seconds=-1),
    lambda r: r[38].update(optimizer_finished_seconds=100000),
    lambda r: r[99].update(validation_mode="sampled"), lambda r: r[99].update(validation_sequence_count=1),
    lambda r: r[99].pop("validation_loss"), lambda r: r.pop(50), lambda r: r.append(dict(r[38])),
    lambda r: r.__setitem__(slice(1, 3), list(reversed(r[1:3]))),
])
def test_invalid_measurements_fail_closed(mutation):
    rows = metric_rows()
    mutation(rows)
    with pytest.raises(profile.ProfileError):
        profile.analyze_rows(rows)


def test_short_window_early_exit_and_ignored_stop_rejected():
    for durations, reason in (([10.] * 100, "invalid sustained window"),
                              ([10.] * 230, "stop boundary"), ([10.] * 241, "continued past"),
                              ([5.] * 38 + [1300.] * 3, "exceeds 3600")):
        with pytest.raises(profile.ProfileError, match=reason):
            profile.analyze_rows(metric_rows(durations))


@pytest.fixture
def artifacts(tmp_path):
    phase = {"training_optimizer_seconds": 2400., "data_wait_seconds": 12.,
             "validation_seconds": 20., "checkpoint_seconds": 10., "process_wall_seconds": 2600.}
    history = {**phase, "schema": "training_phase_timing_invocation_v1", "invocation_index": 0,
               "started_at_update": 0, "completed_updates": 240, "resume_checkpoint": None,
               "metric_invocation_id": "invocation-fixture"}
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in metric_rows()) + "\n")
    for name, data in {"phase_timing.json": phase, "phase_timing_history.jsonl": history,
                       "runner_identity.json": identity(), "latest.pt.manifest.json": {
                           "bytes": 7, "updates_completed": 240, "run_id": "run-fixture"}}.items():
        (tmp_path / name).write_text(json.dumps(data) + "\n")
    (tmp_path / "latest.pt").write_bytes(b"fixture")
    return tmp_path


def test_artifact_timing_reconciliation_and_hashes(artifacts):
    report = profile.validate_training_artifacts(artifacts, identity(), 3000.)
    assert report["trainer_phase_timing"]["data_wait_seconds"] == 12
    assert len(report["artifact_sha256"]) == 5


@pytest.mark.parametrize("name,key,value", [
    ("phase_timing.json", "training_optimizer_seconds", 2399),
    ("phase_timing.json", "data_wait_seconds", -1), ("phase_timing.json", "checkpoint_seconds", 0),
    ("phase_timing.json", "validation_seconds", math.nan),
    ("phase_timing_history.jsonl", "completed_updates", 239),
    ("phase_timing_history.jsonl", "metric_invocation_id", "wrong"),
    ("latest.pt.manifest.json", "updates_completed", 239),
])
def test_artifact_inconsistency_rejected(artifacts, name, key, value):
    path = artifacts / name
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data) + "\n")
    with pytest.raises(profile.ProfileError):
        profile.validate_training_artifacts(artifacts, identity(), 3000.)


@pytest.fixture
def mocked_telemetry(monkeypatch):
    monkeypatch.setattr(profile, "sample_telemetry", lambda: {"sample": "cpu-fixture"})
    monkeypatch.setattr(profile, "summarize_telemetry", lambda samples: {"status": "MEASURED"})
    monkeypatch.setattr(profile, "TELEMETRY_CADENCE_SECONDS", 0.01)


def test_main_is_plan_only_and_rejects_reuse(tmp_path, monkeypatch, mocked_telemetry):
    output = tmp_path / "runs/verification/g4/fixture"
    monkeypatch.setattr(profile, "ROOT", tmp_path)
    monkeypatch.setattr(profile, "prepare", lambda **kwargs: identity())
    monkeypatch.setattr(profile, "snapshot_source", lambda: {"source.py": "fixture"})
    monkeypatch.setattr(profile, "snapshot_inputs", lambda identity: {"input": "fixture"})
    monkeypatch.setattr(profile, "execute", lambda *args: pytest.fail("dry-run launched training"))
    assert profile.main(["--output-dir", str(output)]) == 0
    plan = json.loads((output / "profile_plan.json").read_text())
    assert plan["execute"] is False and "--stop-after-updates" in plan["identity"]["command"]
    assert not (output / "train").exists()
    with pytest.raises(profile.ProfileError, match="fresh and unique"):
        profile.main(["--output-dir", str(output)])


def test_wrong_config_fails_preparation_with_retained_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "ROOT", tmp_path)
    output = tmp_path / "runs/verification/g4/wrong-input"
    with pytest.raises(ValueError, match="bound to baseline"):
        profile.main(["--output-dir", str(output), "--config", str(tmp_path / "wrong.yaml")])
    receipt = json.loads((output / "wrapper_receipt.json").read_text())
    assert receipt["status"] == "FAIL" and receipt["preparation_seconds"] > 0
    assert not (output / "train").exists()


def test_output_cannot_target_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "ROOT", tmp_path)
    output = tmp_path / "runs/reduced_campaign/reduced_baseline_v1/run"
    with pytest.raises(profile.ProfileError, match="engineering directory"):
        profile.main(["--output-dir", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("kind", ["child_failure", "launch_failure", "watchdog", "sampler_failure"])
def test_failures_retain_outer_cost_and_stop_child(tmp_path, monkeypatch, mocked_telemetry, kind):
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    if kind == "launch_failure":
        command = [str(tmp_path / "nonexistent-executable")]
    elif kind == "child_failure":
        command = [sys.executable, "-c", "import sys; print('bad'); sys.exit(7)"]
    elif kind == "sampler_failure":
        def fail_sample():
            raise RuntimeError("simulated collector failure")
        monkeypatch.setattr(profile, "sample_telemetry", fail_sample)
    with pytest.raises(profile.ProfileError):
        profile.execute(tmp_path, command, identity(), watchdog_seconds=0.1 if kind == "watchdog" else 5)
    history = [json.loads(s) for s in (tmp_path / "outer_invocation_history.jsonl").read_text().splitlines()]
    assert len(history) == 1 and history[0]["outer_wall_seconds"] > 0
    assert len(history[0]["stdout_sha256"]) == len(history[0]["stderr_sha256"]) == 64
    assert json.loads((tmp_path / "measurement.json").read_text())["status"] == "FAIL"
    assert json.loads((tmp_path / "active_process.json").read_text())["status"] == "EXITED"
    if kind == "child_failure":
        assert history[0]["returncode"] == 7
    if kind == "launch_failure":
        assert history[0]["returncode"] is None
    if kind == "watchdog":
        assert history[0]["watchdog_expired"] is True
