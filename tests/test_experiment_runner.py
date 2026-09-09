import json
from pathlib import Path

import pytest

from scripts import run_experiment as runner


@pytest.fixture(autouse=True)
def isolated_launch_inputs(monkeypatch):
    from datetime import datetime, timezone

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 9, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(runner, "datetime", FixedDatetime)
    monkeypatch.setattr(runner, "check_bundle", lambda *a, **k: {})


def identity(tmp_path: Path, job="S0"):
    return {
        "identity_schema": "pre_campaign_runner_v2",
        "run_id": "experiment-test",
        "job": job,
        "spec": {"lane": "rtx_4070", "seed": 1001, "lr": 0.0006, "updates": 382, "warmup": 4, "stable": 340, "decay": 38, "mixture": "base"},
        "input_hashes": {},
    }


def test_command_contains_all_materialized_inputs_and_slice_reporting(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    paths = {}
    for name in ("train_manifest", "validation_manifest", "validation_schedule", "shard_root"):
        path = bundle / name
        path.mkdir() if name == "shard_root" else path.write_text("fixture", encoding="utf-8")
        paths[name] = path
    monkeypatch.setattr(runner, "inputs", lambda: {k: v for k, v in paths.items() if k != "shard_root"})
    monkeypatch.setattr(runner, "SHARDS", paths["shard_root"])
    schedule = bundle / "final.base.schedule.json"
    schedule.write_text("fixture", encoding="utf-8")
    command = runner.command_for(bundle, "S0", tmp_path / "run", identity(tmp_path))
    assert "--train-manifest" in command and "--validation-manifest" in command
    assert "--validation-schedule" in command and "--train-schedule" in command
    assert "--experiment-slice-reporting" in command


def test_launch_without_execute_does_not_touch_gpu_or_process(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "inspect_hardware", lambda: pytest.fail("GPU inspected"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("process started"))
    with pytest.raises(ValueError, match="PLAN_ONLY"):
        runner.launch(tmp_path, "S0", identity(tmp_path), lane="rtx_4070", execute=False, smoke=True, resume=False)


def test_launch_rejects_wrong_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "inspect_hardware", lambda: {"name": "RTX 3070 Ti", "uuid": "u", "total_vram_bytes": 12, "free_vram_bytes": 12, "host_ram_bytes": 1, "bf16": True})
    with pytest.raises(ValueError, match="GPU"):
        runner.launch(tmp_path, "S0", identity(tmp_path), lane="rtx_4070", execute=True, smoke=True, resume=False)


def test_full_launch_requires_verified_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "inspect_hardware", lambda: {"name": "RTX 4070", "uuid": "u", "total_vram_bytes": 10**9, "free_vram_bytes": 10**9, "host_ram_bytes": 1, "bf16": True})
    with pytest.raises(ValueError, match="missing endpoint|smoke"):
        runner.launch(tmp_path, "S0", identity(tmp_path), lane="rtx_4070", execute=True, smoke=False, resume=False)


def test_existing_fresh_run_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "inspect_hardware", lambda: {"name": "RTX 4070", "uuid": "u", "total_vram_bytes": 10**9, "free_vram_bytes": 10**9, "host_ram_bytes": 1, "bf16": True})
    (tmp_path / "smoke" / "S0").mkdir(parents=True)
    with pytest.raises(ValueError, match="exists"):
        runner.launch(tmp_path, "S0", identity(tmp_path), lane="rtx_4070", execute=True, smoke=True, resume=False)


class FakeChild:
    def __init__(self, code=1, timeout=False):
        self.code, self.timeout, self.terminated = code, timeout, False
    def wait(self, timeout=None):
        if self.timeout and not self.terminated:
            raise runner.subprocess.TimeoutExpired("x", timeout)
        return self.code
    def poll(self): return None if self.timeout and not self.terminated else self.code
    def terminate(self): self.terminated = True


class FakeLedger:
    def __init__(self, *args, **kwargs):
        self.settled = None
        self.path = Path(kwargs.get("root", ".")) / "fake-ledger.json"
    def remaining_seconds(self): return 6 * 3600
    def reserve(self, *args, **kwargs): return "token"
    def settle(self, token, **kwargs): self.settled = kwargs


def _hardware(monkeypatch):
    monkeypatch.setattr(runner, "inspect_hardware", lambda: {"name": "RTX 4070", "uuid": "u", "total_vram_bytes": 10**9, "free_vram_bytes": 10**9, "host_ram_bytes": 1, "bf16": True})
    monkeypatch.setattr(runner, "BudgetLedger", FakeLedger)


def test_failed_child_writes_exit_and_settles(tmp_path, monkeypatch):
    _hardware(monkeypatch)
    child = FakeChild(code=3)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: child)
    with pytest.raises(ValueError, match="failed"):
        runner.launch(tmp_path, "S0", identity(tmp_path), lane="rtx_4070", execute=True, smoke=True, resume=False)
    exits = list((tmp_path / "smoke" / "S0" / "invocations").glob("*/exit.json"))
    assert exits and json.loads(exits[0].read_text())["exit_code"] == 3


def test_timeout_terminates_child_and_settles(tmp_path, monkeypatch):
    _hardware(monkeypatch)
    child = FakeChild(timeout=True)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: child)
    with pytest.raises(ValueError, match="exceeded"):
        runner.launch(tmp_path, "S0", identity(tmp_path), lane="rtx_4070", execute=True, smoke=True, resume=False)
    assert child.terminated


def test_smoke_rejects_reserved_memory_near_capacity(tmp_path, monkeypatch):
    _hardware(monkeypatch)
    ledger = FakeLedger()
    monkeypatch.setattr(runner, "BudgetLedger", lambda *a, **k: ledger)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: FakeChild(code=0))
    monkeypatch.setattr(runner, "verify_run", lambda *a, **k: {
        "peak_allocated_vram_bytes": 500_000_000,
        "peak_reserved_vram_bytes": 950_000_000,
    })
    with pytest.raises(ValueError, match="headroom"):
        runner.launch(tmp_path, "S0", identity(tmp_path), lane="rtx_4070", execute=True, smoke=True, resume=False)
    assert ledger.settled["exit_code"] == -1
    failures = list((tmp_path / "smoke" / "S0" / "invocations").glob("*/failure.json"))
    assert "headroom" in json.loads(failures[0].read_text())["error"]


def test_resume_completed_smoke_does_not_launch(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from tinybench_lm import checkpointing
    _hardware(monkeypatch)
    root = tmp_path / "smoke" / "S0"
    root.mkdir(parents=True)
    (root / "latest.pt").write_bytes(b"fixture")
    monkeypatch.setattr(checkpointing, "verify_checkpoint", lambda path: SimpleNamespace(ok=True))
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"counters":{"updates_completed":2}})
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="already reached"):
        runner.launch(tmp_path, "S0", identity(tmp_path), lane="rtx_4070", execute=True, smoke=True, resume=True)


def test_smix_checks_entire_predecessor_chain(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(runner, "verify_run", lambda path: observed.append(path.name))
    runner.check_dependencies(tmp_path, "SMIX")
    assert observed == ["S0", "SLR"]


def test_inventory_rejects_missing_device_uuid(monkeypatch):
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "RTX 4070, N/A, 12282, 11000")
    with pytest.raises(ValueError, match="UUID"):
        runner.inspect_hardware()


def test_smoke_receipt_requires_matching_device_and_identity(tmp_path, monkeypatch):
    root = tmp_path / "smoke" / "S0"
    (root / "invocations" / "a").mkdir(parents=True)
    report = {"run_id": "experiment-test", "checkpoint_sha256": "c", "metric_sha256": "m", "runner_identity_sha256": "i"}
    monkeypatch.setattr(runner, "verify_run", lambda *args, **kwargs: report)
    (root / "invocations" / "a" / "verification.json").write_text(json.dumps({**report, "hardware": {"uuid": "other", "name": "RTX 4070"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="different device"):
        runner.smoke_receipt(tmp_path, "S0", identity(tmp_path), {"uuid": "u", "name": "RTX 4070"})


def test_smoke_receipt_requires_identity_match(tmp_path, monkeypatch):
    root = tmp_path / "smoke" / "S0"
    (root / "invocations" / "a").mkdir(parents=True)
    report = {"run_id": "other", "checkpoint_sha256": "c", "metric_sha256": "m", "runner_identity_sha256": "i"}
    monkeypatch.setattr(runner, "verify_run", lambda *args, **kwargs: report)
    (root / "invocations" / "a" / "verification.json").write_text(json.dumps({**report, "hardware": {"uuid": "u", "name": "RTX 4070"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        runner.smoke_receipt(tmp_path, "S0", identity(tmp_path), {"uuid": "u", "name": "RTX 4070"})


def test_estimate_full_seconds_rejects_malformed_timings(tmp_path):
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"step_seconds": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timing"):
        runner.estimate_full_seconds({"outer_wall_seconds": 2}, tmp_path)


def test_estimate_full_seconds_uses_two_steps_and_overhead(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("{\"step_seconds\": 2}\n{\"step_seconds\": 3}\n", encoding="utf-8")
    expected = (3 * 382 + (20 - 5) * 10 + 300) * 1.5
    assert runner.estimate_full_seconds({"outer_wall_seconds": 20}, tmp_path) == expected
