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
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")


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
    def start(self, *args, **kwargs): return "token"
    def finish(self, token, **kwargs): self.settled = kwargs


def _hardware(monkeypatch):
    monkeypatch.setattr(runner, "inspect_hardware", lambda: {"name": "RTX 4070", "uuid": "u", "total_vram_bytes": 10**9, "free_vram_bytes": 10**9, "host_ram_bytes": 1, "bf16": True})
    monkeypatch.setattr(runner, "ExecutionLedger", FakeLedger)


def test_failed_child_writes_exit_and_settles(tmp_path, monkeypatch):
    _hardware(monkeypatch)
    child = FakeChild(code=3)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: child)
    with pytest.raises(ValueError, match="failed"):
        runner.launch(tmp_path, "S0", identity(tmp_path), lane="rtx_4070", execute=True, smoke=True, resume=False)
    exits = list((tmp_path / "smoke" / "S0" / "invocations").glob("*/exit.json"))
    assert exits and json.loads(exits[0].read_text())["exit_code"] == 3


def test_execution_has_no_timeout_even_after_old_deadline(tmp_path, monkeypatch):
    _hardware(monkeypatch)
    from datetime import datetime, timezone
    class Later(datetime):
        @classmethod
        def now(cls, tz=None): return datetime(2026, 10, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(runner, "datetime", Later)
    class UnboundedChild(FakeChild):
        def wait(self, *args, **kwargs):
            assert not args and not kwargs, "must not pass a time limit"
            return 0
    child = UnboundedChild(code=0)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(runner, "verify_run", lambda *a, **k: {"peak_allocated_vram_bytes":1,"peak_reserved_vram_bytes":1})
    runner.launch(tmp_path, "S0", identity(tmp_path), lane="rtx_4070", execute=True, smoke=True, resume=False)
    assert not child.terminated


def test_smoke_rejects_reserved_memory_near_capacity(tmp_path, monkeypatch):
    _hardware(monkeypatch)
    ledger = FakeLedger()
    monkeypatch.setattr(runner, "ExecutionLedger", lambda *a, **k: ledger)
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


@pytest.mark.parametrize("answer", ["", "no"])
def test_declining_does_not_create_run_or_ledger(tmp_path, monkeypatch, answer, capsys):
    _hardware(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    monkeypatch.setattr(runner, "ExecutionLedger", lambda *a, **k: pytest.fail("ledger created before decision"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("child started"))
    result=runner.launch(tmp_path,"S0",identity(tmp_path),lane="rtx_4070",execute=True,smoke=True,resume=False)
    assert result["status"] == "DECLINED_NOT_STARTED"
    assert not (tmp_path/'smoke').exists()
    assert "Estimated runtime" in capsys.readouterr().out


def test_no_stdin_does_not_approve(monkeypatch):
    def closed(prompt): raise EOFError
    monkeypatch.setattr("builtins.input", closed)
    assert not runner.confirm_execution({"estimated_seconds":99999,"basis":"fixture"})


def test_setup_error_finishes_runtime_entry(tmp_path, monkeypatch):
    _hardware(monkeypatch)
    ledger = FakeLedger()
    monkeypatch.setattr(runner, "ExecutionLedger", lambda *a, **k: ledger)
    def broken(*a, **k): raise OSError("disk full")
    monkeypatch.setattr(runner, "write_once", broken)
    with pytest.raises(OSError, match="disk full"):
        runner.launch(tmp_path,"S0",identity(tmp_path),lane="rtx_4070",execute=True,smoke=True,resume=False)
    assert ledger.settled is not None
    assert ledger.settled["exit_code"] != 0


def test_resume_forecast_preserves_fixed_overhead(tmp_path):
    (tmp_path/"metrics.jsonl").write_text('{"step_seconds":2}\n{"step_seconds":3}\n')
    result=runner.estimate_full_seconds({"outer_wall_seconds":20},tmp_path,remaining_updates=1)
    assert result == (3 + 150 + 300)*1.5


def test_estimate_ignores_receipt_from_other_identity(tmp_path, monkeypatch):
    root=tmp_path/"smoke"/"S0"
    (root/"invocations"/"a").mkdir(parents=True)
    (root/"invocations"/"z").mkdir()
    (root/"metrics.jsonl").write_text('{"step_seconds":2}\n{"step_seconds":3}\n')
    report={"run_id":"r","checkpoint_sha256":"c","metric_sha256":"m","runner_identity_sha256":"i"}
    monkeypatch.setattr(runner,"verify_run",lambda *a,**k:report)
    (root/"invocations"/"a"/"verification.json").write_text(json.dumps({**report,"outer_wall_seconds":20}))
    (root/"invocations"/"z"/"verification.json").write_text(json.dumps({**report,"run_id":"wrong","outer_wall_seconds":99999}))
    result=runner.estimate_for_job(tmp_path,"S0",smoke=False)
    assert result["estimated_seconds"] == (3*382+150+300)*1.5


def test_operator_can_accept_estimate_above_six_hours(tmp_path, monkeypatch):
    _hardware(monkeypatch)
    monkeypatch.setattr(runner,"smoke_receipt",lambda *a,**k:{})
    monkeypatch.setattr(runner,"estimate_for_job",lambda *a,**k:{"estimated_seconds":86400,"basis":"fixture"})
    monkeypatch.setattr(runner,"verify_run",lambda *a,**k:{})
    child=FakeChild(code=0)
    monkeypatch.setattr(runner.subprocess,"Popen",lambda *a,**k:child)
    runner.launch(tmp_path,"S0",identity(tmp_path),lane="rtx_4070",execute=True,smoke=False,resume=False)
    assert not child.terminated


def test_resume_plan_prints_resume_command_and_estimate(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner,"identity_for",lambda *a,**k:identity(tmp_path))
    monkeypatch.setattr(runner,"estimate_for_job",lambda *a,**k:{"resume":k["resume"]})
    monkeypatch.setattr(runner.sys,"argv",["runner","plan","--job","S0","--resume","--bundle",str(tmp_path/"runs/pre_campaign/fixture")])
    assert runner.main()==0
    result=json.loads(capsys.readouterr().out)
    assert "--resume" in result["command"]
    assert result["estimate"]["resume"]
