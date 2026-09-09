"""CPU crash-ledger regression; no training or corpus access required."""
import json

import pytest

from tinybench_lm import metric_ledger
from tinybench_lm.training_recipe import (
    BatchPlan, TrainingIntegrityError, WSDSchedule, assert_update_record,
    build_update_record, select_precision_policy,
)


@pytest.fixture
def ledger(tmp_path):
    schedule = WSDSchedule(total_updates=200, warmup_updates=10, decay_updates=20, peak_lr=6e-4)
    plan = BatchPlan(2, 16, 2)
    precision = select_precision_policy(bf16_supported=True, bf16_measured_stable=True)
    kwargs = dict(run_dir=tmp_path, completed_updates=100, run_id="run-test", schedule=schedule,
                  plan=plan, precision=precision, schedule_content_hash="abc",
                  checkpoint_cursor=400, resume_checkpoint=tmp_path / "latest.pt")
    rows = []
    for index in range(120):
        row = build_update_record(run_id="run-test", update_index=index, schedule=schedule,
                                  plan=plan, precision=precision, loss=2., grad_norm=1.,
                                  schedule_content_hash="abc", schedule_cursor=(index + 1) * 4).to_dict()
        rows.append({**row, "step_seconds": 2., "step": index, "tokens": (index + 1) * 64})
    return kwargs, rows


def write_rows(kwargs, rows):
    path = kwargs["run_dir"] / "metrics.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_checkpoint100_log120_preserves_replayed_compute_and_seeds_continuity(ledger):
    kwargs, rows = ledger
    path = write_rows(kwargs, rows)
    previous, invocation = metric_ledger.reconcile_metrics(**kwargs)
    assert previous.update_index == 99
    assert len(path.read_text().splitlines()) == 100
    archives = list((kwargs["run_dir"] / "superseded_metrics").glob("*.json"))
    archived = json.loads(archives[0].read_text())
    assert archived["records"] == rows[100:]
    assert archived["superseded_optimizer_seconds"] == 40
    assert archived["superseded_loss_tokens"] == 20 * 64
    assert archived["source_invocation_ids"][0].startswith("legacy-")
    for row in rows[100:]:
        record = metric_ledger.UpdateRecord(**{field.name: row[field.name] for field in metric_ledger.fields(metric_ledger.UpdateRecord)})
        previous = assert_update_record(record, schedule=kwargs["schedule"], plan=kwargs["plan"], previous=previous)
        with path.open("a") as output:
            output.write(json.dumps({**row, "invocation_id": invocation}) + "\n")
    canonical = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["update_index"] for row in canonical] == list(range(120))
    assert sum(row["step_seconds"] for row in canonical) + archived["superseded_optimizer_seconds"] == 280
    assert sum(row["loss_tokens_per_update"] for row in canonical) + archived["superseded_loss_tokens"] == 140 * 64
    assert json.loads((kwargs["run_dir"] / "metric_invocations" / f"{invocation}.json").read_text())["superseded_segment"] == archives[0].name


@pytest.mark.parametrize("problem", ["gap", "duplicate", "run", "hash", "cursor", "precision", "tokens", "step", "malformed", "partial", "missing", "nan", "fractional"])
def test_bad_ledger_fails_without_modifying_evidence(ledger, problem):
    kwargs, rows = ledger
    if problem == "gap": del rows[50]
    elif problem == "duplicate": rows.insert(50, rows[50])
    elif problem == "run": rows[0]["run_id"] = "wrong"
    elif problem == "hash": rows[0]["schedule_content_hash"] = "wrong"
    elif problem == "cursor": rows[99]["schedule_cursor"] = 399
    elif problem == "precision": rows[0]["precision_dtype"] = "float16"
    elif problem == "tokens": rows[0]["tokens"] = 0
    elif problem == "step": rows[0]["step"] = 1
    elif problem == "missing": rows = rows[:99]
    elif problem == "nan": rows[100]["step_seconds"] = float("nan")
    elif problem == "fractional": rows[99]["update_index"] = 99.1
    path = write_rows(kwargs, rows)
    if problem in ("malformed", "partial"):
        with path.open("a") as output: output.write('{"update_index":' if problem == "partial" else "\n")
    original = path.read_bytes()
    with pytest.raises(TrainingIntegrityError): metric_ledger.reconcile_metrics(**kwargs)
    assert path.read_bytes() == original
    assert not (kwargs["run_dir"] / "superseded_metrics").exists()


def test_archive_before_atomic_truncation_is_idempotent_after_failure(ledger, monkeypatch):
    kwargs, rows = ledger
    path = write_rows(kwargs, rows)
    original = path.read_bytes()
    real_write = metric_ledger._atomic_write
    def fail_replace(target, content):
        if target == path: raise OSError("simulated crash before canonical replacement")
        real_write(target, content)
    with monkeypatch.context() as patch:
        patch.setattr(metric_ledger, "_atomic_write", fail_replace)
        with pytest.raises(OSError): metric_ledger.reconcile_metrics(**kwargs)
    assert path.read_bytes() == original
    with pytest.raises(TrainingIntegrityError, match="previously archived rollback"):
        metric_ledger.reconcile_metrics(**{**kwargs, "completed_updates": 90, "checkpoint_cursor": 360})
    assert path.read_bytes() == original
    metric_ledger.reconcile_metrics(**kwargs)
    metric_ledger.reconcile_metrics(**kwargs)
    assert len(list((kwargs["run_dir"] / "superseded_metrics").glob("*.json"))) == 1
    assert len(path.read_text().splitlines()) == 100


def test_complete_last_json_without_newline_is_append_safe(ledger):
    kwargs, rows = ledger
    path = write_rows(kwargs, rows[:100])
    path.write_bytes(path.read_bytes().rstrip(b"\r\n"))
    metric_ledger.reconcile_metrics(**kwargs)
    assert path.read_bytes().endswith(b"\n")


def test_fresh_run_refuses_existing_log(ledger):
    kwargs, rows = ledger
    write_rows(kwargs, rows)
    kwargs.update(completed_updates=0, checkpoint_cursor=0, resume_checkpoint=None)
    with pytest.raises(TrainingIntegrityError, match="explicit checkpoint"):
        metric_ledger.reconcile_metrics(**kwargs)


def test_cpu_adapted_trainer_subprocess_reconciles_dirty_tail(tmp_path):
    """Exercise actual main/checkpoints/optimizer with only CUDA boundaries adapted.

    This is not CUDA correctness evidence. The production GPU-only guard is removed
    in memory for this test; no production device policy or source file is changed.
    """
    import subprocess
    import sys
    from pathlib import Path
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "tiny.json"
    config.write_text(json.dumps(dict(vocab_size=32, max_seq_len=8, n_layers=1,
                                     d_model=16, n_heads=2, n_kv_heads=2, d_ff=32)))
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.bin", "validation.bin"):
        (np.arange(256, dtype=np.uint16) % 32).tofile(data / name)
    (data / "metadata.json").write_text("{}")
    script = tmp_path / "cpu_trainer.py"
    script.write_text('''import contextlib, sys
from pathlib import Path
from unittest.mock import patch
import torch
torch.set_num_threads(1)
source_path = Path(sys.argv.pop(1))
sys.path.insert(0, str(source_path.parent))
source = source_path.read_text()
guard = '    if device.type != "cuda":\\n        raise RuntimeError("This training configuration expects an NVIDIA GPU")'
assert source.count(guard) == 1
source = source.replace(guard, "    # CPU test adaptation of the GPU-only boundary")
namespace = {"__name__": "cpu_adapted_train", "__file__": str(source_path)}
exec(compile(source, str(source_path), "exec"), namespace)
with contextlib.ExitStack() as stack:
    for name, result in [("is_available", False), ("is_bf16_supported", True), ("get_device_name", "CPU test"), ("reset_peak_memory_stats", None), ("synchronize", None), ("max_memory_allocated", 0)]:
        stack.enter_context(patch.object(torch.cuda, name, return_value=result))
    stack.enter_context(patch.object(torch, "autocast", side_effect=lambda **kwargs: contextlib.nullcontext()))
    namespace["main"]()
''')
    run = tmp_path / "run"
    base = [sys.executable, str(script), str(root / "train.py"), "--config", str(config),
            "--data-dir", str(data), "--run-dir", str(run), "--steps", "4",
            "--micro-batch-size", "1", "--sequence-length", "8", "--gradient-accumulation", "1",
            "--warmup-steps", "1", "--decay-updates", "2", "--eval-batches", "1", "--save-interval", "2"]
    def launch(extra):
        result = subprocess.run(base + extra, cwd=root, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    launch(["--stop-after-updates", "2"])
    # Keep a real checkpoint at update 2, then produce a real third optimizer row.
    import shutil
    for path in run.glob("latest.pt*"):
        shutil.copy2(path, path.with_name("saved-" + path.name))
    launch(["--resume", str(run / "latest.pt"), "--stop-after-updates", "3"])
    for path in run.glob("saved-latest.pt*"):
        shutil.copy2(path, path.with_name(path.name.removeprefix("saved-")))
    launch(["--resume", str(run / "latest.pt"), "--stop-after-updates", "4"])
    canonical = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [row["update_index"] for row in canonical] == [0, 1, 2, 3]
    archive = json.loads(next((run / "superseded_metrics").glob("*.json")).read_text())
    assert [row["update_index"] for row in archive["records"]] == [2]
    assert archive["records"][0]["invocation_id"] != canonical[2]["invocation_id"]
    assert archive["records"][0]["loss"] == canonical[2]["loss"]
    timings = [json.loads(line) for line in (run / "phase_timing_history.jsonl").read_text().splitlines()]
    assert [row["started_at_update"] for row in timings] == [0, 2, 2]
    assert timings[-1]["metric_invocation_id"] == canonical[-1]["invocation_id"]
