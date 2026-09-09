"""Tiny CPU adaptation of the scheduled experiment trainer; no production readiness claim."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def _write_wrapper(tmp_path: Path) -> Path:
    wrapper = tmp_path / "cpu_scheduled_trainer.py"
    wrapper.write_text(
        '''import contextlib, json, sys
from pathlib import Path
from unittest.mock import patch
import torch
import numpy as np
torch.set_num_threads(1)
source_path = Path(sys.argv.pop(1))
root = source_path.parent
source = source_path.read_text()
guard = '    if device.type != "cuda":\\n        raise RuntimeError("This training configuration expects an NVIDIA GPU")'
assert source.count(guard) == 1
source = source.replace(guard, "    # CPU test adaptation of the GPU-only boundary")
namespace = {"__name__": "cpu_scheduled_train", "__file__": str(source_path)}
exec(compile(source, str(source_path), "exec"), namespace)
from tinybench_lm.schedule import MaterializedSchedule, ScheduleEntry, ScheduledTokenStream
from tinybench_lm.shards import NamespaceManifest, ShardRecord, SplitManifest

fixture = Path(sys.argv.pop(1))
slices = ("broad_general", "educational_science", "narrative_coreference", "math_technical")
records = []
for index, name in enumerate(slices):
    shard_id = f"validation/{index}"
    relative = f"validation/{index}.bin"
    records.append(ShardRecord(shard_id, "validation", f"source{index}", "validation_dev", relative,
        "uint16", 9, 1, (f"doc{index}",), (0,), (9,), 1, "", "fixture-counter", (name,)))
manifest = SplitManifest("validation_dev", "validation_dev",
    (NamespaceManifest("validation", "fixture", "validation_dev", tuple(records)),), "fixture-counter")
entries = tuple(ScheduleEntry(record.shard_id, 0, 9, record.source_id, record.namespace) for record in records)
schedule = MaterializedSchedule("fixture-schedule", "validation_dev", "validation_dev", 8, 1, 1001, 1,
    entries, manifest.content_hash(), "fixture-protocol")
def open_sources(args):
    train = ScheduledTokenStream(fixture, manifest, schedule)
    validation = ScheduledTokenStream(fixture, manifest, schedule)
    facts = {"batch_source": "CPU_FIXTURE_SCHEDULE", "train_schedule_content_hash": train.content_hash,
             "validation_schedule_content_hash": validation.content_hash, "validation_schedule_id": "fixture-schedule",
             "train_total_scheduled_sequences": 4, "actual_vocab_size": 32}
    return train, validation, facts
namespace["open_batch_sources"] = open_sources
with contextlib.ExitStack() as stack:
    for name, result in [("is_available", False), ("is_bf16_supported", True), ("get_device_name", "CPU test"),
                         ("reset_peak_memory_stats", None), ("synchronize", None), ("max_memory_allocated", 0)]:
        stack.enter_context(patch.object(torch.cuda, name, return_value=result))
    stack.enter_context(patch.object(torch, "autocast", side_effect=lambda **kwargs: contextlib.nullcontext()))
    namespace["main"]()
''', encoding="utf-8")
    return wrapper


def _run(tmp_path: Path, source: Path, run: Path, wrapper: Path, *extra: str) -> None:
    command = [sys.executable, str(wrapper), str(source), str(tmp_path), "--config", str(tmp_path / "tiny.json"),
               "--run-dir", str(run), "--steps", "4", "--micro-batch-size", "1", "--sequence-length", "8",
               "--gradient-accumulation", "1", "--warmup-steps", "1", "--decay-updates", "2",
               "--learning-rate", "0.001", "--validation-mode", "full-dev", "--eval-interval", "1",
               "--save-interval", "2", "--experiment-slice-reporting", *extra]
    result = subprocess.run(command, cwd=source.parent, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def _payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    else:
        assert left == right


def test_scheduled_cpu_experiment_continuous_equals_interrupted_resume(tmp_path: Path) -> None:
    config = tmp_path / "tiny.json"
    config.write_text(json.dumps({"vocab_size": 32, "max_seq_len": 8, "n_layers": 1, "d_model": 16,
                                 "n_heads": 2, "n_kv_heads": 2, "d_ff": 32}), encoding="utf-8")
    for index in range(4):
        path = tmp_path / "validation" / f"{index}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        (np.arange(9, dtype=np.uint16) + index).tofile(path)
    source = Path(__file__).resolve().parents[1] / "train.py"
    wrapper = _write_wrapper(tmp_path)
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    _run(tmp_path, source, continuous, wrapper)
    _run(tmp_path, source, resumed, wrapper, "--stop-after-updates", "2")
    _run(tmp_path, source, resumed, wrapper, "--resume", str(resumed / "latest.pt"))
    left, right = _payload(continuous / "completed.pt"), _payload(resumed / "completed.pt")
    assert left["counters"] == right["counters"]
    assert left["schedule_cursor"] == right["schedule_cursor"] == 4
    assert left["training_args"]["experiment_endpoint_validation"] == right["training_args"]["experiment_endpoint_validation"]
    for key, tensor in left["model"].items():
        assert torch.equal(tensor, right["model"][key])
    _assert_equal(left["optimizer"], right["optimizer"])
    rows = [json.loads(line) for line in (continuous / "metrics.jsonl").read_text().splitlines()]
    endpoint = rows[-1]
    assert endpoint["validation_slice_metrics"] == left["training_args"]["experiment_endpoint_validation"]["validation_slice_metrics"]
    assert sum(item["token_count"] for item in endpoint["validation_slice_metrics"].values()) == endpoint["validation_scored_token_count"]
