"""Bounded real-shard GPU recovery/export rehearsal; does not claim cross-machine G2."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from tinybench_lm.checkpointing import CHECKPOINT_CHECKSUM_MISMATCH, CheckpointIntegrityError, load_verified_checkpoint, manifest_path_for
from tinybench_lm.provenance import export_release_from_checkpoint, verify_release_export
from tinybench_lm.schedule import load_schedule, training_order_hash


def assert_equal(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--train-schedule", type=Path, required=True)
    parser.add_argument("--validation-schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = [sys.executable, "-u", "train.py", "--config", "configs/final_49m.json",
            "--shard-root", str(args.shard_root),
            "--train-manifest", str(args.shard_root / "stable_train.manifest.json"),
            "--validation-manifest", str(args.shard_root / "validation_dev.manifest.json"),
            "--train-schedule", str(args.train_schedule), "--validation-schedule", str(args.validation_schedule),
            "--train-epochs", "3",
            "--steps", "8", "--micro-batch-size", "4", "--sequence-length", "1024",
            "--bf16-stability", "stable", "--warmup-steps", "1", "--decay-updates", "2",
            "--eval-interval", "4", "--eval-batches", "2", "--save-interval", "4", "--log-interval", "1"]
    uninterrupted = args.output_dir / "uninterrupted"
    resumed = args.output_dir / "resumed"
    for name, extra in [
        ("uninterrupted", ["--run-dir", str(uninterrupted)]),
        ("pause", ["--run-dir", str(resumed), "--stop-after-updates", "4"]),
        ("resume", ["--run-dir", str(resumed), "--resume", str(resumed / "latest.pt")]),
    ]:
        with (args.output_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(base + extra, stdout=log, stderr=subprocess.STDOUT, check=True)
    a = load_verified_checkpoint(uninterrupted / "latest.pt")
    b = load_verified_checkpoint(resumed / "latest.pt")
    # Compare every durable checkpoint field.  The only justified differences are
    # run-location and resume-control CLI metadata embedded in training_args; these
    # do not affect continuation state and are checked for presence by key equality.
    compared = sorted(a.keys())
    assert a.keys() == b.keys(), (sorted(a.keys()), sorted(b.keys()))
    for key in compared:
        if key == "training_args":
            left = dict(a[key])
            right = dict(b[key])
            for excluded in ("run_dir", "resume", "stop_after_updates", "stop_after_training_seconds"):
                left.pop(excluded, None)
                right.pop(excluded, None)
            assert_equal(left, right, f"{key} (excluding run-location/resume controls)")
        else:
            assert_equal(a[key], b[key], key)
    rows = [json.loads(row) for row in (uninterrupted / "metrics.jsonl").read_text().splitlines()]
    resumed_rows = [json.loads(row) for row in (resumed / "metrics.jsonl").read_text().splitlines()]
    expected_cursor = 0
    sequences_per_update = 262144 // 1024
    schedule = load_schedule(args.train_schedule)
    repeated_entries = schedule.entries * 3
    batch_reference_hashes = []
    for index, (continuous, interrupted) in enumerate(zip(rows, resumed_rows)):
        expected_cursor += sequences_per_update
        assert continuous["schedule_cursor"] == expected_cursor, (index, continuous["schedule_cursor"])
        assert interrupted["schedule_cursor"] == expected_cursor, (index, interrupted["schedule_cursor"])
        expected_hash = training_order_hash(repeated_entries[index * sequences_per_update : expected_cursor])
        assert continuous["train_batch_reference_hash"] == expected_hash, (index, "continuous")
        assert interrupted["train_batch_reference_hash"] == expected_hash, (index, "interrupted")
        batch_reference_hashes.append(expected_hash)
    assert len(rows) == len(resumed_rows) == 8
    losses = [float(row["validation_loss"]) for row in rows if "validation_loss" in row]
    assert all(np.isfinite(loss) for loss in losses), losses
    assert losses[-1] < losses[0], losses
    corrupt = args.output_dir / "deliberately_corrupt.pt"
    shutil.copyfile(resumed / "latest.pt", corrupt)
    shutil.copyfile(manifest_path_for(resumed / "latest.pt"), manifest_path_for(corrupt))
    with corrupt.open("r+b") as handle:
        handle.seek(128)
        byte = handle.read(1)
        handle.seek(128)
        handle.write(bytes([byte[0] ^ 1]))
    try:
        load_verified_checkpoint(corrupt)
    except CheckpointIntegrityError as error:
        assert CHECKPOINT_CHECKSUM_MISMATCH in str(error), str(error)
        corruption_rejected = True
    else:
        raise AssertionError("Corrupt checkpoint was accepted")
    export = args.output_dir / "engineering_export.pt"
    export_release_from_checkpoint(resumed / "latest.pt", export,
                                   provenance_path=resumed / "step_zero_provenance.json",
                                   notes="Engineering recovery rehearsal, not a submission model")
    report = verify_release_export(export, expected_parameter_count=49_658_368)
    assert report.ok, report.to_dict()
    run_config = json.loads((resumed / "run_config.json").read_text(encoding="utf-8"))
    precision = run_config.get("precision_policy", {})
    execution = run_config.get("data_metadata", {}).get("execution_policy", {})
    assert precision.get("status") == "PASS" and precision.get("bf16_measured_stable") is True, "BF16 stability evidence missing"
    assert execution.get("strict_deterministic_algorithms") is True, "strict deterministic execution evidence missing"
    result = {"status": "LOCAL_CHECKS_PASS", "other_machine": "NOT_RUN",
              "target_machine_id": platform.node(),
              "target_environment": {"hostname": platform.node(), "python": platform.python_version(), "platform": platform.platform()},
              "precision_policy": precision,
              "execution_policy": execution,
              "exact_resume_fields": compared, "resume_state_complete_equal": True,
              "corruption_rejected": corruption_rejected,
              "training_input_identity": {
                  "base_schedule_content_hash": schedule.content_hash(),
                  "epochs": 3,
                  "sequences_per_update": sequences_per_update,
                  "expected_cursors": [row["schedule_cursor"] for row in rows],
                  "batch_reference_hashes": batch_reference_hashes,
              },
              "validation_losses": losses,
              "export": report.to_dict()}
    (args.output_dir / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
