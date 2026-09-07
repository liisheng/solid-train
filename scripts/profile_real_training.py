"""Measure at least 30 minutes of scheduled final-model training, without a gate claim."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from tinybench_lm.operations import ThroughputMeasurement, throughput_violations
from tinybench_lm.schedule import load_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--train-schedule", type=Path, required=True)
    parser.add_argument("--validation-schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = load_schedule(args.train_schedule)
    epochs = math.ceil(2048 * 256 / schedule.sequence_count)
    command = [sys.executable, "-u", "train.py", "--config", "configs/final_49m.json",
               "--shard-root", str(args.shard_root),
               "--train-manifest", str(args.shard_root / "stable_train.manifest.json"),
               "--validation-manifest", str(args.shard_root / "validation_dev.manifest.json"),
               "--train-schedule", str(args.train_schedule), "--validation-schedule", str(args.validation_schedule),
               "--train-epochs", str(epochs), "--run-dir", str(args.output_dir / "train"),
               "--steps", "2048", "--micro-batch-size", "8", "--sequence-length", "1024",
               "--bf16-stability", "stable", "--warmup-steps", "2", "--eval-interval", "100",
               "--eval-batches", "8", "--save-interval", "100", "--log-interval", "10",
               "--stop-after-training-seconds", "1900"]
    started = time.perf_counter()
    with (args.output_dir / "training.log").open("w", encoding="utf-8") as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
    wall = time.perf_counter() - started
    rows = [json.loads(line) for line in (args.output_dir / "train/metrics.jsonl").read_text().splitlines()]
    steady = rows[4:]
    seconds = sum(row["step_seconds"] for row in steady)
    measurement = ThroughputMeasurement("rtx_4070", tuple(row["tokens_per_second"] for row in steady), seconds, True)
    problems = throughput_violations(measurement)
    result = {"status": "FAIL" if problems else "MEASURED", "problems": list(problems),
              **measurement.to_dict(), "weighted_tokens_per_second": len(steady) * 262144 / seconds,
              "wall_seconds": wall, "optimizer_seconds": sum(row["step_seconds"] for row in rows),
              "peak_allocated_vram_gib": max(row["peak_vram_gib"] for row in rows),
              "command": command, "scope": "ENGINEERING_PROFILE_NOT_SUBMISSION_MODEL",
              "other_machine_profile": "NOT_RUN"}
    (args.output_dir / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
