from __future__ import annotations

import hashlib

import torch

from scripts.run_g3_integration import (
    _all_checks,
    _counter_checks,
    _metric_checks,
    _phase_timing_checks,
    _require_command_success,
    _saved_command_checks,
    _state_differences,
    _timing_history_record_checks,
    _training_command_checks,
)
from train import write_phase_timing


def test_saved_command_checks_recompute_normalized_text_hashes(tmp_path):
    stdout = tmp_path / "stdout.txt"
    stderr = tmp_path / "stderr.txt"
    stdout.write_bytes(b"one passed\r\n")
    stderr.write_bytes(b"")
    record = {
        "returncode": 0,
        "wall_seconds": 0.5,
        "stdout_path": str(stdout),
        "stderr_path": str(stderr),
        "stdout_sha256": hashlib.sha256(b"one passed\n").hexdigest(),
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
    }

    checks = _saved_command_checks(
        record, output_root=tmp_path, expected_returncode=0, required_text="passed"
    )

    assert _all_checks(checks)


def test_saved_command_checks_fail_for_tamper_and_path_escape(tmp_path):
    output_root = tmp_path / "evidence"
    output_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("changed", encoding="utf-8")
    record = {
        "returncode": 1,
        "wall_seconds": 1.0,
        "stdout_path": str(outside),
        "stderr_path": str(outside),
        "stdout_sha256": hashlib.sha256(b"original").hexdigest(),
        "stderr_sha256": hashlib.sha256(b"original").hexdigest(),
    }

    checks = _saved_command_checks(record, output_root=output_root, expected_returncode=0)

    assert checks["returncode_matches"] is False
    assert checks["stdout_in_output_root"] is False
    assert checks["stdout_hash_matches"] is False
    assert not _all_checks(checks)


def test_failed_training_command_stops_chain_and_preserves_machine_record(tmp_path):
    result = {"returncode": 1, "command": ["python", "train.py"]}
    with __import__("pytest").raises(RuntimeError, match="dependent rehearsal commands were not launched"):
        _require_command_success(result, stage="uninterrupted", output_root=tmp_path)
    failure = __import__("json").loads((tmp_path / "integration_failure.json").read_text(encoding="utf-8"))
    assert failure == {
        "schema": "reduced_g3_integration_failure_v1",
        "status": "FAIL",
        "failed_stage": "uninterrupted",
        "command": result,
    }


def test_training_command_checks_bind_stop_lineage_and_resume(tmp_path):
    run_dir = tmp_path / "run"
    resume = run_dir / "latest.pt"
    record = {
        "command": [
            "python", "scripts/run_reduced_baseline.py", "launch", "--execute",
            "--run-dir", str(run_dir), "--resume", str(resume), "--stop-after-updates", "8",
        ]
    }
    assert _all_checks(
        _training_command_checks(record, run_dir=run_dir, stop_after_updates=8, resume=resume)
    )
    assert not _all_checks(
        _training_command_checks(record, run_dir=run_dir, stop_after_updates=4, resume=None)
    )


def test_recursive_state_comparison_checks_nested_tensors_but_excludes_locations():
    left = {
        "training_args": {"run_dir": "first", "resume": None, "steps": 8},
        "optimizer": {"state": [torch.tensor([1.0, 2.0, 3.0])]},
    }
    right = {
        "training_args": {"run_dir": "second", "resume": "latest.pt", "steps": 8},
        "optimizer": {"state": [torch.tensor([1.0, 9.0, 3.0])]},
    }

    assert _state_differences(left, right) == ["state.optimizer.state[0]"]


def test_counter_checks_require_absolute_update_token_and_cursor_values():
    state = {
        "updates_completed": 4,
        "consumed_loss_tokens": 64,
        "schedule_cursor": 16,
        "step": 3,
    }
    assert _all_checks(
        _counter_checks(state, updates=4, sequences_per_update=4, loss_tokens_per_update=16)
    )
    state["schedule_cursor"] = 12
    assert not _all_checks(
        _counter_checks(state, updates=4, sequences_per_update=4, loss_tokens_per_update=16)
    )


def test_metric_checks_bind_order_and_counters_to_expected_schedule():
    hashes = ["first", "second"]
    rows = [
        {
            "update_index": index,
            "step": index,
            "consumed_loss_tokens": (index + 1) * 16,
            "schedule_cursor": (index + 1) * 4,
            "train_batch_reference_hash": expected,
            "run_id": "run-a",
            "schedule_content_hash": "schedule-a",
        }
        for index, expected in enumerate(hashes)
    ]
    assert _all_checks(
        _metric_checks(
            rows,
            expected_hashes=hashes,
            sequences_per_update=4,
            loss_tokens_per_update=16,
            run_id="run-a",
            schedule_hash="schedule-a",
        )
    )
    rows[1]["train_batch_reference_hash"] = "wrong"
    assert not _all_checks(
        _metric_checks(
            rows,
            expected_hashes=hashes,
            sequences_per_update=4,
            loss_tokens_per_update=16,
            run_id="run-a",
            schedule_hash="schedule-a",
        )
    )


def test_phase_timing_and_recursive_decision_fail_on_missing_evidence():
    assert not _all_checks(_phase_timing_checks(None, validation_expected=True))
    timing = {
        "training_optimizer_seconds": 2.0,
        "validation_seconds": 1.0,
        "checkpoint_seconds": 0.5,
        "process_wall_seconds": 4.0,
    }
    assert _all_checks(_phase_timing_checks(timing, validation_expected=True))
    timing["checkpoint_seconds"] = 0.0
    assert not _all_checks(_phase_timing_checks(timing, validation_expected=True))
    assert not _all_checks({"existing_claim": True, "missing_evidence": {}})


def test_phase_timing_history_preserves_each_resume_invocation(tmp_path):
    first = {
        "training_optimizer_seconds": 2.0,
        "validation_seconds": 1.0,
        "checkpoint_seconds": 0.5,
        "process_wall_seconds": 4.0,
    }
    second = {
        "training_optimizer_seconds": 3.0,
        "validation_seconds": 0.0,
        "checkpoint_seconds": 0.5,
        "process_wall_seconds": 4.5,
    }
    write_phase_timing(
        tmp_path, first, started_at_update=0, completed_updates=4, resume_checkpoint=None
    )
    write_phase_timing(
        tmp_path,
        second,
        started_at_update=4,
        completed_updates=8,
        resume_checkpoint=tmp_path / "latest.pt",
    )

    history = [
        __import__("json").loads(line)
        for line in (tmp_path / "phase_timing_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["invocation_index"] for record in history] == [0, 1]
    assert [(record["started_at_update"], record["completed_updates"]) for record in history] == [(0, 4), (4, 8)]
    assert history[0]["resume_checkpoint"] is None
    assert history[1]["resume_checkpoint"] == str(tmp_path / "latest.pt")
    assert __import__("json").loads((tmp_path / "phase_timing.json").read_text(encoding="utf-8")) == second
    assert _all_checks(
        _timing_history_record_checks(
            history[1], invocation_index=1, started_at_update=4, completed_updates=8,
            resume_checkpoint=tmp_path / "latest.pt", timing=second,
        )
    )
    assert not _all_checks(
        _timing_history_record_checks(
            history[1], invocation_index=1, started_at_update=0, completed_updates=8,
            resume_checkpoint=tmp_path / "latest.pt", timing=second,
        )
    )
