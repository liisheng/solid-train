import json

import pytest

from tinybench_lm.experiment_runtime import ExecutionLedger


def test_start_records_advisory_estimate_without_budget(tmp_path):
    ledger = ExecutionLedger(tmp_path, "rtx_4070")
    token = ledger.start("run-a", "RTX 4070", estimate_seconds=120)
    entry = json.loads(ledger.path.read_text(encoding="utf-8"))["entries"][0]
    assert entry["token"] == token
    assert entry["estimate_seconds"] == 120.0
    assert "allowance" not in json.loads(ledger.path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="unfinished"):
        ledger.start("run-b", "RTX 4070", estimate_seconds=1)


def test_lane_is_allowlisted_and_active_attempt_is_unknown_duration(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        ExecutionLedger(tmp_path, "../runtime")
    ledger = ExecutionLedger(tmp_path, "rtx_4070")
    ledger.start("run", "RTX 4070")
    assert ledger.summary() == {
        "measured_elapsed_seconds": 0,
        "unknown_duration_attempts": 1,
        "active_count": 1,
    }


def test_finish_and_summary_report_measured_time_and_unknown_crash_cost(tmp_path):
    ledger = ExecutionLedger(tmp_path, "rtx_3070")
    first = ledger.start("run-a", "RTX 3070", estimate_seconds=300)
    ledger.finish(first, 12.5, 0)
    second = ledger.start("run-b", "RTX 3070")
    ledger.finish(second, 4, None, uncertain=True)
    result = ledger.summary()
    assert result == {
        "measured_elapsed_seconds": 16.5,
        "unknown_duration_attempts": 1,
        "active_count": 0,
    }


def test_recover_marks_duration_unknown_without_charging_estimate(tmp_path):
    ledger = ExecutionLedger(tmp_path, "rtx_4070")
    token = ledger.start("run", "RTX 4070", estimate_seconds=900)
    with pytest.raises(ValueError, match="dead"):
        ledger.recover(token, process_dead=False)
    result = ledger.recover(token, process_dead=True)
    assert result["status"] == "interrupted"
    assert result["duration_unknown"] is True
    assert "elapsed_seconds" not in result
    assert ledger.summary() == {
        "measured_elapsed_seconds": 0,
        "unknown_duration_attempts": 1,
        "active_count": 0,
    }


def test_wrong_hardware_and_concurrent_lock_are_rejected(tmp_path):
    ledger = ExecutionLedger(tmp_path, "rtx_4070")
    token = ledger.start("run", "GPU-A")
    with pytest.raises(ValueError, match="unfinished"):
        ledger.start("next", "GPU-B")
    ledger.finish(token, 1, 1)
    with pytest.raises(ValueError, match="hardware"):
        ledger.start("next", "GPU-B")
    ledger.lock_path.write_text("other", encoding="utf-8")
    with pytest.raises(ValueError, match="concurrent"):
        ledger.start("again", "GPU-A")


def test_legacy_pending_budget_entry_blocks_overlap_and_file_is_untouched(tmp_path):
    ledger = ExecutionLedger(tmp_path, "rtx_3070")
    legacy = {
        "schema": "pre_campaign_budget_v2",
        "lane": "rtx_3070",
        "allowance_seconds": 21600,
        "reservations": [
            {
                "token": "old",
                "status": "uncertain",
                "reserved_seconds": 10,
                "charged_seconds": 10,
                "hardware": "RTX 3070",
            }
        ],
    }
    ledger.old_path.write_text(json.dumps(legacy), encoding="utf-8")
    before = ledger.old_path.read_bytes()
    with pytest.raises(ValueError, match="legacy"):
        ledger.start("new", "RTX 3070")
    assert ledger.old_path.read_bytes() == before


def test_malformed_legacy_budget_ledger_fails_closed(tmp_path):
    ledger = ExecutionLedger(tmp_path, "rtx_3070")
    ledger.old_path.write_text(json.dumps({"schema": "pre_campaign_budget_v2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        ledger.start("new", "RTX 3070")


@pytest.mark.parametrize(
    "override",
    [
        {"schema": "wrong"},
        {"lane": "rtx_4070"},
        {"allowance_seconds": 1},
        {"reservations": [{"token": "x", "status": "bogus"}]},
    ],
)
def test_invalid_legacy_budget_metadata_fails_closed(tmp_path, override):
    ledger = ExecutionLedger(tmp_path, "rtx_3070")
    legacy = {
        "schema": "pre_campaign_budget_v2",
        "lane": "rtx_3070",
        "allowance_seconds": 21600,
        "reservations": [],
    }
    legacy.update(override)
    ledger.old_path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        ledger.start("new", "RTX 3070")


def test_legacy_budget_lock_blocks_start(tmp_path):
    ledger = ExecutionLedger(tmp_path, "rtx_3070")
    ledger.old_lock_path.write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        ledger.start("new", "RTX 3070")


@pytest.mark.parametrize("elapsed", [float("nan"), "NaN", -1])
def test_persisted_finished_elapsed_duration_is_validated(tmp_path, elapsed):
    ledger = ExecutionLedger(tmp_path, "rtx_4070")
    ledger.path.write_text(
        json.dumps(
            {
                "schema": "experiment_runtime_v1",
                "lane": "rtx_4070",
                "entries": [
                    {
                        "token": "x",
                        "run_id": "run",
                        "hardware": "RTX 4070",
                        "status": "completed",
                        "estimate_seconds": None,
                        "elapsed_seconds": elapsed,
                        "created_at": "2026-09-09T00:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="elapsed"):
        ledger.summary()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_estimates_and_elapsed_durations_are_rejected(tmp_path, value):
    ledger = ExecutionLedger(tmp_path, "rtx_4070")
    with pytest.raises(ValueError, match="finite|non-negative"):
        ledger.start("run", "RTX 4070", estimate_seconds=value)
    token = ledger.start("run", "RTX 4070")
    with pytest.raises(ValueError, match="finite|non-negative"):
        ledger.finish(token, value, 0)
