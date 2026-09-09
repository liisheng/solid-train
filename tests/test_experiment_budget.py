from datetime import datetime
import json

import pytest

from tinybench_lm.experiment_budget import BudgetLedger, LANE_SECONDS


def clock(value):
    return lambda: datetime.fromisoformat(value)


def test_reservation_is_lane_bound_and_pending_amount_is_unavailable(tmp_path):
    ledger = BudgetLedger(tmp_path, "rtx_4070", clock=clock("2026-09-10T12:00:00+08:00"))
    token = ledger.reserve("run-a", 100, hardware="rtx_4070")
    assert ledger.remaining_seconds() == LANE_SECONDS - 100
    with pytest.raises(ValueError, match="unfinished"):
        ledger.reserve("run-b", 100, hardware="rtx_4070")
    with pytest.raises(ValueError, match="hardware"):
        BudgetLedger(tmp_path, "rtx_3070").reserve("run", 1, hardware="rtx_4070")
    ledger.settle(token, elapsed_seconds=120, exit_code=1)
    assert ledger.remaining_seconds() == LANE_SECONDS - 120


def test_uncertain_reservation_requires_explicit_dead_process_recovery(tmp_path):
    ledger = BudgetLedger(tmp_path, "rtx_3070", clock=clock("2026-09-10T12:00:00+08:00"))
    token = ledger.reserve("run", 50, hardware="rtx_3070")
    ledger.settle(token, elapsed_seconds=1, exit_code=None, uncertain=True)
    with pytest.raises(ValueError, match="dead"):
        ledger.recover_uncertain(token, process_dead=False)
    result = ledger.recover_uncertain(token, process_dead=True)
    assert result["status"] == "failed"


def test_abandoned_reserved_reservation_recovers_with_full_charge(tmp_path):
    ledger = BudgetLedger(tmp_path, "rtx_3070", clock=clock("2026-09-10T12:00:00+08:00"))
    token = ledger.reserve("run", 50, hardware={"name": "RTX 3070", "uuid": "gpu-1"})
    with pytest.raises(ValueError, match="dead"):
        ledger.recover_uncertain(token, process_dead=False)
    result = ledger.recover_uncertain(token, process_dead=True)
    assert result["status"] == "failed" and result["charged_seconds"] == 50
    assert ledger.remaining_seconds() == LANE_SECONDS - 50


def test_lane_ledger_rejects_hardware_identity_change(tmp_path):
    ledger = BudgetLedger(tmp_path, "rtx_4070", clock=clock("2026-09-10T12:00:00+08:00"))
    token = ledger.reserve("run-a", 10, hardware={"name": "RTX 4070", "uuid": "gpu-a"})
    ledger.settle(token, elapsed_seconds=12, exit_code=1)
    with pytest.raises(ValueError, match="hardware identity"):
        ledger.reserve("run-b", 10, hardware={"name": "RTX 4070", "uuid": "gpu-b"})


def test_uncertain_recovery_keeps_overrun_charge(tmp_path):
    ledger = BudgetLedger(tmp_path, "rtx_4070", clock=clock("2026-09-10T12:00:00+08:00"))
    token = ledger.reserve("run", 10, hardware="rtx_4070")
    ledger.settle(token, elapsed_seconds=20, exit_code=None, uncertain=True)
    result = ledger.recover_uncertain(token, process_dead=True)
    assert result["charged_seconds"] == 20


def test_deadline_is_timezone_aware_and_reservation_cannot_cross_it(tmp_path):
    ledger = BudgetLedger(tmp_path, "rtx_4070", clock=clock("2026-09-12T23:59:58+08:00"))
    with pytest.raises(ValueError, match="deadline"):
        ledger.reserve("run", 2, hardware="rtx_4070")
    with pytest.raises(ValueError, match="finite"):
        BudgetLedger(tmp_path / "nan", "rtx_4070", clock=clock("2026-09-10T12:00:00+08:00")).reserve("run", float("nan"), hardware="rtx_4070")


def test_tampered_allowance_is_refused(tmp_path):
    ledger = BudgetLedger(tmp_path, "rtx_4070", clock=clock("2026-09-10T12:00:00+08:00"))
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    ledger.path.write_text('{"schema":"pre_campaign_budget_v2","lane":"rtx_4070","allowance_seconds":1,"reservations":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        ledger.remaining_seconds()


@pytest.mark.parametrize("field,value", [("charged_seconds", -1), ("reserved_seconds", "NaN")])
def test_corrupt_reservation_values_are_refused(tmp_path, field, value):
    ledger = BudgetLedger(tmp_path, "rtx_4070", clock=clock("2026-09-10T12:00:00+08:00"))
    ledger.path.write_text(json.dumps({"schema": "pre_campaign_budget_v2", "lane": "rtx_4070", "allowance_seconds": LANE_SECONDS, "reservations": [{"token": "x", "status": "failed", "reserved_seconds": 1, "charged_seconds": 0, field: value}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="finite|non-negative"):
        ledger.remaining_seconds()
