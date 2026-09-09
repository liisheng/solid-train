"""Persistent, exclusive, non-transferable lane budget ledger."""

from __future__ import annotations

import json
import math
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LANE_HOURS = 6.0
LANE_SECONDS = LANE_HOURS * 3600.0
DEADLINE = "2026-09-12T23:59:59+08:00"
LANES = {"rtx_4070", "rtx_3070"}


def _parse(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("ledger timestamps must include a timezone")
    return result


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BudgetLedger:
    def __init__(self, root: Path, lane: str, *, clock=None):
        if lane not in LANES:
            raise ValueError("unknown budget lane")
        self.root = Path(root)
        self.lane = lane
        self.clock = clock or _now
        self.path = self.root / f"{lane}.ledger.json"
        self.lock_path = self.root / f"{lane}.ledger.lock"

    @contextmanager
    def _locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            handle = self.lock_path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise ValueError(
                "concurrent budget operation or stale ledger lock"
            ) from exc
        try:
            handle.write(str(os.getpid()))
            handle.flush()
            os.fsync(handle.fileno())
            yield
        finally:
            handle.close()
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def _load(self) -> dict:
        if not self.path.exists():
            return {
                "schema": "pre_campaign_budget_v2",
                "lane": self.lane,
                "allowance_seconds": LANE_SECONDS,
                "reservations": [],
            }
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            data.get("schema") != "pre_campaign_budget_v2"
            or data.get("lane") != self.lane
        ):
            raise ValueError("budget ledger schema/lane mismatch")
        if float(data.get("allowance_seconds")) != LANE_SECONDS:
            raise ValueError("budget allowance is immutable and must be six hours")
        allowed = {"reserved", "uncertain", "complete", "failed"}
        if not isinstance(data.get("reservations"), list):
            raise ValueError("budget reservation history is required")
        tokens = set()
        for item in data.get("reservations", []):
            if (
                item.get("status") not in allowed
                or not item.get("token")
                or item["token"] in tokens
            ):
                raise ValueError("invalid budget reservation history")
            tokens.add(item["token"])
            for key in ("reserved_seconds", "charged_seconds"):
                value = item.get(key, 0.0)
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(
                        "budget reservation values must be finite and non-negative"
                    )
            if item["status"] == "reserved" and item["charged_seconds"] != 0:
                raise ValueError("reserved entry cannot already be charged")
            if (
                item["status"] == "uncertain"
                and item["charged_seconds"] < item["reserved_seconds"]
            ):
                raise ValueError(
                    "uncertain entry must retain its full reservation charge"
                )
        return data

    def _save(self, data: dict) -> None:
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, self.path)

    def reserve(self, run_id: str, seconds: float, *, hardware: str | dict) -> str:
        hardware_identity = (
            hardware
            if isinstance(hardware, str)
            else json.dumps(hardware, sort_keys=True)
        )
        if not hardware_identity:
            raise ValueError("hardware identity is required")
        if (
            isinstance(hardware, str)
            and hardware.startswith("rtx_")
            and hardware != self.lane
        ):
            raise ValueError("hardware identity does not match lane")
        if (
            not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("reservation duration must be finite and positive")
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock must return an aware timestamp")
        remaining_deadline = (_parse(DEADLINE) - now).total_seconds()
        if remaining_deadline <= 0 or seconds > remaining_deadline:
            raise ValueError("reservation would cross the campaign deadline")
        with self._locked():
            data = self._load()
            if any(
                item.get("status") in {"reserved", "uncertain"}
                for item in data["reservations"]
            ):
                raise ValueError("unfinished reservation exists")
            prior_hardware = {
                item.get("hardware")
                for item in data["reservations"]
                if item.get("hardware")
            }
            if prior_hardware and hardware_identity not in prior_hardware:
                raise ValueError("hardware identity changed for lane ledger")
            charged = sum(
                float(item.get("charged_seconds", 0.0)) for item in data["reservations"]
            )
            pending = sum(
                float(item.get("reserved_seconds", 0.0))
                for item in data["reservations"]
                if item.get("status") == "reserved"
            )
            if charged + pending + seconds > LANE_SECONDS:
                raise ValueError("lane budget exhausted")
            token = uuid.uuid4().hex
            data["reservations"].append(
                {
                    "token": token,
                    "run_id": run_id,
                    "hardware": hardware_identity,
                    "lane": self.lane,
                    "reserved_seconds": float(seconds),
                    "charged_seconds": 0.0,
                    "status": "reserved",
                    "created_at": now.isoformat(),
                }
            )
            self._save(data)
            return token

    def settle(
        self,
        token: str,
        *,
        elapsed_seconds: float,
        exit_code: int | None,
        uncertain: bool = False,
    ) -> dict:
        if (
            not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(elapsed_seconds)
            or elapsed_seconds < 0
        ):
            raise ValueError("elapsed duration must be finite and non-negative")
        with self._locked():
            data = self._load()
            item = next(
                (entry for entry in data["reservations"] if entry["token"] == token),
                None,
            )
            if item is None or item.get("status") != "reserved":
                raise ValueError("unknown or already settled reservation")
            charged = (
                max(float(item["reserved_seconds"]), float(elapsed_seconds))
                if uncertain
                else float(elapsed_seconds)
            )
            status = (
                "uncertain"
                if uncertain
                else ("complete" if exit_code == 0 else "failed")
            )
            item.update(
                {
                    "charged_seconds": charged,
                    "elapsed_seconds": float(elapsed_seconds),
                    "status": status,
                    "exit_code": exit_code,
                    "settled_at": self.clock().isoformat(),
                }
            )
            self._save(data)
            return dict(item)

    def recover_uncertain(self, token: str, *, process_dead: bool) -> dict:
        if not process_dead:
            raise ValueError("operator must confirm the child process is dead")
        with self._locked():
            data = self._load()
            item = next(
                (entry for entry in data["reservations"] if entry["token"] == token),
                None,
            )
            if item is None or item.get("status") not in {"reserved", "uncertain"}:
                raise ValueError("reservation is not recoverable")
            item.update(
                {
                    "status": "failed",
                    "charged_seconds": max(
                        float(item.get("charged_seconds", 0.0)),
                        float(item["reserved_seconds"]),
                    ),
                    "recovered_at": self.clock().isoformat(),
                    "process_dead": True,
                }
            )
            self._save(data)
            return dict(item)

    def remaining_seconds(self) -> float:
        data = self._load()
        charged = sum(
            float(item.get("charged_seconds", 0.0)) for item in data["reservations"]
        )
        pending = sum(
            float(item.get("reserved_seconds", 0.0))
            for item in data["reservations"]
            if item.get("status") == "reserved"
        )
        return max(0.0, LANE_SECONDS - charged - pending)
