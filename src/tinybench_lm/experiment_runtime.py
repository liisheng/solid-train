"""Persistent advisory runtime accounting for one experiment lane.

This ledger records what the operator measured.  Estimates are retained as
advice only and never become a reservation, charge, deadline, or allowance.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .experiment_budget import BudgetLedger

LANES = {"rtx_4070", "rtx_3070"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionLedger:
    """An exclusive, append-history ledger for executions in one hardware lane."""

    def __init__(self, root: Path, lane: str, *, clock=None):
        if lane not in LANES:
            raise ValueError("unknown runtime lane")
        self.root = Path(root)
        self.lane = lane
        self.clock = clock or _now
        self.path = self.root / f"{lane}.runtime.json"
        self.lock_path = self.root / f"{lane}.runtime.lock"
        self.old_path = self.root / f"{lane}.ledger.json"
        self.old_lock_path = self.root / f"{lane}.ledger.lock"

    @contextmanager
    def _locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            handle = self.lock_path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise ValueError(
                "concurrent runtime operation or stale ledger lock"
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

    def _timestamp(self) -> str:
        value = self.clock()
        if value.tzinfo is None:
            raise ValueError("clock must return an aware timestamp")
        return value.astimezone(timezone.utc).isoformat()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"schema": "experiment_runtime_v1", "lane": self.lane, "entries": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid runtime ledger") from exc
        if (
            data.get("schema") != "experiment_runtime_v1"
            or data.get("lane") != self.lane
        ):
            raise ValueError("runtime ledger schema/lane mismatch")
        entries = data.get("entries")
        if not isinstance(entries, list):
            raise ValueError("runtime ledger entries are required")
        tokens = set()
        for item in entries:
            if (
                not isinstance(item, dict)
                or not item.get("token")
                or item["token"] in tokens
            ):
                raise ValueError("invalid runtime ledger entry")
            tokens.add(item["token"])
            if item.get("status") not in {
                "running",
                "completed",
                "failed",
                "uncertain",
                "interrupted",
            }:
                raise ValueError("invalid runtime entry status")
            if not isinstance(item.get("hardware"), str) or not item["hardware"]:
                raise ValueError("runtime entry hardware is required")
            elapsed = item.get("elapsed_seconds")
            if (
                item["status"] in {"completed", "failed", "uncertain"}
                and elapsed is None
            ):
                raise ValueError("finished runtime entry requires elapsed duration")
            if elapsed is not None and (
                not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise ValueError(
                    "runtime elapsed duration must be finite and non-negative"
                )
            estimate = item.get("estimate_seconds")
            if estimate is not None and (
                not isinstance(estimate, (int, float))
                or not math.isfinite(estimate)
                or estimate < 0
            ):
                raise ValueError("runtime estimate must be finite and non-negative")
        return data

    def _check_old_ledger(self) -> None:
        if self.old_lock_path.exists():
            raise ValueError("legacy budget operation is in progress")
        if not self.old_path.exists():
            return
        try:
            data = BudgetLedger(self.root, self.lane)._load()
            reservations = data["reservations"]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "cannot reconcile legacy budget ledger before starting"
            ) from exc
        if any(
            item.get("status") in {"reserved", "uncertain"}
            for item in reservations
        ):
            raise ValueError("legacy budget ledger has an unfinished reservation")

    def _save(self, data: dict) -> None:
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, self.path)

    def start(
        self, run_id: str, hardware: str, estimate_seconds: float | None = None
    ) -> str:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run identity is required")
        if not isinstance(hardware, str) or not hardware.strip():
            raise ValueError("hardware identity is required")
        if hardware.startswith("rtx_") and hardware != self.lane:
            raise ValueError("hardware identity does not match runtime lane")
        if estimate_seconds is not None and (
            not isinstance(estimate_seconds, (int, float))
            or not math.isfinite(estimate_seconds)
            or estimate_seconds < 0
        ):
            raise ValueError("runtime estimate must be finite and non-negative")
        with self._locked():
            self._check_old_ledger()
            data = self._load()
            if any(
                item.get("status") in {"running", "uncertain"}
                for item in data["entries"]
            ):
                raise ValueError("unfinished runtime execution exists")
            prior = {item["hardware"] for item in data["entries"]}
            if prior and hardware not in prior:
                raise ValueError("hardware identity changed for runtime lane")
            token = uuid.uuid4().hex
            data["entries"].append(
                {
                    "token": token,
                    "run_id": run_id,
                    "hardware": hardware,
                    "status": "running",
                    "estimate_seconds": (
                        float(estimate_seconds)
                        if estimate_seconds is not None
                        else None
                    ),
                    "created_at": self._timestamp(),
                }
            )
            self._save(data)
            return token

    def finish(
        self,
        token: str,
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
                (entry for entry in data["entries"] if entry["token"] == token), None
            )
            if item is None or item.get("status") != "running":
                raise ValueError("unknown or already finished execution")
            item.update(
                {
                    "elapsed_seconds": float(elapsed_seconds),
                    "exit_code": exit_code,
                    "status": "uncertain"
                    if uncertain
                    else ("completed" if exit_code == 0 else "failed"),
                    "finished_at": self._timestamp(),
                }
            )
            self._save(data)
            return dict(item)

    def recover(self, token: str, *, process_dead: bool) -> dict:
        if not process_dead:
            raise ValueError("operator must confirm the child process is dead")
        with self._locked():
            data = self._load()
            item = next(
                (entry for entry in data["entries"] if entry["token"] == token), None
            )
            if item is None or item.get("status") not in {"running", "uncertain"}:
                raise ValueError("execution is not recoverable")
            item.update(
                {
                    "status": "interrupted",
                    "duration_unknown": True,
                    "recovered_at": self._timestamp(),
                    "process_dead": True,
                }
            )
            self._save(data)
            return dict(item)

    def summary(self) -> dict:
        data = self._load()
        measured = sum(
            float(item.get("elapsed_seconds", 0.0)) for item in data["entries"]
        )
        unknown = sum(
            1
            for item in data["entries"]
            if item.get("duration_unknown")
            or item.get("status") in {"running", "uncertain"}
        )
        active = sum(1 for item in data["entries"] if item.get("status") == "running")
        return {
            "measured_elapsed_seconds": measured,
            "unknown_duration_attempts": unknown,
            "active_count": active,
        }
