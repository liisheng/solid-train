"""Bounded telemetry collection for the G4 real-shard profile.

The collector is deliberately read-only.  A missing platform query is evidence of an
unavailable measurement, never a zero-valued measurement.
"""
from __future__ import annotations

import ctypes
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CADENCE_SECONDS = 5.0
QUERY_TIMEOUT_SECONDS = 5.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _not_run(reason: str) -> dict[str, Any]:
    return {"status": "NOT_RUN", "reason": reason}


def _windows_memory() -> dict[str, Any]:
    class Status(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
            ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
            ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
            ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
            ("extended", ctypes.c_ulonglong),
        ]

    status = Status()
    status.length = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return _not_run("GlobalMemoryStatusEx failed")
    try:
        output = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "$ErrorActionPreference='Stop'; ConvertTo-Json -Compress -InputObject @(@(Get-CimInstance Win32_PageFileUsage | Select-Object CurrentUsage,PeakUsage))"],
            capture_output=True, text=True, check=True, timeout=QUERY_TIMEOUT_SECONDS,
        ).stdout.strip()
        pagefiles = json.loads(output) if output else []
        if isinstance(pagefiles, Mapping):
            pagefiles = [pagefiles]
        if not isinstance(pagefiles, list):
            raise ValueError("Win32_PageFileUsage result is not a list")
        current_mib = sum(float(item["CurrentUsage"]) for item in pagefiles)
        peak_mib = sum(float(item["PeakUsage"]) for item in pagefiles)
        if not all(math.isfinite(value) and value >= 0 for value in (current_mib, peak_mib)):
            raise ValueError("invalid pagefile usage")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return _not_run(f"Win32_PageFileUsage unavailable: {type(error).__name__}: {error}")
    return {
        "status": "MEASURED", "ram_total_bytes": int(status.total),
        "ram_available_bytes": int(status.available), "pagefile_current_bytes": int(current_mib * 2**20),
        "pagefile_peak_bytes": int(peak_mib * 2**20),
    }


def _linux_memory() -> dict[str, Any]:
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, tail = line.partition(":")
            parts = tail.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) * 1024
        return {
            "status": "MEASURED", "ram_total_bytes": values["MemTotal"],
            "ram_available_bytes": values["MemAvailable"], "swap_total_bytes": values["SwapTotal"],
            "swap_available_bytes": values["SwapFree"],
        }
    except (OSError, KeyError, ValueError) as error:
        return _not_run(f"/proc/meminfo unavailable: {type(error).__name__}: {error}")


def system_memory() -> dict[str, Any]:
    """Read machine-level RAM and actual swap/pagefile use, bounded by platform queries."""
    return _windows_memory() if platform.system() == "Windows" else _linux_memory()


def _gpu() -> dict[str, Any]:
    query = "index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,clocks_throttle_reasons.active"
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=QUERY_TIMEOUT_SECONDS,
        ).stdout.strip().splitlines()
        if len(output) != 1:
            raise ValueError(f"expected one GPU, observed {len(output)}")
        fields = [field.strip() for field in output[0].split(",")]
        if len(fields) != 9:
            raise ValueError("unexpected nvidia-smi field count")
        result = {
            "status": "MEASURED", "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
            "driver_version": fields[3], "vram_total_mib": float(fields[4]),
            "vram_used_mib": float(fields[5]), "utilization_percent": float(fields[6]),
            "temperature_c": float(fields[7]), "throttle_reasons_active": fields[8],
        }
        numeric = ("vram_total_mib", "vram_used_mib", "utilization_percent", "temperature_c")
        if not result["uuid"] or any(not math.isfinite(result[name]) or result[name] < 0 for name in numeric):
            raise ValueError("non-finite or negative GPU telemetry")
        return result
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return _not_run(f"nvidia-smi telemetry unavailable: {type(error).__name__}: {error}")


def sample_telemetry() -> dict[str, Any]:
    """Collect one read-only sample; each external command has a five-second timeout."""
    started = time.monotonic()
    sample = {"timestamp_utc": _utc_now(), "monotonic_seconds": started, "system_memory": system_memory(), "gpu": _gpu()}
    sample["collection_seconds"] = time.monotonic() - started
    return sample


def summarize_telemetry(samples: Sequence[Mapping[str, Any]], *, cadence_seconds: float = DEFAULT_CADENCE_SECONDS) -> dict[str, Any]:
    """Summarize machine-level sampled peaks, or name why telemetry cannot be used."""
    if not isinstance(cadence_seconds, (int, float)) or cadence_seconds <= 0 or not math.isfinite(cadence_seconds):
        raise ValueError("cadence_seconds must be finite and positive")
    summary: dict[str, Any] = {
        "cadence_seconds": float(cadence_seconds), "sample_count": len(samples),
        "units": {"memory": "bytes", "gpu_memory": "MiB", "utilization": "percent", "temperature": "degrees_C"},
    }
    if not samples:
        return {**summary, **_not_run("no telemetry samples")}
    gpu = [sample.get("gpu", {}) for sample in samples]
    memory = [sample.get("system_memory", {}) for sample in samples]
    monotonic = [sample.get("monotonic_seconds") for sample in samples]
    failures = [item for item in [*gpu, *memory] if not isinstance(item, Mapping) or item.get("status") != "MEASURED"]
    def number(value: Any, *, positive: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("non-finite numeric telemetry")
        value = float(value)
        if value < 0 or (positive and value == 0):
            raise ValueError("negative or zero physical telemetry")
        return value

    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in monotonic):
        failures.append(_not_run("missing or invalid monotonic timestamp"))
    for sample in samples:
        try:
            datetime.fromisoformat(str(sample["timestamp_utc"]).replace("Z", "+00:00"))
            number(sample["collection_seconds"])
        except (KeyError, TypeError, ValueError) as error:
            failures.append(_not_run(f"invalid timestamp or collection duration: {type(error).__name__}: {error}"))
    gaps = [float(right) - float(left) for left, right in zip(monotonic, monotonic[1:]) if isinstance(left, (int, float)) and isinstance(right, (int, float))]
    if any(gap <= 0 or gap > float(cadence_seconds) * 3 for gap in gaps):
        failures.append(_not_run("telemetry sampling gap exceeds three cadence intervals"))
    if failures:
        return {**summary, "status": "NOT_RUN", "failures": failures, "collection_duration_seconds": None, "max_gap_seconds": max(gaps, default=0.0)}
    try:
        for item in gpu:
            if item.get("index") != 0 or type(item.get("index")) is not int:
                raise ValueError("profile requires GPU index zero")
            if not all(isinstance(item.get(name), str) and item[name].strip() for name in ("uuid", "name", "driver_version")):
                raise ValueError("GPU identity fields are empty")
            total, used = number(item["vram_total_mib"], positive=True), number(item["vram_used_mib"])
            utilization = number(item["utilization_percent"])
            number(item["temperature_c"])
            if used > total or utilization > 100:
                raise ValueError("GPU physical range is invalid")
            if int(str(item["throttle_reasons_active"]), 16) < 0:
                raise ValueError("negative throttle-reason mask")
        if any(item["uuid"] != gpu[0]["uuid"] or item["vram_total_mib"] != gpu[0]["vram_total_mib"] for item in gpu):
            raise ValueError("GPU telemetry identity changed during collection")
        ram_used = []
        for item in memory:
            total, available = number(item["ram_total_bytes"], positive=True), number(item["ram_available_bytes"])
            if available > total:
                raise ValueError("RAM availability exceeds total")
            if "pagefile_current_bytes" in item:
                current, peak = number(item["pagefile_current_bytes"]), number(item["pagefile_peak_bytes"])
                if current > peak:
                    raise ValueError("pagefile current use exceeds observed peak")
            else:
                swap_total, swap_available = number(item["swap_total_bytes"]), number(item["swap_available_bytes"])
                if swap_available > swap_total:
                    raise ValueError("swap availability exceeds total")
            ram_used.append(total - available)
        result: dict[str, Any] = {
            **summary, "status": "MEASURED", "collection_duration_seconds": float(monotonic[-1]) - float(monotonic[0]),
            "max_gap_seconds": max(gaps, default=0.0), "gpu_uuid": gpu[0]["uuid"], "gpu_name": gpu[0]["name"],
            "vram_total_mib": gpu[0]["vram_total_mib"], "peak_vram_used_mib": max(item["vram_used_mib"] for item in gpu),
            "minimum_vram_headroom_mib": min(item["vram_total_mib"] - item["vram_used_mib"] for item in gpu),
            "minimum_vram_headroom_fraction": min((item["vram_total_mib"] - item["vram_used_mib"]) / item["vram_total_mib"] for item in gpu),
            "peak_ram_used_bytes": max(ram_used), "minimum_ram_available_bytes": min(item["ram_available_bytes"] for item in memory),
            "gpu_utilization_percent": {"minimum": min(item["utilization_percent"] for item in gpu), "maximum": max(item["utilization_percent"] for item in gpu)},
            "gpu_temperature_c": {"minimum": min(item["temperature_c"] for item in gpu), "maximum": max(item["temperature_c"] for item in gpu)},
            "throttle_reasons_observed": sorted({item["throttle_reasons_active"] for item in gpu}),
        }
        if "pagefile_current_bytes" in memory[0]:
            result["peak_pagefile_used_bytes"] = max(item["pagefile_current_bytes"] for item in memory)
            result["pagefile_peak_since_boot_bytes"] = max(item["pagefile_peak_bytes"] for item in memory)
        else:
            swap_used = [float(item["swap_total_bytes"]) - float(item["swap_available_bytes"]) for item in memory]
            if any(value < 0 for value in swap_used):
                raise ValueError("swap availability exceeds total")
            result["peak_swap_used_bytes"] = max(swap_used)
        return result
    except (KeyError, TypeError, ValueError) as error:
        return {**summary, **_not_run(f"invalid measured telemetry: {type(error).__name__}: {error}"), "collection_duration_seconds": None, "max_gap_seconds": max(gaps, default=0.0)}
