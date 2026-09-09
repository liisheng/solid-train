import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("profile_telemetry", ROOT / "scripts/profile_telemetry.py")
assert SPEC and SPEC.loader
telemetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(telemetry)


def sample(monotonic: float, *, gpu_status: str = "MEASURED"):
    gpu = {"status": gpu_status}
    if gpu_status == "MEASURED":
        gpu.update({"index": 0, "uuid": "GPU-1", "name": "test", "driver_version": "1", "vram_total_mib": 100.0, "vram_used_mib": 60.0,
                    "utilization_percent": 90.0, "temperature_c": 70.0, "throttle_reasons_active": "0x0000000000000000"})
    else:
        gpu["reason"] = "unavailable"
    return {"timestamp_utc": "2026-09-09T00:00:00+00:00", "monotonic_seconds": monotonic,
            "collection_seconds": 0.1, "gpu": gpu,
            "system_memory": {"status": "MEASURED", "ram_total_bytes": 1000, "ram_available_bytes": 250,
                              "swap_total_bytes": 500, "swap_available_bytes": 400}}


def test_summary_reports_machine_peaks_ranges_and_cadence():
    report = telemetry.summarize_telemetry([sample(10.0), sample(15.0)])
    assert report["status"] == "MEASURED"
    assert report["peak_ram_used_bytes"] == 750
    assert report["peak_swap_used_bytes"] == 100
    assert report["peak_vram_used_mib"] == 60.0
    assert report["minimum_vram_headroom_mib"] == 40.0
    assert report["collection_duration_seconds"] == 5.0
    assert report["max_gap_seconds"] == 5.0


def test_missing_telemetry_and_sampling_gaps_remain_not_run():
    unavailable = telemetry.summarize_telemetry([sample(1.0, gpu_status="NOT_RUN")])
    assert unavailable["status"] == "NOT_RUN"
    assert unavailable["failures"][0]["reason"] == "unavailable"
    gapped = telemetry.summarize_telemetry([sample(1.0), sample(17.0)])
    assert gapped["status"] == "NOT_RUN"
    assert "gap" in gapped["failures"][-1]["reason"]


def test_windows_pagefile_peaks_are_reported_without_calling_host_apis():
    left, right = sample(1.0), sample(6.0)
    for item, current, peak in ((left, 10, 20), (right, 15, 25)):
        item["system_memory"].pop("swap_total_bytes")
        item["system_memory"].pop("swap_available_bytes")
        item["system_memory"].update({"pagefile_current_bytes": current, "pagefile_peak_bytes": peak})
    report = telemetry.summarize_telemetry([left, right])
    assert report["status"] == "MEASURED"
    assert report["peak_pagefile_used_bytes"] == 15
    assert report["pagefile_peak_since_boot_bytes"] == 25


def test_invalid_physical_values_fail_closed():
    invalid = sample(1.0)
    invalid["gpu"]["utilization_percent"] = 101
    report = telemetry.summarize_telemetry([invalid])
    assert report["status"] == "NOT_RUN"
