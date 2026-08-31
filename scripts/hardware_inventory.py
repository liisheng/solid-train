"""Record the Plan Section 9.1 hardware promotion measurements for one machine.

Run this on each machine separately and commit the resulting JSON. Plan Section 9.1 is one
instruction -- "Measure rather than assume" -- so this script measures what it can locally and
records everything else as ``NOT_RUN`` with the reason it could not be taken. It never fills a
field with a plausible number.

What it measures without any corpus:

* exact GPU name, VRAM, compute capability, driver and torch build,
* BF16 support **and** a stability probe (a real optimizer loop checked for NaN/Inf),
* the maximum safe microbatch leaving at least 10% VRAM headroom, by sweeping and recording
  peak allocation at each size,
* short-burst tokens/s at each microbatch, labelled a capacity probe and never a throughput
  claim,
* system RAM, and active clock-throttle reasons from ``nvidia-smi``.

What it deliberately does **not** measure without real shards:

* sustained throughput. Plan Section 9.1 requires a 30-60 minute window on real shards; a
  synthetic loop is not that, and :func:`throughput_violations` will reject one. Pass
  ``--shard-root`` with ``--sustained-seconds`` once G1 shards exist.
* dataloader wait and checkpoint time, which are properties of the real data path.

Usage::

    python scripts/hardware_inventory.py
    python scripts/hardware_inventory.py --machine-id rtx_4070
    python scripts/hardware_inventory.py --summarize

Exit codes:
    0  the inventory was written
    1  a measurement failed, or a supplied sustained window does not satisfy Section 9.1
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch  # noqa: E402

from tinybench_lm import ModelConfig, TinyBenchLM  # noqa: E402
from tinybench_lm.config import FINAL_CONFIG_PATH  # noqa: E402
from tinybench_lm.operations import (  # noqa: E402
    MACHINE_3070,
    MACHINE_4070,
    MicrobatchProbe,
    ThroughputMeasurement,
    minimum_vram_headroom,
    nearest_rank_percentile,
    safe_microbatch,
    throughput_violations,
)
from tinybench_lm.parameters import count_unique_trainable_parameters  # noqa: E402
from tinybench_lm.training_recipe import (  # noqa: E402
    adamw_parameter_groups,
    adamw_settings,
)

NOT_RUN = "NOT_RUN"

#: Inventory records are evidence, not run output, so they live in a tracked directory.
#: `runs/` is gitignored, which would leave the other machine's record uncommittable.
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "evidence" / "hardware"

#: nvidia-smi clock-throttle bits (NVML). 0x1 is simply "GPU idle" and is not a throttle.
_THROTTLE_BITS: dict[int, str] = {
    0x0000000000000002: "applications_clocks_setting",
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000010: "sync_boost",
    0x0000000000000020: "sw_thermal_slowdown",
    0x0000000000000040: "hw_thermal_slowdown",
    0x0000000000000080: "hw_power_brake_slowdown",
    0x0000000000000100: "display_clock_setting",
}


def detect_machine_id(gpu_name: str) -> str | None:
    """Map a GPU name onto one of the plan's two lanes, or return None to force a choice."""
    lowered = gpu_name.lower()
    if "4070" in lowered:
        return MACHINE_4070
    if "3070" in lowered:
        return MACHINE_3070
    return None


def system_memory_bytes() -> dict[str, Any]:
    """Total and available system RAM using only the standard library."""
    if platform.system() == "Windows":
        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            # Windows reports a combined commit limit rather than a separate swap device, so
            # the page-file figure is recorded under its own name instead of as "swap".
            return {
                "total_bytes": int(status.ullTotalPhys),
                "available_bytes": int(status.ullAvailPhys),
                "total_pagefile_bytes": int(status.ullTotalPageFile),
                "available_pagefile_bytes": int(status.ullAvailPageFile),
            }
        return {"total_bytes": NOT_RUN, "reason": "GlobalMemoryStatusEx failed"}

    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) * 1024
        return {
            "total_bytes": values.get("MemTotal", NOT_RUN),
            "available_bytes": values.get("MemAvailable", NOT_RUN),
            "total_swap_bytes": values.get("SwapTotal", NOT_RUN),
            "available_swap_bytes": values.get("SwapFree", NOT_RUN),
        }
    return {"total_bytes": NOT_RUN, "reason": f"unsupported platform {platform.system()!r}"}


def throttle_reasons() -> Any:
    """Active clock-throttle reasons from nvidia-smi, or NOT_RUN when it is unavailable."""
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks_throttle_reasons.active",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError) as error:
        return {"status": NOT_RUN, "reason": f"nvidia-smi unavailable: {type(error).__name__}"}
    try:
        mask = int(output, 16)
    except ValueError:
        return {"status": NOT_RUN, "reason": f"unparsable nvidia-smi output {output!r}"}
    active = sorted(name for bit, name in _THROTTLE_BITS.items() if mask & bit)
    return {"status": "MEASURED", "raw": output, "active": active}


def bf16_stability_probe(model: torch.nn.Module, device: torch.device, steps: int = 20) -> dict[str, Any]:
    """Plan 9.1 asks for BF16 *stability*, not merely support.

    Support is a hardware fact; stability is an observation. A short optimizer loop whose loss
    and gradients stay finite is weak evidence of stability, so it is recorded as exactly that
    -- a bounded probe over ``steps`` updates, not a verdict about the full campaign.
    """
    if not torch.cuda.is_bf16_supported():
        return {"supported": False, "probe": NOT_RUN, "reason": "BF16 unsupported on this device"}

    settings = adamw_settings()
    optimizer = torch.optim.AdamW(
        adamw_parameter_groups(model),
        lr=6e-4,
        betas=tuple(settings["betas"]),
        eps=float(settings["epsilon"]),
    )
    config = model.config
    inputs = torch.randint(0, config.vocab_size, (2, config.max_seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (2, config.max_seq_len), device=device)

    losses: list[float] = []
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(inputs, targets)
        if not torch.isfinite(loss):
            return {"supported": True, "probe": "UNSTABLE", "steps": steps, "reason": "non-finite loss"}
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                return {
                    "supported": True,
                    "probe": "UNSTABLE",
                    "steps": steps,
                    "reason": "non-finite gradient",
                }
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_global_norm"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))

    return {
        "supported": True,
        "probe": "STABLE_OVER_PROBE",
        "steps": steps,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "note": f"bounded {steps}-update probe on synthetic tokens; not a campaign-length verdict",
    }


def sweep_microbatches(
    config: ModelConfig, device: torch.device, sizes: list[int], iterations: int
) -> tuple[list[MicrobatchProbe], list[dict[str, Any]]]:
    """Record peak VRAM and short-burst tokens/s at each microbatch, stopping at OOM."""
    total = torch.cuda.get_device_properties(device).total_memory
    settings = adamw_settings()
    probes: list[MicrobatchProbe] = []
    rows: list[dict[str, Any]] = []

    for size in sizes:
        model = optimizer = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(0)
            model = TinyBenchLM(config).to(device)
            optimizer = torch.optim.AdamW(
                adamw_parameter_groups(model),
                lr=6e-4,
                betas=tuple(settings["betas"]),
                eps=float(settings["epsilon"]),
            )
            inputs = torch.randint(0, config.vocab_size, (size, config.max_seq_len), device=device)
            targets = torch.randint(0, config.vocab_size, (size, config.max_seq_len), device=device)

            for _ in range(2):  # warm up allocator and autotuner before timing
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(inputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for _ in range(iterations):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(inputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(settings["gradient_clip_global_norm"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) / iterations
            peak = int(torch.cuda.max_memory_allocated())

            probe = MicrobatchProbe(
                microbatch=size, peak_vram_bytes=peak, vram_total_bytes=total
            )
            probes.append(probe)
            rows.append(
                {
                    "microbatch": size,
                    "peak_vram_bytes": peak,
                    "headroom_fraction": probe.headroom_fraction,
                    "meets_headroom": probe.headroom_fraction >= minimum_vram_headroom(),
                    "burst_tokens_per_second": size * config.max_seq_len / elapsed,
                    "seconds_per_update": elapsed,
                }
            )
        except torch.cuda.OutOfMemoryError:
            rows.append({"microbatch": size, "result": "OOM"})
            break
        finally:
            del model, optimizer
            torch.cuda.empty_cache()
    return probes, rows


def build_sustained(
    shard_root: Path | None, seconds: float | None, machine_id: str
) -> dict[str, Any]:
    """Sustained throughput needs real shards and a 30-60 minute window, or it is NOT_RUN."""
    if shard_root is None or seconds is None:
        return {
            "status": NOT_RUN,
            "reason": (
                "Plan Section 9.1 measures 30-60 minutes of sustained throughput on real "
                "shards; no shard root was supplied. Re-run with --shard-root and "
                "--sustained-seconds once G1 shards exist."
            ),
            "owner": "operator",
            "next_action": "complete G1 shard production, then re-run this script with --shard-root",
        }
    return {
        "status": NOT_RUN,
        "reason": (
            "sustained measurement over real shards is not implemented here; it belongs to the "
            "G1/G2 run path where a real ScheduledTokenStream is already open"
        ),
        "owner": "operator",
        "next_action": f"run the sustained measurement on {shard_root} for {seconds}s on {machine_id}",
    }


def summarize(output_dir: Path) -> int:
    """Report which Section 9.1 fields are still missing, across every recorded machine."""
    records = sorted(output_dir.glob("*.json"))
    if not records:
        print(f"No inventory records in {output_dir}. Run this script on each machine first.")
        return 1
    print(f"{'machine':<12} {'GPU':<28} {'VRAM':>9} {'safe mb':>8} {'BF16':>18} {'sustained':>10}")
    for path in records:
        payload = json.loads(path.read_text(encoding="utf-8"))
        gpu = payload.get("gpu", {})
        bf16 = payload.get("bf16", {})
        print(
            f"{payload.get('machine_id', '?'):<12} "
            f"{str(gpu.get('name', '?'))[:28]:<28} "
            f"{gpu.get('vram_total_bytes', 0) / 1024**3:>8.2f}G "
            f"{str(payload.get('safe_microbatch', NOT_RUN)):>8} "
            f"{str(bf16.get('probe', NOT_RUN)):>18} "
            f"{str(payload.get('sustained_throughput', {}).get('status', NOT_RUN)):>10}"
        )
    missing = [
        payload["machine_id"]
        for path in records
        if (payload := json.loads(path.read_text(encoding="utf-8")))
        and payload.get("sustained_throughput", {}).get("status") != "MEASURED"
    ]
    print()
    print(f"Machines recorded: {len(records)} of 2 lanes.")
    if missing:
        print(f"Sustained throughput still NOT_RUN for: {missing}")
        print("G0 cannot pass on these records alone; absence of evidence is never a PASS.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=FINAL_CONFIG_PATH)
    parser.add_argument(
        "--machine-id",
        choices=[MACHINE_4070, MACHINE_3070],
        help="which lane this machine is; auto-detected from the GPU name when omitted",
    )
    parser.add_argument(
        "--microbatches",
        default="1,2,4,6,8,12,16",
        help="comma-separated microbatch sizes to sweep",
    )
    parser.add_argument("--iterations", type=int, default=5, help="timed iterations per microbatch")
    parser.add_argument("--bf16-probe-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--sustained-seconds", type=float, default=None)
    parser.add_argument("--summarize", action="store_true", help="summarize recorded machines and exit")
    arguments = parser.parse_args()

    if arguments.summarize:
        return summarize(arguments.output_dir)

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device. Section 9.1 measures the training GPU; nothing to record.")
        return 1

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    machine_id = arguments.machine_id or detect_machine_id(properties.name)
    if machine_id is None:
        print(
            f"FAIL: cannot map GPU {properties.name!r} onto a plan lane. "
            f"Re-run with --machine-id {MACHINE_4070} or --machine-id {MACHINE_3070}."
        )
        return 1

    config = ModelConfig.from_json(arguments.config)
    sizes = [int(value) for value in str(arguments.microbatches).split(",") if value.strip()]

    print(f"Recording Section 9.1 inventory for {machine_id} ({properties.name})\n")

    model = TinyBenchLM(config).to(device)
    parameters = count_unique_trainable_parameters(model)
    bf16 = bf16_stability_probe(model, device, steps=arguments.bf16_probe_steps)
    del model
    torch.cuda.empty_cache()

    probes, rows = sweep_microbatches(config, device, sizes, arguments.iterations)
    try:
        chosen = safe_microbatch(probes)
    except Exception as error:  # noqa: BLE001 - reported, never guessed
        chosen = NOT_RUN
        print(f"safe_microbatch: {error}")

    sustained = build_sustained(arguments.shard_root, arguments.sustained_seconds, machine_id)
    if arguments.shard_root is not None and arguments.sustained_seconds is not None:
        proposed = ThroughputMeasurement(
            machine_id=machine_id,
            samples=(1.0, 1.0, 1.0),
            window_seconds=float(arguments.sustained_seconds),
            used_real_shards=True,
        )
        problems = throughput_violations(proposed)
        if problems:
            print("The supplied sustained window does not satisfy Section 9.1:")
            for problem in problems:
                print(f"  {problem}")
            return 1

    record: dict[str, Any] = {
        "schema_version": 1,
        "machine_id": machine_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "gpu": {
            "name": properties.name,
            "vram_total_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "multi_processor_count": int(properties.multi_processor_count),
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
        },
        "system_memory": system_memory_bytes(),
        "bf16": bf16,
        "model": {"config": str(arguments.config), "unique_trainable_parameters": parameters},
        "microbatch_sweep": rows,
        "safe_microbatch": chosen,
        "minimum_headroom_fraction": minimum_vram_headroom(),
        "burst_probe_note": (
            "burst_tokens_per_second is a short synthetic capacity probe, not a throughput "
            "claim. Section 9.1 sustained throughput is recorded separately and is NOT_RUN "
            "until measured on real shards."
        ),
        "sustained_throughput": sustained,
        "dataloader_wait_seconds": {
            "status": NOT_RUN,
            "reason": "a property of the real data path; needs G1 shards",
        },
        "checkpoint_seconds": {
            "status": NOT_RUN,
            "reason": "measured during a real run; needs G1 shards",
        },
        "throttle": throttle_reasons(),
        "backend": {
            "current": "eager_sdpa",
            "promotion": {
                "status": NOT_RUN,
                "reason": "Plan 9.1 promotes a backend only on correctness plus sustained improvement",
            },
        },
    }

    if isinstance(chosen, int):
        loss_tokens = 262_144
        per_microbatch = chosen * config.max_seq_len
        record["gradient_accumulation_for_frozen_batch"] = (
            loss_tokens // per_microbatch if loss_tokens % per_microbatch == 0 else NOT_RUN
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    destination = arguments.output_dir / f"{machine_id}.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{'micro':>6} {'peak VRAM':>11} {'headroom':>9} {'burst tok/s':>13}  verdict")
    for row in rows:
        if row.get("result") == "OOM":
            print(f"{row['microbatch']:>6} {'OOM':>11}")
            continue
        verdict = "ok" if row["meets_headroom"] else "< 10% headroom"
        print(
            f"{row['microbatch']:>6} {row['peak_vram_bytes'] / 1024**3:>9.2f}GiB "
            f"{row['headroom_fraction']:>8.1%} {row['burst_tokens_per_second']:>13,.0f}  {verdict}"
        )

    memory = record["system_memory"]
    total_ram = memory.get("total_bytes")
    print()
    print(f"safe microbatch          {chosen}")
    if "gradient_accumulation_for_frozen_batch" in record:
        print(f"gradient accumulation    {record['gradient_accumulation_for_frozen_batch']}  (262,144 loss tokens/update)")
    print(f"BF16                     supported={bf16.get('supported')} probe={bf16.get('probe')}")
    if isinstance(total_ram, int):
        print(f"system RAM               {total_ram / 1024**3:.1f} GiB total")
    print(f"throttle reasons         {record['throttle'].get('active', record['throttle'].get('status'))}")
    print(f"sustained throughput     {sustained['status']}  <- {sustained['reason'][:60]}...")
    print()
    print(f"Wrote {destination.relative_to(REPOSITORY_ROOT)}")
    print("Commit it, and run the same command on the other machine. Then --summarize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
