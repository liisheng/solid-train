"""CPU preparation/planning and explicitly acknowledged one-job execution for v2."""

from __future__ import annotations
import argparse
import json
import sys
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from tinybench_lm.experiments import (
    ROOT,
    CONTRACT_SHA256,
    check_bundle,
    contract,
    digest,
    job_spec,
    object_hash,
    quotas,
    source_identity,
    verify_run,
)
from tinybench_lm.experiment_runtime import ExecutionLedger
from tinybench_lm.schedule import (
    assert_schedule_valid,
    build_materialized_schedule,
    load_schedule,
    write_schedule,
)
from tinybench_lm.shards import load_split_manifest, verify_shard_files
from tinybench_lm.training_recipe import model_config_hash

DEFAULT_BUNDLE = ROOT / "runs/pre_campaign/v2-advisory"
SHARDS = ROOT / "data/shards/reduced_5pct_v1"


def inputs() -> dict[str, Path]:
    return {
        "train_manifest": SHARDS / "stable_train.manifest.json",
        "validation_manifest": SHARDS / "validation_dev.manifest.json",
        "validation_schedule": ROOT
        / "data/schedules/reduced_5pct_v1/validation_dev.json",
    }


def verify_inputs() -> None:
    from scripts.run_reduced_baseline import _verified_baseline_config

    baseline = _verified_baseline_config()
    paths = inputs()
    expected = {
        "train_manifest": baseline["data"]["manifests"]["stable_train"]["file_sha256"],
        "validation_manifest": baseline["data"]["manifests"]["validation_dev"][
            "file_sha256"
        ],
        "validation_schedule": baseline["schedule_inputs"]["development_schedule"][
            "file_sha256"
        ],
    }
    for key, path in paths.items():
        if not path.is_file() or digest(path) != expected[key]:
            raise ValueError(f"production input does not match trusted pin: {key}")
    tokenizer = ROOT / baseline["tokenizer"]["artifact_path"]
    if digest(tokenizer) != baseline["tokenizer"]["artifact_sha256"]:
        raise ValueError("tokenizer differs from trusted pin")
    for key in ("train_manifest", "validation_manifest"):
        manifest = load_split_manifest(paths[key])
        failures = [
            item.check_id
            for item in verify_shard_files(SHARDS, manifest)
            if item.failed
        ]
        if failures:
            raise ValueError(f"shard integrity failed for {key}: {failures[:3]}")


def write_once(path: Path, value: dict) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to overwrite evidence: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def prepare(bundle: Path) -> dict:
    policy = contract()
    bundle = bundle.resolve()
    verify_inputs()
    bundle.mkdir(parents=True, exist_ok=True)
    from scripts.run_reduced_baseline import _verified_baseline_config

    trusted = _verified_baseline_config()
    trusted_config = (
        ROOT / trusted["model_config_path"]
        if "model_config_path" in trusted
        else ROOT / "configs/final_49m.json"
    )
    config = trusted_config
    if not config.is_file():
        raise ValueError("missing frozen final_49m config")
    files = {"final.model.json": digest(config)}
    target = bundle / "final.model.json"
    if not target.exists():
        target.write_bytes(config.read_bytes())
    elif digest(target) != files["final.model.json"]:
        raise ValueError("final model config differs")
    manifest = load_split_manifest(inputs()["train_manifest"])
    for mixture, shares in policy["mixtures"].items():
        schedule = build_materialized_schedule(
            manifest,
            sequence_length=1024,
            seed=1001,
            source_sequence_quotas=quotas(policy["final"]["updates"] * 256, shares),
        )
        assert_schedule_valid(manifest, schedule)
        path = bundle / f"final.{mixture}.schedule.json"
        if (
            path.exists()
            and load_schedule(path).content_hash() != schedule.content_hash()
        ):
            raise ValueError(f"existing schedule differs: {mixture}")
        if not path.exists():
            write_schedule(path, schedule)
        files[path.name] = digest(path)
    result = {
        "schema": "pre_campaign_bundle_v2",
        "contract_sha256": CONTRACT_SHA256,
        "source_identity": source_identity(),
        "files": files,
        "input_hashes": {k: digest(v) for k, v in inputs().items()},
        "status": "PREPARED_NOT_TRAINED",
        "jobs": policy["jobs"],
    }
    write_once(bundle / "bundle.json", result)
    return result


def identity_for(bundle: Path, job: str, selected: dict | None = None) -> dict:
    check_bundle(bundle, production=False)
    spec = job_spec(job, selected)
    config = Path(bundle) / "final.model.json"
    input_hashes = {"config": digest(config)}
    mixture_schedule = Path(bundle) / f"final.{spec['mixture']}.schedule.json"
    for name, candidate in {
        "train_manifest": inputs().get("train_manifest"),
        "validation_manifest": inputs().get("validation_manifest"),
        "validation_schedule": inputs().get("validation_schedule"),
        "train_schedule": mixture_schedule,
    }.items():
        if candidate is None:
            continue
        if candidate.is_file():
            input_hashes[name] = digest(candidate)
    result = {
        "identity_schema": "pre_campaign_runner_v2",
        "job": job,
        "spec": spec,
        "selected": selected,
        "contract_sha256": CONTRACT_SHA256,
        "source_identity": source_identity(),
        "bundle_sha256": digest(Path(bundle) / "bundle.json"),
        "input_hashes": input_hashes,
        "model_config_sha256": digest(config),
        "model_config_canonical_hash": digest(config),
        "train_schedule_content_hash": load_schedule(mixture_schedule).content_hash(),
        "validation_schedule_hash": digest(inputs()["validation_schedule"]),
        "validation_mode": "full-dev",
        "eval_interval": 100,
        "save_interval": 100,
        "seed": spec["seed"],
        "total_updates": spec["updates"],
        "peak_lr": spec["lr"],
        "warmup_updates": spec["warmup"],
        "stable_updates": spec["stable"],
        "decay_updates": spec["decay"],
        "micro_batch_size": 8,
        "gradient_accumulation": 32,
        "sequence_length": 1024,
        "precision": {"dtype": "bfloat16", "grad_scaler": False},
    }
    result["model_config_canonical_hash"] = model_config_hash(
        json.loads(config.read_text(encoding="utf-8"))
    )
    result["validation_schedule_hash"] = load_schedule(
        inputs()["validation_schedule"]
    ).content_hash()
    result["run_id"] = "experiment-" + object_hash(result)[:16]
    return result


def command_for(
    bundle: Path, job: str, run: Path, identity: dict, *, resume=False, smoke=False
) -> list[str]:
    spec = identity["spec"]
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config",
        str(Path(bundle) / "final.model.json"),
        "--run-dir",
        str(run),
        "--runner-identity",
        str(run / "runner_identity.json"),
        "--steps",
        str(spec["updates"]),
        "--seed",
        str(spec["seed"]),
        "--learning-rate",
        str(spec["lr"]),
        "--warmup-steps",
        str(spec["warmup"]),
        "--decay-updates",
        str(spec["decay"]),
        "--micro-batch-size",
        "8",
        "--gradient-accumulation",
        "32",
        "--sequence-length",
        "1024",
        "--bf16-stability",
        "stable",
        "--validation-mode",
        "full-dev",
        "--eval-interval",
        "100",
        "--save-interval",
        "100",
    ]
    candidates = {
        "shard_root": SHARDS,
        "train_manifest": inputs()["train_manifest"],
        "validation_manifest": inputs()["validation_manifest"],
        "validation_schedule": inputs()["validation_schedule"],
        "train_schedule": Path(bundle)
        / f"final.{identity['spec']['mixture']}.schedule.json",
    }
    for name, candidate in candidates.items():
        cmd.extend(["--" + name.replace("_", "-"), str(candidate)])
    cmd.append("--experiment-slice-reporting")
    if resume:
        cmd += ["--resume", str(run / "latest.pt")]
    if smoke:
        cmd += ["--stop-after-updates", "2"]
    return cmd


def inspect_hardware() -> dict:
    # Driver inventory does not create a CUDA context or run a GPU probe.
    raw = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
        .strip()
        .splitlines()
    )
    if len(raw) != 1:
        raise ValueError("experiment wrapper requires exactly one visible physical GPU")
    name, device_uuid, total_mib, free_mib = [
        value.strip() for value in raw[0].split(",")
    ]
    if not device_uuid.startswith("GPU-"):
        raise ValueError("stable GPU UUID is unavailable")
    total, free = int(total_mib) * 1024**2, int(free_mib) * 1024**2
    try:
        import psutil

        host_total = int(psutil.virtual_memory().total)
    except ImportError:
        host_total = 0
    return {
        "name": name,
        "uuid": device_uuid,
        "total_vram_bytes": total,
        "free_vram_bytes": int(free),
        "host_ram_bytes": host_total,
        "bf16": "CHECKED_BY_SMOKE_TRAINER",
    }


def _lane_matches(lane: str, name: str) -> bool:
    lowered = name.lower()
    return (lane == "rtx_4070" and "4070" in lowered) or (
        lane == "rtx_3070" and "3070" in lowered
    )


def check_dependencies(bundle: Path, job: str) -> None:
    predecessors = {"SLR": ("S0",), "SMIX": ("S0", "SLR"), "C1": ("C0",)}.get(job, ())
    for predecessor in predecessors:
        verify_run(bundle / "jobs" / predecessor)
    if job in {"C0", "C1"}:
        from scripts.analyze_experiments import select_screen

        if select_screen(bundle)["selected_job"] == "S0":
            raise ValueError("confirmation is unnecessary: screen retained control")


def smoke_receipt(bundle: Path, job: str, identity: dict, hardware: dict) -> dict:
    root = bundle / "smoke" / job
    report = verify_run(root, smoke=True)
    for path in sorted(
        (root / "invocations").glob("*/verification.json"), reverse=True
    ):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if all(
            receipt.get(key) == report.get(key)
            for key in (
                "run_id",
                "checkpoint_sha256",
                "metric_sha256",
                "runner_identity_sha256",
            )
        ):
            previous = receipt.get("hardware", {})
            if previous.get("uuid") != hardware.get("uuid") or previous.get(
                "name"
            ) != hardware.get("name"):
                raise ValueError("successful smoke is bound to a different device")
            if report["run_id"] != identity["run_id"]:
                raise ValueError("successful smoke is bound to a different identity")
            return receipt
    raise ValueError("missing verified smoke invocation receipt")


def stop_child(child) -> None:
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


def estimate_full_seconds(
    receipt: dict, root: Path, *, remaining_updates: int = 382
) -> float:
    rows = [
        json.loads(line)
        for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    steps = [float(row["step_seconds"]) for row in rows]
    import math

    if len(steps) != 2 or any(not math.isfinite(x) or x <= 0 for x in steps):
        raise ValueError("smoke timing evidence is invalid")
    outer = float(receipt["outer_wall_seconds"])
    if not math.isfinite(outer) or outer < sum(steps):
        raise ValueError("smoke outer timing does not reconcile")
    # Repeat the entire non-optimizer overhead ten times, then add startup
    # allowance and 50% margin. This is an advisory forecast, never a timeout.
    if not 0 <= remaining_updates <= 382:
        raise ValueError("remaining updates are outside the declared horizon")
    return (max(steps) * remaining_updates + (outer - sum(steps)) * 10 + 300) * 1.5


def estimate_for_job(
    bundle: Path,
    job: str,
    *,
    smoke: bool,
    resume: bool = False,
    receipt: dict | None = None,
) -> dict:
    result = {
        "advisory_only": True,
        "estimated_seconds": None,
        "remaining_updates": 2 if smoke else 382,
        "basis": "No measured estimate available before this job's smoke.",
        "automatic_time_limit": None,
    }
    if smoke:
        return result
    root = bundle / "smoke" / job
    if not root.exists():
        return result
    report = verify_run(root, smoke=True)
    keys = ("run_id", "checkpoint_sha256", "metric_sha256", "runner_identity_sha256")
    if receipt is None:
        for path in sorted(
            (root / "invocations").glob("*/verification.json"), reverse=True
        ):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if all(candidate.get(key) == report.get(key) for key in keys):
                receipt = candidate
                break
    if receipt is None:
        raise ValueError("missing verified smoke timing receipt")
    if any(receipt.get(key) != report.get(key) for key in keys):
        raise ValueError("smoke timing receipt identity mismatch")
    remaining = 382
    if resume:
        import torch
        from tinybench_lm.checkpointing import verify_checkpoint

        checkpoint = bundle / "jobs" / job / "latest.pt"
        if not verify_checkpoint(checkpoint).ok:
            raise ValueError("resume checkpoint is invalid")
        remaining -= int(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["counters"][
                "updates_completed"
            ]
        )
    result.update(
        estimated_seconds=estimate_full_seconds(
            receipt, root, remaining_updates=remaining
        ),
        remaining_updates=max(0, remaining),
        basis="Same-job two-update smoke; extrapolated optimizer/overhead with 50% margin. Not a guarantee.",
    )
    return result


def confirm_execution(estimate: dict) -> bool:
    seconds = estimate["estimated_seconds"]
    duration = (
        "unavailable until a smoke has been measured"
        if seconds is None
        else f"about {seconds / 3600:.2f} hours ({seconds / 60:.0f} minutes)"
    )
    print(f"Estimated runtime: {duration}. {estimate['basis']}", flush=True)
    print("No automatic time cutoff. Actual elapsed time will be recorded.", flush=True)
    try:
        return input("Start this job? [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, OSError):
        return False


def launch(
    bundle: Path,
    job: str,
    identity: dict,
    *,
    lane: str,
    execute: bool,
    smoke: bool,
    resume: bool,
) -> dict:
    if not execute:
        raise ValueError("GPU action is PLAN_ONLY unless --execute is supplied")
    check_bundle(bundle, production=True)
    check_dependencies(bundle, job)
    hardware = inspect_hardware()
    if lane != identity["spec"]["lane"] or not _lane_matches(lane, hardware["name"]):
        raise ValueError("actual GPU does not match assigned lane")
    root = bundle / ("smoke" if smoke else "jobs") / job
    if not resume and root.exists():
        raise ValueError(
            "run directory exists; use explicit resume after inspecting it"
        )
    if resume:
        from tinybench_lm.checkpointing import verify_checkpoint

        checkpoint = root / "latest.pt"
        if not checkpoint.is_file() or not verify_checkpoint(checkpoint).ok:
            raise ValueError("resume checkpoint is missing or invalid")
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        target_updates = 2 if smoke else identity["spec"]["updates"]
        if payload["counters"]["updates_completed"] >= target_updates:
            raise ValueError("resume checkpoint already reached the requested endpoint")
        existing = json.loads(
            (root / "runner_identity.json").read_text(encoding="utf-8")
        )
        if existing != identity:
            raise ValueError("resume identity differs from frozen job")
    receipt = smoke_receipt(bundle, job, identity, hardware) if not smoke else None
    estimate = estimate_for_job(
        bundle, job, smoke=smoke, resume=resume, receipt=receipt
    )
    if not confirm_execution(estimate):
        return {"status": "DECLINED_NOT_STARTED", "estimate": estimate}
    ledger = ExecutionLedger(ROOT / "runs/pre_campaign", lane)
    token = ledger.start(
        identity["run_id"],
        hardware=str(hardware["uuid"]),
        estimate_seconds=estimate["estimated_seconds"],
    )
    started = time.monotonic()
    attempt = None
    child = None
    exit_code = None
    uncertain = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        write_once(root / "runner_identity.json", identity)
        attempt = (
            root
            / "invocations"
            / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
        )
        attempt.mkdir(parents=True, exist_ok=False)
        command = command_for(bundle, job, root, identity, resume=resume, smoke=smoke)
        write_once(
            attempt / "launch.json",
            {
                "command": command,
                "identity": identity,
                "hardware": hardware,
                "estimate": estimate,
                "reservation_token": token,
                "ledger": str(ledger.path),
            },
        )
        with (attempt / "console.log").open("x", encoding="utf-8") as output:
            child = subprocess.Popen(
                command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT
            )
            exit_code = child.wait()
        write_once(
            attempt / "exit.json",
            {"exit_code": exit_code, "outer_wall_seconds": time.monotonic() - started},
        )
        if exit_code != 0:
            raise ValueError(f"training failed ({exit_code})")
        result = verify_run(root, smoke=smoke)
        result["hardware"] = hardware
        result["outer_wall_seconds"] = time.monotonic() - started
        if (
            smoke
            and max(
                float(result["peak_allocated_vram_bytes"]),
                float(result["peak_reserved_vram_bytes"]),
            )
            + hardware["total_vram_bytes"]
            - hardware["free_vram_bytes"]
            > hardware["total_vram_bytes"] * 0.9
        ):
            raise ValueError("smoke lacks ten percent VRAM headroom")
        write_once(attempt / "verification.json", result)
        return result
    except BaseException as error:
        if child is not None and child.poll() is None:
            uncertain = True
            stop_child(child)
        if attempt is not None and attempt.is_dir():
            write_once(
                attempt / "failure.json",
                {"error": str(error), "child_exit_code": exit_code},
            )
        if exit_code == 0:
            exit_code = -1
        if (
            attempt is not None
            and attempt.is_dir()
            and not (attempt / "exit.json").exists()
        ):
            write_once(
                attempt / "exit.json",
                {
                    "exit_code": exit_code,
                    "outer_wall_seconds": time.monotonic() - started,
                    "uncertain": uncertain,
                },
            )
        raise
    finally:
        ledger.finish(
            token,
            elapsed_seconds=time.monotonic() - started,
            exit_code=exit_code,
            uncertain=uncertain,
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=(
            "prepare",
            "check",
            "plan",
            "smoke",
            "run",
            "verify",
            "budget",
            "usage",
            "recover",
        ),
    )
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--job", choices=tuple(contract()["jobs"]))
    p.add_argument("--lane", choices=("rtx_4070", "rtx_3070"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reservation-token")
    p.add_argument("--confirm-process-dead", action="store_true")
    if len(sys.argv) == 1:
        p.print_help()
        return 0
    args = p.parse_args()
    bundle = args.bundle.resolve()
    if not bundle.is_relative_to((ROOT / "runs/pre_campaign").resolve()):
        raise ValueError("bundle must stay under runs/pre_campaign")
    if args.action in {"budget", "usage", "recover"}:
        if not args.lane:
            p.error("--lane is required")
        ledger = ExecutionLedger(ROOT / "runs/pre_campaign", args.lane)
        if args.action == "recover":
            if not args.reservation_token or not args.confirm_process_dead:
                p.error(
                    "recover requires --reservation-token and --confirm-process-dead after inspection"
                )
            result = ledger.recover(args.reservation_token, process_dead=True)
        else:
            result = {
                "lane": args.lane,
                "ledger": str(ledger.path),
                "usage": ledger.summary(),
            }
    elif args.action == "prepare":
        result = prepare(bundle)
    else:
        check_bundle(bundle, production=args.action in {"check", "verify"})
        if args.action == "check":
            result = {"status": "INPUTS_VERIFIED_NOT_TRAINED"}
        else:
            if not args.job:
                p.error("--job is required")
            selected = None
            if args.job == "C1":
                from scripts.analyze_experiments import select_screen

                selected = select_screen(bundle)["selected_treatment"]
            identity = identity_for(bundle, args.job, selected)
            run = bundle / ("smoke" if args.action == "smoke" else "jobs") / args.job
            if args.action == "plan":
                result = {
                    "status": "PLAN_ONLY",
                    "identity": identity,
                    "command": command_for(bundle, args.job, run, identity, resume=args.resume),
                    "estimate": estimate_for_job(
                        bundle, args.job, smoke=False, resume=args.resume
                    ),
                }
            elif args.action == "verify":
                result = verify_run(run)
            else:
                result = launch(
                    bundle,
                    args.job,
                    identity,
                    lane=args.lane,
                    execute=args.execute,
                    smoke=args.action == "smoke",
                    resume=args.resume,
                )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("source_identity",)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(1)
