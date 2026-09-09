"""Strict evidence-first selection for the approved final-model experiment."""

from __future__ import annotations
import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

SLICES = (
    "broad_general",
    "educational_science",
    "narrative_coreference",
    "math_technical",
)
EXPECTED_JOBS = {
    "S0": (1001, 0.0006, "base"),
    "SLR": (1001, 0.001, "base"),
    "SMIX": (1001, 0.0006, "edu"),
    "C0": (1002, 0.0006, "base"),
}


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def _report(bundle: Path, job: str) -> dict[str, Any]:
    from scripts import run_experiment as runner

    runner.check_bundle(bundle, production=False)
    result = runner.verify_run(bundle / "jobs" / job)
    if result.get("status") not in {
        "TRAINING_COMPLETE",
        "TRAINING_COMPLETE_NOT_SELECTED",
    }:
        raise ValueError(f"{job} is not a completed selectable run")
    if result.get("job") != job:
        raise ValueError(f"verified report job mismatch for {job}")
    if job in EXPECTED_JOBS:
        identity = result.get("identity", result)
        spec = identity.get("spec", identity)
        expected_seed, expected_lr, expected_mix = EXPECTED_JOBS[job]
        if (
            int(spec.get("seed", result.get("seed", -1))) != expected_seed
            or float(spec.get("lr", result.get("lr", float("nan")))) != expected_lr
            or spec.get("mixture", result.get("mixture")) != expected_mix
        ):
            raise ValueError(f"{job} treatment/seed does not match frozen job")
    return result


def _metrics(report: dict[str, Any], job: str) -> tuple[float, dict[str, float]]:
    global_loss = _finite(
        report.get("final_validation_loss", report.get("validation_loss")),
        f"{job}.final_validation_loss",
    )
    if global_loss <= 0:
        raise ValueError(f"{job} global loss must be positive")
    raw = report.get("validation_slice_metrics")
    if not isinstance(raw, dict) or set(raw) != set(SLICES):
        raise ValueError(f"{job} must contain exactly four validation slices")
    slices, total_count, total_sum = {}, 0, 0.0
    for name in SLICES:
        item = raw[name]
        if isinstance(item, (int, float)):
            raise ValueError(f"{job}.{name} slice lacks sums/counts")
        if not isinstance(item, dict):
            raise ValueError(f"{job}.{name} slice is malformed")
        if "token_count" not in item or "loss_sum" not in item or "loss" not in item:
            raise ValueError(f"{job}.{name} slice is missing count/sum/loss")
        count = int(item["token_count"])
        loss_sum = _finite(item["loss_sum"], f"{job}.{name}.loss_sum")
        loss = _finite(item["loss"], f"{job}.{name}.loss")
        if (
            count <= 0
            or loss_sum <= 0
            or abs(loss - loss_sum / count) > 1e-10 * max(1.0, abs(loss))
        ):
            raise ValueError(f"{job}.{name} slice does not reconcile")
        slices[name] = loss
        total_count += count
        total_sum += loss_sum
    if (
        "validation_scored_token_count" not in report
        or int(report["validation_scored_token_count"]) != total_count
        or abs(total_sum / total_count - global_loss)
        > 2e-6 * max(1.0, abs(global_loss))
    ):
        raise ValueError(f"{job} slice totals do not reconcile with global validation")
    return global_loss, slices


def _qualifies(
    control: float,
    candidate: float,
    control_slices: dict[str, float],
    candidate_slices: dict[str, float],
) -> bool:
    return (control - candidate) / control >= 0.003 and all(
        (candidate_slices[n] - control_slices[n]) / control_slices[n] <= 0.01
        for n in SLICES
    )


def select_screen(bundle: Path) -> dict[str, Any]:
    # Perform expensive production input custody once per selection operation.
    from scripts import run_experiment as runner

    if (bundle / "bundle.json").is_file():
        runner.check_bundle(bundle)
    reports = {job: _report(bundle, job) for job in ("S0", "SLR", "SMIX")}
    losses, slices = {}, {}
    for job, report in reports.items():
        losses[job], slices[job] = _metrics(report, job)
    passed = {
        job: _qualifies(losses["S0"], losses[job], slices["S0"], slices[job])
        for job in ("SLR", "SMIX")
    }
    eligible = [job for job in ("SLR", "SMIX") if passed[job]]
    selected = (
        min(eligible, key=lambda j: (losses[j], 0 if j == "SLR" else 1))
        if eligible
        else "S0"
    )
    treatment = (
        {"lr": 0.001, "mixture": "base"}
        if selected == "SLR"
        else {"lr": 0.0006, "mixture": "edu"}
        if selected == "SMIX"
        else {"lr": 0.0006, "mixture": "base"}
    )
    return {
        "stage": "screen",
        "selected_job": selected,
        "selected_treatment": treatment,
        "passed": passed,
        "losses": losses,
        "slices": slices,
        "reports": reports,
    }


def select_final(bundle: Path, close_reason: str | None = None) -> dict[str, Any]:
    if close_reason is not None and close_reason not in {
        "budget",
        "deadline",
        "hardware",
        "failed",
    }:
        raise ValueError("close reason must be budget, deadline, hardware, or failed")
    try:
        screen = select_screen(bundle)
    except ValueError as exc:
        if close_reason is None:
            raise
        return {
            "stage": "final",
            "selected_job": "S0",
            "selected_treatment": {"lr": 0.0006, "mixture": "base"},
            "status": "INCOMPLETE_CONTROL",
            "close_reason": close_reason,
            "failure": str(exc),
        }
    if screen["selected_job"] == "S0":
        return {
            **screen,
            "stage": "final",
            "selected_job": "S0",
            "selected_treatment": {"lr": 0.0006, "mixture": "base"},
            "status": "INCOMPLETE_CONTROL" if close_reason else "CONTROL",
            "close_reason": close_reason,
        }
    try:
        reports = {job: _report(bundle, job) for job in ("C0", "C1")}
        losses, slices = {}, {}
        c1_identity = reports["C1"].get("identity", reports["C1"])
        c1_spec = c1_identity.get("spec", c1_identity)
        treatment = screen["selected_treatment"]
        if (
            float(c1_spec.get("lr", reports["C1"].get("lr", float("nan"))))
            != float(treatment["lr"])
            or c1_spec.get("mixture", reports["C1"].get("mixture"))
            != treatment["mixture"]
        ):
            raise ValueError(
                "C1 treatment does not match the selected screen treatment"
            )
    except ValueError as exc:
        if close_reason is None:
            raise
        return {
            **screen,
            "stage": "final",
            "selected_job": "S0",
            "selected_treatment": {"lr": 0.0006, "mixture": "base"},
            "status": "INCOMPLETE_CONTROL",
            "close_reason": close_reason,
            "failure": str(exc),
        }
    try:
        for job, report in reports.items():
            losses[job], slices[job] = _metrics(report, job)
    except ValueError as exc:
        if close_reason is None:
            raise
        return {
            **screen,
            "stage": "final",
            "selected_job": "S0",
            "selected_treatment": {"lr": 0.0006, "mixture": "base"},
            "status": "INCOMPLETE_CONTROL",
            "close_reason": close_reason,
            "failure": str(exc),
        }
    passed = _qualifies(losses["C0"], losses["C1"], slices["C0"], slices["C1"])
    return {
        **screen,
        "stage": "final",
        "confirmation_passed": passed,
        "selected_job": "C1" if passed and not close_reason else "C0",
        "selected_treatment": screen["selected_treatment"]
        if passed and not close_reason
        else {"lr": 0.0006, "mixture": "base"},
        "status": "INCOMPLETE_CONTROL" if close_reason else "SELECTED" if passed else "CONTROL",
        "confirmation_losses": losses,
        "confirmation_slices": slices,
        "confirmation_reports": reports,
        "close_reason": close_reason,
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"refusing to replace malformed output: {path}") from exc
        if existing != value:
            raise ValueError(f"refusing to overwrite different evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "final"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--close-incomplete")
    args = parser.parse_args()
    result = (
        select_screen(args.bundle)
        if args.stage == "screen"
        else select_final(args.bundle, args.close_incomplete)
    )
    if args.output:
        _atomic_write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
