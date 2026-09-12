"""Verify the pinned final selection and prepare the fresh CONTROL exposure; no training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.analyze_experiments import _atomic_write
from scripts.run_reduced_baseline import ROOT, SELECTED_CONFIG, _verified_baseline_config, sha256
from tinybench_lm.baseline_contract import SELECTED_BASELINE_SHA256
from tinybench_lm.exposure import CompositeExposure, load_exposure_plan, verify_exposure, write_exposure_artifacts
from tinybench_lm.shards import load_split_manifest


def verify_selection(report: dict, settings: dict) -> None:
    digest = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != settings["source_report_canonical_sha256"]:
        raise ValueError("final selection report differs from the reviewed selection")
    if report.get("stage") != "final" or report.get("status") != "CONTROL" or report.get("selected_treatment") != {"lr": 0.0006, "mixture": "base"}:
        raise ValueError("this successor requires the completed CONTROL selection")


def integrate(bundle: Path, output: Path) -> dict:
    config = _verified_baseline_config(config_path=SELECTED_CONFIG)
    settings = json.loads((ROOT / config["selection"]["path"]).read_text())
    # Experiment endpoints require their original source checkout for reanalysis.
    # Consume the exact reviewed report here rather than relabeling old-source evidence.
    report = json.loads((bundle / "selection.final.json").read_text(encoding="utf-8"))
    verify_selection(report, settings)
    _atomic_write(bundle / "selection.final.json", report)
    previous = ROOT / "runs/reduced_campaign/reduced_baseline_v1"
    old = load_exposure_plan(previous / "exposure_plan.json", (previous / "component_1.json", previous / "component_2.json"))
    pins = config["schedule_inputs"]["baseline_exposure_artifacts"]["component_file_sha256"]
    for name, digest in pins.items():
        if sha256(previous / f"{name}.json") != digest:
            raise ValueError("historical component differs from selected CONTROL pin")
    exposure = CompositeExposure(old.components, SELECTED_BASELINE_SHA256, recipe_sha256_normalized_lf=SELECTED_BASELINE_SHA256)
    accounting = verify_exposure(exposure, load_split_manifest(ROOT / config["data"]["manifests"]["stable_train"]["path"]))
    destination = ROOT / config["schedule_inputs"]["baseline_exposure_artifacts"]["root"]
    if destination.exists():
        retained = load_exposure_plan(destination / "exposure_plan.json", (destination / "component_1.json", destination / "component_2.json"))
        if retained.to_dict() != exposure.to_dict():
            raise ValueError("refusing to replace a different successor exposure")
    else:
        write_exposure_artifacts(destination, exposure)
    for name, digest in pins.items():
        if sha256(destination / f"{name}.json") != digest:
            raise ValueError("successor component bytes changed")
    result = {
        "status": "SELECTED_CONTROL_EXPOSURE_PREPARED_NOT_TRAINED",
        "baseline_contract_sha256": SELECTED_BASELINE_SHA256,
        "selection_report_canonical_sha256": settings["source_report_canonical_sha256"],
        "previous_exposure_content_hash": old.content_hash,
        "exposure": accounting,
        "artifacts": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(destination.glob("*.json"))},
        "initialization": "fresh_seeded_random", "seed": 1337,
        "G4": "NOT_RUN", "G5": "NOT_RUN",
    }
    _atomic_write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "runs/pre_campaign/v2-advisory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(integrate(args.bundle, args.output), indent=2, sort_keys=True))
