"""Materialize and verify the finite reduced-baseline exposure plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tinybench_lm.exposure import DEFAULT_QUOTAS, CompositeExposure, verify_exposure, write_exposure_artifacts
from tinybench_lm.schedule import build_materialized_schedule, load_schedule
from tinybench_lm.shards import load_split_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/shards/reduced_5pct_v1/stable_train.manifest.json"))
    parser.add_argument("--base-schedule", type=Path, default=Path("data/schedules/reduced_5pct_v1/stable_train.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/reduced_campaign/reduced_baseline_v1"))
    parser.add_argument("--contract", type=Path, default=Path("docs/g3/BASELINE_CONTRACT.md"))
    parser.add_argument("--recipe", type=Path, default=Path("configs/training/baseline_reduced_v1.yaml"))
    args = parser.parse_args()
    manifest = load_split_manifest(args.manifest)
    base = load_schedule(args.base_schedule)
    component2 = build_materialized_schedule(
        manifest, sequence_length=1024, seed=1337,
        source_sequence_quotas=DEFAULT_QUOTAS, local_shuffle_buffer_sequences=1024,
    )
    recipe_hash = hashlib.sha256(args.recipe.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    exposure = CompositeExposure((base, component2), contract_hash=sha256(args.contract), recipe_sha256_normalized_lf=recipe_hash)
    accounting = verify_exposure(exposure, manifest)
    paths = write_exposure_artifacts(args.output, exposure)
    accounting.update({
        "schema_version": "baseline_exposure_accounting_v1",
        "manifest_content_hash": manifest.content_hash(),
        "base_schedule_file_sha256": sha256(args.base_schedule),
        "component_file_sha256": {name: sha256(path) for name, path in paths.items() if name.startswith("component_")},
        "plan_file_sha256": sha256(paths["plan"]),
        "contract_file_sha256": sha256(args.contract),
        "contract_file_sha256_normalized_lf": hashlib.sha256(args.contract.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "recipe_file_sha256_normalized_lf": hashlib.sha256(args.recipe.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "distinct_selected_corpus_tokens": 550094903,
        "effective_passes": exposure.loss_tokens / 550094903,
        "component_reuse": {
            "cross_component_repeated_reference_count": len(
                {entry.reference for entry in exposure.components[0].entries}
                & {entry.reference for entry in exposure.components[1].entries}
            ),
            "within_component_duplicates_rejected": True,
        },
    })
    accounting_path = args.output / "exposure_accounting.json"
    accounting_path.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "accounting": str(accounting_path), **accounting}, sort_keys=True))


if __name__ == "__main__":
    main()
