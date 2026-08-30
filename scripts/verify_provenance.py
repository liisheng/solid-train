"""Local command line for step-zero provenance and clean release verification.

Everything this script does is local and bounded (Plan Sections 3.3, 10.2, 13 G0/G2/G6):

    python scripts/verify_provenance.py record --config configs/final_49m.json --seed 1337 \
        --output runs/<run>/step_zero_provenance.json
    python scripts/verify_provenance.py verify-provenance runs/<run>/step_zero_provenance.json
    python scripts/verify_provenance.py export-release --checkpoint runs/<run>/best.pt \
        --provenance runs/<run>/step_zero_provenance.json --output releases/<name>.pt
    python scripts/verify_provenance.py verify-release releases/<name>.pt
    python scripts/verify_provenance.py verify-fresh releases/<name>.pt \
        --provenance runs/<run>/step_zero_provenance.json

No hash printed here is a final campaign hash. Each one is derived from the artifact given
on the command line, and nothing is published.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tinybench_lm import ModelConfig
from tinybench_lm.config import FINAL_CONFIG_PATH
from tinybench_lm.provenance import (
    build_initial_model,
    export_release_from_checkpoint,
    format_verification_report,
    read_step_zero_provenance,
    record_step_zero_provenance,
    verify_fresh_initialization_claim,
    verify_release_export,
    verify_step_zero_provenance,
    write_step_zero_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Record step-zero provenance for a fresh init")
    record.add_argument("--config", type=Path, default=FINAL_CONFIG_PATH)
    record.add_argument("--seed", type=int, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--overwrite", action="store_true")

    verify_provenance = subparsers.add_parser(
        "verify-provenance", help="Reproduce a step-zero provenance record"
    )
    verify_provenance.add_argument("path", type=Path)

    export = subparsers.add_parser(
        "export-release", help="Strip a local checkpoint into a clean release export"
    )
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--provenance", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--notes")

    verify_release = subparsers.add_parser(
        "verify-release", help="Reload and re-derive every claim of a release export"
    )
    verify_release.add_argument("path", type=Path)
    verify_release.add_argument("--expected-parameter-count", type=int)

    verify_fresh = subparsers.add_parser(
        "verify-fresh", help="Reject an artifact that is presented as a fresh initialization"
    )
    verify_fresh.add_argument("path", type=Path)
    verify_fresh.add_argument("--provenance", type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "record":
        config = ModelConfig.from_json(args.config)
        model = build_initial_model(config, args.seed)
        record = record_step_zero_provenance(
            model, config, seed=args.seed, config_path=args.config
        )
        write_step_zero_provenance(args.output, record, overwrite=args.overwrite)
        if args.json:
            print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"Recorded step-zero provenance at {args.output}")
            print(f"  seed:                 {record.seed}")
            print(f"  unique parameters:    {record.unique_parameter_count:,}")
            print(f"  cap headroom:         {record.cap_headroom:,}")
            print(f"  step-zero weight hash: {record.weight_sha256}")
        return

    if args.command == "verify-provenance":
        report = verify_step_zero_provenance(read_step_zero_provenance(args.path))
        title = f"Step-zero provenance verification: {args.path}"
    elif args.command == "export-release":
        payload = export_release_from_checkpoint(
            args.checkpoint,
            args.output,
            provenance_path=args.provenance,
            notes=args.notes,
        )
        report = verify_release_export(args.output)
        title = f"Release export {args.output} ({payload['unique_parameter_count']:,} parameters)"
    elif args.command == "verify-release":
        report = verify_release_export(
            args.path, expected_parameter_count=args.expected_parameter_count
        )
        title = f"Release verification: {args.path}"
    else:
        payload = torch.load(args.path, map_location="cpu", weights_only=False)
        record = read_step_zero_provenance(args.provenance) if args.provenance else None
        report = verify_fresh_initialization_claim(
            payload, record=record, source=str(args.path)
        )
        title = f"Fresh-initialization claim: {args.path}"

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_verification_report(report, title))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
