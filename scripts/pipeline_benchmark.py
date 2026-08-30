"""Turn measured pipeline-stage counters into an auditable benchmark artifact.

Plan Section 5.5 requires the complete pipeline to be benchmarked on a stratified 1% slice
and a configurable 2-5% slice. This script is the reporting half of that contract: it reads
counters an operator measured while running the pipeline, derives the six required
per-stage metrics, builds one explicit forecast, audits the degradation policy, and writes
the artifact.

It never times, streams, or executes a pipeline stage, and it never invents a counter. With
no measurement file it prints the frozen contract and the readiness blockers instead of a
result, because absence of evidence is ``NOT_RUN``, not ``PASS``.

Usage (PowerShell)::

    .venv\\Scripts\\python.exe scripts\\pipeline_benchmark.py --show-contract
    .venv\\Scripts\\python.exe scripts\\pipeline_benchmark.py ^
        --measurements runs\\bench\\slice_1pct.measurements.json ^
        --artifact runs\\bench\\slice_1pct.artifact.json

The measurement file is JSON::

    {
      "mode_id": "slice_1pct",
      "slice_fraction": 0.01,
      "stratified": true,
      "deadline_seconds": 86400,
      "stages": [
        {"stage_id": "stream_and_filter", "documents": 1000, "input_bytes": 5000000,
         "output_bytes": 4000000, "elapsed_seconds": 12.5, "peak_rss_bytes": 900000000,
         "peak_temporary_disk_bytes": 2500000}
      ],
      "omitted_stage_ids": [],
      "scope_reduction": null
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tinybench_lm.pipeline_bench import (
    FAIL,
    REQUIRED_STAGE_METRICS,
    BenchmarkRun,
    PipelineBenchContractError,
    ScopeReductionDecision,
    StageMeasurement,
    build_benchmark_report,
    format_pipeline_bench_report,
    load_pipeline_bench_protocol,
    mandatory_stage_ids,
    mode_index,
    never_omit_stage_ids,
    omittable_stage_ids,
    write_benchmark_artifact,
)

_STAGE_FIELDS = (
    "documents",
    "input_bytes",
    "output_bytes",
    "elapsed_seconds",
    "peak_rss_bytes",
    "peak_temporary_disk_bytes",
)


def parse_run(payload: Mapping[str, Any]) -> BenchmarkRun:
    """Build a :class:`BenchmarkRun` from a measurement file. Absent counters stay absent."""
    stages = []
    for entry in payload.get("stages", []):
        stages.append(
            StageMeasurement(
                stage_id=str(entry["stage_id"]),
                **{name: entry.get(name) for name in _STAGE_FIELDS},
            )
        )
    record = payload.get("scope_reduction")
    decision = None
    if isinstance(record, Mapping):
        decision = ScopeReductionDecision(
            omitted_stage_ids=tuple(str(name) for name in record.get("omitted_stage_ids", ())),
            date=record.get("date"),
            owner=record.get("owner"),
            reason=record.get("reason"),
            forecast_reference=record.get("forecast_reference"),
        )
    return BenchmarkRun(
        mode_id=str(payload["mode_id"]),
        slice_fraction=float(payload["slice_fraction"]),
        stratified=bool(payload.get("stratified", True)),
        measurements=tuple(stages),
        omitted_stage_ids=tuple(str(name) for name in payload.get("omitted_stage_ids", ())),
        deadline_seconds=payload.get("deadline_seconds"),
        scope_reduction=decision,
        slice_id=payload.get("slice_id"),
    )


def print_contract() -> None:
    protocol = load_pipeline_bench_protocol()
    print(f"frozen contract: {protocol['_source']}")
    print(f"digest:          {protocol['_digest']}")
    print("modes:")
    for mode_id, declared in mode_index(protocol).items():
        window = (
            f"exactly {declared['exact_fraction']}"
            if "exact_fraction" in declared
            else f"{declared['minimum_fraction']} to {declared['maximum_fraction']}"
        )
        print(f"  {mode_id:<18} slice fraction {window}")
    print("required per-stage metrics:")
    for metric in REQUIRED_STAGE_METRICS:
        print(f"  {metric}")
    print("mandatory stages (every profile):")
    for stage_id in mandatory_stage_ids(protocol):
        print(f"  {stage_id}")
    print("omittable, in omission order:")
    for stage_id in omittable_stage_ids(protocol):
        print(f"  {stage_id}")
    print("never omitted under any forecast pressure:")
    for stage_id in never_omit_stage_ids(protocol):
        print(f"  {stage_id}")
    readiness = protocol["readiness"]
    print("readiness:")
    for name in ("measured_1pct_benchmark", "measured_2_to_5pct_benchmark", "measured_pipeline_forecast"):
        print(f"  {name:<30} {readiness[name]}")
    print(f"  blocker     {readiness['blocker']}")
    print(f"  owner       {readiness['owner']}")
    print(f"  next_action {readiness['next_action']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report a bounded pipeline benchmark from measured counters.")
    parser.add_argument("--measurements", type=Path, default=None, help="JSON file of measured stage counters")
    parser.add_argument("--artifact", type=Path, default=None, help="where to write the benchmark artifact JSON")
    parser.add_argument("--show-contract", action="store_true", help="print the frozen contract and readiness only")
    args = parser.parse_args()

    if args.measurements is None or args.show_contract:
        print_contract()
        if args.measurements is None:
            print("\nno measurement file supplied: no benchmark result is reported (NOT_RUN)")
            return

    payload = json.loads(args.measurements.read_text(encoding="utf-8"))
    try:
        report = build_benchmark_report(parse_run(payload))
    except PipelineBenchContractError as error:
        raise SystemExit(f"benchmark rejected: {error}") from error

    print(format_pipeline_bench_report(report.results))
    print(f"\nforecast: {report.forecast.status} ({report.forecast.reason})")
    if args.artifact is not None:
        written = write_benchmark_artifact(args.artifact, report)
        print(f"artifact: {written}")
    if any(result.status == FAIL for result in report.results):
        raise SystemExit("pipeline benchmark audit failed")


if __name__ == "__main__":
    main()
