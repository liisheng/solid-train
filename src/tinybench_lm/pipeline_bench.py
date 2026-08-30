"""Bounded pipeline benchmarks, forecasts, and the frozen degradation policy.

Plan Section 5.5 requires the complete data pipeline to be run over a stratified 1% slice
and a configurable 2-5% slice, recording **per-stage documents/s, input GB/s, peak RSS,
temporary disk amplification, output size, and extrapolated wall time**. If the forecast
misses the Section 12 schedule, exactly one scope reduction is permitted first (the
optional expanded within-source full-corpus near-dedup pass); the frozen cross-source
near-dedup, validation/reserved isolation, and benchmark decontamination stages can never
be dropped. Section 13 G1 is the gate those artifacts feed.

This module is the mechanism, backed by one frozen config::

    configs/data/pipeline_bench_v1.yaml

Guarantees, mirroring :mod:`tinybench_lm.data_protocols`, :mod:`tinybench_lm.source_manifest`,
and :mod:`tinybench_lm.shards`:

1. **Immutable.** The config is verified against a pinned SHA-256 digest
   (:data:`FROZEN_PIPELINE_BENCH_SHA256`) on every load, so a stage identity, a
   mandatory-stage flag, or the omission order cannot drift. Changing one means publishing
   ``pipeline_bench_v2.yaml``.
2. **Pure calculators.** Every metric, extrapolation, and gate decision is a function of
   supplied measurements. Nothing in this module times, streams, or executes a pipeline
   stage, so tests drive the whole surface from fixture timings.
3. **Mandatory stages are identified in the artifact.** A benchmark artifact carries the
   frozen mandatory / omittable / never-omit stage identities alongside the measurements
   and one explicit forecast, so feasibility and any scope reduction are auditable from
   measured counters.
4. **Fail closed.** Omitting a mandatory stage, omitting out of the frozen order, or
   claiming a scope reduction without a dated decision record raises or reports ``FAIL``.
5. **Absence of evidence is never PASS.** A missing metric is ``BENCH_METRIC_MISSING``, a
   benchmark with no measurements is ``NOT_RUN``, and a forecast without a deadline is
   ``NOT_RUN`` - never a pass.

No real slice has been benchmarked. The ``readiness`` section of the frozen config reports
``NOT_RUN``/``DEFERRED``, and :func:`assert_ready_for_real_pipeline_benchmark` fails closed
until an operator acquires the corpus and builds the final tokenizer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data_protocols import (
    DATA_PROTOCOL_DIR,
    ProtocolError,
    ProtocolNotReadyError,
    load_protocol,
)
from .environment import CheckResult

PIPELINE_BENCH_PROTOCOL_PATH = DATA_PROTOCOL_DIR / "pipeline_bench_v1.yaml"

#: SHA-256 of the frozen pipeline benchmark contract, file bytes with CRLF normalized to LF.
FROZEN_PIPELINE_BENCH_SHA256: Mapping[str, str] = {
    "pipeline_bench_v1.yaml": "e789960c7730c216e6c0a3a228fad24e0d328bb4197d85609e64b6cec01a07da",
}

# --------------------------------------------------------------------------------------
# Statuses and the reason-code vocabulary
# --------------------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
DEFERRED = "DEFERRED"
NOT_RUN = "NOT_RUN"

#: Forecast verdicts. ``NOT_RUN`` is used whenever the evidence to decide is absent.
FIT = "FIT"
MISS = "MISS"

MODE_1PCT = "slice_1pct"
MODE_2_TO_5PCT = "slice_2_to_5pct"

#: The eight pipeline stages the frozen contract declares, in execution order.
EXPECTED_STAGES: tuple[str, ...] = (
    "stream_and_filter",
    "exact_and_mirror_dedup",
    "within_source_near_dedup",
    "expanded_within_source_full_corpus_near_dedup",
    "cross_source_near_dedup",
    "split_reserved_isolation",
    "benchmark_decontamination",
    "tokenize_and_pack",
)

#: The one omittable stage (Plan Section 5.5).
EXPECTED_OMITTABLE_STAGES: tuple[str, ...] = ("expanded_within_source_full_corpus_near_dedup",)

#: Stages Plan Section 5.5 forbids dropping under any forecast pressure.
EXPECTED_NEVER_OMIT_STAGES: tuple[str, ...] = (
    "cross_source_near_dedup",
    "split_reserved_isolation",
    "benchmark_decontamination",
)

#: The six per-stage counters Plan Section 5.5 lists verbatim.
REQUIRED_STAGE_METRICS: tuple[str, ...] = (
    "documents_per_second",
    "input_gigabytes_per_second",
    "peak_rss_bytes",
    "temporary_disk_amplification",
    "output_bytes",
    "extrapolated_wall_time_seconds",
)

BENCH_OK = "BENCH_OK"
BENCH_MODE_FRACTION_OUT_OF_RANGE = "BENCH_MODE_FRACTION_OUT_OF_RANGE"
BENCH_SLICE_NOT_STRATIFIED = "BENCH_SLICE_NOT_STRATIFIED"
BENCH_STAGE_UNREGISTERED = "BENCH_STAGE_UNREGISTERED"
BENCH_STAGE_MISSING = "BENCH_STAGE_MISSING"
BENCH_STAGE_ORDER_MISMATCH = "BENCH_STAGE_ORDER_MISMATCH"
BENCH_STAGE_DUPLICATED = "BENCH_STAGE_DUPLICATED"
BENCH_METRIC_MISSING = "BENCH_METRIC_MISSING"
BENCH_METRIC_NOT_MEASURED = "BENCH_METRIC_NOT_MEASURED"
BENCH_ZERO_ELAPSED = "BENCH_ZERO_ELAPSED"
BENCH_MANDATORY_STAGE_OMITTED = "BENCH_MANDATORY_STAGE_OMITTED"
BENCH_MANDATORY_STAGE_DISABLED = "BENCH_MANDATORY_STAGE_DISABLED"
BENCH_OMISSION_ORDER_VIOLATED = "BENCH_OMISSION_ORDER_VIOLATED"
BENCH_FORECAST_MISSES_SCHEDULE = "BENCH_FORECAST_MISSES_SCHEDULE"
BENCH_SCOPE_DECISION_RECORD_MISSING = "BENCH_SCOPE_DECISION_RECORD_MISSING"
BENCH_ARTIFACT_INCOMPLETE = "BENCH_ARTIFACT_INCOMPLETE"
BENCH_ARTIFACT_PROTOCOL_MISMATCH = "BENCH_ARTIFACT_PROTOCOL_MISMATCH"

BENCH_FAIL_CLOSED_REASON_CODES: frozenset[str] = frozenset(
    {
        BENCH_MODE_FRACTION_OUT_OF_RANGE,
        BENCH_SLICE_NOT_STRATIFIED,
        BENCH_STAGE_UNREGISTERED,
        BENCH_STAGE_MISSING,
        BENCH_STAGE_ORDER_MISMATCH,
        BENCH_STAGE_DUPLICATED,
        BENCH_METRIC_MISSING,
        BENCH_METRIC_NOT_MEASURED,
        BENCH_ZERO_ELAPSED,
        BENCH_MANDATORY_STAGE_OMITTED,
        BENCH_MANDATORY_STAGE_DISABLED,
        BENCH_OMISSION_ORDER_VIOLATED,
        BENCH_FORECAST_MISSES_SCHEDULE,
        BENCH_SCOPE_DECISION_RECORD_MISSING,
        BENCH_ARTIFACT_INCOMPLETE,
        BENCH_ARTIFACT_PROTOCOL_MISMATCH,
    }
)

ARTIFACT_SCHEMA_VERSION = "pipeline_bench_v1"


class PipelineBenchContractError(ProtocolError):
    """The frozen benchmark contract is malformed, or a run/artifact violates it."""


class MandatoryStageOmittedError(PipelineBenchContractError):
    """A scope reduction tried to drop a stage the plan marks mandatory."""


class PipelineBenchNotReadyError(ProtocolNotReadyError):
    """Real-slice benchmarking is gated behind corpus acquisition and the final tokenizer."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def load_pipeline_bench_protocol(
    path: Path = PIPELINE_BENCH_PROTOCOL_PATH,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    """Load the frozen benchmark contract, verifying its pinned digest by default."""
    protocol = load_protocol(path, verify=verify, registry=FROZEN_PIPELINE_BENCH_SHA256)
    required = (
        "modes",
        "mode_rules",
        "stages",
        "required_stage_metrics",
        "metric_definitions",
        "metric_rules",
        "forecast",
        "degradation_policy",
        "scope_reduction_decision",
        "artifact",
        "readiness",
    )
    for section in required:
        if section not in protocol:
            raise PipelineBenchContractError(f"pipeline benchmark protocol is missing required section {section!r}")

    stages = stage_index(protocol)
    if tuple(stages) != EXPECTED_STAGES:
        raise PipelineBenchContractError(
            f"pipeline benchmark protocol declares stages {tuple(stages)}, expected {EXPECTED_STAGES}"
        )
    orders = tuple(int(entry["order"]) for entry in stages.values())
    if orders != tuple(range(1, len(EXPECTED_STAGES) + 1)):
        raise PipelineBenchContractError(f"stage orders must be 1..{len(EXPECTED_STAGES)} in sequence, found {orders}")

    metrics = tuple(str(name) for name in protocol["required_stage_metrics"])
    if metrics != REQUIRED_STAGE_METRICS:
        raise PipelineBenchContractError(
            f"protocol requires stage metrics {metrics}, expected the Plan 5.5 list {REQUIRED_STAGE_METRICS}"
        )

    omittable = omittable_stage_ids(protocol)
    if omittable != EXPECTED_OMITTABLE_STAGES:
        raise PipelineBenchContractError(
            f"protocol marks {omittable} omittable, expected exactly {EXPECTED_OMITTABLE_STAGES}"
        )
    order = tuple(str(name) for name in protocol["degradation_policy"]["omission_order"])
    if order != EXPECTED_OMITTABLE_STAGES:
        raise PipelineBenchContractError(f"omission order {order} must match the omittable stages {EXPECTED_OMITTABLE_STAGES}")
    never = never_omit_stage_ids(protocol)
    if never != EXPECTED_NEVER_OMIT_STAGES:
        raise PipelineBenchContractError(f"protocol never-omit list {never}, expected {EXPECTED_NEVER_OMIT_STAGES}")
    for stage_id in never:
        declared = stages[stage_id]
        if bool(declared["omittable"]) or not bool(declared["plan_never_bypass"]):
            raise PipelineBenchContractError(
                f"stage {stage_id!r} is in the never-omit list but is not flagged mandatory and never-bypass"
            )
    if not bool(protocol["degradation_policy"]["full_integrity_stages_enabled_in_every_profile"]):
        raise PipelineBenchContractError("the frozen contract must keep full-integrity stages enabled in every profile")
    if not bool(protocol["forecast"]["absence_of_evidence_is_never_pass"]):
        raise PipelineBenchContractError("the frozen contract must state that absence of evidence is never a pass")

    modes = mode_index(protocol)
    if tuple(modes) != (MODE_1PCT, MODE_2_TO_5PCT):
        raise PipelineBenchContractError(f"protocol declares modes {tuple(modes)}, expected {(MODE_1PCT, MODE_2_TO_5PCT)}")
    return protocol


def stage_index(protocol: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Stage ID -> its frozen declaration, in declared execution order."""
    resolved = protocol or load_pipeline_bench_protocol()
    index: dict[str, dict[str, Any]] = {}
    for entry in resolved["stages"]:
        stage_id = str(entry["stage_id"])
        if stage_id in index:
            raise PipelineBenchContractError(f"{BENCH_STAGE_DUPLICATED}: stage {stage_id!r} is declared twice")
        index[stage_id] = dict(entry)
    return index


def mode_index(protocol: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Mode ID -> its frozen declaration."""
    resolved = protocol or load_pipeline_bench_protocol()
    return {str(entry["mode_id"]): dict(entry) for entry in resolved["modes"]}


def mandatory_stage_ids(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Stages that must run in every profile, in execution order."""
    return tuple(
        stage_id for stage_id, entry in stage_index(protocol).items() if not bool(entry["omittable"])
    )


def omittable_stage_ids(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Stages that may be dropped, ordered by their frozen ``omission_order``."""
    entries = [entry for entry in stage_index(protocol).values() if bool(entry["omittable"])]
    entries.sort(key=lambda entry: int(entry.get("omission_order", 0)))
    return tuple(str(entry["stage_id"]) for entry in entries)


def never_omit_stage_ids(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """The stages Plan Section 5.5 forbids dropping, in the frozen declaration order."""
    resolved = protocol or load_pipeline_bench_protocol()
    return tuple(str(name) for name in resolved["degradation_policy"]["never_omit"])


def resolve_slice_fraction(
    mode_id: str,
    slice_fraction: float,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> float:
    """Validate a slice fraction against its frozen mode window and return it."""
    resolved = protocol or load_pipeline_bench_protocol()
    modes = mode_index(resolved)
    declared = modes.get(str(mode_id))
    if declared is None:
        raise PipelineBenchContractError(f"benchmark mode {mode_id!r} is not declared; expected one of {tuple(modes)}")
    if not _is_measured(slice_fraction) or slice_fraction <= 0.0:
        raise PipelineBenchContractError(
            f"{BENCH_MODE_FRACTION_OUT_OF_RANGE}: slice fraction {slice_fraction!r} is not a positive measurement"
        )
    tolerance = float(resolved["mode_rules"]["fraction_tolerance"])
    if "exact_fraction" in declared:
        exact = float(declared["exact_fraction"])
        if abs(float(slice_fraction) - exact) > tolerance:
            raise PipelineBenchContractError(
                f"{BENCH_MODE_FRACTION_OUT_OF_RANGE}: mode {mode_id!r} is fixed at {exact}, got {slice_fraction}"
            )
        return float(slice_fraction)
    low = float(declared["minimum_fraction"])
    high = float(declared["maximum_fraction"])
    if float(slice_fraction) < low - tolerance or float(slice_fraction) > high + tolerance:
        raise PipelineBenchContractError(
            f"{BENCH_MODE_FRACTION_OUT_OF_RANGE}: mode {mode_id!r} accepts [{low}, {high}], got {slice_fraction}"
        )
    return float(slice_fraction)


def assert_ready_for_real_pipeline_benchmark(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: a real slice benchmark needs an acquired corpus and the final tokenizer."""
    resolved = protocol or load_pipeline_bench_protocol()
    readiness = resolved["readiness"]
    blocked = [
        name
        for name in ("measured_1pct_benchmark", "measured_2_to_5pct_benchmark", "measured_pipeline_forecast")
        if str(readiness.get(name)) != PASS
    ]
    if blocked:
        raise PipelineBenchNotReadyError(
            f"real-slice pipeline benchmarking is not ready: {blocked}. "
            f"blocker={readiness.get('blocker')} owner={readiness.get('owner')} "
            f"next_action={readiness.get('next_action')}"
        )


# --------------------------------------------------------------------------------------
# Pure metric calculators
# --------------------------------------------------------------------------------------


def _is_measured(value: Any) -> bool:
    """A measurement is a finite, non-negative real number. Anything else is absent."""
    if isinstance(value, bool) or value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) >= 0.0


def documents_per_second(documents: float, elapsed_seconds: float) -> float:
    """Documents processed per wall-clock second."""
    if not _is_measured(documents):
        raise PipelineBenchContractError(f"{BENCH_METRIC_NOT_MEASURED}: documents={documents!r}")
    if not _is_measured(elapsed_seconds) or float(elapsed_seconds) == 0.0:
        raise PipelineBenchContractError(f"{BENCH_ZERO_ELAPSED}: elapsed_seconds={elapsed_seconds!r}")
    return float(documents) / float(elapsed_seconds)


def input_gigabytes_per_second(
    input_bytes: float,
    elapsed_seconds: float,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> float:
    """Input GB/s using the frozen decimal gigabyte definition (1e9 bytes)."""
    resolved = protocol or load_pipeline_bench_protocol()
    gigabyte = float(resolved["metric_definitions"]["gigabyte_bytes"])
    if not _is_measured(input_bytes):
        raise PipelineBenchContractError(f"{BENCH_METRIC_NOT_MEASURED}: input_bytes={input_bytes!r}")
    if not _is_measured(elapsed_seconds) or float(elapsed_seconds) == 0.0:
        raise PipelineBenchContractError(f"{BENCH_ZERO_ELAPSED}: elapsed_seconds={elapsed_seconds!r}")
    return float(input_bytes) / gigabyte / float(elapsed_seconds)


def temporary_disk_amplification(peak_temporary_disk_bytes: float, input_bytes: float) -> float:
    """Peak temporary disk usage as a multiple of the stage's input size."""
    if not _is_measured(peak_temporary_disk_bytes):
        raise PipelineBenchContractError(
            f"{BENCH_METRIC_NOT_MEASURED}: peak_temporary_disk_bytes={peak_temporary_disk_bytes!r}"
        )
    if not _is_measured(input_bytes) or float(input_bytes) == 0.0:
        raise PipelineBenchContractError(f"{BENCH_METRIC_NOT_MEASURED}: input_bytes={input_bytes!r}")
    return float(peak_temporary_disk_bytes) / float(input_bytes)


def extrapolate_wall_time(elapsed_seconds: float, slice_fraction: float) -> float:
    """Linear full-corpus extrapolation: ``elapsed / slice_fraction`` (Plan Section 5.5)."""
    if not _is_measured(elapsed_seconds):
        raise PipelineBenchContractError(f"{BENCH_METRIC_NOT_MEASURED}: elapsed_seconds={elapsed_seconds!r}")
    if not _is_measured(slice_fraction) or float(slice_fraction) == 0.0:
        raise PipelineBenchContractError(f"{BENCH_METRIC_NOT_MEASURED}: slice_fraction={slice_fraction!r}")
    return float(elapsed_seconds) / float(slice_fraction)


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StageMeasurement:
    """Raw counters observed for one pipeline stage on one slice.

    Every numeric field may be ``None``, which means *not measured*. A missing counter is
    reported as ``BENCH_METRIC_MISSING`` and never silently treated as zero.
    """

    stage_id: str
    documents: int | None = None
    input_bytes: int | None = None
    output_bytes: int | None = None
    elapsed_seconds: float | None = None
    peak_rss_bytes: int | None = None
    peak_temporary_disk_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "documents": self.documents,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_temporary_disk_bytes": self.peak_temporary_disk_bytes,
        }


@dataclass(frozen=True)
class StageReport:
    """Derived per-stage metrics plus the frozen identity of the stage."""

    stage_id: str
    order: int
    mandatory: bool
    never_bypass: bool
    protocol_refs: tuple[str, ...]
    measurement: StageMeasurement
    documents_per_second: float | None
    input_gigabytes_per_second: float | None
    peak_rss_bytes: int | None
    temporary_disk_amplification: float | None
    output_bytes: int | None
    extrapolated_wall_time_seconds: float | None
    missing_metrics: tuple[str, ...]
    reason_codes: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "order": self.order,
            "mandatory": self.mandatory,
            "never_bypass": self.never_bypass,
            "protocol_refs": list(self.protocol_refs),
            "measurement": self.measurement.to_dict(),
            "documents_per_second": self.documents_per_second,
            "input_gigabytes_per_second": self.input_gigabytes_per_second,
            "peak_rss_bytes": self.peak_rss_bytes,
            "temporary_disk_amplification": self.temporary_disk_amplification,
            "output_bytes": self.output_bytes,
            "extrapolated_wall_time_seconds": self.extrapolated_wall_time_seconds,
            "missing_metrics": list(self.missing_metrics),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ScopeReductionDecision:
    """A dated decision record for omitting the one optional pipeline stage."""

    omitted_stage_ids: tuple[str, ...] = ()
    date: str | None = None
    owner: str | None = None
    reason: str | None = None
    forecast_reference: str | None = None

    def missing_fields(self, required: Sequence[str]) -> tuple[str, ...]:
        missing: list[str] = []
        for name in required:
            value = getattr(self, str(name), None)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(str(name))
            elif isinstance(value, tuple) and not value:
                missing.append(str(name))
        return tuple(missing)

    def parsed_date(self, date_format: str) -> date | None:
        if not self.date:
            return None
        try:
            return datetime.strptime(self.date, date_format).date()
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "omitted_stage_ids": list(self.omitted_stage_ids),
            "date": self.date,
            "owner": self.owner,
            "reason": self.reason,
            "forecast_reference": self.forecast_reference,
        }


@dataclass(frozen=True)
class BenchmarkRun:
    """One bounded benchmark: a mode, a slice, its per-stage counters, and its deadline."""

    mode_id: str
    slice_fraction: float
    stratified: bool = True
    measurements: tuple[StageMeasurement, ...] = ()
    omitted_stage_ids: tuple[str, ...] = ()
    deadline_seconds: float | None = None
    scope_reduction: ScopeReductionDecision | None = None
    slice_id: str | None = None


@dataclass(frozen=True)
class ForecastReport:
    """The explicit full-corpus forecast. ``NOT_RUN`` whenever the evidence is absent."""

    status: str
    measured_seconds: float | None
    extrapolated_seconds: float | None
    deadline_seconds: float | None
    headroom_seconds: float | None
    reason: str

    @property
    def misses_schedule(self) -> bool:
        return self.status == MISS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "measured_seconds": self.measured_seconds,
            "extrapolated_seconds": self.extrapolated_seconds,
            "deadline_seconds": self.deadline_seconds,
            "headroom_seconds": self.headroom_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PipelineBenchmarkReport:
    """A complete, auditable benchmark artifact: identities, metrics, forecast, verdicts."""

    mode_id: str
    slice_fraction: float
    stratified: bool
    stages: tuple[StageReport, ...]
    omitted_stage_ids: tuple[str, ...]
    mandatory_stage_ids: tuple[str, ...]
    omittable_stage_ids: tuple[str, ...]
    never_omit_stage_ids: tuple[str, ...]
    forecast: ForecastReport
    results: tuple[CheckResult, ...]
    protocol_digest: str
    slice_id: str | None = None
    scope_reduction: ScopeReductionDecision | None = None

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status == FAIL)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def peak_rss_bytes(self) -> int | None:
        observed = [stage.peak_rss_bytes for stage in self.stages if stage.peak_rss_bytes is not None]
        return max(observed) if observed else None

    @property
    def peak_temporary_disk_amplification(self) -> float | None:
        observed = [
            stage.temporary_disk_amplification
            for stage in self.stages
            if stage.temporary_disk_amplification is not None
        ]
        return max(observed) if observed else None

    @property
    def total_output_bytes(self) -> int | None:
        observed = [stage.output_bytes for stage in self.stages if stage.output_bytes is not None]
        if len(observed) != len(self.stages) or not observed:
            return None
        return int(sum(observed))

    def stage(self, stage_id: str) -> StageReport:
        for report in self.stages:
            if report.stage_id == stage_id:
                return report
        raise KeyError(stage_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "protocol_digest": self.protocol_digest,
            "mode_id": self.mode_id,
            "slice_fraction": self.slice_fraction,
            "slice_id": self.slice_id,
            "stratified": self.stratified,
            "mandatory_stage_ids": list(self.mandatory_stage_ids),
            "omittable_stage_ids": list(self.omittable_stage_ids),
            "never_omit_stage_ids": list(self.never_omit_stage_ids),
            "omitted_stage_ids": list(self.omitted_stage_ids),
            "stages": [stage.to_dict() for stage in self.stages],
            "forecast": self.forecast.to_dict(),
            "scope_reduction": self.scope_reduction.to_dict() if self.scope_reduction else None,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_temporary_disk_amplification": self.peak_temporary_disk_amplification,
            "total_output_bytes": self.total_output_bytes,
            "results": [result.__dict__ for result in self.results],
            "ok": self.ok,
        }


# --------------------------------------------------------------------------------------
# Pure report construction
# --------------------------------------------------------------------------------------


def _verdict(check_id: str, requirement: str, observed: Any, ok: bool, reason: str) -> CheckResult:
    return CheckResult(check_id, requirement, str(observed), PASS if ok else FAIL, reason)


def _not_run(check_id: str, requirement: str, observed: Any, reason: str) -> CheckResult:
    return CheckResult(check_id, requirement, str(observed), NOT_RUN, reason)


def stage_report(
    measurement: StageMeasurement,
    slice_fraction: float,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> StageReport:
    """Derive the six Plan 5.5 metrics for one stage. Pure: measurements in, report out."""
    resolved = protocol or load_pipeline_bench_protocol()
    declared = stage_index(resolved).get(measurement.stage_id)
    if declared is None:
        raise PipelineBenchContractError(
            f"{BENCH_STAGE_UNREGISTERED}: stage {measurement.stage_id!r} is not in the frozen contract"
        )

    missing: list[str] = []
    codes: list[str] = []

    def _try(metric: str, compute: Any) -> Any:
        try:
            return compute()
        except PipelineBenchContractError as error:
            missing.append(metric)
            codes.append(str(error).split(":", 1)[0])
            return None

    docs_per_second = _try(
        "documents_per_second",
        lambda: documents_per_second(measurement.documents, measurement.elapsed_seconds),
    )
    gb_per_second = _try(
        "input_gigabytes_per_second",
        lambda: input_gigabytes_per_second(measurement.input_bytes, measurement.elapsed_seconds, protocol=resolved),
    )
    amplification = _try(
        "temporary_disk_amplification",
        lambda: temporary_disk_amplification(measurement.peak_temporary_disk_bytes, measurement.input_bytes),
    )
    extrapolated = _try(
        "extrapolated_wall_time_seconds",
        lambda: extrapolate_wall_time(measurement.elapsed_seconds, slice_fraction),
    )

    peak_rss = measurement.peak_rss_bytes if _is_measured(measurement.peak_rss_bytes) else None
    if peak_rss is None:
        missing.append("peak_rss_bytes")
        codes.append(BENCH_METRIC_MISSING)
    output_bytes = measurement.output_bytes if _is_measured(measurement.output_bytes) else None
    if output_bytes is None:
        missing.append("output_bytes")
        codes.append(BENCH_METRIC_MISSING)

    ordered_missing = tuple(name for name in REQUIRED_STAGE_METRICS if name in set(missing))
    unique_codes = tuple(dict.fromkeys(codes)) if ordered_missing else (BENCH_OK,)
    return StageReport(
        stage_id=str(measurement.stage_id),
        order=int(declared["order"]),
        mandatory=not bool(declared["omittable"]),
        never_bypass=bool(declared["plan_never_bypass"]),
        protocol_refs=tuple(str(ref) for ref in declared.get("protocol_refs", ())),
        measurement=measurement,
        documents_per_second=docs_per_second,
        input_gigabytes_per_second=gb_per_second,
        peak_rss_bytes=int(peak_rss) if peak_rss is not None else None,
        temporary_disk_amplification=amplification,
        output_bytes=int(output_bytes) if output_bytes is not None else None,
        extrapolated_wall_time_seconds=extrapolated,
        missing_metrics=ordered_missing,
        reason_codes=unique_codes,
    )


def build_forecast(
    stages: Sequence[StageReport],
    deadline_seconds: float | None,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> ForecastReport:
    """Sum the enabled stage extrapolations into one explicit forecast.

    ``NOT_RUN`` when no stage was measured, when any enabled stage lacks its extrapolation,
    or when no deadline was supplied. Absence of evidence is never ``FIT``.
    """
    resolved = protocol or load_pipeline_bench_protocol()
    if not bool(resolved["forecast"]["run_total_is_sum_of_enabled_stage_extrapolations"]):
        raise PipelineBenchContractError("the frozen forecast method is no longer the sum of stage extrapolations")
    if not stages:
        return ForecastReport(NOT_RUN, None, None, deadline_seconds, None, "no stage measurement was supplied")
    incomplete = tuple(stage.stage_id for stage in stages if stage.extrapolated_wall_time_seconds is None)
    if incomplete:
        return ForecastReport(
            NOT_RUN,
            None,
            None,
            deadline_seconds,
            None,
            f"{BENCH_METRIC_MISSING}: stages without an extrapolation {incomplete}",
        )
    measured = float(sum(float(stage.measurement.elapsed_seconds or 0.0) for stage in stages))
    extrapolated = float(sum(float(stage.extrapolated_wall_time_seconds or 0.0) for stage in stages))
    if deadline_seconds is None:
        return ForecastReport(
            NOT_RUN,
            measured,
            extrapolated,
            None,
            None,
            "no schedule deadline was supplied, so fit against the Section 12 calendar is undecided",
        )
    if not _is_measured(deadline_seconds) or float(deadline_seconds) == 0.0:
        raise PipelineBenchContractError(f"{BENCH_METRIC_NOT_MEASURED}: deadline_seconds={deadline_seconds!r}")
    headroom = float(deadline_seconds) - extrapolated
    if extrapolated > float(deadline_seconds):
        return ForecastReport(
            MISS,
            measured,
            extrapolated,
            float(deadline_seconds),
            headroom,
            f"{BENCH_FORECAST_MISSES_SCHEDULE}: forecast {extrapolated:.3f}s exceeds deadline {float(deadline_seconds):.3f}s",
        )
    return ForecastReport(
        FIT,
        measured,
        extrapolated,
        float(deadline_seconds),
        headroom,
        f"{BENCH_OK}: forecast {extrapolated:.3f}s fits deadline {float(deadline_seconds):.3f}s",
    )


def assert_scope_reduction_allowed(
    omitted_stage_ids: Sequence[str],
    *,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Fail closed on any omission the plan forbids; return the omissions in frozen order.

    Only the optional expanded within-source full-corpus near-dedup pass may be dropped,
    and omissions must follow the frozen ``omission_order`` prefix.
    """
    resolved = protocol or load_pipeline_bench_protocol()
    stages = stage_index(resolved)
    order = omittable_stage_ids(resolved)
    never = set(never_omit_stage_ids(resolved))

    requested: list[str] = []
    for stage_id in omitted_stage_ids:
        name = str(stage_id)
        if name in requested:
            raise PipelineBenchContractError(f"{BENCH_STAGE_DUPLICATED}: stage {name!r} was omitted twice")
        requested.append(name)

    for name in requested:
        declared = stages.get(name)
        if declared is None:
            raise PipelineBenchContractError(f"{BENCH_STAGE_UNREGISTERED}: stage {name!r} is not in the frozen contract")
        if name in never:
            raise MandatoryStageOmittedError(
                f"{BENCH_MANDATORY_STAGE_OMITTED}: stage {name!r} can never be bypassed (Plan Section 5.5)"
            )
        if not bool(declared["omittable"]):
            raise MandatoryStageOmittedError(
                f"{BENCH_MANDATORY_STAGE_OMITTED}: stage {name!r} is mandatory in every profile"
            )

    ordered = tuple(name for name in order if name in set(requested))
    expected_prefix = order[: len(ordered)]
    if ordered != expected_prefix:
        raise PipelineBenchContractError(
            f"{BENCH_OMISSION_ORDER_VIOLATED}: omissions {ordered} do not follow the frozen order {order}"
        )
    return ordered


def scope_reduction_results(
    omitted_stage_ids: Sequence[str],
    decision: ScopeReductionDecision | None,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[CheckResult, ...]:
    """Audit a scope reduction: allowed stages, frozen order, and a dated decision record."""
    resolved = protocol or load_pipeline_bench_protocol()
    policy = resolved["scope_reduction_decision"]
    results: list[CheckResult] = []

    requested = tuple(str(name) for name in omitted_stage_ids)
    try:
        ordered = assert_scope_reduction_allowed(requested, protocol=resolved)
        results.append(
            _verdict(
                "DEGRADATION_POLICY",
                "only the optional expanded within-source near-dedup pass may be omitted, in the frozen order",
                ordered or "no omission requested",
                True,
                f"{BENCH_OK}: mandatory stages remain enabled",
            )
        )
    except PipelineBenchContractError as error:
        results.append(
            _verdict(
                "DEGRADATION_POLICY",
                "only the optional expanded within-source near-dedup pass may be omitted, in the frozen order",
                requested,
                False,
                str(error),
            )
        )

    if not requested:
        results.append(
            _not_run(
                "SCOPE_DECISION_RECORD",
                "a scope reduction requires a dated decision record",
                "no omission requested",
                "no scope reduction was claimed, so no decision record is required yet",
            )
        )
        return tuple(results)

    if decision is None:
        results.append(
            _verdict(
                "SCOPE_DECISION_RECORD",
                "a scope reduction requires a dated decision record",
                requested,
                False,
                f"{BENCH_SCOPE_DECISION_RECORD_MISSING}: no decision record was supplied",
            )
        )
        return tuple(results)

    required = tuple(str(name) for name in policy["required_fields"])
    missing = decision.missing_fields(required)
    parsed = decision.parsed_date(str(policy["date_format"]))
    mismatch = tuple(sorted(set(decision.omitted_stage_ids) ^ set(requested)))
    problems: list[str] = []
    if missing:
        problems.append(f"missing fields {missing}")
    if parsed is None:
        problems.append(f"date {decision.date!r} is not {policy['date_format']}")
    if mismatch:
        problems.append(f"record and run disagree on omitted stages {mismatch}")
    results.append(
        _verdict(
            "SCOPE_DECISION_RECORD",
            "a scope reduction requires a dated decision record naming the omitted stages",
            decision.to_dict(),
            not problems,
            f"{BENCH_OK}: dated scope-reduction record is complete"
            if not problems
            else f"{BENCH_SCOPE_DECISION_RECORD_MISSING}: " + "; ".join(problems),
        )
    )
    return tuple(results)


def build_benchmark_report(
    run: BenchmarkRun,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> PipelineBenchmarkReport:
    """Turn one bounded run's counters into a complete, auditable benchmark artifact."""
    resolved = protocol or load_pipeline_bench_protocol()
    stages_declared = stage_index(resolved)
    mandatory = mandatory_stage_ids(resolved)
    omittable = omittable_stage_ids(resolved)
    never = never_omit_stage_ids(resolved)
    results: list[CheckResult] = []

    try:
        fraction = resolve_slice_fraction(run.mode_id, run.slice_fraction, protocol=resolved)
        results.append(
            _verdict(
                "MODE_SLICE_FRACTION",
                f"mode {run.mode_id!r} slice fraction is inside its frozen window",
                fraction,
                True,
                f"{BENCH_OK}: slice fraction accepted",
            )
        )
    except PipelineBenchContractError as error:
        fraction = float(run.slice_fraction) if _is_measured(run.slice_fraction) else 0.0
        results.append(
            _verdict(
                "MODE_SLICE_FRACTION",
                f"mode {run.mode_id!r} slice fraction is inside its frozen window",
                run.slice_fraction,
                False,
                str(error),
            )
        )

    stratified_required = bool(mode_index(resolved).get(str(run.mode_id), {}).get("stratified_required", True))
    results.append(
        _verdict(
            "SLICE_STRATIFIED",
            "the benchmark slice is stratified across every declared source",
            run.stratified,
            bool(run.stratified) or not stratified_required,
            f"{BENCH_OK}: slice is stratified"
            if run.stratified
            else f"{BENCH_SLICE_NOT_STRATIFIED}: an unstratified slice cannot forecast the real mixture",
        )
    )

    reports: list[StageReport] = []
    seen: list[str] = []
    duplicated: list[str] = []
    for measurement in run.measurements:
        if measurement.stage_id in seen:
            duplicated.append(measurement.stage_id)
            continue
        seen.append(measurement.stage_id)
        reports.append(stage_report(measurement, fraction if fraction > 0 else run.slice_fraction, protocol=resolved))
    stage_reports = tuple(reports)

    if duplicated:
        results.append(
            _verdict(
                "STAGE_UNIQUENESS",
                "each pipeline stage is measured at most once per run",
                tuple(duplicated),
                False,
                f"{BENCH_STAGE_DUPLICATED}: repeated stage measurements {tuple(duplicated)}",
            )
        )

    observed_orders = tuple(report.order for report in stage_reports)
    results.append(
        _verdict(
            "STAGE_ORDER",
            "measured stages appear in the frozen execution order",
            observed_orders,
            list(observed_orders) == sorted(observed_orders),
            f"{BENCH_OK}: stage order matches the frozen contract"
            if list(observed_orders) == sorted(observed_orders)
            else f"{BENCH_STAGE_ORDER_MISMATCH}: observed stage orders {observed_orders}",
        )
    )

    omitted = tuple(str(name) for name in run.omitted_stage_ids)
    measured_ids = {report.stage_id for report in stage_reports}
    if not stage_reports:
        results.append(
            _not_run(
                "STAGE_COVERAGE",
                f"every mandatory stage {mandatory} is measured on the slice",
                "no stage measurement supplied",
                "the pipeline benchmark has not been run, so coverage is undecided",
            )
        )
    else:
        absent = tuple(name for name in mandatory if name not in measured_ids)
        results.append(
            _verdict(
                "STAGE_COVERAGE",
                f"every mandatory stage {mandatory} is measured on the slice",
                sorted(measured_ids),
                not absent,
                f"{BENCH_OK}: all mandatory stages measured"
                if not absent
                else f"{BENCH_STAGE_MISSING}: mandatory stages without measurements {absent}",
            )
        )

    disabled = tuple(name for name in never if name in set(omitted) or (stage_reports and name not in measured_ids))
    if stage_reports:
        results.append(
            _verdict(
                "FULL_INTEGRITY_STAGES_ENABLED",
                f"cross-source near-dedup, split/reserved isolation, and decontamination always run {never}",
                sorted(name for name in never if name in measured_ids),
                not disabled,
                f"{BENCH_OK}: never-omit stages ran on this slice"
                if not disabled
                else f"{BENCH_MANDATORY_STAGE_DISABLED}: never-omit stages absent or omitted {disabled}",
            )
        )
    else:
        results.append(
            _not_run(
                "FULL_INTEGRITY_STAGES_ENABLED",
                f"cross-source near-dedup, split/reserved isolation, and decontamination always run {never}",
                "no stage measurement supplied",
                "the pipeline benchmark has not been run, so stage enablement is undecided",
            )
        )

    for report in stage_reports:
        results.append(
            _verdict(
                f"STAGE_METRICS::{report.stage_id}",
                f"all six Plan 5.5 counters are measured for {report.stage_id!r}",
                {
                    "documents_per_second": report.documents_per_second,
                    "input_gigabytes_per_second": report.input_gigabytes_per_second,
                    "peak_rss_bytes": report.peak_rss_bytes,
                    "temporary_disk_amplification": report.temporary_disk_amplification,
                    "output_bytes": report.output_bytes,
                    "extrapolated_wall_time_seconds": report.extrapolated_wall_time_seconds,
                },
                report.complete,
                f"{BENCH_OK}: every required counter measured"
                if report.complete
                else f"{BENCH_METRIC_MISSING}: {report.missing_metrics} ({report.reason_codes})",
            )
        )

    for stage_id in sorted(name for name in stages_declared if name not in measured_ids and name not in set(omitted)):
        if stage_id in mandatory and stage_reports:
            continue
        if not stage_reports:
            continue
        results.append(
            _not_run(
                f"STAGE_METRICS::{stage_id}",
                f"all six Plan 5.5 counters are measured for {stage_id!r}",
                "no measurement supplied",
                "this optional stage was neither measured nor formally omitted",
            )
        )

    results.extend(scope_reduction_results(omitted, run.scope_reduction, protocol=resolved))

    forecast = build_forecast(stage_reports, run.deadline_seconds, protocol=resolved)
    results.append(
        CheckResult(
            "FORECAST",
            "the extrapolated full-corpus wall time fits the Section 12 schedule",
            str(forecast.to_dict()),
            {FIT: PASS, MISS: FAIL, NOT_RUN: NOT_RUN}[forecast.status],
            forecast.reason,
        )
    )

    return PipelineBenchmarkReport(
        mode_id=str(run.mode_id),
        slice_fraction=float(run.slice_fraction),
        stratified=bool(run.stratified),
        stages=stage_reports,
        omitted_stage_ids=omitted,
        mandatory_stage_ids=mandatory,
        omittable_stage_ids=omittable,
        never_omit_stage_ids=never,
        forecast=forecast,
        results=tuple(results),
        protocol_digest=str(resolved["_digest"]),
        slice_id=run.slice_id,
        scope_reduction=run.scope_reduction,
    )


# --------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------


def write_benchmark_artifact(path: Path, report: PipelineBenchmarkReport) -> Path:
    """Write one benchmark artifact as deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_benchmark_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PipelineBenchContractError(f"{BENCH_ARTIFACT_INCOMPLETE}: {path} does not contain a JSON object")
    return payload


def verify_benchmark_artifact(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[CheckResult, ...]:
    """Audit a stored artifact: required fields, frozen identities, and one explicit forecast."""
    resolved = protocol or load_pipeline_bench_protocol()
    contract = resolved["artifact"]
    results: list[CheckResult] = []

    required = tuple(str(name) for name in contract["required_fields"])
    absent = tuple(name for name in required if name not in payload)
    results.append(
        _verdict(
            "ARTIFACT_FIELDS",
            f"the artifact carries every required field {required}",
            sorted(payload),
            not absent,
            f"{BENCH_OK}: artifact is complete"
            if not absent
            else f"{BENCH_ARTIFACT_INCOMPLETE}: missing fields {absent}",
        )
    )

    expected_digest = str(resolved["_digest"])
    observed_digest = str(payload.get("protocol_digest", ""))
    results.append(
        _verdict(
            "ARTIFACT_PROTOCOL_DIGEST",
            "the artifact was produced under the frozen benchmark contract",
            observed_digest or "absent",
            observed_digest == expected_digest,
            f"{BENCH_OK}: artifact matches {ARTIFACT_SCHEMA_VERSION}"
            if observed_digest == expected_digest
            else f"{BENCH_ARTIFACT_PROTOCOL_MISMATCH}: expected {expected_digest}, observed {observed_digest or 'absent'}",
        )
    )

    declared_mandatory = tuple(str(name) for name in payload.get("mandatory_stage_ids", ()))
    declared_never = tuple(str(name) for name in payload.get("never_omit_stage_ids", ()))
    frozen_mandatory = mandatory_stage_ids(resolved)
    frozen_never = never_omit_stage_ids(resolved)
    identities_ok = declared_mandatory == frozen_mandatory and declared_never == frozen_never
    results.append(
        _verdict(
            "ARTIFACT_MANDATORY_STAGE_IDENTITY",
            "the artifact names the frozen mandatory and never-omit stages",
            {"mandatory": declared_mandatory, "never_omit": declared_never},
            identities_ok,
            f"{BENCH_OK}: mandatory stage identities match the frozen contract"
            if identities_ok
            else f"{BENCH_ARTIFACT_INCOMPLETE}: expected mandatory {frozen_mandatory} and never-omit {frozen_never}",
        )
    )

    forecast = payload.get("forecast")
    statuses = tuple(str(name) for name in resolved["forecast"]["statuses"])
    if not isinstance(forecast, Mapping) or "status" not in forecast:
        results.append(
            _verdict(
                "ARTIFACT_FORECAST",
                "the artifact states one explicit forecast",
                forecast,
                False,
                f"{BENCH_ARTIFACT_INCOMPLETE}: no explicit forecast is present",
            )
        )
    elif str(forecast["status"]) not in statuses:
        results.append(
            _verdict(
                "ARTIFACT_FORECAST",
                "the artifact states one explicit forecast",
                forecast.get("status"),
                False,
                f"{BENCH_ARTIFACT_INCOMPLETE}: forecast status must be one of {statuses}",
            )
        )
    elif str(forecast["status"]) == NOT_RUN:
        results.append(
            _not_run(
                "ARTIFACT_FORECAST",
                "the artifact states one explicit forecast",
                forecast.get("reason", ""),
                "the forecast is explicitly NOT_RUN, which is never a pass",
            )
        )
    else:
        fit = str(forecast["status"]) == FIT
        results.append(
            _verdict(
                "ARTIFACT_FORECAST",
                "the artifact states one explicit forecast",
                {k: forecast.get(k) for k in ("status", "extrapolated_seconds", "deadline_seconds")},
                fit,
                f"{BENCH_OK}: forecast fits the schedule"
                if fit
                else f"{BENCH_FORECAST_MISSES_SCHEDULE}: {forecast.get('reason', '')}",
            )
        )
    return tuple(results)


def format_pipeline_bench_report(results: Sequence[CheckResult]) -> str:
    """Human-readable summary of a benchmark audit."""
    width = max((len(result.check_id) for result in results), default=0)
    lines = ["pipeline benchmark audit"]
    for result in results:
        lines.append(f"  [{result.status:<8}] {result.check_id:<{width}}  {result.reason}")
    return "\n".join(lines)
