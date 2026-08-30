"""Fixture-timing tests for bounded pipeline benchmarks, forecasts, and degradation policy.

Plan Section 5.5, Sections 12 and 13 G1. Nothing here streams a corpus, times a real stage,
or records a real throughput: every number is a fixture timing supplied to a pure
calculator. The tests prove that:

- the 1% mode is fixed and the 2-5% mode is configurable inside its frozen window,
- all six Plan 5.5 counters are derived exactly from measured inputs and a missing counter
  fails instead of defaulting to zero,
- the forecast is an explicit ``FIT``/``MISS``/``NOT_RUN`` derived only from measurements,
- the one optional expanded within-source near-dedup pass is the only omittable stage and a
  reduction needs a dated decision record,
- cross-source near-dedup, split/reserved isolation, and benchmark decontamination can
  never be bypassed,
- a benchmark artifact identifies the mandatory stages plus one forecast and is verifiable,
- absence of evidence is ``NOT_RUN``, never ``PASS``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tinybench_lm.data_protocols import ProtocolMutatedError, protocol_digest
from tinybench_lm.environment import CheckResult
from tinybench_lm.pipeline_bench import (
    ARTIFACT_SCHEMA_VERSION,
    BENCH_ARTIFACT_INCOMPLETE,
    BENCH_ARTIFACT_PROTOCOL_MISMATCH,
    BENCH_FORECAST_MISSES_SCHEDULE,
    BENCH_MANDATORY_STAGE_DISABLED,
    BENCH_MANDATORY_STAGE_OMITTED,
    BENCH_METRIC_MISSING,
    BENCH_SCOPE_DECISION_RECORD_MISSING,
    BENCH_STAGE_MISSING,
    BENCH_STAGE_ORDER_MISMATCH,
    BENCH_STAGE_UNREGISTERED,
    DEFERRED,
    EXPECTED_NEVER_OMIT_STAGES,
    EXPECTED_OMITTABLE_STAGES,
    EXPECTED_STAGES,
    FAIL,
    FIT,
    FROZEN_PIPELINE_BENCH_SHA256,
    MISS,
    MODE_1PCT,
    MODE_2_TO_5PCT,
    NOT_RUN,
    PASS,
    PIPELINE_BENCH_PROTOCOL_PATH,
    REQUIRED_STAGE_METRICS,
    BenchmarkRun,
    MandatoryStageOmittedError,
    PipelineBenchContractError,
    PipelineBenchNotReadyError,
    PipelineBenchmarkReport,
    ScopeReductionDecision,
    StageMeasurement,
    assert_ready_for_real_pipeline_benchmark,
    assert_scope_reduction_allowed,
    build_benchmark_report,
    build_forecast,
    documents_per_second,
    extrapolate_wall_time,
    format_pipeline_bench_report,
    input_gigabytes_per_second,
    load_benchmark_artifact,
    load_pipeline_bench_protocol,
    mandatory_stage_ids,
    never_omit_stage_ids,
    omittable_stage_ids,
    resolve_slice_fraction,
    stage_report,
    temporary_disk_amplification,
    verify_benchmark_artifact,
    write_benchmark_artifact,
)

PROTOCOL = load_pipeline_bench_protocol()

OPTIONAL_STAGE = "expanded_within_source_full_corpus_near_dedup"

# --------------------------------------------------------------------------------------
# Fixture timings. Chosen so every derived metric is exact in binary floating point.
# These are invented inputs for pure calculators, never observed measurements.
# --------------------------------------------------------------------------------------

FIXTURE_ELAPSED: dict[str, float] = {
    "stream_and_filter": 5.0,
    "exact_and_mirror_dedup": 4.0,
    "within_source_near_dedup": 8.0,
    OPTIONAL_STAGE: 16.0,
    "cross_source_near_dedup": 10.0,
    "split_reserved_isolation": 2.0,
    "benchmark_decontamination": 1.0,
    "tokenize_and_pack": 4.0,
}

FIXTURE_TOTAL_ELAPSED = 50.0
FIXTURE_TOTAL_WITHOUT_OPTIONAL = 34.0


def measurement(stage_id: str, **overrides: object) -> StageMeasurement:
    """One stage's fixture counters, with any counter overridable (including to ``None``)."""
    fields: dict[str, object] = {
        "documents": 2500,
        "input_bytes": 2_000_000_000,
        "output_bytes": 1_500_000_000,
        "elapsed_seconds": FIXTURE_ELAPSED[stage_id],
        "peak_rss_bytes": 900_000_000,
        "peak_temporary_disk_bytes": 5_000_000_000,
    }
    fields.update(overrides)
    return StageMeasurement(stage_id=stage_id, **fields)  # type: ignore[arg-type]


def full_measurements(*, skip: tuple[str, ...] = ()) -> tuple[StageMeasurement, ...]:
    return tuple(measurement(stage_id) for stage_id in EXPECTED_STAGES if stage_id not in skip)


def run(**overrides: object) -> BenchmarkRun:
    fields: dict[str, object] = {
        "mode_id": MODE_1PCT,
        "slice_fraction": 0.01,
        "stratified": True,
        "measurements": full_measurements(),
        "deadline_seconds": 6000.0,
    }
    fields.update(overrides)
    return BenchmarkRun(**fields)  # type: ignore[arg-type]


def result_for(report: PipelineBenchmarkReport, check_id: str) -> CheckResult:
    for result in report.results:
        if result.check_id == check_id:
            return result
    raise AssertionError(f"{check_id} is not in {[result.check_id for result in report.results]}")


DATED_DECISION = ScopeReductionDecision(
    omitted_stage_ids=(OPTIONAL_STAGE,),
    date="2025-08-29",
    owner="data lane operator",
    reason="fixture forecast exceeded the Section 12 data-lane window",
    forecast_reference="runs/bench/slice_1pct.artifact.json",
)


# --------------------------------------------------------------------------------------
# Frozen contract
# --------------------------------------------------------------------------------------


def test_frozen_contract_matches_its_pinned_digest() -> None:
    observed = protocol_digest(PIPELINE_BENCH_PROTOCOL_PATH)
    assert observed == FROZEN_PIPELINE_BENCH_SHA256[PIPELINE_BENCH_PROTOCOL_PATH.name]
    assert PROTOCOL["_digest"] == observed
    assert PROTOCOL["frozen"] is True


def test_editing_the_frozen_contract_fails_closed(tmp_path: Path) -> None:
    mutated = tmp_path / PIPELINE_BENCH_PROTOCOL_PATH.name
    text = PIPELINE_BENCH_PROTOCOL_PATH.read_text(encoding="utf-8")
    mutated.write_text(text.replace("omittable: false", "omittable: true", 1), encoding="utf-8")
    with pytest.raises(ProtocolMutatedError):
        load_pipeline_bench_protocol(mutated)


def test_contract_declares_the_plan_stage_identities_and_counters() -> None:
    assert mandatory_stage_ids(PROTOCOL) == tuple(s for s in EXPECTED_STAGES if s != OPTIONAL_STAGE)
    assert omittable_stage_ids(PROTOCOL) == EXPECTED_OMITTABLE_STAGES == (OPTIONAL_STAGE,)
    assert never_omit_stage_ids(PROTOCOL) == EXPECTED_NEVER_OMIT_STAGES
    assert REQUIRED_STAGE_METRICS == (
        "documents_per_second",
        "input_gigabytes_per_second",
        "peak_rss_bytes",
        "temporary_disk_amplification",
        "output_bytes",
        "extrapolated_wall_time_seconds",
    )
    # Each stage names the frozen protocol it exercises, so an artifact identifies mechanisms.
    for stage_id in EXPECTED_NEVER_OMIT_STAGES:
        refs = stage_report(measurement(stage_id), 0.01, protocol=PROTOCOL).protocol_refs
        assert refs, f"{stage_id} must reference at least one frozen protocol"
    assert "configs/data/decontam_v1.yaml" in stage_report(
        measurement("benchmark_decontamination"), 0.01, protocol=PROTOCOL
    ).protocol_refs
    assert "configs/data/shards_v1.yaml" in stage_report(
        measurement("split_reserved_isolation"), 0.01, protocol=PROTOCOL
    ).protocol_refs


def test_real_slice_benchmark_is_not_ready() -> None:
    readiness = PROTOCOL["readiness"]
    assert readiness["measured_1pct_benchmark"] == NOT_RUN
    assert readiness["measured_2_to_5pct_benchmark"] == NOT_RUN
    assert readiness["measured_pipeline_forecast"] == NOT_RUN
    assert readiness["real_corpus_slice_benchmark"] == DEFERRED
    with pytest.raises(PipelineBenchNotReadyError) as error:
        assert_ready_for_real_pipeline_benchmark(PROTOCOL)
    assert "next_action" in str(error.value)


# --------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------


def test_one_percent_mode_is_fixed_and_the_second_mode_is_configurable() -> None:
    assert resolve_slice_fraction(MODE_1PCT, 0.01, protocol=PROTOCOL) == 0.01
    for fraction in (0.02, 0.03, 0.04, 0.05):
        assert resolve_slice_fraction(MODE_2_TO_5PCT, fraction, protocol=PROTOCOL) == fraction
    for mode_id, fraction in ((MODE_1PCT, 0.02), (MODE_2_TO_5PCT, 0.01), (MODE_2_TO_5PCT, 0.051), (MODE_1PCT, 0.0)):
        with pytest.raises(PipelineBenchContractError):
            resolve_slice_fraction(mode_id, fraction, protocol=PROTOCOL)
    with pytest.raises(PipelineBenchContractError):
        resolve_slice_fraction("slice_50pct", 0.5, protocol=PROTOCOL)


def test_unstratified_slice_fails_the_report() -> None:
    report = build_benchmark_report(run(stratified=False), protocol=PROTOCOL)
    assert result_for(report, "SLICE_STRATIFIED").status == FAIL


# --------------------------------------------------------------------------------------
# Metric calculators
# --------------------------------------------------------------------------------------


def test_stage_metrics_are_derived_exactly_from_fixture_timings() -> None:
    report = stage_report(measurement("stream_and_filter"), 0.01, protocol=PROTOCOL)
    assert report.documents_per_second == 500.0  # 2500 docs / 5 s
    assert report.input_gigabytes_per_second == 0.4  # 2 GB / 5 s
    assert report.peak_rss_bytes == 900_000_000
    assert report.temporary_disk_amplification == 2.5  # 5e9 temp / 2e9 input
    assert report.output_bytes == 1_500_000_000
    assert math.isclose(report.extrapolated_wall_time_seconds, 500.0, rel_tol=1e-12)  # 5 s / 0.01
    assert report.complete and report.missing_metrics == ()
    assert report.mandatory is True and report.never_bypass is False


def test_zero_elapsed_and_absent_inputs_are_rejected_by_the_calculators() -> None:
    with pytest.raises(PipelineBenchContractError):
        documents_per_second(10, 0.0)
    with pytest.raises(PipelineBenchContractError):
        documents_per_second(None, 1.0)  # type: ignore[arg-type]
    with pytest.raises(PipelineBenchContractError):
        input_gigabytes_per_second(1_000, 0.0, protocol=PROTOCOL)
    with pytest.raises(PipelineBenchContractError):
        temporary_disk_amplification(1_000, 0)
    with pytest.raises(PipelineBenchContractError):
        extrapolate_wall_time(1.0, 0.0)
    with pytest.raises(PipelineBenchContractError):
        extrapolate_wall_time(float("inf"), 0.01)


def test_unregistered_stage_is_rejected() -> None:
    with pytest.raises(PipelineBenchContractError) as error:
        stage_report(StageMeasurement(stage_id="frontend_build"), 0.01, protocol=PROTOCOL)
    assert BENCH_STAGE_UNREGISTERED in str(error.value)


def test_a_missing_counter_fails_and_is_never_a_pass() -> None:
    measurements = tuple(
        measurement(stage_id, peak_rss_bytes=None) if stage_id == "tokenize_and_pack" else measurement(stage_id)
        for stage_id in EXPECTED_STAGES
    )
    report = build_benchmark_report(run(measurements=measurements), protocol=PROTOCOL)
    stage = report.stage("tokenize_and_pack")
    assert stage.missing_metrics == ("peak_rss_bytes",)
    assert stage.peak_rss_bytes is None
    verdict = result_for(report, "STAGE_METRICS::tokenize_and_pack")
    assert verdict.status == FAIL and BENCH_METRIC_MISSING in verdict.reason
    assert report.ok is False


def test_missing_elapsed_time_makes_the_forecast_not_run() -> None:
    measurements = tuple(
        measurement(stage_id, elapsed_seconds=None) if stage_id == "cross_source_near_dedup" else measurement(stage_id)
        for stage_id in EXPECTED_STAGES
    )
    report = build_benchmark_report(run(measurements=measurements), protocol=PROTOCOL)
    assert report.forecast.status == NOT_RUN
    assert report.forecast.extrapolated_seconds is None
    assert result_for(report, "FORECAST").status == NOT_RUN


# --------------------------------------------------------------------------------------
# Complete runs and forecasts
# --------------------------------------------------------------------------------------


def test_complete_one_percent_run_passes_and_forecasts_the_full_corpus() -> None:
    report = build_benchmark_report(run(), protocol=PROTOCOL)
    assert report.ok, format_pipeline_bench_report(report.results)
    assert report.forecast.status == FIT
    assert report.forecast.measured_seconds == FIXTURE_TOTAL_ELAPSED
    assert math.isclose(report.forecast.extrapolated_seconds, 5000.0, rel_tol=1e-12)
    assert math.isclose(report.forecast.headroom_seconds, 1000.0, rel_tol=1e-9)
    assert report.mandatory_stage_ids == mandatory_stage_ids(PROTOCOL)
    assert report.never_omit_stage_ids == EXPECTED_NEVER_OMIT_STAGES
    assert report.peak_rss_bytes == 900_000_000
    assert report.peak_temporary_disk_amplification == 2.5
    assert report.total_output_bytes == 8 * 1_500_000_000
    assert result_for(report, "FULL_INTEGRITY_STAGES_ENABLED").status == PASS
    assert result_for(report, "STAGE_COVERAGE").status == PASS


def test_configurable_three_percent_mode_extrapolates_from_its_own_fraction() -> None:
    report = build_benchmark_report(
        run(mode_id=MODE_2_TO_5PCT, slice_fraction=0.04, deadline_seconds=2000.0), protocol=PROTOCOL
    )
    assert result_for(report, "MODE_SLICE_FRACTION").status == PASS
    assert math.isclose(report.forecast.extrapolated_seconds, 1250.0, rel_tol=1e-12)
    assert report.forecast.status == FIT
    assert report.ok


def test_forecast_misses_the_schedule_when_extrapolation_exceeds_the_deadline() -> None:
    report = build_benchmark_report(run(deadline_seconds=4000.0), protocol=PROTOCOL)
    assert report.forecast.status == MISS
    assert math.isclose(report.forecast.headroom_seconds, -1000.0, rel_tol=1e-9)
    verdict = result_for(report, "FORECAST")
    assert verdict.status == FAIL and BENCH_FORECAST_MISSES_SCHEDULE in verdict.reason


def test_forecast_without_a_deadline_is_not_run_not_a_pass() -> None:
    report = build_benchmark_report(run(deadline_seconds=None), protocol=PROTOCOL)
    assert report.forecast.status == NOT_RUN
    assert math.isclose(report.forecast.extrapolated_seconds, 5000.0, rel_tol=1e-12)
    assert result_for(report, "FORECAST").status == NOT_RUN
    assert report.ok  # NOT_RUN is not a failure, but it is also not a pass
    assert result_for(report, "FORECAST").status != PASS


def test_a_benchmark_with_no_measurements_is_not_run() -> None:
    report = build_benchmark_report(run(measurements=()), protocol=PROTOCOL)
    assert report.stages == ()
    assert result_for(report, "STAGE_COVERAGE").status == NOT_RUN
    assert result_for(report, "FULL_INTEGRITY_STAGES_ENABLED").status == NOT_RUN
    assert result_for(report, "FORECAST").status == NOT_RUN
    assert report.forecast.extrapolated_seconds is None
    assert not any(
        result.status == PASS and result.check_id in {"STAGE_COVERAGE", "FULL_INTEGRITY_STAGES_ENABLED", "FORECAST"}
        for result in report.results
    )


def test_missing_mandatory_stage_measurement_fails_closed() -> None:
    report = build_benchmark_report(
        run(measurements=full_measurements(skip=("benchmark_decontamination",))), protocol=PROTOCOL
    )
    coverage = result_for(report, "STAGE_COVERAGE")
    assert coverage.status == FAIL and BENCH_STAGE_MISSING in coverage.reason
    integrity = result_for(report, "FULL_INTEGRITY_STAGES_ENABLED")
    assert integrity.status == FAIL and BENCH_MANDATORY_STAGE_DISABLED in integrity.reason


def test_out_of_order_stage_measurements_fail() -> None:
    measurements = full_measurements()
    shuffled = (measurements[3], measurements[0]) + measurements[4:]
    report = build_benchmark_report(run(measurements=shuffled), protocol=PROTOCOL)
    order = result_for(report, "STAGE_ORDER")
    assert order.status == FAIL and BENCH_STAGE_ORDER_MISMATCH in order.reason


# --------------------------------------------------------------------------------------
# Degradation policy
# --------------------------------------------------------------------------------------


def test_the_expanded_within_source_pass_is_the_only_allowed_omission() -> None:
    assert assert_scope_reduction_allowed((), protocol=PROTOCOL) == ()
    assert assert_scope_reduction_allowed((OPTIONAL_STAGE,), protocol=PROTOCOL) == (OPTIONAL_STAGE,)
    for stage_id in EXPECTED_STAGES:
        if stage_id == OPTIONAL_STAGE:
            continue
        with pytest.raises(MandatoryStageOmittedError) as error:
            assert_scope_reduction_allowed((stage_id,), protocol=PROTOCOL)
        assert BENCH_MANDATORY_STAGE_OMITTED in str(error.value)


def test_never_omit_stages_cannot_be_bypassed_even_with_a_decision_record() -> None:
    for stage_id in EXPECTED_NEVER_OMIT_STAGES:
        report = build_benchmark_report(
            run(
                measurements=full_measurements(skip=(stage_id,)),
                omitted_stage_ids=(stage_id,),
                scope_reduction=ScopeReductionDecision(
                    omitted_stage_ids=(stage_id,),
                    date="2025-08-29",
                    owner="data lane operator",
                    reason="fixture pressure",
                    forecast_reference="runs/bench/slice_1pct.artifact.json",
                ),
            ),
            protocol=PROTOCOL,
        )
        policy = result_for(report, "DEGRADATION_POLICY")
        assert policy.status == FAIL and BENCH_MANDATORY_STAGE_OMITTED in policy.reason
        assert result_for(report, "FULL_INTEGRITY_STAGES_ENABLED").status == FAIL
        assert report.ok is False


def test_allowed_omission_with_a_dated_record_passes_and_shortens_the_forecast() -> None:
    report = build_benchmark_report(
        run(
            measurements=full_measurements(skip=(OPTIONAL_STAGE,)),
            omitted_stage_ids=(OPTIONAL_STAGE,),
            scope_reduction=DATED_DECISION,
            deadline_seconds=4000.0,
        ),
        protocol=PROTOCOL,
    )
    assert result_for(report, "DEGRADATION_POLICY").status == PASS
    assert result_for(report, "SCOPE_DECISION_RECORD").status == PASS
    assert result_for(report, "FULL_INTEGRITY_STAGES_ENABLED").status == PASS
    assert report.forecast.measured_seconds == FIXTURE_TOTAL_WITHOUT_OPTIONAL
    assert math.isclose(report.forecast.extrapolated_seconds, 3400.0, rel_tol=1e-12)
    assert report.forecast.status == FIT
    assert report.ok, format_pipeline_bench_report(report.results)


def test_scope_reduction_requires_a_complete_dated_record() -> None:
    base = dict(
        measurements=full_measurements(skip=(OPTIONAL_STAGE,)),
        omitted_stage_ids=(OPTIONAL_STAGE,),
        deadline_seconds=4000.0,
    )
    absent = build_benchmark_report(run(**base), protocol=PROTOCOL)
    verdict = result_for(absent, "SCOPE_DECISION_RECORD")
    assert verdict.status == FAIL and BENCH_SCOPE_DECISION_RECORD_MISSING in verdict.reason

    undated = build_benchmark_report(
        run(**base, scope_reduction=ScopeReductionDecision(omitted_stage_ids=(OPTIONAL_STAGE,), owner="operator", reason="pressure", forecast_reference="artifact.json")),
        protocol=PROTOCOL,
    )
    assert result_for(undated, "SCOPE_DECISION_RECORD").status == FAIL

    malformed = build_benchmark_report(
        run(
            **base,
            scope_reduction=ScopeReductionDecision(
                omitted_stage_ids=(OPTIONAL_STAGE,),
                date="29-08-2025",
                owner="operator",
                reason="pressure",
                forecast_reference="artifact.json",
            ),
        ),
        protocol=PROTOCOL,
    )
    assert result_for(malformed, "SCOPE_DECISION_RECORD").status == FAIL

    disagreeing = build_benchmark_report(
        run(
            **base,
            scope_reduction=ScopeReductionDecision(
                omitted_stage_ids=("cross_source_near_dedup",),
                date="2025-08-29",
                owner="operator",
                reason="pressure",
                forecast_reference="artifact.json",
            ),
        ),
        protocol=PROTOCOL,
    )
    assert result_for(disagreeing, "SCOPE_DECISION_RECORD").status == FAIL


def test_no_omission_leaves_the_decision_record_check_not_run() -> None:
    report = build_benchmark_report(run(), protocol=PROTOCOL)
    assert result_for(report, "SCOPE_DECISION_RECORD").status == NOT_RUN


# --------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------


def test_artifact_round_trips_and_identifies_mandatory_stages_and_the_forecast(tmp_path: Path) -> None:
    report = build_benchmark_report(run(), protocol=PROTOCOL)
    path = write_benchmark_artifact(tmp_path / "slice_1pct.artifact.json", report)
    payload = load_benchmark_artifact(path)
    assert payload["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert tuple(payload["mandatory_stage_ids"]) == mandatory_stage_ids(PROTOCOL)
    assert tuple(payload["never_omit_stage_ids"]) == EXPECTED_NEVER_OMIT_STAGES
    assert payload["forecast"]["status"] == FIT
    assert math.isclose(payload["forecast"]["extrapolated_seconds"], 5000.0, rel_tol=1e-12)
    results = verify_benchmark_artifact(payload, protocol=PROTOCOL)
    assert [result.status for result in results] == [PASS, PASS, PASS, PASS], format_pipeline_bench_report(results)
    # Writing is deterministic, so an artifact can be hashed as evidence.
    again = write_benchmark_artifact(tmp_path / "again.json", build_benchmark_report(run(), protocol=PROTOCOL))
    assert again.read_bytes() == path.read_bytes()


def test_artifact_verifier_rejects_tampered_or_incomplete_artifacts(tmp_path: Path) -> None:
    payload = build_benchmark_report(run(), protocol=PROTOCOL).to_dict()

    tampered = dict(payload, mandatory_stage_ids=["stream_and_filter"])
    statuses = {r.check_id: (r.status, r.reason) for r in verify_benchmark_artifact(tampered, protocol=PROTOCOL)}
    assert statuses["ARTIFACT_MANDATORY_STAGE_IDENTITY"][0] == FAIL

    stale = dict(payload, protocol_digest="0" * 64)
    statuses = {r.check_id: (r.status, r.reason) for r in verify_benchmark_artifact(stale, protocol=PROTOCOL)}
    assert statuses["ARTIFACT_PROTOCOL_DIGEST"][0] == FAIL
    assert BENCH_ARTIFACT_PROTOCOL_MISMATCH in statuses["ARTIFACT_PROTOCOL_DIGEST"][1]

    truncated = {key: value for key, value in payload.items() if key != "forecast"}
    statuses = {r.check_id: (r.status, r.reason) for r in verify_benchmark_artifact(truncated, protocol=PROTOCOL)}
    assert statuses["ARTIFACT_FIELDS"][0] == FAIL
    assert BENCH_ARTIFACT_INCOMPLETE in statuses["ARTIFACT_FIELDS"][1]
    assert statuses["ARTIFACT_FORECAST"][0] == FAIL


def test_artifact_forecast_that_was_not_run_is_never_reported_as_pass() -> None:
    payload = build_benchmark_report(run(deadline_seconds=None), protocol=PROTOCOL).to_dict()
    statuses = {result.check_id: result.status for result in verify_benchmark_artifact(payload, protocol=PROTOCOL)}
    assert statuses["ARTIFACT_FORECAST"] == NOT_RUN


# --------------------------------------------------------------------------------------
# Property-based tests
# --------------------------------------------------------------------------------------

_counts = st.integers(min_value=0, max_value=10_000_000)
_bytes = st.integers(min_value=1, max_value=10**13)
_elapsed = st.floats(min_value=1e-3, max_value=1e5, allow_nan=False, allow_infinity=False)
_fraction = st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False)


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
@settings(max_examples=60, deadline=None)
@given(
    documents=_counts,
    input_bytes=_bytes,
    output_bytes=_bytes,
    temp_bytes=st.integers(min_value=0, max_value=10**14),
    peak_rss=_bytes,
    elapsed=_elapsed,
    fraction=_fraction,
)
def test_stage_metrics_are_exact_functions_of_the_supplied_counters(
    documents: int,
    input_bytes: int,
    output_bytes: int,
    temp_bytes: int,
    peak_rss: int,
    elapsed: float,
    fraction: float,
) -> None:
    observed = stage_report(
        StageMeasurement(
            stage_id="stream_and_filter",
            documents=documents,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            elapsed_seconds=elapsed,
            peak_rss_bytes=peak_rss,
            peak_temporary_disk_bytes=temp_bytes,
        ),
        fraction,
        protocol=PROTOCOL,
    )
    assert observed.complete
    assert math.isclose(observed.documents_per_second, documents / elapsed, rel_tol=1e-12)
    assert math.isclose(observed.input_gigabytes_per_second, input_bytes / 1e9 / elapsed, rel_tol=1e-12)
    assert math.isclose(observed.temporary_disk_amplification, temp_bytes / input_bytes, rel_tol=1e-12)
    assert observed.peak_rss_bytes == peak_rss
    assert observed.output_bytes == output_bytes
    assert math.isclose(observed.extrapolated_wall_time_seconds, elapsed / fraction, rel_tol=1e-12)
    # A slice is a fraction of the corpus, so the forecast never shrinks below the measurement.
    assert observed.extrapolated_wall_time_seconds >= elapsed


# **Validates: Requirements 2.1, 2.4, 2.5**
@settings(max_examples=60, deadline=None)
@given(
    elapsed=st.lists(_elapsed, min_size=1, max_size=8),
    fraction=_fraction,
    deadline=st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False),
)
def test_forecast_verdict_follows_only_from_the_measurements(
    elapsed: list[float], fraction: float, deadline: float
) -> None:
    stage_ids = EXPECTED_STAGES[: len(elapsed)]
    stages = tuple(
        stage_report(measurement(stage_id, elapsed_seconds=seconds), fraction, protocol=PROTOCOL)
        for stage_id, seconds in zip(stage_ids, elapsed)
    )
    forecast = build_forecast(stages, deadline, protocol=PROTOCOL)
    expected = sum(seconds / fraction for seconds in elapsed)
    assert math.isclose(forecast.extrapolated_seconds, expected, rel_tol=1e-9)
    assert math.isclose(forecast.measured_seconds, sum(elapsed), rel_tol=1e-9)
    assert forecast.status == (MISS if forecast.extrapolated_seconds > deadline else FIT)
    assert math.isclose(forecast.headroom_seconds, deadline - forecast.extrapolated_seconds, rel_tol=1e-9, abs_tol=1e-9)
    # Without a deadline there is no evidence to decide the fit.
    assert build_forecast(stages, None, protocol=PROTOCOL).status == NOT_RUN


# **Validates: Requirements 1.1, 2.1, 3.3**
@settings(max_examples=60, deadline=None)
@given(requested=st.lists(st.sampled_from(EXPECTED_STAGES), min_size=0, max_size=4, unique=True))
def test_no_mandatory_stage_can_ever_be_omitted(requested: list[str]) -> None:
    mandatory = set(mandatory_stage_ids(PROTOCOL))
    try:
        allowed = assert_scope_reduction_allowed(tuple(requested), protocol=PROTOCOL)
    except MandatoryStageOmittedError:
        assert mandatory & set(requested)
        return
    assert not mandatory & set(allowed)
    assert set(allowed) <= set(EXPECTED_OMITTABLE_STAGES)
    assert not set(allowed) & set(EXPECTED_NEVER_OMIT_STAGES)
    assert set(allowed) == set(requested)


# **Validates: Requirements 2.4, 2.5, 3.3**
@settings(max_examples=40, deadline=None)
@given(
    mode=st.sampled_from([(MODE_1PCT, 0.01), (MODE_2_TO_5PCT, 0.02), (MODE_2_TO_5PCT, 0.05)]),
    deadline=st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False),
)
def test_every_report_identifies_the_mandatory_stages_and_one_explicit_forecast(
    mode: tuple[str, float], deadline: float
) -> None:
    mode_id, fraction = mode
    report = build_benchmark_report(
        run(mode_id=mode_id, slice_fraction=fraction, deadline_seconds=deadline), protocol=PROTOCOL
    )
    assert report.mandatory_stage_ids == mandatory_stage_ids(PROTOCOL)
    assert report.never_omit_stage_ids == EXPECTED_NEVER_OMIT_STAGES
    assert result_for(report, "FULL_INTEGRITY_STAGES_ENABLED").status == PASS
    assert result_for(report, "FORECAST").status in {PASS, FAIL}
    assert report.forecast.status in {FIT, MISS}
    payload = json.loads(json.dumps(report.to_dict()))
    assert tuple(payload["mandatory_stage_ids"]) == mandatory_stage_ids(PROTOCOL)
    assert payload["forecast"]["status"] == report.forecast.status
