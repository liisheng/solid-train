"""Two-machine measurement, horizon calculators, and G0-G6 gates (Plan Sections 7.1, 9, 10.4, 12-13).

Every measurement here is synthetic. Nothing profiles a GPU, promotes a backend, rehearses
takeover, schedules a dated campaign, or passes a real gate. What the tests prove is that the
tooling cannot manufacture a claim:

- percentiles are nearest-rank, so p10/median/p90 are always observed samples and two machines
  summarizing the same run agree,
- a throughput number is rejected unless its window, sample count, and real-shard provenance
  earn it,
- the safe microbatch is the largest that still leaves at least 10% VRAM headroom, not the
  largest that merely ran,
- every horizon and efficiency calculator refuses ``None``, a sentinel string, a bool, NaN,
  and infinity rather than turning a placeholder into a number,
- a gate is PASS only when every one of its requirements is PASS, an unrecorded requirement is
  NOT_RUN, and a blocker without an owner and next action is FAIL,
- Plan Section 7.1's 20B and 402.8 tokens/parameter stay NOT_REACHED targets, and an
  unmeasured 8 GB fit, CPU RAM, or throughput cannot be stated as measured.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tinybench_lm.data_protocols import ProtocolMutatedError, protocol_digest
from tinybench_lm.operations import (
    BLOCKED,
    FLOPS_PER_PARAMETER_PER_TOKEN,
    FROZEN_OPERATIONS_PROTOCOL_SHA256,
    GATE_IDS,
    GATE_STATUSES,
    MACHINE_3070,
    MACHINE_4070,
    OPERATIONS_PROTOCOL_PATH,
    OPS_BACKEND_PROMOTION_UNJUSTIFIED,
    OPS_BLOCKER_DETAIL_INCOMPLETE,
    OPS_FAIL_CLOSED_REASON_CODES,
    OPS_INSUFFICIENT_SAMPLES,
    OPS_TAKEOVER_NOT_REHEARSED,
    OPS_TARGET_CLAIMED_AS_MEASURED,
    OPS_THROUGHPUT_WINDOW_TOO_SHORT,
    OPS_UNMEASURED_INPUT,
    OPS_VRAM_HEADROOM_INSUFFICIENT,
    BackendComparison,
    Evidence,
    MachineProfile,
    MicrobatchProbe,
    OperationsContractError,
    OperationsNotReadyError,
    ThroughputMeasurement,
    UnmeasuredInputError,
    active_gpu_hours,
    approximate_flops,
    assert_ready_for_horizon_claim,
    backend_promotion_violations,
    branch_time,
    build_horizon_plan,
    consumed_tokens,
    effective_passes,
    evaluate_all_gates,
    evaluate_gate,
    evaluation_reserve,
    evaluation_time,
    gate_requirements,
    is_measured,
    load_operations_protocol,
    mainline_parent_time,
    minimum_vram_headroom,
    nearest_rank_percentile,
    profile_violations,
    readiness_results,
    require_measured,
    safe_microbatch,
    stretch_target_violations,
    takeover_rehearsal_violations,
    target_reference,
    throughput_violations,
    tokens_per_parameter,
    wall_time,
)
from tinybench_lm.shards import FAIL, NOT_RUN, PASS

GIGABYTE = 1024**3
LOSS_TOKENS_PER_UPDATE = 262_144
FINAL_PARAMETERS = 49_658_368


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_operations_protocol()


def _throughput(machine_id: str = MACHINE_4070, **overrides) -> ThroughputMeasurement:
    defaults = {
        "machine_id": machine_id,
        "samples": (14_000.0, 15_000.0, 15_500.0, 16_000.0, 16_500.0),
        "window_seconds": 2400.0,
        "used_real_shards": True,
    }
    return ThroughputMeasurement(**{**defaults, "machine_id": machine_id, **overrides})


def _profile(**overrides) -> MachineProfile:
    defaults = {
        "machine_id": MACHINE_4070,
        "gpu_name": "SYNTHETIC-GPU",
        "vram_total_bytes": 12.0 * GIGABYTE,
        "bf16_supported": True,
        "bf16_stable": True,
        "backend": "eager_sdpa",
        "safe_microbatch": 8,
        "throughput": _throughput(),
        "peak_ram_bytes": 20.0 * GIGABYTE,
        "peak_swap_bytes": 0.0,
        "peak_vram_bytes": 10.0 * GIGABYTE,  # ~16.7% headroom
        "dataloader_wait_seconds": 12.5,
        "checkpoint_seconds": 30.0,
        "thermal_throttle_reasons": (),
    }
    return MachineProfile(**{**defaults, **overrides})


# --------------------------------------------------------------------------------------
# The frozen contract
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_frozen_operations_protocol_is_pinned_and_immutable(tmp_path: Path) -> None:
    assert (
        protocol_digest(OPERATIONS_PROTOCOL_PATH)
        == FROZEN_OPERATIONS_PROTOCOL_SHA256["measurement_v1.yaml"]
    )
    text = OPERATIONS_PROTOCOL_PATH.read_text(encoding="utf-8")

    mutated = tmp_path / "measurement_v1.yaml"
    mutated.write_text(text.replace("owner: operator", "owner: nobody"), encoding="utf-8")
    with pytest.raises(ProtocolMutatedError):
        load_operations_protocol(mutated)
    assert str(load_operations_protocol(mutated, verify=False)["readiness"]["owner"]) == "nobody"


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
@pytest.mark.parametrize(
    "old,new",
    [
        ("minimum_vram_headroom_fraction: 0.10", "minimum_vram_headroom_fraction: 0.02"),
        ("minimum_sustained_seconds: 1800", "minimum_sustained_seconds: 60"),
        ("percentile_method: nearest_rank", "percentile_method: linear"),
        ("evaluation_reserve_multiplier: 1.25", "evaluation_reserve_multiplier: 1.0"),
        ("absence_of_evidence_is_never_pass: true", "absence_of_evidence_is_never_pass: false"),
        ("gate_passes_only_if_all_requirements_pass: true", "gate_passes_only_if_all_requirements_pass: false"),
        ("missing_evidence_status: NOT_RUN", "missing_evidence_status: PASS"),
        ("  status: NOT_REACHED", "  status: REACHED"),
        ("  is_measurement: false", "  is_measurement: true"),
        ("approximate_flops_is_estimate: true", "approximate_flops_is_estimate: false"),
        ("rehearsed_before: G4", "rehearsed_before: G6"),
        ("requires_throughput_improvement: true", "requires_throughput_improvement: false"),
    ],
)
def test_weakening_the_contract_fails_before_the_digest_is_consulted(
    tmp_path: Path, old: str, new: str
) -> None:
    text = OPERATIONS_PROTOCOL_PATH.read_text(encoding="utf-8")
    assert old in text, f"replacement would be a no-op: {old!r}"
    weakened = tmp_path / "weakened_v1.yaml"
    weakened.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(OperationsContractError):
        load_operations_protocol(weakened, verify=False)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_contract_declares_both_lanes_and_all_seven_gates(protocol: dict) -> None:
    machines = {str(machine["machine_id"]) for machine in protocol["machines"]}
    assert machines == {MACHINE_4070, MACHINE_3070}
    assert tuple(str(gate["gate_id"]) for gate in protocol["gates"]) == GATE_IDS
    assert GATE_STATUSES == (PASS, FAIL, BLOCKED, NOT_RUN)
    assert set(protocol["reason_codes"]["fail_closed"]) == set(OPS_FAIL_CLOSED_REASON_CODES)


# --------------------------------------------------------------------------------------
# Deterministic percentiles (Plan Section 9.1)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4**
def test_nearest_rank_percentile_matches_worked_examples() -> None:
    samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # ceil(p/100 * 10)-th smallest, 1-indexed.
    assert nearest_rank_percentile(samples, 10) == 1
    assert nearest_rank_percentile(samples, 50) == 5
    assert nearest_rank_percentile(samples, 90) == 9
    assert nearest_rank_percentile(samples, 100) == 10
    # Order of the input never matters.
    assert nearest_rank_percentile(list(reversed(samples)), 90) == 9
    # A single sample is its own every percentile.
    assert nearest_rank_percentile([42.0], 10) == 42.0
    assert nearest_rank_percentile([42.0], 90) == 42.0

    # Sample counts where nearest-rank and a round-to-nearest rank disagree. Pinning these
    # keeps the contract's "nearest_rank" from silently becoming another convention, which
    # would make the 4070 and the 3070 report different p90s for the same run.
    eight = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    assert nearest_rank_percentile(eight, 90) == 80.0  # ceil(7.2) = 8, not round(7.2) = 7
    assert nearest_rank_percentile(eight, 10) == 10.0  # ceil(0.8) = 1
    three = [1.0, 2.0, 3.0]
    assert nearest_rank_percentile(three, 10) == 1.0  # ceil(0.3) = 1
    assert nearest_rank_percentile(three, 90) == 3.0  # ceil(2.7) = 3
    seven = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert nearest_rank_percentile(seven, 90) == 7.0  # ceil(6.3) = 7, not round(6.3) = 6

    with pytest.raises(UnmeasuredInputError, match=OPS_INSUFFICIENT_SAMPLES):
        nearest_rank_percentile([], 50)
    with pytest.raises(OperationsContractError):
        nearest_rank_percentile(samples, 0)
    with pytest.raises(OperationsContractError):
        nearest_rank_percentile(samples, 101)


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4**
@given(
    samples=st.lists(
        st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=40,
    )
)
@settings(max_examples=40, deadline=None, derandomize=True)
def test_percentiles_are_observed_samples_and_ordered(samples: list[float]) -> None:
    p10 = nearest_rank_percentile(samples, 10)
    median = nearest_rank_percentile(samples, 50)
    p90 = nearest_rank_percentile(samples, 90)
    # Nearest rank never invents a value between samples, which is why two machines agree.
    for value in (p10, median, p90):
        assert value in samples
    assert p10 <= median <= p90
    assert min(samples) <= p10
    assert p90 <= max(samples)


# --------------------------------------------------------------------------------------
# Throughput and hardware profile (Plan Section 9.1)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_a_throughput_number_must_earn_its_window_and_samples(protocol: dict) -> None:
    assert throughput_violations(_throughput(), protocol) == ()

    too_few = _throughput(samples=(15_000.0, 15_500.0))
    assert any(OPS_INSUFFICIENT_SAMPLES in p for p in throughput_violations(too_few, protocol))

    # Plan 9.1 measures 30-60 minutes; a 60-second burst is not sustained throughput.
    too_short = _throughput(window_seconds=60.0)
    assert any(OPS_THROUGHPUT_WINDOW_TOO_SHORT in p for p in throughput_violations(too_short, protocol))

    synthetic = _throughput(used_real_shards=False)
    assert any(OPS_THROUGHPUT_WINDOW_TOO_SHORT in p for p in throughput_violations(synthetic, protocol))


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_safe_microbatch_requires_ten_percent_headroom(protocol: dict) -> None:
    total = 12.0 * GIGABYTE
    probes = [
        MicrobatchProbe(microbatch=2, peak_vram_bytes=4.0 * GIGABYTE, vram_total_bytes=total),
        MicrobatchProbe(microbatch=4, peak_vram_bytes=8.0 * GIGABYTE, vram_total_bytes=total),
        # 10.5 GB of 12 GB leaves 12.5%: comfortably qualifying.
        MicrobatchProbe(microbatch=8, peak_vram_bytes=10.5 * GIGABYTE, vram_total_bytes=total),
        # 11.5 GB leaves ~4%: it ran, but it is not safe.
        MicrobatchProbe(microbatch=16, peak_vram_bytes=11.5 * GIGABYTE, vram_total_bytes=total),
    ]
    assert minimum_vram_headroom(protocol) == 0.10
    # The largest QUALIFYING probe, not the largest that merely fit.
    assert safe_microbatch(probes, protocol) == 8

    overcommitted = [
        MicrobatchProbe(microbatch=16, peak_vram_bytes=11.9 * GIGABYTE, vram_total_bytes=total)
    ]
    with pytest.raises(OperationsContractError, match=OPS_VRAM_HEADROOM_INSUFFICIENT):
        safe_microbatch(overcommitted, protocol)


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_headroom_boundary_rounds_toward_more_headroom(protocol: dict) -> None:
    """A representable-error tie must never be resolved in favour of less headroom."""
    total = 12.0 * GIGABYTE
    # (12 - 10.8) / 12 is 0.09999999999999998 in binary floating point, not 0.1.
    boundary = MicrobatchProbe(microbatch=8, peak_vram_bytes=10.8 * GIGABYTE, vram_total_bytes=total)
    assert boundary.headroom_fraction < 0.10
    with pytest.raises(OperationsContractError, match=OPS_VRAM_HEADROOM_INSUFFICIENT):
        safe_microbatch([boundary], protocol)

    # A hair more headroom qualifies, so the rule is a boundary and not an exclusion.
    generous = MicrobatchProbe(microbatch=8, peak_vram_bytes=10.79 * GIGABYTE, vram_total_bytes=total)
    assert generous.headroom_fraction > 0.10
    assert safe_microbatch([generous], protocol) == 8


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_machine_profile_records_every_required_measurement(protocol: dict) -> None:
    profile = _profile()
    assert profile_violations(profile, protocol) == ()
    recorded = profile.to_dict()
    for name in protocol["hardware_promotion"]["required_measurements"]:
        assert name in recorded, name

    starved = _profile(peak_vram_bytes=11.9 * GIGABYTE)
    assert any(OPS_VRAM_HEADROOM_INSUFFICIENT in p for p in profile_violations(starved, protocol))

    # A profile carrying a short window is rejected through the same path.
    short = _profile(throughput=_throughput(window_seconds=120.0))
    assert any(OPS_THROUGHPUT_WINDOW_TOO_SHORT in p for p in profile_violations(short, protocol))


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_backend_is_promoted_only_on_correctness_and_improvement(protocol: dict) -> None:
    incumbent = BackendComparison("eager_sdpa", correctness_passed=True, reliability_adjusted_throughput=15_000.0)
    better = BackendComparison("flash", correctness_passed=True, reliability_adjusted_throughput=17_000.0)
    assert backend_promotion_violations(incumbent, better, protocol) == ()

    incorrect = replace(better, correctness_passed=False)
    assert any(OPS_BACKEND_PROMOTION_UNJUSTIFIED in p for p in backend_promotion_violations(incumbent, incorrect, protocol))

    slower = replace(better, reliability_adjusted_throughput=14_000.0)
    assert any(OPS_BACKEND_PROMOTION_UNJUSTIFIED in p for p in backend_promotion_violations(incumbent, slower, protocol))

    # A tie is not an improvement, so it is not a promotion.
    tied = replace(better, reliability_adjusted_throughput=15_000.0)
    assert backend_promotion_violations(incumbent, tied, protocol) != ()


# --------------------------------------------------------------------------------------
# Pure calculators (Plan Sections 7.1 and 10.4)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4**
def test_horizon_calculators_match_the_plan_formulas(protocol: dict) -> None:
    assert mainline_parent_time(150_000_000, 15_000) == pytest.approx(10_000.0)
    assert branch_time(200_015_872, 12_000) == pytest.approx(200_015_872 / 12_000)
    assert evaluation_time(600.0, 4) == pytest.approx(2400.0)
    # Plan 10.4: p90 x candidate count x 1.25
    assert evaluation_reserve(600.0, 4, protocol) == pytest.approx(600.0 * 4 * 1.25)

    assert consumed_tokens(573, LOSS_TOKENS_PER_UPDATE) == 150_208_512
    assert consumed_tokens(0, LOSS_TOKENS_PER_UPDATE) == 0
    assert effective_passes(20_000_000_000, 10_000_000_000) == pytest.approx(2.0)
    assert tokens_per_parameter(20_000_000_000, FINAL_PARAMETERS) == pytest.approx(
        20_000_000_000 / FINAL_PARAMETERS
    )
    assert active_gpu_hours(7200.0) == pytest.approx(2.0)
    assert wall_time(100.0, 200.0, 300.0) == pytest.approx(600.0)
    assert approximate_flops(FINAL_PARAMETERS, 1_000_000) == pytest.approx(
        FLOPS_PER_PARAMETER_PER_TOKEN * FINAL_PARAMETERS * 1_000_000
    )


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
@pytest.mark.parametrize(
    "placeholder",
    [None, "NOT_RUN", "BLOCKED", "TBD", "", "15000", True, False, float("nan"), float("inf"), -float("inf")],
)
def test_every_calculator_refuses_a_placeholder(placeholder: object) -> None:
    """A placeholder that silently becomes a number is how an unmeasured claim ships."""
    assert not is_measured(placeholder)
    with pytest.raises(UnmeasuredInputError, match=OPS_UNMEASURED_INPUT):
        require_measured("value", placeholder)

    for call in (
        lambda: mainline_parent_time(placeholder, 15_000),
        lambda: mainline_parent_time(150_000, placeholder),
        lambda: branch_time(placeholder, 12_000),
        lambda: branch_time(150_000, placeholder),
        lambda: evaluation_time(placeholder, 4),
        lambda: evaluation_time(600.0, placeholder),
        lambda: evaluation_reserve(placeholder, 4),
        lambda: consumed_tokens(placeholder, LOSS_TOKENS_PER_UPDATE),
        lambda: effective_passes(placeholder, 1_000),
        lambda: tokens_per_parameter(1_000, placeholder),
        lambda: active_gpu_hours(placeholder),
        lambda: wall_time(placeholder),
        lambda: approximate_flops(placeholder, 1_000),
    ):
        with pytest.raises(UnmeasuredInputError):
            call()


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_a_real_measurement_is_accepted() -> None:
    for value in (1, 1.5, 15_000, 262_144):
        assert is_measured(value)
        assert require_measured("value", value) == float(value)
    # Zero is a real measurement, but not a valid divisor.
    assert is_measured(0)
    assert require_measured("value", 0, positive=False) == 0.0
    with pytest.raises(UnmeasuredInputError):
        require_measured("rate", 0)


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_horizon_plan_totals_and_calendar_fit(protocol: dict) -> None:
    plan = build_horizon_plan(
        parent_tokens=150_000_000,
        measured_4070_tps=15_000,
        total_branch_tokens=200_015_872,
        measured_3070_tps=12_000,
        measured_eval_runtime=600.0,
        candidate_count=4,
        recovery_reserve_seconds=3600.0,
        protocol=protocol,
    )
    expected = 10_000.0 + 200_015_872 / 12_000 + 2400.0 + 3600.0
    assert plan.total_seconds == pytest.approx(expected)
    assert plan.total_hours == pytest.approx(expected / 3600)
    assert plan.fits_within(expected + 1)
    assert not plan.fits_within(expected - 1)

    # One unmeasured input makes the whole plan unbuildable, rather than partly invented.
    with pytest.raises(UnmeasuredInputError):
        build_horizon_plan(
            parent_tokens=150_000_000,
            measured_4070_tps="NOT_RUN",
            total_branch_tokens=200_015_872,
            measured_3070_tps=12_000,
            measured_eval_runtime=600.0,
            candidate_count=4,
            recovery_reserve_seconds=3600.0,
            protocol=protocol,
        )


# --------------------------------------------------------------------------------------
# Stretch targets stay targets (Plan Section 7.1)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5, 3.3**
def test_stretch_targets_are_never_reported_as_measured(protocol: dict) -> None:
    reference = target_reference(protocol)
    assert reference["status"] == "NOT_REACHED"
    assert reference["is_measurement"] is False
    assert reference["consumed_tokens"] == 20_000_000_000
    assert reference["tokens_per_parameter"] == pytest.approx(402.8)

    # Nothing claimed: nothing to report.
    assert stretch_target_violations({}, protocol) == ()
    assert stretch_target_violations({"eight_gigabyte_vram_fit": NOT_RUN}, protocol) == ()

    # Plan 15 preservation: an unmeasured fit stays a labeled target.
    for name in ("eight_gigabyte_vram_fit", "cpu_ram_sufficiency", "sustained_throughput"):
        problems = stretch_target_violations({name: 8.0}, protocol)
        assert any(OPS_TARGET_CLAIMED_AS_MEASURED in problem for problem in problems), name


# --------------------------------------------------------------------------------------
# Gates G0-G6 (Plan Section 13)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_no_gate_passes_on_absent_evidence(protocol: dict) -> None:
    """The single rule the whole module exists to enforce."""
    reports = evaluate_all_gates({}, protocol)
    assert tuple(report.gate_id for report in reports) == GATE_IDS
    for report in reports:
        assert report.status == NOT_RUN, report.gate_id
        assert all(result.status == NOT_RUN for result in report.results)
    assert not any(report.status == PASS for report in reports)


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_gate_status_reflects_its_requirements(protocol: dict) -> None:
    requirements = gate_requirements("G2", protocol)
    assert requirements
    passing = {requirement.evidence_key: Evidence(passed=True) for requirement in requirements}
    key = requirements[0].evidence_key

    assert evaluate_gate("G2", passing, protocol).status == PASS

    failing = {**passing, key: Evidence(passed=False)}
    assert evaluate_gate("G2", failing, protocol).status == FAIL

    missing = {name: value for name, value in passing.items() if name != key}
    assert evaluate_gate("G2", missing, protocol).status == NOT_RUN

    # A recorded-but-empty placeholder is not the same as an omitted key, and must reach the
    # same verdict. Filing an empty Evidence() is exactly how a gate would get talked into
    # passing on nothing.
    placeholder = {**passing, key: Evidence()}
    report = evaluate_gate("G2", placeholder, protocol)
    assert report.status == NOT_RUN
    assert all(result.status != PASS for result in report.results if result.check_id.endswith(requirements[0].requirement_id))

    blocked = {**passing, key: Evidence(blocker="organizer reply", owner="operator", next_action="email organizer")}
    assert evaluate_gate("G2", blocked, protocol).status == BLOCKED

    # A blocker with no owner and no next action is an excuse, not a status.
    bare = {**passing, key: Evidence(blocker="organizer reply")}
    report = evaluate_gate("G2", bare, protocol)
    assert report.status == FAIL
    assert any(OPS_BLOCKER_DETAIL_INCOMPLETE in result.reason for result in report.results)

    with pytest.raises(OperationsContractError):
        gate_requirements("G9", protocol)


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
@given(data=st.data())
@settings(max_examples=40, deadline=None, derandomize=True)
def test_a_gate_passes_only_when_every_requirement_passes(data: st.DataObject) -> None:
    protocol = load_operations_protocol()
    gate_id = data.draw(st.sampled_from(GATE_IDS))
    requirements = gate_requirements(gate_id, protocol)

    choices = st.sampled_from(["pass", "fail", "missing", "blocked"])
    assignments = data.draw(st.lists(choices, min_size=len(requirements), max_size=len(requirements)))

    evidence: dict[str, Evidence] = {}
    for requirement, choice in zip(requirements, assignments):
        if choice == "pass":
            evidence[requirement.evidence_key] = Evidence(passed=True)
        elif choice == "fail":
            evidence[requirement.evidence_key] = Evidence(passed=False)
        elif choice == "blocked":
            evidence[requirement.evidence_key] = Evidence(
                blocker="external", owner="operator", next_action="follow up"
            )
        # "missing" records nothing at all.

    report = evaluate_gate(gate_id, evidence, protocol)
    everything_passed = all(choice == "pass" for choice in assignments)
    assert (report.status == PASS) == everything_passed
    # Severity ordering: any FAIL dominates, then BLOCKED, then NOT_RUN.
    if "fail" in assignments:
        assert report.status == FAIL
    elif "blocked" in assignments:
        assert report.status == BLOCKED
    elif "missing" in assignments:
        assert report.status == NOT_RUN


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_g4_requires_throughput_and_a_takeover_rehearsal(protocol: dict) -> None:
    keys = {requirement.evidence_key for requirement in gate_requirements("G4", protocol)}
    assert "real_shard_throughput_measured" in keys
    assert "takeover_rehearsal_passes" in keys
    assert "two_person_freeze_bundle_approval" in keys
    assert "no_failed_correctness_gate_bypassed" in keys
    assert str(protocol["takeover_rehearsal"]["rehearsed_before"]) == "G4"


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_takeover_rehearsal_requires_every_step(protocol: dict) -> None:
    required = [str(step) for step in protocol["takeover_rehearsal"]["required_steps"]]
    assert takeover_rehearsal_violations(required, protocol) == ()
    assert takeover_rehearsal_violations([], protocol) != ()
    partial = takeover_rehearsal_violations(required[:-1], protocol)
    assert any(OPS_TAKEOVER_NOT_REHEARSED in problem for problem in partial)
    assert str(protocol["takeover_rehearsal"]["status"]) == NOT_RUN


# --------------------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_operations_readiness_is_not_run_or_blocked_with_an_owner(protocol: dict) -> None:
    results = readiness_results(protocol)
    assert results
    statuses = {result.status for result in results}
    assert PASS not in statuses, "no hardware measurement has been taken yet"
    assert statuses <= {NOT_RUN, BLOCKED}
    for result in results:
        assert "blocker=" in result.reason
        assert "owner=" in result.reason
        assert "next_action=" in result.reason


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_horizon_claims_refuse_to_run_without_measured_profiles(protocol: dict) -> None:
    with pytest.raises(OperationsNotReadyError) as excinfo:
        assert_ready_for_horizon_claim(protocol)
    message = str(excinfo.value)
    assert "measured_4070_profile" in message
    assert "next_action=" in message
