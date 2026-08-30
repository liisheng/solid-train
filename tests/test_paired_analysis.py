"""Paired document-bootstrap analysis and protected-slice claim checks (Plan Sections 6.1, 8.6, 15).

Every score here is synthetic. Nothing evaluates a model, opens ``validation_final``, runs an
arm, or selects a release, and no fixture outcome is used to choose a hypothesis, a branch, or
any public wording. What the tests prove is that the analysis cannot be talked into a claim:

- the bootstrap is paired and deterministic, so the same scores and seed always give the same
  interval, and two arms scored on different documents fail closed,
- a claim needs every preregistered condition -- CI entirely below zero, at least 0.3%
  relative reduction, no protected slice past 1%, and a same-direction earlier confirmation --
  and each one is independently falsifiable,
- all five preregistered interpretations are reachable: quality-and-timing, quality-only,
  harmful, null, and incomplete,
- a missing confirmation yields INCOMPLETE rather than a quietly-supported claim,
- ``claim_allowed`` is true for exactly one interpretation, and a harmful primary is reported
  as harmful even when the secondary looks good,
- an undecayed peak-LR mainline checkpoint can never be the release fallback.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from tinybench_lm.analysis import (
    ANALYSIS_CONFIRMATION_MISSING,
    ANALYSIS_DOCUMENTS_NOT_PAIRED,
    ANALYSIS_FAIL_CLOSED_REASON_CODES,
    ANALYSIS_INSUFFICIENT_DOCUMENTS,
    ANALYSIS_PROTOCOL_PATH,
    ANALYSIS_UNDECAYED_FALLBACK,
    ANNEALING_HARMFUL,
    BLOCKED,
    FROZEN_ANALYSIS_PROTOCOL_SHA256,
    HYPOTHESIS_PRIMARY,
    HYPOTHESIS_SECONDARY,
    INCOMPLETE,
    INTERPRETATION_IDS,
    NULL_RESULT,
    QUALITY_AND_TIMING,
    QUALITY_ONLY,
    AnalysisContractError,
    AnalysisNotReadyError,
    ConfirmationResult,
    PairedScores,
    UnpairedScoresError,
    assert_documents_paired,
    assert_ready_for_claim,
    default_seed,
    evaluate_hypothesis,
    fallback_violations,
    format_interpretation,
    interpret,
    load_analysis_protocol,
    paired_bootstrap,
    protected_slice_regressions,
    readiness_results,
    relative_nll_change,
    threshold_drift_violations,
)
from tinybench_lm.campaign import decision_thresholds
from tinybench_lm.data_protocols import ProtocolMutatedError, protocol_digest
from tinybench_lm.shards import EXPECTED_PROTECTED_SLICES, FAIL, NOT_RUN, PASS

DOCUMENTS = 240


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_analysis_protocol()


def _slices(count: int) -> tuple[str, ...]:
    return tuple(EXPECTED_PROTECTED_SLICES[index % len(EXPECTED_PROTECTED_SLICES)] for index in range(count))


def _scores(
    *,
    shift: float,
    noise: float = 0.02,
    count: int = DOCUMENTS,
    seed: int = 7,
    slice_shift: dict[str, float] | None = None,
) -> PairedScores:
    """Paired scores where the candidate is ``shift`` better (negative shift = better)."""
    rng = np.random.default_rng(seed)
    control = rng.uniform(3.5, 4.5, count)
    tags = _slices(count)
    candidate = control + shift + rng.normal(0.0, noise, count)
    if slice_shift:
        for index, tag in enumerate(tags):
            candidate[index] += slice_shift.get(tag, 0.0)
    return PairedScores(
        document_ids=tuple(f"doc_{index:04d}" for index in range(count)),
        candidate_nll=tuple(float(value) for value in candidate),
        control_nll=tuple(float(value) for value in control),
        slices=tags,
    )


# --------------------------------------------------------------------------------------
# The frozen contract
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_frozen_analysis_protocol_is_pinned_and_immutable(tmp_path: Path) -> None:
    assert (
        protocol_digest(ANALYSIS_PROTOCOL_PATH)
        == FROZEN_ANALYSIS_PROTOCOL_SHA256["paired_analysis_v1.yaml"]
    )
    text = ANALYSIS_PROTOCOL_PATH.read_text(encoding="utf-8")
    mutated = tmp_path / "paired_analysis_v1.yaml"
    mutated.write_text(text.replace("owner: operator", "owner: nobody"), encoding="utf-8")
    with pytest.raises(ProtocolMutatedError):
        load_analysis_protocol(mutated)
    assert str(load_analysis_protocol(mutated, verify=False)["readiness"]["owner"]) == "nobody"


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
@pytest.mark.parametrize(
    "old,new",
    [
        ("    hypothesis_id: H2", "    hypothesis_id: H1"),
        ("  emphasis_fixed_before_results: true", "  emphasis_fixed_before_results: false"),
        ("  pairing: per_document_difference", "  pairing: independent_resample"),
        ("  resamples: 10000", "  resamples: 100"),
        ("  confidence_level: 0.95", "  confidence_level: 0.8"),
        ("  deterministic_seed_required: true", "  deterministic_seed_required: false"),
        ("  lower_percentile: 2.5", "  lower_percentile: 5.0"),
        ("  minimum_documents: 2", "  minimum_documents: 1"),
        ("  null_and_incomplete_are_valid_outcomes: true", "  null_and_incomplete_are_valid_outcomes: false"),
        ("  never_use_undecayed_peak_lr_mainline: true", "  never_use_undecayed_peak_lr_mainline: false"),
    ],
)
def test_weakening_the_analysis_contract_fails_closed(tmp_path: Path, old: str, new: str) -> None:
    text = ANALYSIS_PROTOCOL_PATH.read_text(encoding="utf-8")
    assert old in text, f"replacement would be a no-op: {old!r}"
    weakened = tmp_path / "weakened_v1.yaml"
    weakened.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(AnalysisContractError):
        load_analysis_protocol(weakened, verify=False)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
def test_thresholds_reconcile_with_the_campaign_preregistration(protocol: dict) -> None:
    """Two frozen files, one number each. Drift between them would be invisible otherwise."""
    assert threshold_drift_violations(protocol) == ()
    campaign = decision_thresholds()
    thresholds = protocol["thresholds"]
    assert float(thresholds["confidence_level"]) == campaign.confidence_level == 0.95
    assert float(thresholds["minimum_relative_reduction"]) == campaign.minimum_relative_reduction == 0.003
    assert float(thresholds["maximum_slice_regression"]) == campaign.maximum_slice_regression == 0.01
    assert tuple(thresholds["protected_slices"]) == campaign.protected_slices == EXPECTED_PROTECTED_SLICES
    assert set(protocol["reason_codes"]["fail_closed"]) == set(ANALYSIS_FAIL_CLOSED_REASON_CODES)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
@pytest.mark.parametrize(
    "field,drifted",
    [
        ("confidence_level", 0.9),
        ("minimum_relative_reduction", 0.001),
        ("maximum_slice_regression", 0.05),
    ],
)
def test_threshold_drift_between_the_two_contracts_is_reported(
    protocol: dict, field: str, drifted: float
) -> None:
    """The contracts agree today, so drift them deliberately to prove the check is not vacuous."""
    diverged = {**protocol, "thresholds": {**protocol["thresholds"], field: drifted}}
    problems = threshold_drift_violations(diverged)
    assert problems, f"a drifted {field} must be reported"
    assert any(field in problem for problem in problems)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_protected_slice_drift_between_the_two_contracts_is_reported(protocol: dict) -> None:
    diverged = {
        **protocol,
        "thresholds": {**protocol["thresholds"], "protected_slices": ["broad_general"]},
    }
    problems = threshold_drift_violations(diverged)
    assert any("protected slices disagree" in problem for problem in problems)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_endpoints_are_fixed_before_any_result(protocol: dict) -> None:
    endpoints = protocol["endpoints"]
    assert endpoints["primary"]["hypothesis_id"] == HYPOTHESIS_PRIMARY
    assert endpoints["primary"]["delta"] == "NLL_C - NLL_B"
    assert endpoints["primary"]["split"] == "validation_final"
    assert endpoints["secondary"]["hypothesis_id"] == HYPOTHESIS_SECONDARY
    assert endpoints["secondary"]["delta"] == "NLL_B - NLL_A"
    assert tuple(
        item["interpretation_id"] for item in protocol["interpretations"]["definitions"]
    ) == INTERPRETATION_IDS


# --------------------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_unpaired_or_malformed_scores_fail_closed() -> None:
    with pytest.raises(UnpairedScoresError, match=ANALYSIS_DOCUMENTS_NOT_PAIRED):
        PairedScores(("a", "b"), (1.0, 2.0), (1.0,))
    with pytest.raises(UnpairedScoresError, match=ANALYSIS_DOCUMENTS_NOT_PAIRED):
        PairedScores(("a", "b"), (1.0, 2.0), (1.0, 2.0), ("broad_general",))
    # Repeated ids make the pairing ambiguous rather than merely redundant.
    with pytest.raises(UnpairedScoresError, match=ANALYSIS_DOCUMENTS_NOT_PAIRED):
        PairedScores(("a", "a"), (1.0, 2.0), (1.0, 2.0))
    for bad in (float("nan"), float("inf")):
        with pytest.raises(AnalysisContractError):
            PairedScores(("a", "b"), (1.0, bad), (1.0, 2.0))

    assert_documents_paired(("a", "b"), ("a", "b"))
    with pytest.raises(UnpairedScoresError, match=ANALYSIS_DOCUMENTS_NOT_PAIRED):
        assert_documents_paired(("a", "b"), ("b", "a"))


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_a_bootstrap_needs_enough_documents(protocol: dict) -> None:
    single = PairedScores(("a",), (1.0,), (2.0,))
    with pytest.raises(AnalysisContractError, match=ANALYSIS_INSUFFICIENT_DOCUMENTS):
        paired_bootstrap(single, protocol=protocol)


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_the_bootstrap_is_deterministic_for_a_given_seed(protocol: dict) -> None:
    """A confidence interval that moves between runs is not evidence."""
    scores = _scores(shift=-0.05)
    first = paired_bootstrap(scores, seed=12345, protocol=protocol)
    second = paired_bootstrap(scores, seed=12345, protocol=protocol)
    assert first == second
    assert (first.low, first.high) == (second.low, second.high)

    # The default seed is frozen, so an unseeded call is reproducible too.
    assert paired_bootstrap(scores, protocol=protocol) == paired_bootstrap(scores, protocol=protocol)
    assert paired_bootstrap(scores, protocol=protocol).seed == default_seed(protocol)

    # A different seed gives a different resample draw but the same point estimate.
    other = paired_bootstrap(scores, seed=999, protocol=protocol)
    assert other.delta == pytest.approx(first.delta)
    assert other.resamples == first.resamples


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_the_point_estimate_is_the_paired_mean_difference(protocol: dict) -> None:
    scores = _scores(shift=-0.05)
    expected = scores.mean_candidate - scores.mean_control
    assert scores.delta == pytest.approx(expected)
    assert float(np.mean(scores.differences)) == pytest.approx(expected)
    interval = paired_bootstrap(scores, protocol=protocol)
    assert interval.delta == pytest.approx(expected)
    # The interval brackets the point estimate.
    assert interval.low <= interval.delta <= interval.high


# --------------------------------------------------------------------------------------
# The three outcome shapes
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_a_clear_improvement_puts_the_interval_below_zero(protocol: dict) -> None:
    interval = paired_bootstrap(_scores(shift=-0.05), protocol=protocol)
    assert interval.entirely_below_zero
    assert not interval.entirely_above_zero
    assert not interval.spans_zero
    assert interval.high < 0


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_a_null_result_spans_zero(protocol: dict) -> None:
    interval = paired_bootstrap(_scores(shift=0.0), protocol=protocol)
    assert interval.spans_zero
    assert not interval.entirely_below_zero
    assert not interval.entirely_above_zero


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_a_harmful_result_puts_the_interval_above_zero(protocol: dict) -> None:
    interval = paired_bootstrap(_scores(shift=+0.05), protocol=protocol)
    assert interval.entirely_above_zero
    assert not interval.entirely_below_zero


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_relative_change_sign_convention() -> None:
    """Positive means the candidate improved, because these are losses."""
    better = PairedScores(("a", "b"), (3.9, 3.9), (4.0, 4.0))
    assert relative_nll_change(better) == pytest.approx(0.025)
    worse = PairedScores(("a", "b"), (4.1, 4.1), (4.0, 4.0))
    assert relative_nll_change(worse) == pytest.approx(-0.025)
    identical = PairedScores(("a", "b"), (4.0, 4.0), (4.0, 4.0))
    assert relative_nll_change(identical) == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# Protected slices (Plan Section 4.4)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_protected_slice_regression_is_measured_per_slice(protocol: dict) -> None:
    clean = _scores(shift=-0.05)
    regressions = protected_slice_regressions(clean, protocol)
    assert {item.slice_id for item in regressions} == set(EXPECTED_PROTECTED_SLICES)
    assert not any(item.breaches_limit for item in regressions)
    assert sum(item.documents for item in regressions) == DOCUMENTS

    # One slice pushed past the 1% limit while the overall delta still improves.
    regressed = _scores(shift=-0.05, slice_shift={"math_technical": 0.15})
    results = {item.slice_id: item for item in protected_slice_regressions(regressed, protocol)}
    assert results["math_technical"].breaches_limit
    assert results["math_technical"].relative_regression > 0.01
    assert not results["broad_general"].breaches_limit


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_the_slice_limit_is_a_strict_greater_than(protocol: dict) -> None:
    """"more than 1%" means a slice sitting exactly at 1% has not breached."""
    tags = ("broad_general",) * 4
    # 100/101 keeps the ratio exactly representable, so the boundary is a real boundary and
    # not a rounding artefact.
    control = (100.0, 100.0, 100.0, 100.0)
    at_limit = (101.0, 101.0, 101.0, 101.0)  # exactly +1%
    exact = PairedScores(("a", "b", "c", "d"), at_limit, control, tags)
    result = protected_slice_regressions(exact, protocol)[0]
    assert result.relative_regression == 0.01
    assert not result.breaches_limit

    over = PairedScores(("a", "b", "c", "d"), (102.0,) * 4, control, tags)
    assert protected_slice_regressions(over, protocol)[0].breaches_limit

    # Where the ratio is not exactly representable the comparison errs toward protecting the
    # slice, which is the safe direction for a regression guard.
    nearly = PairedScores(("a", "b", "c", "d"), (4.04,) * 4, (4.0,) * 4, tags)
    assert protected_slice_regressions(nearly, protocol)[0].relative_regression > 0.01


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_untagged_scores_report_no_slices(protocol: dict) -> None:
    untagged = PairedScores(("a", "b"), (3.9, 3.9), (4.0, 4.0))
    assert protected_slice_regressions(untagged, protocol) == ()


# --------------------------------------------------------------------------------------
# Conditions are individually falsifiable
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_every_annealing_condition_can_fail_on_its_own(protocol: dict) -> None:
    confirmation = ConfirmationResult(delta=-0.02)

    supported = evaluate_hypothesis(
        _scores(shift=-0.05), hypothesis_id=HYPOTHESIS_PRIMARY, confirmation=confirmation, protocol=protocol
    )
    assert supported.supported
    assert set(supported.conditions) == {
        "ci_below_zero",
        "relative_reduction_met",
        "no_protected_slice_regression",
        "confirmation_same_direction",
    }
    assert supported.unmet_conditions == ()

    # 1. Interval spans zero.
    null = evaluate_hypothesis(
        _scores(shift=0.0), hypothesis_id=HYPOTHESIS_PRIMARY, confirmation=confirmation, protocol=protocol
    )
    assert not null.conditions["ci_below_zero"]
    assert not null.supported

    # 2. A real but too-small improvement clears the CI bar and misses the 0.3% bar.
    tiny = evaluate_hypothesis(
        _scores(shift=-0.002, noise=0.001),
        hypothesis_id=HYPOTHESIS_PRIMARY,
        confirmation=confirmation,
        protocol=protocol,
    )
    assert tiny.conditions["ci_below_zero"]
    assert not tiny.conditions["relative_reduction_met"]
    assert tiny.relative_change < 0.003
    assert not tiny.supported

    # 3. A protected slice regresses even though the overall delta improves.
    regressed = evaluate_hypothesis(
        _scores(shift=-0.05, slice_shift={"math_technical": 0.15}),
        hypothesis_id=HYPOTHESIS_PRIMARY,
        confirmation=confirmation,
        protocol=protocol,
    )
    assert not regressed.conditions["no_protected_slice_regression"]
    assert not regressed.supported

    # 4. The earlier confirmation is absent, or points the other way.
    unconfirmed = evaluate_hypothesis(
        _scores(shift=-0.05), hypothesis_id=HYPOTHESIS_PRIMARY, confirmation=None, protocol=protocol
    )
    assert not unconfirmed.conditions["confirmation_same_direction"]
    assert not unconfirmed.supported

    opposite = evaluate_hypothesis(
        _scores(shift=-0.05),
        hypothesis_id=HYPOTHESIS_PRIMARY,
        confirmation=ConfirmationResult(delta=+0.02),
        protocol=protocol,
    )
    assert not opposite.conditions["confirmation_same_direction"]
    assert not opposite.supported


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_the_secondary_hypothesis_has_no_confirmation_condition(protocol: dict) -> None:
    """Plan 8.6 attaches the earlier confirmation to the annealing claim, not to H1."""
    result = evaluate_hypothesis(
        _scores(shift=-0.05), hypothesis_id=HYPOTHESIS_SECONDARY, protocol=protocol
    )
    assert "confirmation_same_direction" not in result.conditions
    assert result.supported
    assert (result.candidate_arm, result.control_arm) == ("B", "A")

    with pytest.raises(AnalysisContractError):
        evaluate_hypothesis(_scores(shift=-0.05), hypothesis_id="H9", protocol=protocol)


# --------------------------------------------------------------------------------------
# Interpretation (Plan Section 8.6)
# --------------------------------------------------------------------------------------


def _result(shift: float, hypothesis_id: str, protocol: dict, confirmation=None, **kwargs):
    return evaluate_hypothesis(
        _scores(shift=shift, **kwargs),
        hypothesis_id=hypothesis_id,
        confirmation=confirmation,
        protocol=protocol,
    )


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_all_five_interpretations_are_reachable(protocol: dict) -> None:
    confirmation = ConfirmationResult(delta=-0.02)
    good_h2 = _result(-0.05, HYPOTHESIS_PRIMARY, protocol, confirmation)
    good_h1 = _result(-0.05, HYPOTHESIS_SECONDARY, protocol)
    null_h2 = _result(0.0, HYPOTHESIS_PRIMARY, protocol, confirmation)
    null_h1 = _result(0.0, HYPOTHESIS_SECONDARY, protocol)
    bad_h2 = _result(+0.05, HYPOTHESIS_PRIMARY, protocol, confirmation)

    both = interpret(good_h2, good_h1, confirmation=confirmation, protocol=protocol)
    assert both.interpretation_id == QUALITY_AND_TIMING
    assert both.claim_allowed

    quality_only = interpret(null_h2, good_h1, confirmation=confirmation, protocol=protocol)
    assert quality_only.interpretation_id == QUALITY_ONLY
    assert not quality_only.claim_allowed

    harmful = interpret(bad_h2, good_h1, confirmation=confirmation, protocol=protocol)
    assert harmful.interpretation_id == ANNEALING_HARMFUL
    assert not harmful.claim_allowed

    null = interpret(null_h2, null_h1, confirmation=confirmation, protocol=protocol)
    assert null.interpretation_id == NULL_RESULT
    assert not null.claim_allowed

    incomplete = interpret(good_h2, good_h1, confirmation=None, protocol=protocol)
    assert incomplete.interpretation_id == INCOMPLETE
    assert not incomplete.claim_allowed
    assert any(ANALYSIS_CONFIRMATION_MISSING in reason for reason in incomplete.reasons)

    missing_endpoint = interpret(None, good_h1, confirmation=confirmation, protocol=protocol)
    assert missing_endpoint.interpretation_id == INCOMPLETE

    # Every reachable code is one the frozen contract declared in advance.
    for result in (both, quality_only, harmful, null, incomplete, missing_endpoint):
        assert result.interpretation_id in INTERPRETATION_IDS
        assert format_interpretation(result)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_a_harmful_primary_is_reported_even_when_the_secondary_looks_good(protocol: dict) -> None:
    """"C loses to B" is a finding about annealing that a favourable H1 does not soften."""
    confirmation = ConfirmationResult(delta=-0.02)
    harmful = interpret(
        _result(+0.05, HYPOTHESIS_PRIMARY, protocol, confirmation),
        _result(-0.05, HYPOTHESIS_SECONDARY, protocol),
        confirmation=confirmation,
        protocol=protocol,
    )
    assert harmful.interpretation_id == ANNEALING_HARMFUL
    assert harmful.secondary is not None and harmful.secondary.supported
    assert not harmful.claim_allowed


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_a_slice_regression_alone_blocks_the_claim(protocol: dict) -> None:
    confirmation = ConfirmationResult(delta=-0.02)
    regressed = _result(
        -0.08, HYPOTHESIS_PRIMARY, protocol, confirmation, slice_shift={"math_technical": 0.15}
    )
    assert regressed.interval.entirely_below_zero
    assert regressed.relative_change > 0.003
    # Everything else passes; the slice alone is disqualifying.
    assert regressed.unmet_conditions == ("no_protected_slice_regression",)
    outcome = interpret(
        regressed, _result(0.0, HYPOTHESIS_SECONDARY, protocol), confirmation=confirmation, protocol=protocol
    )
    assert outcome.interpretation_id == NULL_RESULT
    assert not outcome.claim_allowed


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
@given(
    h2_shift=st.sampled_from([-0.05, 0.0, 0.05]),
    h1_shift=st.sampled_from([-0.05, 0.0, 0.05]),
    has_confirmation=st.booleans(),
)
@settings(max_examples=18, deadline=None, derandomize=True)
def test_a_claim_is_allowed_only_when_both_endpoints_are_supported(
    h2_shift: float, h1_shift: float, has_confirmation: bool
) -> None:
    protocol = load_analysis_protocol()
    confirmation = ConfirmationResult(delta=-0.02) if has_confirmation else None
    h2 = _result(h2_shift, HYPOTHESIS_PRIMARY, protocol, confirmation)
    h1 = _result(h1_shift, HYPOTHESIS_SECONDARY, protocol)
    outcome = interpret(h2, h1, confirmation=confirmation, protocol=protocol)

    assert outcome.interpretation_id in INTERPRETATION_IDS
    # The one invariant that matters: a claim requires both endpoints supported.
    if outcome.claim_allowed:
        assert h2.supported and h1.supported
        assert outcome.interpretation_id == QUALITY_AND_TIMING
    if not has_confirmation:
        assert outcome.interpretation_id == INCOMPLETE
        assert not outcome.claim_allowed


# --------------------------------------------------------------------------------------
# Fallback and readiness (Plan Section 15)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_an_undecayed_peak_lr_mainline_is_never_the_fallback(protocol: dict) -> None:
    assert fallback_violations(is_undecayed_peak_lr_mainline=False, protocol=protocol) == ()
    problems = fallback_violations(is_undecayed_peak_lr_mainline=True, protocol=protocol)
    assert any(ANALYSIS_UNDECAYED_FALLBACK in problem for problem in problems)
    # Plan 15: prefer arm A when no experimental effect is supported.
    assert str(protocol["fallback"]["prefer_when_no_effect_supported"]) == "A"


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_analysis_readiness_is_not_run_with_an_owner(protocol: dict) -> None:
    results = readiness_results(protocol)
    assert results
    statuses = {result.status for result in results}
    assert PASS not in statuses, "no arm has been trained and validation_final is unopened"
    assert statuses <= {NOT_RUN, BLOCKED}
    for result in results:
        assert "blocker=" in result.reason
        assert "owner=" in result.reason
        assert "next_action=" in result.reason


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_no_claim_can_be_made_before_the_branches_run(protocol: dict) -> None:
    with pytest.raises(AnalysisNotReadyError) as excinfo:
        assert_ready_for_claim(protocol)
    message = str(excinfo.value)
    assert "validation_final_opened" in message
    assert "next_action=" in message
