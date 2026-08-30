"""Paired document-bootstrap analysis and protected-slice claim checks (Plan Sections 6.1, 8.6, 15).

Plan Section 8.6 fixes the primary endpoint as ``delta_H2 = NLL_C - NLL_B`` and the secondary
as ``delta_H1 = NLL_B - NLL_A``, then adds the sentence this whole module is built around:
"do not choose which hypothesis to emphasize after seeing results."

So this module deliberately cannot write a conclusion. :func:`interpret` returns one of five
*preregistered interpretation codes* -- every one of which exists in the frozen config before
any delta exists -- together with the evidence for each condition. It never produces prose,
never ranks hypotheses, and never selects a branch or a release. Null, harmful, and incomplete
are first-class outcomes with the same standing as a supported claim, because Plan Section 15
requires that a non-replicating innovation "be reported as null/inconclusive" rather than
quietly dropped.

The statistics are equally constrained:

*Paired.* The same documents are scored under both arms, and the bootstrap resamples the
per-document *difference*. Resampling the two arms independently would throw away the pairing
and widen the interval, which is the kind of error that turns a null into a claim.

*Deterministic.* One seeded :func:`numpy.random.default_rng` stream drives every resample, so
the same scores and seed always yield the same interval. A confidence interval that moves
between runs is not evidence.

*Thresholded in advance.* The bars come from ``configs/campaign/preregistration_v1.yaml`` --
the same frozen numbers task 3.14 pinned -- and :func:`threshold_drift_violations` fails
closed if this contract and that one ever disagree.

The contract is backed by one frozen config::

    configs/analysis/paired_analysis_v1.yaml

Nothing here scores a model, opens ``validation_final``, runs an arm, or selects a release.
Tests drive it entirely with synthetic paired documents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .campaign import decision_thresholds
from .data_protocols import (
    REPOSITORY_ROOT,
    ProtocolError,
    ProtocolNotReadyError,
    load_protocol,
)
from .environment import CheckResult
from .shards import EXPECTED_PROTECTED_SLICES, FAIL, NOT_RUN, PASS

ANALYSIS_PROTOCOL_DIR = REPOSITORY_ROOT / "configs" / "analysis"
ANALYSIS_PROTOCOL_PATH = ANALYSIS_PROTOCOL_DIR / "paired_analysis_v1.yaml"

#: SHA-256 of the frozen analysis contract, over file bytes with CRLF normalized to LF.
FROZEN_ANALYSIS_PROTOCOL_SHA256: Mapping[str, str] = {
    "paired_analysis_v1.yaml": "6de7a2e0bb66c29427cff13d469af5a8624ff88a09c5926acffd835bfe6a88a2",
}

BLOCKED = "BLOCKED"

HYPOTHESIS_PRIMARY = "H2"
HYPOTHESIS_SECONDARY = "H1"

ARM_A = "A"
ARM_B = "B"
ARM_C = "C"

#: The five preregistered interpretations. Every outcome has a name before any outcome exists.
QUALITY_AND_TIMING = "QUALITY_AND_TIMING"
QUALITY_ONLY = "QUALITY_ONLY"
ANNEALING_HARMFUL = "ANNEALING_HARMFUL"
NULL_RESULT = "NULL_RESULT"
INCOMPLETE = "INCOMPLETE"

INTERPRETATION_IDS: tuple[str, ...] = (
    QUALITY_AND_TIMING,
    QUALITY_ONLY,
    ANNEALING_HARMFUL,
    NULL_RESULT,
    INCOMPLETE,
)

# --------------------------------------------------------------------------------------
# Reason-code vocabulary
# --------------------------------------------------------------------------------------

ANALYSIS_OK = "ANALYSIS_OK"
ANALYSIS_CLAIM_NOT_SUPPORTED = "ANALYSIS_CLAIM_NOT_SUPPORTED"
ANALYSIS_CONFIRMATION_MISSING = "ANALYSIS_CONFIRMATION_MISSING"
ANALYSIS_DOCUMENTS_NOT_PAIRED = "ANALYSIS_DOCUMENTS_NOT_PAIRED"
ANALYSIS_ENDPOINT_REORDERED = "ANALYSIS_ENDPOINT_REORDERED"
ANALYSIS_INSUFFICIENT_DOCUMENTS = "ANALYSIS_INSUFFICIENT_DOCUMENTS"
ANALYSIS_NONDETERMINISTIC = "ANALYSIS_NONDETERMINISTIC"
ANALYSIS_POST_HOC_SELECTION = "ANALYSIS_POST_HOC_SELECTION"
ANALYSIS_THRESHOLD_DRIFT = "ANALYSIS_THRESHOLD_DRIFT"
ANALYSIS_UNDECAYED_FALLBACK = "ANALYSIS_UNDECAYED_FALLBACK"

ANALYSIS_FAIL_CLOSED_REASON_CODES: frozenset[str] = frozenset(
    {
        ANALYSIS_CLAIM_NOT_SUPPORTED,
        ANALYSIS_CONFIRMATION_MISSING,
        ANALYSIS_DOCUMENTS_NOT_PAIRED,
        ANALYSIS_ENDPOINT_REORDERED,
        ANALYSIS_INSUFFICIENT_DOCUMENTS,
        ANALYSIS_NONDETERMINISTIC,
        ANALYSIS_POST_HOC_SELECTION,
        ANALYSIS_THRESHOLD_DRIFT,
        ANALYSIS_UNDECAYED_FALLBACK,
    }
)


class AnalysisContractError(ProtocolError):
    """The frozen analysis contract is malformed, or supplied scores violate it."""


class UnpairedScoresError(AnalysisContractError):
    """Two arms were scored on different documents, so no paired comparison exists."""


class AnalysisNotReadyError(ProtocolNotReadyError):
    """A claim needs branch outcomes and an opened validation_final that do not exist yet."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def load_analysis_protocol(
    path: Path = ANALYSIS_PROTOCOL_PATH, *, verify: bool = True
) -> dict[str, Any]:
    """Load the frozen analysis contract, verifying its pinned digest by default."""
    protocol = load_protocol(path, verify=verify, registry=FROZEN_ANALYSIS_PROTOCOL_SHA256)
    required = (
        "endpoints",
        "bootstrap",
        "annealing_claim",
        "quality_claim",
        "thresholds",
        "interpretations",
        "fallback",
        "readiness",
    )
    for section in required:
        if section not in protocol:
            raise AnalysisContractError(f"analysis protocol is missing required section {section!r}")

    endpoints = protocol["endpoints"]
    primary = endpoints["primary"]
    secondary = endpoints["secondary"]
    if str(primary["hypothesis_id"]) != HYPOTHESIS_PRIMARY:
        raise AnalysisContractError(
            f"{ANALYSIS_ENDPOINT_REORDERED}: Plan Section 8.6 makes {HYPOTHESIS_PRIMARY} primary"
        )
    if str(secondary["hypothesis_id"]) != HYPOTHESIS_SECONDARY:
        raise AnalysisContractError(
            f"{ANALYSIS_ENDPOINT_REORDERED}: Plan Section 8.6 makes {HYPOTHESIS_SECONDARY} secondary"
        )
    if (str(primary["candidate_arm"]), str(primary["control_arm"])) != (ARM_C, ARM_B):
        raise AnalysisContractError("delta_H2 is NLL_C - NLL_B")
    if (str(secondary["candidate_arm"]), str(secondary["control_arm"])) != (ARM_B, ARM_A):
        raise AnalysisContractError("delta_H1 is NLL_B - NLL_A")
    if not bool(endpoints["emphasis_fixed_before_results"]):
        raise AnalysisContractError(
            f"{ANALYSIS_POST_HOC_SELECTION}: which hypothesis is primary is fixed in advance"
        )

    bootstrap = protocol["bootstrap"]
    if str(bootstrap["pairing"]) != "per_document_difference":
        raise AnalysisContractError(
            f"{ANALYSIS_DOCUMENTS_NOT_PAIRED}: the bootstrap must resample paired differences"
        )
    if int(bootstrap["resamples"]) < 1000:
        raise AnalysisContractError("a 95% percentile interval needs at least 1000 resamples")
    if float(bootstrap["confidence_level"]) != 0.95:
        raise AnalysisContractError("Plan Sections 6.1 and 8.6 freeze a 95% confidence level")
    if not bool(bootstrap["deterministic_seed_required"]):
        raise AnalysisContractError(
            f"{ANALYSIS_NONDETERMINISTIC}: an interval that moves between runs is not evidence"
        )
    lower = float(bootstrap["lower_percentile"])
    upper = float(bootstrap["upper_percentile"])
    if (lower, upper) != (2.5, 97.5):
        raise AnalysisContractError("a 95% percentile interval spans 2.5 to 97.5")
    if int(bootstrap["minimum_documents"]) < 2:
        raise AnalysisContractError(
            f"{ANALYSIS_INSUFFICIENT_DOCUMENTS}: a bootstrap needs at least 2 documents"
        )

    for section_name in ("annealing_claim", "quality_claim"):
        claim = protocol[section_name]
        if not bool(claim["all_conditions_required"]):
            raise AnalysisContractError(f"{section_name} requires every condition, none optional")
    annealing_conditions = tuple(
        str(condition["condition_id"]) for condition in protocol["annealing_claim"]["required_conditions"]
    )
    if annealing_conditions != (
        "ci_below_zero",
        "relative_reduction_met",
        "no_protected_slice_regression",
        "confirmation_same_direction",
    ):
        raise AnalysisContractError(
            "Plan Section 8.6 lists four annealing conditions, including the earlier confirmation"
        )

    declared = tuple(
        str(item["interpretation_id"]) for item in protocol["interpretations"]["definitions"]
    )
    if declared != INTERPRETATION_IDS:
        raise AnalysisContractError(
            f"analysis protocol must declare interpretations {INTERPRETATION_IDS}, found {declared}"
        )
    if not bool(protocol["interpretations"]["null_and_incomplete_are_valid_outcomes"]):
        raise AnalysisContractError(
            "Plan Section 15 keeps null and incomplete as valid reportable outcomes"
        )

    thresholds = protocol["thresholds"]
    if tuple(str(name) for name in thresholds["protected_slices"]) != EXPECTED_PROTECTED_SLICES:
        raise AnalysisContractError(
            f"analysis protocol declares protected slices "
            f"{tuple(thresholds['protected_slices'])}, expected {EXPECTED_PROTECTED_SLICES}"
        )
    if not bool(protocol["fallback"]["never_use_undecayed_peak_lr_mainline"]):
        raise AnalysisContractError(
            f"{ANALYSIS_UNDECAYED_FALLBACK}: Plan Section 15 forbids an undecayed peak-LR fallback"
        )
    return protocol


def _resolved(protocol: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return protocol if protocol is not None else load_analysis_protocol()


def threshold_drift_violations(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """The bars here must equal the bars task 3.14 froze. Two sources, one number each."""
    thresholds = _resolved(protocol)["thresholds"]
    campaign = decision_thresholds()
    problems: list[str] = []
    checks = (
        ("confidence_level", float(thresholds["confidence_level"]), campaign.confidence_level),
        (
            "minimum_relative_reduction",
            float(thresholds["minimum_relative_reduction"]),
            campaign.minimum_relative_reduction,
        ),
        (
            "maximum_slice_regression",
            float(thresholds["maximum_slice_regression"]),
            campaign.maximum_slice_regression,
        ),
    )
    for name, here, there in checks:
        if here != there:
            problems.append(
                f"{ANALYSIS_THRESHOLD_DRIFT}: {name} is {here} here and {there} in the "
                "campaign preregistration"
            )
    if tuple(str(name) for name in thresholds["protected_slices"]) != campaign.protected_slices:
        problems.append(f"{ANALYSIS_THRESHOLD_DRIFT}: protected slices disagree")
    return tuple(problems)


# --------------------------------------------------------------------------------------
# Paired scores
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedScores:
    """Per-document NLL for one candidate and one control, over the same documents."""

    document_ids: tuple[str, ...]
    candidate_nll: tuple[float, ...]
    control_nll: tuple[float, ...]
    slices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        count = len(self.document_ids)
        if len(self.candidate_nll) != count or len(self.control_nll) != count:
            raise UnpairedScoresError(
                f"{ANALYSIS_DOCUMENTS_NOT_PAIRED}: {count} documents but "
                f"{len(self.candidate_nll)} candidate and {len(self.control_nll)} control scores"
            )
        if self.slices and len(self.slices) != count:
            raise UnpairedScoresError(
                f"{ANALYSIS_DOCUMENTS_NOT_PAIRED}: {len(self.slices)} slice tags for {count} documents"
            )
        if len(set(self.document_ids)) != count:
            raise UnpairedScoresError(
                f"{ANALYSIS_DOCUMENTS_NOT_PAIRED}: document ids repeat, so the pairing is ambiguous"
            )
        for label, values in (("candidate", self.candidate_nll), ("control", self.control_nll)):
            for index, value in enumerate(values):
                if not math.isfinite(float(value)):
                    raise AnalysisContractError(
                        f"{label} NLL for document {self.document_ids[index]!r} is {value!r}, "
                        "which is not a finite score"
                    )

    @property
    def count(self) -> int:
        return len(self.document_ids)

    @property
    def differences(self) -> np.ndarray:
        """``candidate - control`` per document. This is what the bootstrap resamples."""
        return np.asarray(self.candidate_nll, dtype=float) - np.asarray(
            self.control_nll, dtype=float
        )

    @property
    def mean_candidate(self) -> float:
        return float(np.mean(self.candidate_nll))

    @property
    def mean_control(self) -> float:
        return float(np.mean(self.control_nll))

    @property
    def delta(self) -> float:
        """The point estimate. Negative favours the candidate, because these are losses."""
        return self.mean_candidate - self.mean_control


def assert_documents_paired(candidate_ids: Sequence[str], control_ids: Sequence[str]) -> None:
    """Two arms must be scored on the same documents in the same order."""
    if tuple(candidate_ids) != tuple(control_ids):
        raise UnpairedScoresError(
            f"{ANALYSIS_DOCUMENTS_NOT_PAIRED}: the two arms were scored on different documents, "
            "so no paired comparison exists"
        )


# --------------------------------------------------------------------------------------
# Deterministic paired document bootstrap
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile confidence interval and the point estimate it surrounds."""

    delta: float
    low: float
    high: float
    confidence_level: float
    resamples: int
    seed: int

    @property
    def entirely_below_zero(self) -> bool:
        """The Plan Section 8.6 condition: the whole interval favours the candidate."""
        return self.high < 0.0

    @property
    def entirely_above_zero(self) -> bool:
        """The candidate is reliably worse -- a harmful result, not merely an unsupported one."""
        return self.low > 0.0

    @property
    def spans_zero(self) -> bool:
        return not (self.entirely_below_zero or self.entirely_above_zero)

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta": self.delta,
            "ci_low": self.low,
            "ci_high": self.high,
            "confidence_level": self.confidence_level,
            "resamples": self.resamples,
            "seed": self.seed,
        }


def default_seed(protocol: Mapping[str, Any] | None = None) -> int:
    return int(_resolved(protocol)["bootstrap"]["default_seed"])


def paired_bootstrap(
    scores: PairedScores,
    *,
    seed: int | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> BootstrapInterval:
    """Resample the paired per-document differences with one seeded generator.

    Drawing indices sequentially from a single :func:`numpy.random.default_rng` stream keeps
    the result reproducible without materializing a ``resamples x documents`` index matrix,
    which for a real validation split would be gigabytes.
    """
    resolved = _resolved(protocol)
    bootstrap = resolved["bootstrap"]
    minimum = int(bootstrap["minimum_documents"])
    if scores.count < minimum:
        raise AnalysisContractError(
            f"{ANALYSIS_INSUFFICIENT_DOCUMENTS}: {scores.count} documents, {minimum} required"
        )

    resamples = int(bootstrap["resamples"])
    chosen_seed = default_seed(resolved) if seed is None else int(seed)
    generator = np.random.default_rng(chosen_seed)
    differences = scores.differences
    count = scores.count

    means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        picks = generator.integers(0, count, size=count)
        means[index] = differences[picks].mean()

    low = float(np.percentile(means, float(bootstrap["lower_percentile"])))
    high = float(np.percentile(means, float(bootstrap["upper_percentile"])))
    return BootstrapInterval(
        delta=scores.delta,
        low=low,
        high=high,
        confidence_level=float(bootstrap["confidence_level"]),
        resamples=resamples,
        seed=chosen_seed,
    )


def relative_nll_change(scores: PairedScores) -> float:
    """``(control - candidate) / control``. Positive means the candidate improved."""
    control = scores.mean_control
    if control <= 0 or not math.isfinite(control):
        raise AnalysisContractError(
            f"mean control NLL is {control!r}; a relative change is undefined"
        )
    return (control - scores.mean_candidate) / control


# --------------------------------------------------------------------------------------
# Protected-slice regression (Plan Section 4.4)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceRegression:
    """One protected slice's relative regression under the candidate."""

    slice_id: str
    documents: int
    mean_candidate: float
    mean_control: float
    relative_regression: float
    breaches_limit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "documents": self.documents,
            "mean_candidate": self.mean_candidate,
            "mean_control": self.mean_control,
            "relative_regression": self.relative_regression,
            "breaches_limit": self.breaches_limit,
        }


def protected_slice_regressions(
    scores: PairedScores, protocol: Mapping[str, Any] | None = None
) -> tuple[SliceRegression, ...]:
    """Per-slice ``(candidate - control) / control``, against the frozen 1% limit."""
    resolved = _resolved(protocol)
    limit = float(resolved["thresholds"]["maximum_slice_regression"])
    if not scores.slices:
        return ()

    candidate = np.asarray(scores.candidate_nll, dtype=float)
    control = np.asarray(scores.control_nll, dtype=float)
    tags = np.asarray(scores.slices)

    results: list[SliceRegression] = []
    for slice_id in EXPECTED_PROTECTED_SLICES:
        mask = tags == slice_id
        if not mask.any():
            continue
        mean_candidate = float(candidate[mask].mean())
        mean_control = float(control[mask].mean())
        if mean_control <= 0:
            raise AnalysisContractError(
                f"mean control NLL for slice {slice_id!r} is {mean_control!r}; "
                "a relative regression is undefined"
            )
        regression = (mean_candidate - mean_control) / mean_control
        results.append(
            SliceRegression(
                slice_id=slice_id,
                documents=int(mask.sum()),
                mean_candidate=mean_candidate,
                mean_control=mean_control,
                relative_regression=regression,
                breaches_limit=regression > limit,
            )
        )
    return tuple(results)


# --------------------------------------------------------------------------------------
# Hypothesis evaluation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisResult:
    """One endpoint's evidence, condition by condition. No prose, no ranking."""

    hypothesis_id: str
    candidate_arm: str
    control_arm: str
    interval: BootstrapInterval
    relative_change: float
    slice_regressions: tuple[SliceRegression, ...]
    conditions: Mapping[str, bool]

    @property
    def supported(self) -> bool:
        """Every preregistered condition holds. Not "most", not "the important ones"."""
        return all(self.conditions.values())

    @property
    def harmful(self) -> bool:
        """The candidate is reliably worse, which is a finding rather than a non-result."""
        return self.interval.entirely_above_zero

    @property
    def unmet_conditions(self) -> tuple[str, ...]:
        return tuple(name for name, ok in self.conditions.items() if not ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "candidate_arm": self.candidate_arm,
            "control_arm": self.control_arm,
            "relative_change": self.relative_change,
            "supported": self.supported,
            "harmful": self.harmful,
            "conditions": dict(self.conditions),
            "unmet_conditions": list(self.unmet_conditions),
            "slice_regressions": [item.to_dict() for item in self.slice_regressions],
            **self.interval.to_dict(),
        }


@dataclass(frozen=True)
class ConfirmationResult:
    """The earlier short A/B/C confirmation, scored on validation_dev."""

    delta: float
    split: str = "validation_dev"

    @property
    def favours_candidate(self) -> bool:
        return self.delta < 0.0


def evaluate_hypothesis(
    scores: PairedScores,
    *,
    hypothesis_id: str,
    seed: int | None = None,
    confirmation: ConfirmationResult | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> HypothesisResult:
    """Evaluate one endpoint against every preregistered condition, and report each."""
    resolved = _resolved(protocol)
    endpoints = resolved["endpoints"]
    if hypothesis_id == HYPOTHESIS_PRIMARY:
        spec = endpoints["primary"]
        required = tuple(
            str(condition["condition_id"])
            for condition in resolved["annealing_claim"]["required_conditions"]
        )
    elif hypothesis_id == HYPOTHESIS_SECONDARY:
        spec = endpoints["secondary"]
        required = tuple(
            str(condition["condition_id"])
            for condition in resolved["quality_claim"]["required_conditions"]
        )
    else:
        raise AnalysisContractError(f"unknown hypothesis {hypothesis_id!r}")

    interval = paired_bootstrap(scores, seed=seed, protocol=resolved)
    relative = relative_nll_change(scores)
    regressions = protected_slice_regressions(scores, resolved)
    minimum_relative = float(resolved["thresholds"]["minimum_relative_reduction"])

    evaluated: dict[str, bool] = {}
    for condition_id in required:
        if condition_id == "ci_below_zero":
            evaluated[condition_id] = interval.entirely_below_zero
        elif condition_id == "relative_reduction_met":
            evaluated[condition_id] = relative >= minimum_relative
        elif condition_id == "no_protected_slice_regression":
            evaluated[condition_id] = not any(item.breaches_limit for item in regressions)
        elif condition_id == "confirmation_same_direction":
            # An absent confirmation cannot satisfy a condition. It fails it.
            evaluated[condition_id] = (
                confirmation is not None and confirmation.favours_candidate
            )
        else:  # pragma: no cover - the loader pins the condition vocabulary
            raise AnalysisContractError(f"unknown condition {condition_id!r}")

    return HypothesisResult(
        hypothesis_id=hypothesis_id,
        candidate_arm=str(spec["candidate_arm"]),
        control_arm=str(spec["control_arm"]),
        interval=interval,
        relative_change=relative,
        slice_regressions=regressions,
        conditions=evaluated,
    )


# --------------------------------------------------------------------------------------
# Interpretation (Plan Section 8.6)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Interpretation:
    """One preregistered interpretation code plus the evidence that selected it."""

    interpretation_id: str
    claim_allowed: bool
    primary: HypothesisResult | None
    secondary: HypothesisResult | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "interpretation_id": self.interpretation_id,
            "claim_allowed": self.claim_allowed,
            "reasons": list(self.reasons),
            "primary": self.primary.to_dict() if self.primary else None,
            "secondary": self.secondary.to_dict() if self.secondary else None,
        }


def _claim_allowed(interpretation_id: str, protocol: Mapping[str, Any]) -> bool:
    for item in protocol["interpretations"]["definitions"]:
        if str(item["interpretation_id"]) == interpretation_id:
            return bool(item["claim_allowed"])
    raise AnalysisContractError(f"unknown interpretation {interpretation_id!r}")


def interpret(
    primary: HypothesisResult | None,
    secondary: HypothesisResult | None,
    *,
    confirmation: ConfirmationResult | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> Interpretation:
    """Return the one preregistered code the evidence justifies, and nothing more.

    The ordering below is Plan Section 8.6's own: a harmful primary is reported as harmful
    even when the secondary looks good, because "C loses to B" is a finding about annealing
    that a favourable H1 does not soften.
    """
    resolved = _resolved(protocol)
    reasons: list[str] = []

    if primary is None or secondary is None:
        reasons.append(f"{ANALYSIS_CLAIM_NOT_SUPPORTED}: a required endpoint was not evaluated")
        return Interpretation(INCOMPLETE, _claim_allowed(INCOMPLETE, resolved), primary, secondary, tuple(reasons))

    if confirmation is None:
        reasons.append(
            f"{ANALYSIS_CONFIRMATION_MISSING}: Plan Section 8.6 requires a same-direction "
            "earlier confirmation on validation_dev"
        )
        return Interpretation(INCOMPLETE, _claim_allowed(INCOMPLETE, resolved), primary, secondary, tuple(reasons))

    if primary.harmful:
        reasons.append("C loses to B: annealing is harmful; ship B or A by development validation")
        return Interpretation(
            ANNEALING_HARMFUL, _claim_allowed(ANNEALING_HARMFUL, resolved), primary, secondary, tuple(reasons)
        )

    if primary.supported and secondary.supported:
        reasons.append("quality and timing both help")
        return Interpretation(
            QUALITY_AND_TIMING, _claim_allowed(QUALITY_AND_TIMING, resolved), primary, secondary, tuple(reasons)
        )

    if secondary.supported:
        reasons.append("quality helps; annealing timing unsupported")
        reasons.append(f"{ANALYSIS_CLAIM_NOT_SUPPORTED}: H2 unmet {list(primary.unmet_conditions)}")
        return Interpretation(
            QUALITY_ONLY, _claim_allowed(QUALITY_ONLY, resolved), primary, secondary, tuple(reasons)
        )

    reasons.append("no reliable differences; report null and ship the best valid branch by validation_dev")
    reasons.append(f"{ANALYSIS_CLAIM_NOT_SUPPORTED}: H2 unmet {list(primary.unmet_conditions)}")
    reasons.append(f"{ANALYSIS_CLAIM_NOT_SUPPORTED}: H1 unmet {list(secondary.unmet_conditions)}")
    return Interpretation(
        NULL_RESULT, _claim_allowed(NULL_RESULT, resolved), primary, secondary, tuple(reasons)
    )


def fallback_violations(
    *, is_undecayed_peak_lr_mainline: bool, protocol: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Plan Section 15: never release an undecayed peak-LR mainline checkpoint as fallback."""
    resolved = _resolved(protocol)
    if bool(resolved["fallback"]["never_use_undecayed_peak_lr_mainline"]) and is_undecayed_peak_lr_mainline:
        return (
            f"{ANALYSIS_UNDECAYED_FALLBACK}: an undecayed peak-LR mainline checkpoint may "
            "never be the release fallback",
        )
    return ()


# --------------------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------------------

_BLOCKER_FIELDS = ("blocker", "owner", "next_action")

_READINESS_GATED = (
    "branch_runs_completed",
    "validation_final_opened",
    "earlier_confirmation_completed",
    "real_paired_scores_available",
)


def assert_ready_for_claim(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: a public claim needs branch outcomes that do not exist yet."""
    readiness = _resolved(protocol)["readiness"]
    blocked = [name for name in _READINESS_GATED if str(readiness.get(name)) != PASS]
    if blocked:
        raise AnalysisNotReadyError(
            f"no claim can be made yet: {blocked}. "
            f"blocker={readiness.get('blocker')} owner={readiness.get('owner')} "
            f"next_action={readiness.get('next_action')}"
        )


def readiness_results(protocol: Mapping[str, Any] | None = None) -> tuple[CheckResult, ...]:
    """Report each unmet prerequisite as NOT_RUN/BLOCKED with its own next action."""
    readiness = _resolved(protocol)["readiness"]
    detail = {name: readiness.get(name) for name in _BLOCKER_FIELDS}
    named = all(str(value or "").strip() for value in detail.values())
    results: list[CheckResult] = []
    for name, value in readiness.items():
        if name in _BLOCKER_FIELDS or not isinstance(value, str):
            continue
        status = str(value)
        if status == PASS:
            results.append(
                CheckResult(f"analysis.readiness.{name}", "evidence exists", status, PASS, ANALYSIS_OK)
            )
            continue
        if not named:
            results.append(
                CheckResult(
                    f"analysis.readiness.{name}",
                    "an unmet prerequisite must name its blocker, owner, and next action",
                    status,
                    FAIL,
                    f"{ANALYSIS_CLAIM_NOT_SUPPORTED}: readiness detail is incomplete",
                )
            )
            continue
        results.append(
            CheckResult(
                f"analysis.readiness.{name}",
                "evidence exists",
                status,
                status,
                f"blocker={detail['blocker']} owner={detail['owner']} next_action={detail['next_action']}",
            )
        )
    return tuple(results)


def format_interpretation(interpretation: Interpretation) -> str:
    """Render the interpretation code and its evidence. Never a headline."""
    lines = [f"{interpretation.interpretation_id}  claim_allowed={interpretation.claim_allowed}"]
    for result in (interpretation.primary, interpretation.secondary):
        if result is None:
            continue
        lines.append(
            f"  {result.hypothesis_id} ({result.candidate_arm} vs {result.control_arm}): "
            f"delta={result.interval.delta:+.6f} "
            f"CI=[{result.interval.low:+.6f}, {result.interval.high:+.6f}] "
            f"relative={result.relative_change:+.4%} supported={result.supported}"
        )
        for name, ok in result.conditions.items():
            lines.append(f"      {'PASS' if ok else 'FAIL'}  {name}")
    for reason in interpretation.reasons:
        lines.append(f"  reason: {reason}")
    return "\n".join(lines)


__all__ = [
    "ANALYSIS_CLAIM_NOT_SUPPORTED",
    "ANALYSIS_CONFIRMATION_MISSING",
    "ANALYSIS_DOCUMENTS_NOT_PAIRED",
    "ANALYSIS_ENDPOINT_REORDERED",
    "ANALYSIS_FAIL_CLOSED_REASON_CODES",
    "ANALYSIS_INSUFFICIENT_DOCUMENTS",
    "ANALYSIS_NONDETERMINISTIC",
    "ANALYSIS_OK",
    "ANALYSIS_POST_HOC_SELECTION",
    "ANALYSIS_PROTOCOL_PATH",
    "ANALYSIS_THRESHOLD_DRIFT",
    "ANALYSIS_UNDECAYED_FALLBACK",
    "ANNEALING_HARMFUL",
    "ARM_A",
    "ARM_B",
    "ARM_C",
    "BLOCKED",
    "FROZEN_ANALYSIS_PROTOCOL_SHA256",
    "HYPOTHESIS_PRIMARY",
    "HYPOTHESIS_SECONDARY",
    "INCOMPLETE",
    "INTERPRETATION_IDS",
    "NULL_RESULT",
    "QUALITY_AND_TIMING",
    "QUALITY_ONLY",
    "AnalysisContractError",
    "AnalysisNotReadyError",
    "BootstrapInterval",
    "ConfirmationResult",
    "HypothesisResult",
    "Interpretation",
    "PairedScores",
    "SliceRegression",
    "UnpairedScoresError",
    "assert_documents_paired",
    "assert_ready_for_claim",
    "default_seed",
    "evaluate_hypothesis",
    "fallback_violations",
    "format_interpretation",
    "interpret",
    "load_analysis_protocol",
    "paired_bootstrap",
    "protected_slice_regressions",
    "readiness_results",
    "relative_nll_change",
    "threshold_drift_violations",
]
