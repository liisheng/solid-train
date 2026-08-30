"""Two-machine measurement, horizon calculators, and G0-G6 gates (Plan Sections 7.1, 9, 10.4, 12-13).

Plan Section 9.1 gives this module its single instruction: "Measure rather than assume." Every
speed, memory, horizon, or efficiency number the submission reports has to trace back to a
recorded measurement, and every gate has to name the evidence it requires. So the calculators
here are deliberately *pure and total*: they accept measured inputs and return derived values,
and they refuse -- loudly -- to accept a missing, sentinel, or non-finite input in place of a
measurement. There is no default, no fallback, and no "approximately" that silently becomes a
claim.

The gate engine encodes the rule the whole plan rests on::

    absence of evidence is NOT_RUN, never PASS

A requirement with no recorded evidence is ``NOT_RUN``. One that depends on an external party
-- teammate eligibility, public artifact access, release approval -- is ``BLOCKED``, and only
when it names its own blocker, owner, and next action. A gate is ``PASS`` only when every one
of its requirements is ``PASS``. Nothing in :func:`evaluate_gate` can turn an empty evidence
mapping into a passing gate, which is exactly the failure the plan's fail-closed posture is
guarding against.

The contract is backed by one frozen config::

    configs/operations/measurement_v1.yaml

Guarantees, mirroring :mod:`tinybench_lm.pipeline_bench` and :mod:`tinybench_lm.campaign`:

1. **Immutable.** The config is verified against a pinned SHA-256 digest
   (:data:`FROZEN_OPERATIONS_PROTOCOL_SHA256`) on every load.
2. **Deterministic.** Percentiles use nearest-rank, not interpolation, so two machines
   summarizing the same samples report the same p10/median/p90.
3. **Fail closed.** Insufficient VRAM headroom, a too-short throughput window, too few
   samples, an unjustified backend promotion, and an unmeasured calculator input all raise
   or report ``FAIL``.
4. **Targets stay targets.** Plan Section 7.1: "'20B' and '402.8 tokens/parameter' remain
   stretch targets until reached." :func:`stretch_target_violations` refuses to let an
   unmeasured 8 GB fit, CPU RAM headroom, or throughput be stated as measured.

Nothing here measures hardware, promotes a backend, rehearses takeover, schedules a dated
campaign, or passes a gate. Tests drive it entirely with synthetic measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .data_protocols import (
    REPOSITORY_ROOT,
    ProtocolError,
    ProtocolNotReadyError,
    load_protocol,
)
from .environment import CheckResult
from .shards import FAIL, NOT_RUN, PASS

OPERATIONS_PROTOCOL_DIR = REPOSITORY_ROOT / "configs" / "operations"
OPERATIONS_PROTOCOL_PATH = OPERATIONS_PROTOCOL_DIR / "measurement_v1.yaml"

#: SHA-256 of the frozen operations contract, over file bytes with CRLF normalized to LF.
FROZEN_OPERATIONS_PROTOCOL_SHA256: Mapping[str, str] = {
    "measurement_v1.yaml": "0adb1c65db9d47deb373e3610489143898a12b485a4b92380b4c6f5dfc2ebf19",
}

#: A pending external dependency is BLOCKED only when it names its own next step.
BLOCKED = "BLOCKED"

#: The four statuses a gate requirement may hold. Only FAIL is a failure.
GATE_STATUSES: tuple[str, ...] = (PASS, FAIL, BLOCKED, NOT_RUN)

GATE_IDS: tuple[str, ...] = ("G0", "G1", "G2", "G3", "G4", "G5", "G6")

MACHINE_4070 = "rtx_4070"
MACHINE_3070 = "rtx_3070"

SECONDS_PER_HOUR = 3600

#: Plan Section 7.1's 6ND forward-plus-backward estimate. An estimate, never a measurement.
FLOPS_PER_PARAMETER_PER_TOKEN = 6

# --------------------------------------------------------------------------------------
# Reason-code vocabulary
# --------------------------------------------------------------------------------------

OPS_OK = "OPS_OK"
OPS_BACKEND_PROMOTION_UNJUSTIFIED = "OPS_BACKEND_PROMOTION_UNJUSTIFIED"
OPS_BLOCKER_DETAIL_INCOMPLETE = "OPS_BLOCKER_DETAIL_INCOMPLETE"
OPS_DEADLINE_OVERRUN = "OPS_DEADLINE_OVERRUN"
OPS_INSUFFICIENT_SAMPLES = "OPS_INSUFFICIENT_SAMPLES"
OPS_TAKEOVER_NOT_REHEARSED = "OPS_TAKEOVER_NOT_REHEARSED"
OPS_TARGET_CLAIMED_AS_MEASURED = "OPS_TARGET_CLAIMED_AS_MEASURED"
OPS_THROUGHPUT_WINDOW_TOO_SHORT = "OPS_THROUGHPUT_WINDOW_TOO_SHORT"
OPS_UNMEASURED_INPUT = "OPS_UNMEASURED_INPUT"
OPS_UNSUPPORTED_PASS = "OPS_UNSUPPORTED_PASS"
OPS_VRAM_HEADROOM_INSUFFICIENT = "OPS_VRAM_HEADROOM_INSUFFICIENT"

OPS_FAIL_CLOSED_REASON_CODES: frozenset[str] = frozenset(
    {
        OPS_BACKEND_PROMOTION_UNJUSTIFIED,
        OPS_BLOCKER_DETAIL_INCOMPLETE,
        OPS_DEADLINE_OVERRUN,
        OPS_INSUFFICIENT_SAMPLES,
        OPS_TAKEOVER_NOT_REHEARSED,
        OPS_TARGET_CLAIMED_AS_MEASURED,
        OPS_THROUGHPUT_WINDOW_TOO_SHORT,
        OPS_UNMEASURED_INPUT,
        OPS_UNSUPPORTED_PASS,
        OPS_VRAM_HEADROOM_INSUFFICIENT,
    }
)

#: Values that look like data but are not a measurement.
_SENTINELS = frozenset({"", NOT_RUN, BLOCKED, FAIL, "TBD", "N/A", "None", "null"})


class OperationsContractError(ProtocolError):
    """The frozen operations contract is malformed, or a measurement violates it."""


class UnmeasuredInputError(OperationsContractError):
    """A calculator was given a sentinel, missing, or non-finite value in place of a measurement."""


class OperationsNotReadyError(ProtocolNotReadyError):
    """A horizon, efficiency, or gate claim needs evidence that does not exist yet."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def load_operations_protocol(
    path=OPERATIONS_PROTOCOL_PATH, *, verify: bool = True
) -> dict[str, Any]:
    """Load the frozen operations contract, verifying its pinned digest by default."""
    protocol = load_protocol(path, verify=verify, registry=FROZEN_OPERATIONS_PROTOCOL_SHA256)
    required = (
        "machines",
        "hardware_promotion",
        "horizon",
        "stretch_targets",
        "gates",
        "gate_policy",
        "takeover_rehearsal",
        "readiness",
    )
    for section in required:
        if section not in protocol:
            raise OperationsContractError(
                f"operations protocol is missing required section {section!r}"
            )

    promotion = protocol["hardware_promotion"]
    headroom = float(promotion["minimum_vram_headroom_fraction"])
    if headroom < 0.10:
        raise OperationsContractError(
            f"{OPS_VRAM_HEADROOM_INSUFFICIENT}: Plan Section 9.1 requires at least 10% VRAM "
            f"headroom, contract declares {headroom}"
        )
    minimum_seconds = int(promotion["minimum_sustained_seconds"])
    maximum_seconds = int(promotion["maximum_sustained_seconds"])
    if minimum_seconds < 1800 or maximum_seconds < minimum_seconds:
        raise OperationsContractError(
            f"{OPS_THROUGHPUT_WINDOW_TOO_SHORT}: Plan Section 9.1 measures 30-60 minutes of "
            f"sustained real-shard throughput, contract declares {minimum_seconds}-{maximum_seconds}s"
        )
    if str(promotion["percentile_method"]) != "nearest_rank":
        raise OperationsContractError(
            "percentiles must be nearest-rank so two machines summarize samples identically"
        )
    if tuple(int(p) for p in promotion["required_percentiles"]) != (10, 50, 90):
        raise OperationsContractError("Plan Section 9.1 requires p10, median, and p90")
    backend = promotion["backend_promotion"]
    if not (bool(backend["requires_correctness_pass"]) and bool(backend["requires_throughput_improvement"])):
        raise OperationsContractError(
            f"{OPS_BACKEND_PROMOTION_UNJUSTIFIED}: a backend is promoted only when correctness "
            "passes and sustained throughput improves"
        )

    horizon = protocol["horizon"]
    if float(horizon["evaluation_reserve_multiplier"]) != 1.25:
        raise OperationsContractError("Plan Section 10.4 reserves p90 x candidates x 1.25")
    if not bool(horizon["approximate_flops_is_estimate"]):
        raise OperationsContractError("the 6ND FLOPs figure is an estimate, never a measurement")

    targets = protocol["stretch_targets"]
    if bool(targets["is_measurement"]):
        raise OperationsContractError(
            f"{OPS_TARGET_CLAIMED_AS_MEASURED}: Plan Section 7.1 keeps 20B and 402.8 "
            "tokens/parameter as stretch targets until reached"
        )
    if str(targets["status"]) != "NOT_REACHED":
        raise OperationsContractError(
            f"{OPS_TARGET_CLAIMED_AS_MEASURED}: the stretch target status must stay NOT_REACHED"
        )

    declared = tuple(str(gate["gate_id"]) for gate in protocol["gates"])
    if declared != GATE_IDS:
        raise OperationsContractError(
            f"operations protocol must declare gates {GATE_IDS}, found {declared}"
        )
    for gate in protocol["gates"]:
        if not gate.get("requirements"):
            raise OperationsContractError(f"gate {gate['gate_id']} declares no requirements")

    policy = protocol["gate_policy"]
    if not bool(policy["absence_of_evidence_is_never_pass"]):
        raise OperationsContractError(
            f"{OPS_UNSUPPORTED_PASS}: absence of evidence may never become PASS"
        )
    if str(policy["missing_evidence_status"]) != NOT_RUN:
        raise OperationsContractError("missing evidence is NOT_RUN")
    if not bool(policy["gate_passes_only_if_all_requirements_pass"]):
        raise OperationsContractError("a gate passes only when every requirement passes")
    if tuple(str(status) for status in policy["statuses"]) != GATE_STATUSES:
        raise OperationsContractError(f"gate statuses must be {GATE_STATUSES}")

    rehearsal = protocol["takeover_rehearsal"]
    if str(rehearsal["rehearsed_before"]) != "G4":
        raise OperationsContractError("Plan Section 9 rehearses takeover before G4")
    return protocol


def _resolved(protocol: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return protocol if protocol is not None else load_operations_protocol()


# --------------------------------------------------------------------------------------
# Measured-input discipline
# --------------------------------------------------------------------------------------


def is_measured(value: Any) -> bool:
    """True only for a real, finite, usable measurement.

    A sentinel string, ``None``, a bool, a NaN, and an infinity are all *absences* wearing
    the shape of data. Treating any of them as a number is how an unmeasured claim gets into
    a report, so they are rejected here once rather than at each call site.
    """
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def require_measured(name: str, value: Any, *, positive: bool = True) -> float:
    """Return the measurement as a float, or raise naming exactly what is missing."""
    if not is_measured(value):
        raise UnmeasuredInputError(
            f"{OPS_UNMEASURED_INPUT}: {name} is {value!r}, which is not a measurement. "
            "Record the measurement instead of supplying a placeholder."
        )
    number = float(value)
    if positive and number <= 0:
        raise UnmeasuredInputError(
            f"{OPS_UNMEASURED_INPUT}: {name} must be positive, got {number}"
        )
    return number


# --------------------------------------------------------------------------------------
# Deterministic percentiles (Plan Section 9.1)
# --------------------------------------------------------------------------------------


def nearest_rank_percentile(samples: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile: ``ceil(p/100 * n)``-th smallest sample, 1-indexed.

    Interpolating percentiles disagree between libraries and between machines. Nearest rank
    always returns an *observed* sample, so the 4070 and the 3070 summarizing the same run
    report the same p90.
    """
    if not samples:
        raise UnmeasuredInputError(f"{OPS_INSUFFICIENT_SAMPLES}: no throughput samples supplied")
    for index, sample in enumerate(samples):
        require_measured(f"sample[{index}]", sample, positive=False)
    if not 0 < percentile <= 100:
        raise OperationsContractError(f"percentile must be in (0, 100], got {percentile}")
    ordered = sorted(float(sample) for sample in samples)
    rank = math.ceil(percentile / 100 * len(ordered))
    return ordered[max(rank, 1) - 1]


@dataclass(frozen=True)
class ThroughputMeasurement:
    """A sustained real-shard throughput window and its deterministic summary."""

    machine_id: str
    samples: tuple[float, ...]
    window_seconds: float
    used_real_shards: bool

    def __post_init__(self) -> None:
        for index, sample in enumerate(self.samples):
            require_measured(f"{self.machine_id}.sample[{index}]", sample)
        require_measured(f"{self.machine_id}.window_seconds", self.window_seconds)

    @property
    def p10(self) -> float:
        return nearest_rank_percentile(self.samples, 10)

    @property
    def median(self) -> float:
        return nearest_rank_percentile(self.samples, 50)

    @property
    def p90(self) -> float:
        return nearest_rank_percentile(self.samples, 90)

    @property
    def sustained(self) -> float:
        """The median is the sustained rate. The p10 is what a schedule should trust."""
        return self.median

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "sample_count": len(self.samples),
            "window_seconds": self.window_seconds,
            "used_real_shards": self.used_real_shards,
            "sustained_tokens_per_second": self.sustained,
            "throughput_p10": self.p10,
            "throughput_median": self.median,
            "throughput_p90": self.p90,
        }


def throughput_violations(
    measurement: ThroughputMeasurement, protocol: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """A throughput number is only usable if the window and sample count earn it."""
    promotion = _resolved(protocol)["hardware_promotion"]
    problems: list[str] = []
    minimum_samples = int(promotion["minimum_samples"])
    if len(measurement.samples) < minimum_samples:
        problems.append(
            f"{OPS_INSUFFICIENT_SAMPLES}: {len(measurement.samples)} samples, "
            f"{minimum_samples} required for p10/median/p90"
        )
    minimum_seconds = int(promotion["minimum_sustained_seconds"])
    if measurement.window_seconds < minimum_seconds:
        problems.append(
            f"{OPS_THROUGHPUT_WINDOW_TOO_SHORT}: {measurement.window_seconds}s window, "
            f"Plan Section 9.1 measures at least {minimum_seconds}s"
        )
    if bool(promotion["sustained_must_use_real_shards"]) and not measurement.used_real_shards:
        problems.append(
            f"{OPS_THROUGHPUT_WINDOW_TOO_SHORT}: sustained throughput must be measured on "
            "real shards, not a synthetic loop"
        )
    return tuple(problems)


# --------------------------------------------------------------------------------------
# Hardware profile (Plan Section 9.1)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MicrobatchProbe:
    """One microbatch size and the peak VRAM it actually used."""

    microbatch: int
    peak_vram_bytes: float
    vram_total_bytes: float

    def __post_init__(self) -> None:
        require_measured("microbatch", self.microbatch)
        require_measured("peak_vram_bytes", self.peak_vram_bytes)
        require_measured("vram_total_bytes", self.vram_total_bytes)

    @property
    def headroom_fraction(self) -> float:
        """``(total - peak) / total``. Negative when the probe overcommitted."""
        return (self.vram_total_bytes - self.peak_vram_bytes) / self.vram_total_bytes


def minimum_vram_headroom(protocol: Mapping[str, Any] | None = None) -> float:
    return float(_resolved(protocol)["hardware_promotion"]["minimum_vram_headroom_fraction"])


def safe_microbatch(
    probes: Sequence[MicrobatchProbe], protocol: Mapping[str, Any] | None = None
) -> int:
    """The largest microbatch that still leaves the frozen VRAM headroom.

    Plan Section 9.1 requires "at least 10% VRAM headroom", so a probe that fits exactly is
    not safe. Returning the largest *qualifying* probe -- rather than the largest that ran --
    is the difference between a measured safe size and an optimistic one.

    A probe sitting exactly on the boundary may be excluded, because a ratio like
    ``(12 - 10.8) / 12`` is ``0.09999999999999998`` in binary floating point. That is
    deliberate: for a headroom rule the representable error must round toward *more* headroom,
    never less, so the comparison stays a strict ``>=`` on the computed fraction.
    """
    minimum = minimum_vram_headroom(protocol)
    qualifying = [probe for probe in probes if probe.headroom_fraction >= minimum]
    if not qualifying:
        raise OperationsContractError(
            f"{OPS_VRAM_HEADROOM_INSUFFICIENT}: no probe leaves at least "
            f"{minimum:.0%} VRAM headroom"
        )
    return max(probe.microbatch for probe in qualifying)


@dataclass(frozen=True)
class MachineProfile:
    """Every Plan Section 9.1 measurement for one machine. No field has a default."""

    machine_id: str
    gpu_name: str
    vram_total_bytes: float
    bf16_supported: bool
    bf16_stable: bool
    backend: str
    safe_microbatch: int
    throughput: ThroughputMeasurement
    peak_ram_bytes: float
    peak_swap_bytes: float
    peak_vram_bytes: float
    dataloader_wait_seconds: float
    checkpoint_seconds: float
    thermal_throttle_reasons: tuple[str, ...]

    @property
    def vram_headroom_fraction(self) -> float:
        return (self.vram_total_bytes - self.peak_vram_bytes) / self.vram_total_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "gpu_name": self.gpu_name,
            "vram_total_bytes": self.vram_total_bytes,
            "bf16_supported": self.bf16_supported,
            "bf16_stable": self.bf16_stable,
            "backend": self.backend,
            "safe_microbatch": self.safe_microbatch,
            "peak_ram_bytes": self.peak_ram_bytes,
            "peak_swap_bytes": self.peak_swap_bytes,
            "peak_vram_bytes": self.peak_vram_bytes,
            "vram_headroom_fraction": self.vram_headroom_fraction,
            "dataloader_wait_seconds": self.dataloader_wait_seconds,
            "checkpoint_seconds": self.checkpoint_seconds,
            "thermal_throttle_reasons": list(self.thermal_throttle_reasons),
            **self.throughput.to_dict(),
        }


def required_measurements(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    return tuple(
        str(name) for name in _resolved(protocol)["hardware_promotion"]["required_measurements"]
    )


def profile_violations(
    profile: MachineProfile, protocol: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Every required measurement must be present, real, and within the frozen limits."""
    resolved = _resolved(protocol)
    problems: list[str] = []

    recorded = profile.to_dict()
    for name in required_measurements(resolved):
        value = recorded.get(name)
        if isinstance(value, bool):
            continue
        if value is None or (isinstance(value, str) and value.strip() in _SENTINELS):
            problems.append(f"{OPS_UNMEASURED_INPUT}: {name} is not recorded")

    minimum = minimum_vram_headroom(resolved)
    if profile.vram_headroom_fraction < minimum:
        problems.append(
            f"{OPS_VRAM_HEADROOM_INSUFFICIENT}: {profile.machine_id} leaves "
            f"{profile.vram_headroom_fraction:.1%} VRAM headroom, {minimum:.0%} required"
        )
    problems.extend(throughput_violations(profile.throughput, resolved))
    return tuple(problems)


@dataclass(frozen=True)
class BackendComparison:
    """One candidate backend measured against the incumbent."""

    backend: str
    correctness_passed: bool
    reliability_adjusted_throughput: float


def backend_promotion_violations(
    incumbent: BackendComparison,
    candidate: BackendComparison,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Plan 9.1: promote only if correctness passes *and* sustained throughput improves."""
    rules = _resolved(protocol)["hardware_promotion"]["backend_promotion"]
    problems: list[str] = []
    if bool(rules["requires_correctness_pass"]) and not candidate.correctness_passed:
        problems.append(
            f"{OPS_BACKEND_PROMOTION_UNJUSTIFIED}: {candidate.backend} did not pass correctness"
        )
    if bool(rules["requires_throughput_improvement"]):
        require_measured("incumbent throughput", incumbent.reliability_adjusted_throughput)
        require_measured("candidate throughput", candidate.reliability_adjusted_throughput)
        if candidate.reliability_adjusted_throughput <= incumbent.reliability_adjusted_throughput:
            problems.append(
                f"{OPS_BACKEND_PROMOTION_UNJUSTIFIED}: {candidate.backend} at "
                f"{candidate.reliability_adjusted_throughput} does not improve on "
                f"{incumbent.backend} at {incumbent.reliability_adjusted_throughput}"
            )
    return tuple(problems)


# --------------------------------------------------------------------------------------
# Pure horizon and efficiency calculators (Plan Sections 7.1 and 10.4)
# --------------------------------------------------------------------------------------


def mainline_parent_time(parent_tokens: Any, measured_4070_tps: Any) -> float:
    """``parent_tokens / measured_4070_tps``, in seconds."""
    tokens = require_measured("parent_tokens", parent_tokens)
    rate = require_measured("measured_4070_tps", measured_4070_tps)
    return tokens / rate


def branch_time(total_branch_tokens: Any, measured_3070_tps: Any) -> float:
    """``total_branch_tokens / measured_3070_tps``, in seconds."""
    tokens = require_measured("total_branch_tokens", total_branch_tokens)
    rate = require_measured("measured_3070_tps", measured_3070_tps)
    return tokens / rate


def evaluation_time(measured_eval_runtime: Any, candidate_count: Any) -> float:
    """``measured_eval_runtime * candidate_count``, in seconds."""
    runtime = require_measured("measured_eval_runtime", measured_eval_runtime)
    count = require_measured("candidate_count", candidate_count)
    return runtime * count


def evaluation_reserve(
    p90_runtime: Any, candidate_count: Any, protocol: Mapping[str, Any] | None = None
) -> float:
    """Plan Section 10.4: ``p90 measured runtime x exact candidate count x 1.25``."""
    multiplier = float(_resolved(protocol)["horizon"]["evaluation_reserve_multiplier"])
    runtime = require_measured("p90_runtime", p90_runtime)
    count = require_measured("candidate_count", candidate_count)
    return runtime * count * multiplier


def consumed_tokens(updates: Any, loss_tokens_per_update: Any) -> int:
    """``updates * loss_tokens_per_update``, counted rather than estimated."""
    count = require_measured("updates", updates, positive=False)
    per_update = require_measured("loss_tokens_per_update", loss_tokens_per_update)
    if count < 0:
        raise UnmeasuredInputError(f"{OPS_UNMEASURED_INPUT}: updates must be nonnegative")
    return int(count) * int(per_update)


def effective_passes(tokens_consumed: Any, corpus_tokens: Any) -> float:
    """How many times the corpus was effectively seen."""
    consumed = require_measured("consumed_tokens", tokens_consumed, positive=False)
    corpus = require_measured("corpus_tokens", corpus_tokens)
    return consumed / corpus


def tokens_per_parameter(tokens_consumed: Any, unique_trainable_parameters: Any) -> float:
    """The README efficiency figure. Derived from actual consumed tokens, never a target."""
    consumed = require_measured("consumed_tokens", tokens_consumed, positive=False)
    parameters = require_measured("unique_trainable_parameters", unique_trainable_parameters)
    return consumed / parameters


def active_gpu_hours(active_gpu_seconds: Any) -> float:
    """Active GPU seconds converted to hours."""
    seconds = require_measured("active_gpu_seconds", active_gpu_seconds, positive=False)
    return seconds / SECONDS_PER_HOUR


def wall_time(*durations: Any) -> float:
    """Total wall time across sequential phases, in seconds."""
    if not durations:
        raise UnmeasuredInputError(f"{OPS_UNMEASURED_INPUT}: no durations supplied")
    return sum(
        require_measured(f"duration[{index}]", value, positive=False)
        for index, value in enumerate(durations)
    )


def approximate_flops(unique_trainable_parameters: Any, tokens_consumed: Any) -> float:
    """The 6ND estimate. Approximate by construction; never report it as a measurement."""
    parameters = require_measured("unique_trainable_parameters", unique_trainable_parameters)
    consumed = require_measured("consumed_tokens", tokens_consumed, positive=False)
    return FLOPS_PER_PARAMETER_PER_TOKEN * parameters * consumed


@dataclass(frozen=True)
class HorizonPlan:
    """Every derived duration for one campaign plan, plus whether it fits the calendar."""

    mainline_seconds: float
    branch_seconds: float
    evaluation_seconds: float
    recovery_reserve_seconds: float

    @property
    def total_seconds(self) -> float:
        return (
            self.mainline_seconds
            + self.branch_seconds
            + self.evaluation_seconds
            + self.recovery_reserve_seconds
        )

    @property
    def total_hours(self) -> float:
        return self.total_seconds / SECONDS_PER_HOUR

    def fits_within(self, available_seconds: Any) -> bool:
        """Plan 7.1: all dependencies plus recovery reserve must fit before September 17."""
        available = require_measured("available_seconds", available_seconds)
        return self.total_seconds <= available

    def to_dict(self) -> dict[str, Any]:
        return {
            "mainline_seconds": self.mainline_seconds,
            "branch_seconds": self.branch_seconds,
            "evaluation_seconds": self.evaluation_seconds,
            "recovery_reserve_seconds": self.recovery_reserve_seconds,
            "total_seconds": self.total_seconds,
            "total_hours": self.total_hours,
        }


def build_horizon_plan(
    *,
    parent_tokens: Any,
    measured_4070_tps: Any,
    total_branch_tokens: Any,
    measured_3070_tps: Any,
    measured_eval_runtime: Any,
    candidate_count: Any,
    recovery_reserve_seconds: Any,
    protocol: Mapping[str, Any] | None = None,
) -> HorizonPlan:
    """Assemble a horizon from measured inputs only; any placeholder raises."""
    del protocol
    return HorizonPlan(
        mainline_seconds=mainline_parent_time(parent_tokens, measured_4070_tps),
        branch_seconds=branch_time(total_branch_tokens, measured_3070_tps),
        evaluation_seconds=evaluation_time(measured_eval_runtime, candidate_count),
        recovery_reserve_seconds=require_measured(
            "recovery_reserve_seconds", recovery_reserve_seconds, positive=False
        ),
    )


# --------------------------------------------------------------------------------------
# Stretch targets stay targets (Plan Section 7.1)
# --------------------------------------------------------------------------------------


def stretch_target_violations(
    claims: Mapping[str, Any], protocol: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Refuse to let an unmeasured target be stated as a measured result."""
    targets = _resolved(protocol)["stretch_targets"]
    unmeasured = {str(name) for name in targets["unmeasured_targets"]}
    problems: list[str] = []
    for name, value in claims.items():
        if name in unmeasured and is_measured(value):
            problems.append(
                f"{OPS_TARGET_CLAIMED_AS_MEASURED}: {name} is an unmeasured target; "
                f"reporting {value!r} would state it as measured"
            )
    return tuple(problems)


def target_reference(protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The stretch numbers, returned with their NOT_REACHED status attached."""
    targets = _resolved(protocol)["stretch_targets"]
    return {
        "status": str(targets["status"]),
        "is_measurement": bool(targets["is_measurement"]),
        "consumed_tokens": int(targets["consumed_tokens"]),
        "tokens_per_parameter": float(targets["tokens_per_parameter"]),
    }


# --------------------------------------------------------------------------------------
# Gates G0-G6 (Plan Section 13)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateRequirement:
    """One requirement and the evidence key that can satisfy it."""

    gate_id: str
    requirement_id: str
    evidence_key: str
    externally_blocked: bool = False


@dataclass(frozen=True)
class Evidence:
    """One recorded outcome. ``passed=None`` means the check has not been run."""

    passed: bool | None = None
    blocker: str = ""
    owner: str = ""
    next_action: str = ""
    detail: str = ""

    @property
    def has_blocker_detail(self) -> bool:
        return all(field_.strip() for field_ in (self.blocker, self.owner, self.next_action))


@dataclass(frozen=True)
class GateReport:
    """One gate's status and the per-requirement verdicts behind it."""

    gate_id: str
    title: str
    status: str
    results: tuple[CheckResult, ...]

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status == FAIL)


def gate_requirements(
    gate_id: str, protocol: Mapping[str, Any] | None = None
) -> tuple[GateRequirement, ...]:
    """Every requirement declared for one gate."""
    resolved = _resolved(protocol)
    for gate in resolved["gates"]:
        if str(gate["gate_id"]) != gate_id:
            continue
        return tuple(
            GateRequirement(
                gate_id=gate_id,
                requirement_id=str(requirement["requirement_id"]),
                evidence_key=str(requirement["evidence_key"]),
                externally_blocked=bool(requirement.get("externally_blocked", False)),
            )
            for requirement in gate["requirements"]
        )
    raise OperationsContractError(f"unknown gate {gate_id!r}")


def _requirement_verdict(
    requirement: GateRequirement, evidence: Evidence | None
) -> CheckResult:
    check_id = f"{requirement.gate_id}.{requirement.requirement_id}"
    detail = "evidence required"

    # Absence of evidence is NOT_RUN. This branch is the reason the module exists.
    # Nothing recorded at all -- including for an externally blocked requirement, which is
    # only BLOCKED once someone records who is blocked on what.
    if evidence is None:
        return CheckResult(
            check_id, detail, NOT_RUN, NOT_RUN, f"{OPS_UNSUPPORTED_PASS}: no evidence recorded"
        )

    if evidence.passed is None:
        claims_blocker = bool(evidence.blocker.strip()) or requirement.externally_blocked
        if not claims_blocker:
            return CheckResult(
                check_id, detail, NOT_RUN, NOT_RUN, f"{OPS_UNSUPPORTED_PASS}: no evidence recorded"
            )
        # A blocker without an owner and a next action is an excuse, not a status.
        if not evidence.has_blocker_detail:
            return CheckResult(
                check_id,
                detail,
                "blocked without detail",
                FAIL,
                f"{OPS_BLOCKER_DETAIL_INCOMPLETE}: a BLOCKED requirement must name its "
                "blocker, owner, and next action",
            )
        return CheckResult(
            check_id,
            detail,
            BLOCKED,
            BLOCKED,
            f"blocker={evidence.blocker} owner={evidence.owner} next_action={evidence.next_action}",
        )

    if evidence.passed:
        return CheckResult(check_id, detail, PASS, PASS, evidence.detail or OPS_OK)
    return CheckResult(check_id, detail, FAIL, FAIL, evidence.detail or "recorded evidence failed")


def evaluate_gate(
    gate_id: str,
    evidence: Mapping[str, Evidence],
    protocol: Mapping[str, Any] | None = None,
) -> GateReport:
    """Evaluate one gate. PASS only when every requirement is PASS."""
    resolved = _resolved(protocol)
    title = next(
        str(gate.get("title", "")) for gate in resolved["gates"] if str(gate["gate_id"]) == gate_id
    )
    requirements = gate_requirements(gate_id, resolved)
    results = tuple(
        _requirement_verdict(requirement, evidence.get(requirement.evidence_key))
        for requirement in requirements
    )
    statuses = {result.status for result in results}
    if FAIL in statuses:
        status = FAIL
    elif BLOCKED in statuses:
        status = BLOCKED
    elif NOT_RUN in statuses:
        status = NOT_RUN
    else:
        status = PASS
    return GateReport(gate_id, title, status, results)


def evaluate_all_gates(
    evidence: Mapping[str, Evidence], protocol: Mapping[str, Any] | None = None
) -> tuple[GateReport, ...]:
    """Evaluate G0-G6 in order."""
    resolved = _resolved(protocol)
    return tuple(evaluate_gate(gate_id, evidence, resolved) for gate_id in GATE_IDS)


def takeover_rehearsal_violations(
    completed_steps: Iterable[str], protocol: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Plan Section 9: the 4070-to-3070 takeover is rehearsed before G4."""
    rehearsal = _resolved(protocol)["takeover_rehearsal"]
    required = [str(step) for step in rehearsal["required_steps"]]
    done = {str(step) for step in completed_steps}
    missing = [step for step in required if step not in done]
    if missing:
        return (f"{OPS_TAKEOVER_NOT_REHEARSED}: missing rehearsal steps {missing}",)
    return ()


# --------------------------------------------------------------------------------------
# Readiness. Absence of evidence is never PASS.
# --------------------------------------------------------------------------------------

_BLOCKER_FIELDS = ("blocker", "owner", "next_action")

_READINESS_GATED = (
    "measured_4070_profile",
    "measured_3070_profile",
    "sustained_real_shard_throughput",
    "takeover_rehearsed",
)


def assert_ready_for_horizon_claim(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: a horizon or efficiency claim needs measurements that do not exist."""
    readiness = _resolved(protocol)["readiness"]
    blocked = [name for name in _READINESS_GATED if str(readiness.get(name)) != PASS]
    if blocked:
        raise OperationsNotReadyError(
            f"horizon and gate claims are not ready: {blocked}. "
            f"blocker={readiness.get('blocker')} owner={readiness.get('owner')} "
            f"next_action={readiness.get('next_action')}"
        )


def readiness_results(protocol: Mapping[str, Any] | None = None) -> tuple[CheckResult, ...]:
    """Report each unmeasured prerequisite as NOT_RUN/BLOCKED with its own next action."""
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
                CheckResult(f"operations.readiness.{name}", "measured evidence exists", status, PASS, OPS_OK)
            )
            continue
        if not named:
            results.append(
                CheckResult(
                    f"operations.readiness.{name}",
                    "an unmeasured prerequisite must name its blocker, owner, and next action",
                    status,
                    FAIL,
                    f"{OPS_BLOCKER_DETAIL_INCOMPLETE}: readiness detail is incomplete",
                )
            )
            continue
        results.append(
            CheckResult(
                f"operations.readiness.{name}",
                "measured evidence exists",
                status,
                status,
                f"blocker={detail['blocker']} owner={detail['owner']} next_action={detail['next_action']}",
            )
        )
    return tuple(results)


def format_gate_report(reports: Sequence[GateReport]) -> str:
    """Render the gate ladder as one aligned, greppable status block."""
    lines: list[str] = []
    for report in reports:
        lines.append(f"{report.status:<8} {report.gate_id}  {report.title}")
        for result in report.results:
            lines.append(f"    {result.status:<8} {result.check_id}  {result.reason}")
    return "\n".join(lines)


__all__ = [
    "BLOCKED",
    "FLOPS_PER_PARAMETER_PER_TOKEN",
    "FROZEN_OPERATIONS_PROTOCOL_SHA256",
    "GATE_IDS",
    "GATE_STATUSES",
    "MACHINE_3070",
    "MACHINE_4070",
    "OPERATIONS_PROTOCOL_PATH",
    "OPS_BACKEND_PROMOTION_UNJUSTIFIED",
    "OPS_BLOCKER_DETAIL_INCOMPLETE",
    "OPS_DEADLINE_OVERRUN",
    "OPS_FAIL_CLOSED_REASON_CODES",
    "OPS_INSUFFICIENT_SAMPLES",
    "OPS_OK",
    "OPS_TAKEOVER_NOT_REHEARSED",
    "OPS_TARGET_CLAIMED_AS_MEASURED",
    "OPS_THROUGHPUT_WINDOW_TOO_SHORT",
    "OPS_UNMEASURED_INPUT",
    "OPS_UNSUPPORTED_PASS",
    "OPS_VRAM_HEADROOM_INSUFFICIENT",
    "BackendComparison",
    "Evidence",
    "GateReport",
    "GateRequirement",
    "HorizonPlan",
    "MachineProfile",
    "MicrobatchProbe",
    "OperationsContractError",
    "OperationsNotReadyError",
    "ThroughputMeasurement",
    "UnmeasuredInputError",
    "active_gpu_hours",
    "approximate_flops",
    "assert_ready_for_horizon_claim",
    "backend_promotion_violations",
    "branch_time",
    "build_horizon_plan",
    "consumed_tokens",
    "effective_passes",
    "evaluate_all_gates",
    "evaluate_gate",
    "evaluation_reserve",
    "evaluation_time",
    "format_gate_report",
    "gate_requirements",
    "is_measured",
    "load_operations_protocol",
    "mainline_parent_time",
    "minimum_vram_headroom",
    "nearest_rank_percentile",
    "profile_violations",
    "readiness_results",
    "require_measured",
    "required_measurements",
    "safe_microbatch",
    "stretch_target_violations",
    "takeover_rehearsal_violations",
    "target_reference",
    "throughput_violations",
    "tokens_per_parameter",
    "wall_time",
]
