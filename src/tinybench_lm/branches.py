"""A/B/C exposure construction and the exact annealing schedule (Plan Sections 8.1-8.5).

The primary innovation experiment compares three arms that must differ in **exactly one**
thing each:

* **B versus A** isolates *data quality*. Plan Section 8.3: "At every B reserved-data
  position, A consumes the corresponding ``stable_control`` exposure; at every other position
  A and B consume the same ordered common-stable exposure." So A is B with the reserved
  sequences swapped, position for position, for disjoint stable sequences -- not an unrelated
  data draw.
* **C versus B** isolates *temporal placement*. Both consume the identical stable and
  reserved exposure multisets, including multiplicity; only the order differs. B holds a
  constant 128 stable / 128 reserved split in every 256-sequence update, C follows the
  linear anneal of Plan Section 8.4.

That is the whole point of this module, and it is why everything here is hash-checked rather
than merely constructed: a claim about annealing is only meaningful if the arms are provably
comparable. Three hashes make each intended difference visible and every unintended one
fatal:

``stable_exposure_hash`` / ``reserved_exposure_hash``
    order-independent hashes of the consumed reference multisets. Equal across B and C.
``training_order_hash``
    order-sensitive hash of the consumed reference sequence. Different across B and C.

The contract is backed by one frozen config::

    configs/branches/exposure_v1.yaml

Guarantees, mirroring :mod:`tinybench_lm.data_protocols`, :mod:`tinybench_lm.shards`, and
:mod:`tinybench_lm.schedule`:

1. **Immutable.** The config is verified against a pinned SHA-256 digest
   (:data:`FROZEN_BRANCH_PROTOCOL_SHA256`) on every load. Exposure lists and arm schedules
   are frozen dataclasses carrying content hashes. A post-hoc arm redefinition would convert
   a preregistered comparison into a chosen one, so the freeze is the integrity control.
2. **Exact.** :func:`reserved_in_update` is evaluated in integer arithmetic only. No float
   ever rounds, so the half-up cumulative formula is reproducible on any machine.
3. **Fail closed.** Unequal B/C multisets, identical B/C order hashes, overlapping exposure
   lists, unmatched A/B positions, a wrong per-update sequence count, a diverging shared
   policy, and insufficient supply all raise or report ``FAIL`` rather than degrade.
4. **Absence of evidence is never PASS.** No throughput has been measured, no branch size is
   selected, and no parent checkpoint hash exists. :func:`assert_ready_for_branch_runs` and
   :func:`select_branch_size` keep those ``NOT_RUN`` explicit instead of guessing a band.

Nothing here trains an arm, binds a parent hash, or observes an outcome. The append-only
parent-hash manifest is task 3.14's scope; this module only fixes the pending sentinel and
proves one set's arms declare the same parent binding.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data_protocols import (
    REPOSITORY_ROOT,
    ProtocolError,
    ProtocolNotReadyError,
    load_protocol,
)
from .environment import CheckResult
from .schedule import (
    MaterializedSchedule,
    ScheduleCursor,
    ScheduleEntry,
    canonical_payload_bytes,
    exposure_reference_hash,
    training_order_hash,
)
from .shards import DEFERRED, FAIL, NOT_RUN, PASS, RESERVED, STABLE_TRAIN

BRANCH_PROTOCOL_DIR = REPOSITORY_ROOT / "configs" / "branches"
BRANCH_PROTOCOL_PATH = BRANCH_PROTOCOL_DIR / "exposure_v1.yaml"

#: SHA-256 of the frozen exposure contract, over file bytes with CRLF normalized to LF.
FROZEN_BRANCH_PROTOCOL_SHA256: Mapping[str, str] = {
    "exposure_v1.yaml": "b6d688959b999e8c358012e4ccf639f8aea43c6f16735d0f90b276044e4f9a9d",
}

_BRANCH_SCHEMA_VERSION = "branch_exposure_v1"

ARM_A = "A"
ARM_B = "B"
ARM_C = "C"

#: Arms in the frozen declaration order. A is derived from B, so B is built first.
ARM_IDS: tuple[str, str, str] = (ARM_A, ARM_B, ARM_C)

COMMON_STABLE = "common_stable"
RESERVED_LIST = "reserved"
STABLE_CONTROL = "stable_control"

EXPOSURE_LIST_IDS: tuple[str, str, str] = (COMMON_STABLE, RESERVED_LIST, STABLE_CONTROL)

#: Plan Section 8.5: the parent checkpoint hash does not exist when the arms are frozen.
PENDING_PARENT_HASH = "PENDING_PARENT_HASH"

# --------------------------------------------------------------------------------------
# Reason-code vocabulary
# --------------------------------------------------------------------------------------

BRANCH_OK = "BRANCH_OK"
BRANCH_ARM_POLICY_DIVERGED = "BRANCH_ARM_POLICY_DIVERGED"
BRANCH_EXPOSURE_NOT_DISJOINT = "BRANCH_EXPOSURE_NOT_DISJOINT"
BRANCH_ORDER_HASH_IDENTICAL = "BRANCH_ORDER_HASH_IDENTICAL"
BRANCH_POSITION_NOT_MATCHED = "BRANCH_POSITION_NOT_MATCHED"
BRANCH_RESERVED_COUNT_MISMATCH = "BRANCH_RESERVED_COUNT_MISMATCH"
BRANCH_RESERVED_MULTISET_MISMATCH = "BRANCH_RESERVED_MULTISET_MISMATCH"
BRANCH_SEQUENCES_PER_UPDATE_MISMATCH = "BRANCH_SEQUENCES_PER_UPDATE_MISMATCH"
BRANCH_STABLE_MULTISET_MISMATCH = "BRANCH_STABLE_MULTISET_MISMATCH"
BRANCH_SUPPLY_EXHAUSTED = "BRANCH_SUPPLY_EXHAUSTED"
BRANCH_TOKENS_NOT_UPDATE_ALIGNED = "BRANCH_TOKENS_NOT_UPDATE_ALIGNED"
BRANCH_UPDATE_COUNT_INVALID = "BRANCH_UPDATE_COUNT_INVALID"
BRANCH_BOUNDARY_WRONG = "BRANCH_BOUNDARY_WRONG"
BRANCH_CONTENT_HASH_MISMATCH = "BRANCH_CONTENT_HASH_MISMATCH"

BRANCH_FAIL_CLOSED_REASON_CODES: frozenset[str] = frozenset(
    {
        BRANCH_ARM_POLICY_DIVERGED,
        BRANCH_BOUNDARY_WRONG,
        BRANCH_CONTENT_HASH_MISMATCH,
        BRANCH_EXPOSURE_NOT_DISJOINT,
        BRANCH_ORDER_HASH_IDENTICAL,
        BRANCH_POSITION_NOT_MATCHED,
        BRANCH_RESERVED_COUNT_MISMATCH,
        BRANCH_RESERVED_MULTISET_MISMATCH,
        BRANCH_SEQUENCES_PER_UPDATE_MISMATCH,
        BRANCH_STABLE_MULTISET_MISMATCH,
        BRANCH_SUPPLY_EXHAUSTED,
        BRANCH_TOKENS_NOT_UPDATE_ALIGNED,
        BRANCH_UPDATE_COUNT_INVALID,
    }
)


class BranchContractError(ProtocolError):
    """The frozen exposure contract is malformed, or an arm/exposure list violates it."""


class BranchExposureError(BranchContractError):
    """Two exposure lists overlap, or an arm consumes an exposure it must not."""


class BranchesNotReadyError(ProtocolNotReadyError):
    """A branch size, a parent hash, or an arm run needs evidence that does not exist yet."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def load_branch_protocol(
    path: Path = BRANCH_PROTOCOL_PATH, *, verify: bool = True
) -> dict[str, Any]:
    """Load the frozen exposure contract, verifying its pinned digest by default."""
    protocol = load_protocol(path, verify=verify, registry=FROZEN_BRANCH_PROTOCOL_SHA256)
    required = (
        "batch_layout",
        "arms",
        "exposure_lists",
        "identity",
        "annealing",
        "shared_policy",
        "branch_sizes",
        "parent_binding",
        "readiness",
    )
    for section in required:
        if section not in protocol:
            raise BranchContractError(f"branch protocol is missing required section {section!r}")

    layout = protocol["batch_layout"]
    sequences = int(layout["sequences_per_update"])
    length = int(layout["sequence_length"])
    if sequences % 2 != 0:
        raise BranchContractError(
            f"{BRANCH_SEQUENCES_PER_UPDATE_MISMATCH}: an even 50/50 split needs an even "
            f"sequences_per_update, got {sequences}"
        )
    if int(layout["loss_tokens_per_update"]) != sequences * length:
        raise BranchContractError(
            f"{BRANCH_SEQUENCES_PER_UPDATE_MISMATCH}: loss_tokens_per_update must equal "
            f"{sequences} x {length}"
        )
    if not bool(layout["identical_across_arms"]):
        raise BranchContractError("the batch layout must be identical across arms")

    declared = tuple(str(arm["arm_id"]) for arm in protocol["arms"])
    if declared != ARM_IDS:
        raise BranchContractError(f"branch protocol must declare arms {ARM_IDS}, found {declared}")
    arms = arm_index(protocol)
    stable_per_update = int(arms[ARM_B]["stable_sequences_per_update"])
    reserved_per_update = int(arms[ARM_B]["reserved_sequences_per_update"])
    if stable_per_update + reserved_per_update != sequences:
        raise BranchContractError(
            f"{BRANCH_SEQUENCES_PER_UPDATE_MISMATCH}: arm B declares "
            f"{stable_per_update} + {reserved_per_update} sequences, not {sequences}"
        )
    if stable_per_update != reserved_per_update:
        raise BranchContractError(
            f"arm B is a 50/50 split, so {stable_per_update} stable must equal "
            f"{reserved_per_update} reserved"
        )
    if float(arms[ARM_A]["reserved_share"]) != 0.0 or float(arms[ARM_A]["stable_share"]) != 1.0:
        raise BranchContractError("arm A is the 100% stable ordinary-decay control")
    if str(arms[ARM_A]["derives_from"]) != ARM_B:
        raise BranchContractError("arm A must be derived from arm B position for position")
    if str(arms[ARM_B]["reserved_placement"]) != "constant":
        raise BranchContractError("arm B places reserved exposure at a constant rate")
    if str(arms[ARM_C]["reserved_placement"]) != "linearly_annealed":
        raise BranchContractError("arm C places reserved exposure on the linear anneal")

    if str(protocol["exposure_lists"][COMMON_STABLE]["boundary"]) != STABLE_TRAIN:
        raise BranchContractError(f"{COMMON_STABLE} must be drawn from {STABLE_TRAIN!r}")
    if str(protocol["exposure_lists"][RESERVED_LIST]["boundary"]) != RESERVED:
        raise BranchContractError(f"{RESERVED_LIST} must be drawn from {RESERVED!r}")
    if str(protocol["exposure_lists"][STABLE_CONTROL]["boundary"]) != STABLE_TRAIN:
        raise BranchContractError(f"{STABLE_CONTROL} must be drawn from {STABLE_TRAIN!r}")
    if sorted(str(name) for name in protocol["exposure_lists"][STABLE_CONTROL]["disjoint_from"]) != [
        COMMON_STABLE,
        RESERVED_LIST,
    ]:
        raise BranchContractError(
            f"{STABLE_CONTROL} must be declared disjoint from {COMMON_STABLE} and {RESERVED_LIST}"
        )

    annealing = protocol["annealing"]
    if int(annealing["minimum_update_count"]) < 2:
        raise BranchContractError(
            f"{BRANCH_UPDATE_COUNT_INVALID}: the anneal divides by (K - 1), so K must be at least 2"
        )
    if str(annealing["rounding"]) != "nonnegative_half_up":
        raise BranchContractError("the frozen anneal uses exact nonnegative half-up rounding")
    for flag in (
        "update_0_all_stable",
        "update_K_minus_1_all_reserved",
        "nonnegative_per_update",
        "at_most_sequences_per_update",
    ):
        if not bool(annealing["invariants"][flag]):
            raise BranchContractError(f"the frozen anneal must assert {flag}")

    if not bool(protocol["shared_policy"]["learning_rate"]["identical_across_arms"]):
        raise BranchContractError("LR must be identical across arms")
    if str(protocol["parent_binding"]["pending_sentinel"]) != PENDING_PARENT_HASH:
        raise BranchContractError(
            f"the pending parent sentinel must be {PENDING_PARENT_HASH!r} until the parent exists"
        )
    if bool(protocol["parent_binding"]["hash_exists_at_freeze_time"]):
        raise BranchContractError("Plan Section 8.5: the parent hash does not exist at freeze time")
    if not bool(protocol["parent_binding"]["bind_before_any_arm_outcome_is_observed"]):
        raise BranchContractError("the parent hash must be bound before any arm outcome is observed")
    return protocol


def arm_index(protocol: Mapping[str, Any] | None = None) -> dict[str, Mapping[str, Any]]:
    """Arm ID -> its frozen declaration."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    return {str(arm["arm_id"]): arm for arm in resolved["arms"]}


def sequences_per_update(protocol: Mapping[str, Any] | None = None) -> int:
    """The 256 sequences every update consumes, identical for every arm."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    return int(resolved["batch_layout"]["sequences_per_update"])


def stable_sequences_per_update(protocol: Mapping[str, Any] | None = None) -> int:
    """The 128 stable sequences arm B consumes per update (Plan Section 8.4)."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    return int(arm_index(resolved)[ARM_B]["stable_sequences_per_update"])


def reserved_sequences_per_update(protocol: Mapping[str, Any] | None = None) -> int:
    """The 128 reserved sequences arm B consumes per update (Plan Section 8.4)."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    return int(arm_index(resolved)[ARM_B]["reserved_sequences_per_update"])


def assert_ready_for_branch_runs(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: an arm run needs a measured throughput and a bound parent hash."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    readiness = resolved["readiness"]
    blocked = [
        name
        for name in (
            "measured_3070_throughput",
            "selected_branch_size_band",
            "bound_parent_checkpoint_hash",
        )
        if str(readiness.get(name)) != PASS
    ]
    if blocked:
        raise BranchesNotReadyError(
            f"branch arms are not ready to run: {blocked}. "
            f"blocker={readiness.get('blocker')} owner={readiness.get('owner')} "
            f"next_action={readiness.get('next_action')}"
        )


# --------------------------------------------------------------------------------------
# The exact annealing schedule (Plan Section 8.4)
# --------------------------------------------------------------------------------------


def _validate_update_count(update_count: int, protocol: Mapping[str, Any] | None = None) -> int:
    resolved = protocol if protocol is not None else load_branch_protocol()
    minimum = int(resolved["annealing"]["minimum_update_count"])
    total = int(update_count)
    if total < minimum:
        raise BranchContractError(
            f"{BRANCH_UPDATE_COUNT_INVALID}: the anneal divides by (K - 1), so K must be at "
            f"least {minimum}, got {total}"
        )
    return total


def cumulative_reserved(
    k: int, update_count: int, *, sequences: int | None = None, protocol: Mapping[str, Any] | None = None
) -> int:
    """``floor(S x k x (k + 1) / (2 x (K - 1)) + 0.5)`` in exact integer arithmetic.

    Plan Section 8.4 writes the formula with a ``+ 0.5``; evaluating that in floating point
    would make the rounding boundary machine dependent for large ``K``. For integers
    ``n >= 0`` and ``d > 0``, ``floor(n / d + 1 / 2) == (2 * n + d) // (2 * d)`` exactly, so
    the whole schedule is computed without a single float.

    ``cumulative_reserved(-1) == 0`` is part of the frozen definition.
    """
    resolved = protocol if protocol is not None else load_branch_protocol()
    total = _validate_update_count(update_count, resolved)
    per_update = int(sequences_per_update(resolved) if sequences is None else sequences)
    index = int(k)
    if index < -1 or index > total - 1:
        raise BranchContractError(
            f"{BRANCH_UPDATE_COUNT_INVALID}: update index {index} is outside [-1, {total - 1}]"
        )
    if index < 0:
        return 0
    numerator = per_update * index * (index + 1)
    denominator = 2 * (total - 1)
    return (2 * numerator + denominator) // (2 * denominator)


def reserved_in_update(
    k: int, update_count: int, *, sequences: int | None = None, protocol: Mapping[str, Any] | None = None
) -> int:
    """Reserved sequences in update ``k``: ``cumulative(k) - cumulative(k - 1)``.

    Nonnegative by construction because ``k(k + 1)`` is increasing, zero at ``k == 0``, and
    exactly ``sequences_per_update`` at ``k == K - 1``.
    """
    resolved = protocol if protocol is not None else load_branch_protocol()
    return cumulative_reserved(k, update_count, sequences=sequences, protocol=resolved) - cumulative_reserved(
        int(k) - 1, update_count, sequences=sequences, protocol=resolved
    )


def annealed_reserved_schedule(
    update_count: int, *, sequences: int | None = None, protocol: Mapping[str, Any] | None = None
) -> tuple[int, ...]:
    """The whole arm-C anneal: ``reserved_in_update(k)`` for every ``k`` in ``[0, K)``."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    total = _validate_update_count(update_count, resolved)
    per_update = int(sequences_per_update(resolved) if sequences is None else sequences)
    cumulative = [
        cumulative_reserved(index, total, sequences=per_update, protocol=resolved) for index in range(total)
    ]
    previous = 0
    schedule: list[int] = []
    for value in cumulative:
        schedule.append(value - previous)
        previous = value
    return tuple(schedule)


def constant_reserved_schedule(
    update_count: int, *, protocol: Mapping[str, Any] | None = None
) -> tuple[int, ...]:
    """The whole arm-B split: a constant 128 reserved sequences in every update."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    total = _validate_update_count(update_count, resolved)
    return (reserved_sequences_per_update(resolved),) * total


def total_reserved_sequences(
    update_count: int, *, protocol: Mapping[str, Any] | None = None
) -> int:
    """``128 x K`` -- exactly half the branch, identical for arms B and C."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    return reserved_sequences_per_update(resolved) * _validate_update_count(update_count, resolved)


def branch_learning_rate(parent_lr: float, k: int, update_count: int) -> float:
    """Linear decay from the parent LR to zero, identical for every arm (Plan Section 8.4).

    Task 3.12 owns the mainline WSD recipe; this is only the branch decay shape the arms must
    share, exposed so an arm-level divergence can be detected rather than assumed away.
    """
    total = int(update_count)
    if total < 2:
        raise BranchContractError(f"{BRANCH_UPDATE_COUNT_INVALID}: branch LR decay needs K >= 2")
    index = int(k)
    if index < 0 or index > total - 1:
        raise BranchContractError(
            f"{BRANCH_UPDATE_COUNT_INVALID}: update index {index} is outside [0, {total - 1}]"
        )
    return float(parent_lr) * (total - 1 - index) / (total - 1)


# --------------------------------------------------------------------------------------
# Branch sizes (Plan Section 8.1)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchSizeBand:
    """One frozen measured-throughput band and the arm sizes it earns."""

    band_id: str
    measured_tps_minimum: int | None
    measured_tps_exclusive_maximum: int | None
    primary_tokens_per_arm: int
    primary_updates_per_arm: int
    confirmation_tokens_per_arm: int
    confirmation_updates_per_arm: int
    requires_calendar_reserve: bool = False

    def contains(self, measured_tps: float) -> bool:
        value = float(measured_tps)
        if self.measured_tps_minimum is not None and value < float(self.measured_tps_minimum):
            return False
        if self.measured_tps_exclusive_maximum is not None and value >= float(
            self.measured_tps_exclusive_maximum
        ):
            return False
        return True


def branch_size_bands(protocol: Mapping[str, Any] | None = None) -> tuple[BranchSizeBand, ...]:
    """The frozen Plan Section 8.1 table, in declaration order."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    bands: list[BranchSizeBand] = []
    for band in resolved["branch_sizes"]["bands"]:
        minimum = band.get("measured_tps_minimum")
        maximum = band.get("measured_tps_exclusive_maximum")
        bands.append(
            BranchSizeBand(
                band_id=str(band["band_id"]),
                measured_tps_minimum=None if minimum is None else int(minimum),
                measured_tps_exclusive_maximum=None if maximum is None else int(maximum),
                primary_tokens_per_arm=int(band["primary_tokens_per_arm"]),
                primary_updates_per_arm=int(band["primary_updates_per_arm"]),
                confirmation_tokens_per_arm=int(band["confirmation_tokens_per_arm"]),
                confirmation_updates_per_arm=int(band["confirmation_updates_per_arm"]),
                requires_calendar_reserve=bool(band.get("requires_calendar_reserve", False)),
            )
        )
    return tuple(bands)


def branch_size_alignment_problems(
    protocol: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Every frozen branch size whose token total is not ``updates x 256 x 1024``."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    per_update = int(resolved["batch_layout"]["loss_tokens_per_update"])
    problems: list[str] = []
    for band in branch_size_bands(resolved):
        for scope, tokens, updates in (
            ("primary", band.primary_tokens_per_arm, band.primary_updates_per_arm),
            ("confirmation", band.confirmation_tokens_per_arm, band.confirmation_updates_per_arm),
        ):
            expected = updates * per_update
            if tokens != expected:
                problems.append(
                    f"{BRANCH_TOKENS_NOT_UPDATE_ALIGNED}: {band.band_id} {scope} declares "
                    f"{tokens} tokens for {updates} updates, expected {expected}"
                )
    return tuple(problems)


def select_branch_size(
    measured_tps: float | None, *, protocol: Mapping[str, Any] | None = None
) -> BranchSizeBand:
    """Pick the band a **measured** 3070 throughput earns.

    ``None`` is not a band. No throughput has been measured, so this raises rather than
    quietly defaulting to the smallest branch -- an unmeasured band selection would put a
    fabricated horizon into the campaign plan.
    """
    resolved = protocol if protocol is not None else load_branch_protocol()
    if measured_tps is None:
        readiness = resolved["readiness"]
        raise BranchesNotReadyError(
            "a branch size requires a measured sustained 3070 throughput; "
            f"measured_3070_throughput={readiness['measured_3070_throughput']} "
            f"blocker={readiness['blocker']} owner={readiness['owner']} "
            f"next_action={readiness['next_action']}"
        )
    if float(measured_tps) <= 0:
        raise BranchContractError(f"a measured throughput must be positive, got {measured_tps}")
    for band in branch_size_bands(resolved):
        if band.contains(measured_tps):
            return band
    raise BranchContractError(f"no frozen branch-size band covers {measured_tps} tokens/s")


# --------------------------------------------------------------------------------------
# Shared policy: everything that must be identical across arms
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedBranchPolicy:
    """Parent, batch layout, LR, optimizer, and RNG policy -- identical across A/B/C.

    Plan Section 8.3: "A and B otherwise share parent, LR, batch layout, optimizer, and RNG
    policy." :meth:`fingerprint` reduces that to one comparable string so an arm-level
    divergence is a hash mismatch instead of a code review.
    """

    update_count: int
    parent_lr: float
    parent_binding_id: str = PENDING_PARENT_HASH
    sequences_per_update: int = 256
    sequence_length: int = 1024
    optimizer_policy: str = "adamw_groups_frozen_by_task_3_12"
    rng_policy: str = "identical_seed_and_stream_derivation_across_arms"

    def __post_init__(self) -> None:
        if int(self.update_count) < 2:
            raise BranchContractError(
                f"{BRANCH_UPDATE_COUNT_INVALID}: a branch needs at least 2 updates, got {self.update_count}"
            )
        if int(self.sequences_per_update) < 2 or int(self.sequences_per_update) % 2 != 0:
            raise BranchContractError(
                f"{BRANCH_SEQUENCES_PER_UPDATE_MISMATCH}: sequences_per_update must be even, "
                f"got {self.sequences_per_update}"
            )
        if float(self.parent_lr) <= 0:
            raise BranchContractError("the parent LR must be positive")

    @property
    def loss_tokens_per_update(self) -> int:
        return int(self.sequences_per_update) * int(self.sequence_length)

    @property
    def total_sequences(self) -> int:
        return int(self.sequences_per_update) * int(self.update_count)

    def learning_rate(self, k: int) -> float:
        return branch_learning_rate(self.parent_lr, k, self.update_count)

    def learning_rate_schedule(self) -> tuple[float, ...]:
        return tuple(self.learning_rate(index) for index in range(int(self.update_count)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_count": int(self.update_count),
            "parent_lr": float(self.parent_lr),
            "parent_binding_id": str(self.parent_binding_id),
            "sequences_per_update": int(self.sequences_per_update),
            "sequence_length": int(self.sequence_length),
            "optimizer_policy": str(self.optimizer_policy),
            "rng_policy": str(self.rng_policy),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SharedBranchPolicy":
        return cls(
            update_count=int(payload["update_count"]),
            parent_lr=float(payload["parent_lr"]),
            parent_binding_id=str(payload["parent_binding_id"]),
            sequences_per_update=int(payload["sequences_per_update"]),
            sequence_length=int(payload["sequence_length"]),
            optimizer_policy=str(payload["optimizer_policy"]),
            rng_policy=str(payload["rng_policy"]),
        )

    def fingerprint(self) -> str:
        """One hash over every field that must not differ between arms."""
        return hashlib.sha256(canonical_payload_bytes(self.to_dict())).hexdigest()


# --------------------------------------------------------------------------------------
# Exposure lists (Plan Section 8.3)
# --------------------------------------------------------------------------------------


def _references(entries: Iterable[ScheduleEntry]) -> list[tuple[str, int, int]]:
    return [entry.reference for entry in entries]


@dataclass(frozen=True)
class ExposureLists:
    """The three fixed exposure lists every arm is built from.

    ``common_stable`` and ``reserved`` are consumed by B and C in the same order at different
    positions. ``stable_control`` is the disjoint stable list arm A substitutes at B's
    reserved positions, and it holds exactly as many sequences as ``reserved``.
    """

    update_count: int
    sequences_per_update: int
    stable_per_update: int
    reserved_per_update: int
    common_stable: tuple[ScheduleEntry, ...]
    reserved: tuple[ScheduleEntry, ...]
    stable_control: tuple[ScheduleEntry, ...]
    stable_schedule_hash: str
    reserved_schedule_hash: str
    protocol_digest: str
    schema_version: str = _BRANCH_SCHEMA_VERSION

    # -- derived identity ---------------------------------------------------------------

    @property
    def stable_exposure_hash(self) -> str:
        """``hash(sorted(reference, multiplicity))`` over the common-stable list."""
        return exposure_reference_hash(self.common_stable)

    @property
    def reserved_exposure_hash(self) -> str:
        return exposure_reference_hash(self.reserved)

    @property
    def stable_control_exposure_hash(self) -> str:
        return exposure_reference_hash(self.stable_control)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "update_count": int(self.update_count),
            "sequences_per_update": int(self.sequences_per_update),
            "stable_per_update": int(self.stable_per_update),
            "reserved_per_update": int(self.reserved_per_update),
            "stable_schedule_hash": self.stable_schedule_hash,
            "reserved_schedule_hash": self.reserved_schedule_hash,
            "protocol_digest": self.protocol_digest,
            "stable_exposure_hash": self.stable_exposure_hash,
            "reserved_exposure_hash": self.reserved_exposure_hash,
            "stable_control_exposure_hash": self.stable_control_exposure_hash,
            COMMON_STABLE: [entry.to_dict() for entry in self.common_stable],
            RESERVED_LIST: [entry.to_dict() for entry in self.reserved],
            STABLE_CONTROL: [entry.to_dict() for entry in self.stable_control],
        }

    def content_hash(self) -> str:
        return hashlib.sha256(canonical_payload_bytes(self.payload())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload()
        payload["content_hash"] = self.content_hash()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExposureLists":
        return cls(
            update_count=int(payload["update_count"]),
            sequences_per_update=int(payload["sequences_per_update"]),
            stable_per_update=int(payload["stable_per_update"]),
            reserved_per_update=int(payload["reserved_per_update"]),
            common_stable=tuple(ScheduleEntry.from_dict(item) for item in payload[COMMON_STABLE]),
            reserved=tuple(ScheduleEntry.from_dict(item) for item in payload[RESERVED_LIST]),
            stable_control=tuple(ScheduleEntry.from_dict(item) for item in payload[STABLE_CONTROL]),
            stable_schedule_hash=str(payload["stable_schedule_hash"]),
            reserved_schedule_hash=str(payload["reserved_schedule_hash"]),
            protocol_digest=str(payload["protocol_digest"]),
            schema_version=str(payload.get("schema_version", _BRANCH_SCHEMA_VERSION)),
        )


def disjointness_problems(lists: ExposureLists) -> tuple[str, ...]:
    """Every overlap that would break the A-versus-B replacement or leak reserved data."""
    common = set(_references(lists.common_stable))
    control = set(_references(lists.stable_control))
    reserved = set(_references(lists.reserved))
    problems: list[str] = []
    for left_name, left, right_name, right in (
        (COMMON_STABLE, common, STABLE_CONTROL, control),
        (COMMON_STABLE, common, RESERVED_LIST, reserved),
        (STABLE_CONTROL, control, RESERVED_LIST, reserved),
    ):
        shared = sorted(left & right)
        if shared:
            problems.append(
                f"{BRANCH_EXPOSURE_NOT_DISJOINT}: {left_name} and {right_name} share "
                f"{len(shared)} references, first {shared[0]}"
            )
    for name, entries in (
        (COMMON_STABLE, lists.common_stable),
        (STABLE_CONTROL, lists.stable_control),
        (RESERVED_LIST, lists.reserved),
    ):
        references = _references(entries)
        if len(set(references)) != len(references):
            problems.append(
                f"{BRANCH_EXPOSURE_NOT_DISJOINT}: {name} repeats a reference, so its "
                "multiplicity is not one and A/B position matching is ambiguous"
            )
    return tuple(problems)


def build_exposure_lists(
    stable_schedule: MaterializedSchedule,
    reserved_schedule: MaterializedSchedule,
    *,
    update_count: int,
    protocol: Mapping[str, Any] | None = None,
) -> ExposureLists:
    """Materialize ``common_stable``, disjoint ``stable_control``, and ``reserved``.

    The lists are prefixes of two materialized schedules, so the whole construction inherits
    task 3.10's determinism: identical manifests and seeds give identical exposure lists and
    therefore identical exposure hashes. ``common_stable`` takes the first ``128 x K``
    stable references and ``stable_control`` the next ``128 x K``, which makes them disjoint
    by construction while both come from the same stable mixture -- so A's substitute data is
    the same *kind* of data, not a differently mixed draw.
    """
    resolved = protocol if protocol is not None else load_branch_protocol()
    total_updates = _validate_update_count(update_count, resolved)
    per_update = sequences_per_update(resolved)
    stable_per = stable_sequences_per_update(resolved)
    reserved_per = reserved_sequences_per_update(resolved)

    if stable_schedule.boundary != STABLE_TRAIN:
        raise BranchContractError(
            f"{BRANCH_BOUNDARY_WRONG}: the stable exposure schedule is {stable_schedule.boundary!r}, "
            f"expected {STABLE_TRAIN!r}"
        )
    if reserved_schedule.boundary != RESERVED:
        raise BranchContractError(
            f"{BRANCH_BOUNDARY_WRONG}: the reserved exposure schedule is {reserved_schedule.boundary!r}, "
            f"expected {RESERVED!r}"
        )
    if stable_schedule.sequence_length != reserved_schedule.sequence_length:
        raise BranchContractError(
            f"{BRANCH_SEQUENCES_PER_UPDATE_MISMATCH}: the arms need one batch layout, but the "
            f"stable schedule is {stable_schedule.sequence_length} tokens and the reserved "
            f"schedule is {reserved_schedule.sequence_length}"
        )

    stable_needed = stable_per * total_updates * 2  # common_stable + stable_control
    reserved_needed = reserved_per * total_updates
    if stable_schedule.sequence_count < stable_needed:
        raise BranchContractError(
            f"{BRANCH_SUPPLY_EXHAUSTED}: {total_updates} updates need {stable_needed} stable "
            f"sequences ({COMMON_STABLE} plus disjoint {STABLE_CONTROL}), the schedule holds "
            f"{stable_schedule.sequence_count}"
        )
    if reserved_schedule.sequence_count < reserved_needed:
        raise BranchContractError(
            f"{BRANCH_SUPPLY_EXHAUSTED}: {total_updates} updates need {reserved_needed} reserved "
            f"sequences, the schedule holds {reserved_schedule.sequence_count}"
        )

    common = stable_schedule.entries[:stable_needed // 2]
    control = stable_schedule.entries[stable_needed // 2 : stable_needed]
    reserved = reserved_schedule.entries[:reserved_needed]

    lists = ExposureLists(
        update_count=total_updates,
        sequences_per_update=per_update,
        stable_per_update=stable_per,
        reserved_per_update=reserved_per,
        common_stable=tuple(common),
        reserved=tuple(reserved),
        stable_control=tuple(control),
        stable_schedule_hash=stable_schedule.content_hash(),
        reserved_schedule_hash=reserved_schedule.content_hash(),
        protocol_digest=str(resolved.get("_digest", "")),
    )
    problems = disjointness_problems(lists)
    if problems:
        raise BranchExposureError("; ".join(problems))
    return lists


# --------------------------------------------------------------------------------------
# Arm schedules
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureSlot:
    """One consumed sequence: where in the branch it sits and which list it came from."""

    update_index: int
    position_in_update: int
    list_id: str
    list_index: int
    entry: ScheduleEntry

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_index": int(self.update_index),
            "position_in_update": int(self.position_in_update),
            "list_id": str(self.list_id),
            "list_index": int(self.list_index),
            "entry": self.entry.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExposureSlot":
        return cls(
            update_index=int(payload["update_index"]),
            position_in_update=int(payload["position_in_update"]),
            list_id=str(payload["list_id"]),
            list_index=int(payload["list_index"]),
            entry=ScheduleEntry.from_dict(payload["entry"]),
        )


@dataclass(frozen=True)
class ArmSchedule:
    """One arm's complete ordered exposure, with the hashes Plan Section 8.3 requires."""

    arm_id: str
    slots: tuple[ExposureSlot, ...]
    policy: SharedBranchPolicy
    exposure_lists_hash: str
    protocol_digest: str
    schema_version: str = _BRANCH_SCHEMA_VERSION

    # -- derived views ------------------------------------------------------------------

    @property
    def entries(self) -> tuple[ScheduleEntry, ...]:
        return tuple(slot.entry for slot in self.slots)

    def entries_from(self, list_id: str) -> tuple[ScheduleEntry, ...]:
        return tuple(slot.entry for slot in self.slots if slot.list_id == list_id)

    @property
    def reserved_positions(self) -> tuple[int, ...]:
        """Global slot indices where this arm consumes reserved data."""
        return tuple(index for index, slot in enumerate(self.slots) if slot.list_id == RESERVED_LIST)

    @property
    def reserved_sequence_count(self) -> int:
        return len(self.reserved_positions)

    @property
    def stable_sequence_count(self) -> int:
        return len(self.entries_from(COMMON_STABLE))

    @property
    def stable_control_sequence_count(self) -> int:
        return len(self.entries_from(STABLE_CONTROL))

    def sequences_in_update(self, k: int) -> int:
        return sum(1 for slot in self.slots if slot.update_index == int(k))

    def reserved_in_update(self, k: int) -> int:
        return sum(
            1 for slot in self.slots if slot.update_index == int(k) and slot.list_id == RESERVED_LIST
        )

    @property
    def reserved_per_update(self) -> tuple[int, ...]:
        return tuple(self.reserved_in_update(index) for index in range(int(self.policy.update_count)))

    @property
    def sequences_per_update_observed(self) -> tuple[int, ...]:
        return tuple(self.sequences_in_update(index) for index in range(int(self.policy.update_count)))

    # -- identity ----------------------------------------------------------------------

    @property
    def stable_exposure_hash(self) -> str:
        """Order-independent hash of the consumed common-stable multiset."""
        return exposure_reference_hash(self.entries_from(COMMON_STABLE))

    @property
    def reserved_exposure_hash(self) -> str:
        """Order-independent hash of the consumed reserved multiset."""
        return exposure_reference_hash(self.entries_from(RESERVED_LIST))

    @property
    def stable_control_exposure_hash(self) -> str:
        """Order-independent hash of the consumed ``stable_control`` multiset."""
        return exposure_reference_hash(self.entries_from(STABLE_CONTROL))

    @property
    def training_order_hash(self) -> str:
        """Order-sensitive hash of the whole consumed sequence: temporal placement."""
        return training_order_hash(self.entries)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arm_id": self.arm_id,
            "policy": self.policy.to_dict(),
            "policy_fingerprint": self.policy.fingerprint(),
            "exposure_lists_hash": self.exposure_lists_hash,
            "protocol_digest": self.protocol_digest,
            "stable_exposure_hash": self.stable_exposure_hash,
            "reserved_exposure_hash": self.reserved_exposure_hash,
            "stable_control_exposure_hash": self.stable_control_exposure_hash,
            "training_order_hash": self.training_order_hash,
            "sequence_count": len(self.slots),
            "slots": [slot.to_dict() for slot in self.slots],
        }

    def content_hash(self) -> str:
        return hashlib.sha256(canonical_payload_bytes(self.payload())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload()
        payload["content_hash"] = self.content_hash()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArmSchedule":
        return cls(
            arm_id=str(payload["arm_id"]),
            slots=tuple(ExposureSlot.from_dict(item) for item in payload["slots"]),
            policy=SharedBranchPolicy.from_dict(payload["policy"]),
            exposure_lists_hash=str(payload["exposure_lists_hash"]),
            protocol_digest=str(payload["protocol_digest"]),
            schema_version=str(payload.get("schema_version", _BRANCH_SCHEMA_VERSION)),
        )

    def cursor(self, position: int = 0) -> ScheduleCursor:
        """One integer cursor bound to this arm's content hash, as task 3.10 defined it."""
        return ScheduleCursor(self.content_hash(), position, len(self.slots))


def _layout_slots(
    reserved_counts: Sequence[int],
    lists: ExposureLists,
) -> tuple[ExposureSlot, ...]:
    """Place stable then reserved sequences inside each update, consuming lists in order.

    Placement inside an update is stable-first and deterministic. Only the *number* of
    reserved sequences per update differs between B and C, which is exactly the temporal
    difference Plan Section 8.2 wants to measure.
    """
    slots: list[ExposureSlot] = []
    stable_index = 0
    reserved_index = 0
    for update_index, reserved_count in enumerate(reserved_counts):
        stable_count = int(lists.sequences_per_update) - int(reserved_count)
        if stable_count < 0:
            raise BranchContractError(
                f"{BRANCH_RESERVED_COUNT_MISMATCH}: update {update_index} asks for "
                f"{reserved_count} reserved sequences of {lists.sequences_per_update}"
            )
        for position in range(stable_count):
            if stable_index >= len(lists.common_stable):
                raise BranchContractError(
                    f"{BRANCH_SUPPLY_EXHAUSTED}: {COMMON_STABLE} holds "
                    f"{len(lists.common_stable)} sequences, update {update_index} needs more"
                )
            slots.append(
                ExposureSlot(update_index, position, COMMON_STABLE, stable_index, lists.common_stable[stable_index])
            )
            stable_index += 1
        for offset in range(int(reserved_count)):
            if reserved_index >= len(lists.reserved):
                raise BranchContractError(
                    f"{BRANCH_SUPPLY_EXHAUSTED}: {RESERVED_LIST} holds "
                    f"{len(lists.reserved)} sequences, update {update_index} needs more"
                )
            slots.append(
                ExposureSlot(
                    update_index,
                    stable_count + offset,
                    RESERVED_LIST,
                    reserved_index,
                    lists.reserved[reserved_index],
                )
            )
            reserved_index += 1
    return tuple(slots)


def build_arm_b(lists: ExposureLists, policy: SharedBranchPolicy, *, protocol: Mapping[str, Any] | None = None) -> ArmSchedule:
    """Arm B: a constant 128 stable / 128 reserved split in every update."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    _assert_policy_matches_lists(policy, lists)
    counts = constant_reserved_schedule(lists.update_count, protocol=resolved)
    return ArmSchedule(
        arm_id=ARM_B,
        slots=_layout_slots(counts, lists),
        policy=policy,
        exposure_lists_hash=lists.content_hash(),
        protocol_digest=str(resolved.get("_digest", "")),
    )


def build_arm_c(lists: ExposureLists, policy: SharedBranchPolicy, *, protocol: Mapping[str, Any] | None = None) -> ArmSchedule:
    """Arm C: the same multisets as B, placed on the exact Plan 8.4 linear anneal."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    _assert_policy_matches_lists(policy, lists)
    counts = annealed_reserved_schedule(
        lists.update_count, sequences=lists.sequences_per_update, protocol=resolved
    )
    return ArmSchedule(
        arm_id=ARM_C,
        slots=_layout_slots(counts, lists),
        policy=policy,
        exposure_lists_hash=lists.content_hash(),
        protocol_digest=str(resolved.get("_digest", "")),
    )


def build_arm_a(
    arm_b: ArmSchedule, lists: ExposureLists, *, protocol: Mapping[str, Any] | None = None
) -> ArmSchedule:
    """Arm A: arm B with every reserved slot replaced by the matching ``stable_control``.

    Position matching is literal -- same update index, same position inside the update, same
    slot order. Only the data at B's reserved positions changes, so ``delta_H1 = NLL_B -
    NLL_A`` measures a stable-to-reserved replacement rather than an unrelated draw.
    """
    resolved = protocol if protocol is not None else load_branch_protocol()
    if arm_b.arm_id != ARM_B:
        raise BranchContractError(f"arm A derives from arm B, not {arm_b.arm_id!r}")
    if arm_b.exposure_lists_hash != lists.content_hash():
        raise BranchContractError(
            f"{BRANCH_CONTENT_HASH_MISMATCH}: arm B was built over exposure lists "
            f"{arm_b.exposure_lists_hash!r}, not {lists.content_hash()!r}"
        )
    slots: list[ExposureSlot] = []
    for slot in arm_b.slots:
        if slot.list_id != RESERVED_LIST:
            slots.append(slot)
            continue
        if slot.list_index >= len(lists.stable_control):
            raise BranchContractError(
                f"{BRANCH_SUPPLY_EXHAUSTED}: {STABLE_CONTROL} holds {len(lists.stable_control)} "
                f"sequences but arm B reserved position {slot.list_index} needs a replacement"
            )
        slots.append(
            ExposureSlot(
                slot.update_index,
                slot.position_in_update,
                STABLE_CONTROL,
                slot.list_index,
                lists.stable_control[slot.list_index],
            )
        )
    return ArmSchedule(
        arm_id=ARM_A,
        slots=tuple(slots),
        policy=arm_b.policy,
        exposure_lists_hash=lists.content_hash(),
        protocol_digest=str(resolved.get("_digest", "")),
    )


def _assert_policy_matches_lists(policy: SharedBranchPolicy, lists: ExposureLists) -> None:
    if int(policy.update_count) != int(lists.update_count):
        raise BranchContractError(
            f"{BRANCH_UPDATE_COUNT_INVALID}: the policy declares {policy.update_count} updates, "
            f"the exposure lists were built for {lists.update_count}"
        )
    if int(policy.sequences_per_update) != int(lists.sequences_per_update):
        raise BranchContractError(
            f"{BRANCH_SEQUENCES_PER_UPDATE_MISMATCH}: the policy declares "
            f"{policy.sequences_per_update} sequences/update, the exposure lists "
            f"{lists.sequences_per_update}"
        )


def build_arm_schedules(
    lists: ExposureLists, policy: SharedBranchPolicy, *, protocol: Mapping[str, Any] | None = None
) -> dict[str, ArmSchedule]:
    """Build all three arms from one exposure-list set and one shared policy."""
    resolved = protocol if protocol is not None else load_branch_protocol()
    arm_b = build_arm_b(lists, policy, protocol=resolved)
    arm_c = build_arm_c(lists, policy, protocol=resolved)
    arm_a = build_arm_a(arm_b, lists, protocol=resolved)
    return {ARM_A: arm_a, ARM_B: arm_b, ARM_C: arm_c}


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def write_exposure_lists(path: Path, lists: ExposureLists) -> Path:
    """Write the fixed exposure lists as canonical JSON, content hash included."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(canonical_payload_bytes(lists.to_dict()))
        handle.write(b"\n")
    return target


def load_exposure_lists(path: Path) -> ExposureLists:
    """Load exposure lists, failing closed when the recorded content hash does not match."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = payload.pop("content_hash", None)
    lists = ExposureLists.from_dict(payload)
    if recorded is None:
        raise BranchContractError(f"{BRANCH_CONTENT_HASH_MISMATCH}: {path} records no content hash")
    if recorded != lists.content_hash():
        raise BranchContractError(f"{BRANCH_CONTENT_HASH_MISMATCH}: {path} does not match its payload")
    return lists


def write_arm_schedule(path: Path, arm: ArmSchedule) -> Path:
    """Write one arm schedule as canonical JSON, content hash included."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(canonical_payload_bytes(arm.to_dict()))
        handle.write(b"\n")
    return target


def load_arm_schedule(path: Path) -> ArmSchedule:
    """Load one arm schedule, failing closed on content-hash drift."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = payload.pop("content_hash", None)
    arm = ArmSchedule.from_dict(payload)
    if recorded is None:
        raise BranchContractError(f"{BRANCH_CONTENT_HASH_MISMATCH}: {path} records no content hash")
    if recorded != arm.content_hash():
        raise BranchContractError(f"{BRANCH_CONTENT_HASH_MISMATCH}: {path} does not match its payload")
    return arm


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


def _verdict(check_id: str, requirement: str, observed: Any, ok: bool, reason: str) -> CheckResult:
    return CheckResult(check_id, requirement, str(observed), PASS if ok else FAIL, reason)


def position_matching_problems(arm_a: ArmSchedule, arm_b: ArmSchedule) -> tuple[str, ...]:
    """Every way A stops being B-with-a-substitution."""
    problems: list[str] = []
    if len(arm_a.slots) != len(arm_b.slots):
        problems.append(
            f"{BRANCH_POSITION_NOT_MATCHED}: A holds {len(arm_a.slots)} slots, B holds {len(arm_b.slots)}"
        )
        return tuple(problems)
    for index, (left, right) in enumerate(zip(arm_a.slots, arm_b.slots)):
        if (left.update_index, left.position_in_update) != (right.update_index, right.position_in_update):
            problems.append(
                f"{BRANCH_POSITION_NOT_MATCHED}: slot {index} sits at "
                f"{(left.update_index, left.position_in_update)} in A and "
                f"{(right.update_index, right.position_in_update)} in B"
            )
            continue
        if right.list_id == RESERVED_LIST:
            if left.list_id != STABLE_CONTROL or left.list_index != right.list_index:
                problems.append(
                    f"{BRANCH_POSITION_NOT_MATCHED}: B reserved slot {index} "
                    f"(list index {right.list_index}) is answered by A with "
                    f"{left.list_id}[{left.list_index}]"
                )
        elif left.list_id != right.list_id or left.entry != right.entry:
            problems.append(
                f"{BRANCH_POSITION_NOT_MATCHED}: non-reserved slot {index} differs -- "
                f"A {left.list_id}[{left.list_index}] vs B {right.list_id}[{right.list_index}]"
            )
    return tuple(problems)


def verify_branch_arms(
    arms: Mapping[str, ArmSchedule],
    lists: ExposureLists,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[CheckResult, ...]:
    """Reconcile the three arms against Plan Sections 8.1-8.5 and the frozen contract.

    Every check that could mask a non-comparable experiment is a ``FAIL``, not a warning.
    The two unmeasurable prerequisites -- a measured throughput and a bound parent hash --
    are reported ``DEFERRED`` with their blocker, never ``PASS``.
    """
    resolved = protocol if protocol is not None else load_branch_protocol()
    results: list[CheckResult] = []

    missing = [arm_id for arm_id in ARM_IDS if arm_id not in arms]
    results.append(
        _verdict(
            "branch.arms_present",
            f"one schedule per arm {ARM_IDS}",
            sorted(arms) if not missing else f"missing {missing}",
            not missing,
            "an A/B/C set needs all three arms to be comparable",
        )
    )
    if missing:
        return tuple(results)

    arm_a, arm_b, arm_c = arms[ARM_A], arms[ARM_B], arms[ARM_C]
    total_updates = int(lists.update_count)
    per_update = int(lists.sequences_per_update)
    expected_reserved = int(lists.reserved_per_update) * total_updates

    results.append(
        _verdict(
            "branch.exposure_lists_binding",
            lists.content_hash(),
            sorted({arm.exposure_lists_hash for arm in arms.values()}),
            all(arm.exposure_lists_hash == lists.content_hash() for arm in arms.values()),
            f"{BRANCH_CONTENT_HASH_MISMATCH} unless every arm names the exposure lists it was built from",
        )
    )
    results.append(
        _verdict(
            "branch.protocol_binding",
            str(resolved.get("_digest", "")),
            sorted({arm.protocol_digest for arm in arms.values()}),
            all(arm.protocol_digest == str(resolved.get("_digest", "")) for arm in arms.values()),
            f"{BRANCH_CONTENT_HASH_MISMATCH} unless every arm names the frozen contract it obeyed",
        )
    )

    disjoint = disjointness_problems(lists)
    results.append(
        _verdict(
            "branch.exposure_lists_disjoint",
            f"{COMMON_STABLE}, {STABLE_CONTROL}, and {RESERVED_LIST} pairwise disjoint",
            disjoint[:3] or "disjoint",
            not disjoint,
            f"{BRANCH_EXPOSURE_NOT_DISJOINT} unless A's substitute data is genuinely held out of B",
        )
    )
    results.append(
        _verdict(
            "branch.stable_control_matches_reserved_size",
            expected_reserved,
            len(lists.stable_control),
            len(lists.stable_control) == expected_reserved,
            f"{BRANCH_POSITION_NOT_MATCHED} unless every B reserved position has one A replacement",
        )
    )

    # -- shared policy: parent, LR, batch layout, optimizer, RNG ------------------------
    fingerprints = sorted({arm.policy.fingerprint() for arm in arms.values()})
    results.append(
        _verdict(
            "branch.shared_policy_identical",
            "one policy fingerprint across A/B/C",
            fingerprints,
            len(fingerprints) == 1,
            f"{BRANCH_ARM_POLICY_DIVERGED} unless parent, LR, batch layout, optimizer, and RNG match",
        )
    )
    lr_schedules = sorted({arm.policy.learning_rate_schedule() for arm in arms.values()})
    results.append(
        _verdict(
            "branch.lr_identical_across_arms",
            "one linear parent-LR-to-zero schedule across A/B/C",
            len(lr_schedules),
            len(lr_schedules) == 1 and lr_schedules[0][0] > 0.0 and lr_schedules[0][-1] == 0.0,
            f"{BRANCH_ARM_POLICY_DIVERGED} unless every arm decays the same LR to zero",
        )
    )
    parents = sorted({arm.policy.parent_binding_id for arm in arms.values()})
    results.append(
        _verdict(
            "branch.common_parent_binding",
            "one parent binding across A/B/C",
            parents,
            len(parents) == 1,
            f"{BRANCH_ARM_POLICY_DIVERGED} unless all three arms branch from one parent",
        )
    )

    # -- batch layout -------------------------------------------------------------------
    layout_ok = all(
        set(arm.sequences_per_update_observed) == {per_update}
        and len(arm.sequences_per_update_observed) == total_updates
        for arm in arms.values()
    )
    results.append(
        _verdict(
            "branch.sequences_per_update",
            f"{per_update} sequences in each of {total_updates} updates, every arm",
            {arm_id: sorted(set(arm.sequences_per_update_observed)) for arm_id, arm in sorted(arms.items())},
            layout_ok,
            f"{BRANCH_SEQUENCES_PER_UPDATE_MISMATCH} unless the batch layout is identical across arms",
        )
    )

    # -- B/C identity (Plan Section 8.3) ------------------------------------------------
    results.append(
        _verdict(
            "branch.bc_stable_exposure_hash_equal",
            "B.stable_exposure_hash == C.stable_exposure_hash",
            (arm_b.stable_exposure_hash[:16], arm_c.stable_exposure_hash[:16]),
            arm_b.stable_exposure_hash == arm_c.stable_exposure_hash,
            f"{BRANCH_STABLE_MULTISET_MISMATCH} unless B and C consume the same stable multiset",
        )
    )
    results.append(
        _verdict(
            "branch.bc_reserved_exposure_hash_equal",
            "B.reserved_exposure_hash == C.reserved_exposure_hash",
            (arm_b.reserved_exposure_hash[:16], arm_c.reserved_exposure_hash[:16]),
            arm_b.reserved_exposure_hash == arm_c.reserved_exposure_hash,
            f"{BRANCH_RESERVED_MULTISET_MISMATCH} unless B and C consume the same reserved multiset",
        )
    )
    results.append(
        _verdict(
            "branch.bc_training_order_hash_differs",
            "B.training_order_hash != C.training_order_hash",
            (arm_b.training_order_hash[:16], arm_c.training_order_hash[:16]),
            arm_b.training_order_hash != arm_c.training_order_hash,
            f"{BRANCH_ORDER_HASH_IDENTICAL} would mean C is not a different temporal placement",
        )
    )

    # -- A/B comparability (Plan Section 8.3) ------------------------------------------
    results.append(
        _verdict(
            "branch.ab_stable_exposure_hash_equal",
            "A.stable_exposure_hash == B.stable_exposure_hash",
            (arm_a.stable_exposure_hash[:16], arm_b.stable_exposure_hash[:16]),
            arm_a.stable_exposure_hash == arm_b.stable_exposure_hash,
            f"{BRANCH_STABLE_MULTISET_MISMATCH} unless A and B share the ordered common-stable exposure",
        )
    )
    results.append(
        _verdict(
            "branch.a_consumes_no_reserved",
            "A.reserved_sequence_count == 0",
            arm_a.reserved_sequence_count,
            arm_a.reserved_sequence_count == 0,
            "arm A is the 100% stable ordinary-decay control",
        )
    )
    matching = position_matching_problems(arm_a, arm_b)
    results.append(
        _verdict(
            "branch.ab_position_matched_replacement",
            f"every B reserved position answered by the matching {STABLE_CONTROL} sequence",
            matching[:3] or "position matched",
            not matching,
            f"{BRANCH_POSITION_NOT_MATCHED} would make B-vs-A an unrelated data draw",
        )
    )
    results.append(
        _verdict(
            "branch.ab_training_order_hash_differs",
            "A.training_order_hash != B.training_order_hash",
            (arm_a.training_order_hash[:16], arm_b.training_order_hash[:16]),
            arm_a.training_order_hash != arm_b.training_order_hash,
            f"{BRANCH_ORDER_HASH_IDENTICAL} would mean the reserved replacement did not happen",
        )
    )
    results.append(
        _verdict(
            "branch.a_stable_control_count",
            f"A consumes {expected_reserved} {STABLE_CONTROL} sequences",
            arm_a.stable_control_sequence_count,
            arm_a.stable_control_sequence_count == expected_reserved,
            f"{BRANCH_POSITION_NOT_MATCHED} unless A replaces every reserved position exactly once",
        )
    )

    # -- the exact anneal (Plan Section 8.4) -------------------------------------------
    expected_anneal = annealed_reserved_schedule(
        total_updates, sequences=per_update, protocol=resolved
    )
    observed_anneal = arm_c.reserved_per_update
    results.append(
        _verdict(
            "branch.c_follows_exact_anneal",
            "reserved_in_update(k) for every k",
            observed_anneal[:4] + ("...",) + observed_anneal[-2:] if total_updates > 6 else observed_anneal,
            observed_anneal == expected_anneal,
            f"{BRANCH_RESERVED_COUNT_MISMATCH} unless C matches the frozen half-up cumulative formula",
        )
    )
    results.append(
        _verdict(
            "branch.anneal_nonnegative",
            "reserved_in_update(k) >= 0 for every k",
            min(observed_anneal),
            min(observed_anneal) >= 0,
            f"{BRANCH_RESERVED_COUNT_MISMATCH} on a negative reserved count",
        )
    )
    results.append(
        _verdict(
            "branch.anneal_update_zero_all_stable",
            "reserved_in_update(0) == 0",
            observed_anneal[0],
            observed_anneal[0] == 0,
            f"{BRANCH_RESERVED_COUNT_MISMATCH} unless the anneal starts at 0% reserved",
        )
    )
    results.append(
        _verdict(
            "branch.anneal_final_update_all_reserved",
            f"reserved_in_update(K - 1) == {per_update}",
            observed_anneal[-1],
            observed_anneal[-1] == per_update,
            f"{BRANCH_RESERVED_COUNT_MISMATCH} unless the anneal ends at 100% reserved",
        )
    )
    results.append(
        _verdict(
            "branch.reserved_total_is_half_the_branch",
            f"{lists.reserved_per_update} x {total_updates} = {expected_reserved} reserved sequences, arms B and C",
            {ARM_B: arm_b.reserved_sequence_count, ARM_C: arm_c.reserved_sequence_count},
            arm_b.reserved_sequence_count == expected_reserved
            and arm_c.reserved_sequence_count == expected_reserved,
            f"{BRANCH_RESERVED_COUNT_MISMATCH} unless both arms consume exactly half reserved data",
        )
    )
    results.append(
        _verdict(
            "branch.b_constant_split",
            f"{lists.reserved_per_update} reserved in every update",
            sorted(set(arm_b.reserved_per_update)),
            set(arm_b.reserved_per_update) == {int(lists.reserved_per_update)},
            f"{BRANCH_RESERVED_COUNT_MISMATCH} unless B holds a constant 50/50 split",
        )
    )

    # -- frozen branch sizes (Plan Section 8.1) ----------------------------------------
    alignment = branch_size_alignment_problems(resolved)
    results.append(
        _verdict(
            "branch.frozen_sizes_update_aligned",
            "every frozen branch size equals updates x loss tokens per update",
            alignment[:3] or "aligned",
            not alignment,
            f"{BRANCH_TOKENS_NOT_UPDATE_ALIGNED} unless a branch ends on an accumulation boundary",
        )
    )

    # -- unmeasurable prerequisites stay explicit -------------------------------------
    readiness = resolved["readiness"]
    for check_id, key, requirement in (
        ("branch.measured_throughput", "measured_3070_throughput", "measured sustained 3070 tokens/s"),
        ("branch.selected_size_band", "selected_branch_size_band", "branch size selected from a measured band"),
        ("branch.bound_parent_hash", "bound_parent_checkpoint_hash", "parent checkpoint hash bound before any arm runs"),
        ("branch.arm_runs", "arm_runs_completed", "A/B/C arms trained"),
    ):
        observed = str(readiness[key])
        results.append(
            CheckResult(
                check_id,
                requirement,
                observed,
                DEFERRED if observed == NOT_RUN else NOT_RUN,
                f"blocker={readiness['blocker']} owner={readiness['owner']} next_action={readiness['next_action']}",
            )
        )
    return tuple(results)


def assert_branch_arms_valid(
    arms: Mapping[str, ArmSchedule],
    lists: ExposureLists,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed on any arm check that is not deferred."""
    failures = [result for result in verify_branch_arms(arms, lists, protocol=protocol) if result.failed]
    if failures:
        raise BranchContractError(
            "; ".join(f"{result.check_id}: {result.reason}" for result in failures)
        )


def format_branch_report(results: Sequence[CheckResult]) -> str:
    """Human-readable summary of A/B/C arm verification."""
    width = max((len(result.check_id) for result in results), default=0)
    lines = [
        f"{result.status:<9} {result.check_id:<{width}}  {result.requirement} -> {result.observed}"
        for result in results
    ]
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    failures = [result for result in results if result.failed]
    lines.append("")
    lines.append("Summary: " + ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))
    lines.append("RESULT: " + ("PASS" if not failures else "FAIL"))
    if failures:
        lines.append("Failures:")
        lines.extend(f"  {result.check_id}: {result.reason}" for result in failures)
    return "\n".join(lines)


__all__ = [
    "ARM_A",
    "ARM_B",
    "ARM_C",
    "ARM_IDS",
    "BRANCH_ARM_POLICY_DIVERGED",
    "BRANCH_BOUNDARY_WRONG",
    "BRANCH_CONTENT_HASH_MISMATCH",
    "BRANCH_EXPOSURE_NOT_DISJOINT",
    "BRANCH_FAIL_CLOSED_REASON_CODES",
    "BRANCH_OK",
    "BRANCH_ORDER_HASH_IDENTICAL",
    "BRANCH_POSITION_NOT_MATCHED",
    "BRANCH_PROTOCOL_DIR",
    "BRANCH_PROTOCOL_PATH",
    "BRANCH_RESERVED_COUNT_MISMATCH",
    "BRANCH_RESERVED_MULTISET_MISMATCH",
    "BRANCH_SEQUENCES_PER_UPDATE_MISMATCH",
    "BRANCH_STABLE_MULTISET_MISMATCH",
    "BRANCH_SUPPLY_EXHAUSTED",
    "BRANCH_TOKENS_NOT_UPDATE_ALIGNED",
    "BRANCH_UPDATE_COUNT_INVALID",
    "COMMON_STABLE",
    "EXPOSURE_LIST_IDS",
    "FROZEN_BRANCH_PROTOCOL_SHA256",
    "PENDING_PARENT_HASH",
    "RESERVED_LIST",
    "STABLE_CONTROL",
    "ArmSchedule",
    "BranchContractError",
    "BranchExposureError",
    "BranchSizeBand",
    "BranchesNotReadyError",
    "ExposureLists",
    "ExposureSlot",
    "SharedBranchPolicy",
    "annealed_reserved_schedule",
    "arm_index",
    "assert_branch_arms_valid",
    "assert_ready_for_branch_runs",
    "branch_learning_rate",
    "branch_size_alignment_problems",
    "branch_size_bands",
    "build_arm_a",
    "build_arm_b",
    "build_arm_c",
    "build_arm_schedules",
    "build_exposure_lists",
    "constant_reserved_schedule",
    "cumulative_reserved",
    "disjointness_problems",
    "format_branch_report",
    "load_arm_schedule",
    "load_branch_protocol",
    "load_exposure_lists",
    "position_matching_problems",
    "reserved_in_update",
    "reserved_sequences_per_update",
    "select_branch_size",
    "sequences_per_update",
    "stable_sequences_per_update",
    "total_reserved_sequences",
    "verify_branch_arms",
    "write_arm_schedule",
    "write_exposure_lists",
]
