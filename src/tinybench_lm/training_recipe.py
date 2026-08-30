"""The frozen training recipe: WSD learning rate, AdamW decay groups, precision, counters.

Plan Section 7 fixes the starting values of the final campaign, Section 8.4 fixes the branch
LR shape, and Section 15 fixes what must fail closed. Two of those were previously not
implemented: ``train.py`` decayed the LR on a cosine curve toward a nonzero
``--minimum-learning-rate`` floor, and it built AdamW from one undifferentiated
``parameters()`` call, so weight decay was applied to the embedding matrix and to every
RMSNorm gain. This module is the frozen replacement, backed by one config::

    configs/training/recipe_v1.yaml

Four properties matter here, mirroring :mod:`tinybench_lm.data_protocols`,
:mod:`tinybench_lm.shards`, :mod:`tinybench_lm.schedule`, and :mod:`tinybench_lm.branches`:

1. **Immutable.** Every load verifies the config bytes against a pinned SHA-256 digest
   (:data:`FROZEN_TRAINING_RECIPE_SHA256`). The digest is also a semantic field of the run
   ID, so editing the recipe under a live run ID is detected rather than absorbed. Changing
   the recipe means publishing ``recipe_v2.yaml``.
2. **Exact.** The warmup length is derived with integer half-up arithmetic, and the LR is a
   pure function of ``(k, K, warmup_updates, decay_updates, peak_lr)``. Nothing depends on
   accumulated float state, so a resumed update reproduces the same LR bit for bit.
3. **One decay shape.** WSD only: linear warmup, stable peak, linear decay to *exactly*
   zero. A branch (Plan Section 8.4) is the full-horizon case of the same function --
   ``warmup_updates = 0`` and ``decay_updates = K`` reproduces
   :func:`tinybench_lm.branches.branch_learning_rate` exactly, which is what makes "identical
   branch LR decay" a checkable statement instead of a comment.
4. **Fail closed.** Non-finite loss or gradient norm, out-of-range token IDs, impossible
   counters, schedule/config hash drift, and a semantic change under an existing run ID all
   raise. Absence of a measurement is never a pass: BF16 stability has not been measured, so
   a final-scope run refuses to start rather than assuming BF16 is stable.

Nothing here starts training, measures throughput, measures BF16 stability, or selects a peak
LR or a horizon. Those remain ``NOT_RUN`` under the config's ``readiness`` section.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from .data_protocols import (
    REPOSITORY_ROOT,
    ProtocolError,
    ProtocolNotReadyError,
    load_protocol,
)
from .environment import CheckResult
from .model import LOSS_IGNORE_INDEX, RMSNorm
from .schedule import canonical_payload_bytes
from .shards import DEFERRED, FAIL, NOT_RUN, PASS

TRAINING_RECIPE_DIR = REPOSITORY_ROOT / "configs" / "training"
TRAINING_RECIPE_PATH = TRAINING_RECIPE_DIR / "recipe_v1.yaml"

#: SHA-256 of the frozen training recipe, over file bytes with CRLF normalized to LF.
FROZEN_TRAINING_RECIPE_SHA256: Mapping[str, str] = {
    "recipe_v1.yaml": "fdd5f9ab42d280515456e0caddc8d77bf3e5bfe8be0bf6bf83cf69681250c0f8",
}

#: Status used when a measurement is missing and the scope forbids guessing.
BLOCKED = "BLOCKED"

# --------------------------------------------------------------------------------------
# Reason-code vocabulary
# --------------------------------------------------------------------------------------

RECIPE_OK = "RECIPE_OK"
RECIPE_BATCH_NOT_TARGET = "RECIPE_BATCH_NOT_TARGET"
RECIPE_COUNTER_IMPOSSIBLE = "RECIPE_COUNTER_IMPOSSIBLE"
RECIPE_DECAY_GROUP_VIOLATION = "RECIPE_DECAY_GROUP_VIOLATION"
RECIPE_HASH_DRIFT = "RECIPE_HASH_DRIFT"
RECIPE_HORIZON_NOT_UPDATE_ALIGNED = "RECIPE_HORIZON_NOT_UPDATE_ALIGNED"
RECIPE_LR_PHASE_INVALID = "RECIPE_LR_PHASE_INVALID"
RECIPE_LR_SHAPE_FORBIDDEN = "RECIPE_LR_SHAPE_FORBIDDEN"
RECIPE_NON_FINITE = "RECIPE_NON_FINITE"
RECIPE_PRECISION_UNMEASURED = "RECIPE_PRECISION_UNMEASURED"
RECIPE_RUN_ID_SEMANTIC_CHANGE = "RECIPE_RUN_ID_SEMANTIC_CHANGE"
RECIPE_TOKEN_ID_OUT_OF_RANGE = "RECIPE_TOKEN_ID_OUT_OF_RANGE"
RECIPE_UNDECAYED_RELEASE_CANDIDATE = "RECIPE_UNDECAYED_RELEASE_CANDIDATE"

RECIPE_FAIL_CLOSED_REASON_CODES: frozenset[str] = frozenset(
    {
        RECIPE_BATCH_NOT_TARGET,
        RECIPE_COUNTER_IMPOSSIBLE,
        RECIPE_DECAY_GROUP_VIOLATION,
        RECIPE_HASH_DRIFT,
        RECIPE_HORIZON_NOT_UPDATE_ALIGNED,
        RECIPE_LR_PHASE_INVALID,
        RECIPE_LR_SHAPE_FORBIDDEN,
        RECIPE_NON_FINITE,
        RECIPE_PRECISION_UNMEASURED,
        RECIPE_RUN_ID_SEMANTIC_CHANGE,
        RECIPE_TOKEN_ID_OUT_OF_RANGE,
        RECIPE_UNDECAYED_RELEASE_CANDIDATE,
    }
)

#: LR phases, in the order WSD traverses them.
PHASE_WARMUP = "warmup"
PHASE_STABLE = "stable"
PHASE_DECAY = "linear_decay"
LR_PHASES: tuple[str, str, str] = (PHASE_WARMUP, PHASE_STABLE, PHASE_DECAY)

#: The optimizer policy identifier that ``configs/branches/exposure_v1.yaml`` and
#: :class:`tinybench_lm.branches.SharedBranchPolicy` name as the thing every arm must share.
#: :func:`adamw_parameter_groups` is that policy.
OPTIMIZER_POLICY_ID = "adamw_groups_frozen_by_task_3_12"

#: Parameter-group names. Exactly two, never one undifferentiated group.
DECAY_GROUP = "decay"
NO_DECAY_GROUP = "no_decay"
PARAMETER_GROUP_NAMES: tuple[str, str] = (DECAY_GROUP, NO_DECAY_GROUP)

BF16 = "bfloat16"
FP16 = "float16"

#: Run scopes. FINAL refuses unmeasured precision; PILOT labels it DEFERRED instead.
SCOPE_FINAL = "FINAL"
SCOPE_PILOT = "PILOT"
RUN_SCOPES: tuple[str, str] = (SCOPE_FINAL, SCOPE_PILOT)

#: Substrings that mark a forbidden final-campaign LR shape in a training entry point.
FORBIDDEN_LR_SHAPE_MARKERS: tuple[str, ...] = (
    "math.cos",
    "cosine",
    "minimum_learning_rate",
    "min_lr",
)

_RELATIVE_TOLERANCE = 1e-9
_ABSOLUTE_TOLERANCE = 1e-12


class TrainingRecipeError(ProtocolError):
    """The frozen recipe is malformed, or a schedule/optimizer/batch violates it."""


class TrainingIntegrityError(TrainingRecipeError):
    """A Plan Section 15 fail-closed condition fired: NaN/Inf, bad IDs, counters, drift."""


class TrainingRecipeNotReadyError(ProtocolNotReadyError):
    """A final-scope run needs a measurement (peak LR, horizon, BF16 stability) that does not exist."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def load_training_recipe(
    path: Path = TRAINING_RECIPE_PATH, *, verify: bool = True
) -> dict[str, Any]:
    """Load the frozen training recipe, verifying its pinned digest by default."""
    protocol = load_protocol(path, verify=verify, registry=FROZEN_TRAINING_RECIPE_SHA256)
    required = (
        "optimizer",
        "batch",
        "learning_rate",
        "precision",
        "fail_closed",
        "run_identity",
        "update_record",
        "readiness",
    )
    for section in required:
        if section not in protocol:
            raise TrainingRecipeError(f"training recipe is missing required section {section!r}")

    optimizer = protocol["optimizer"]
    if str(optimizer["algorithm"]) != "adamw":
        raise TrainingRecipeError("Plan Section 7 freezes AdamW as the optimizer")
    groups = {str(group["group_name"]): group for group in optimizer["groups"]}
    if tuple(sorted(groups)) != tuple(sorted(PARAMETER_GROUP_NAMES)):
        raise TrainingRecipeError(
            f"{RECIPE_DECAY_GROUP_VIOLATION}: the recipe must declare exactly "
            f"{sorted(PARAMETER_GROUP_NAMES)}, found {sorted(groups)}"
        )
    if float(groups[NO_DECAY_GROUP]["weight_decay"]) != 0.0:
        raise TrainingRecipeError(
            f"{RECIPE_DECAY_GROUP_VIOLATION}: the {NO_DECAY_GROUP} group must carry zero weight decay"
        )
    if float(groups[DECAY_GROUP]["weight_decay"]) != float(optimizer["weight_decay"]):
        raise TrainingRecipeError(
            f"{RECIPE_DECAY_GROUP_VIOLATION}: the {DECAY_GROUP} group must carry the declared weight decay"
        )
    if sorted(str(item) for item in optimizer["decay_exclusions"]) != [
        "embedding_weights",
        "normalization_weights",
    ]:
        raise TrainingRecipeError(
            f"{RECIPE_DECAY_GROUP_VIOLATION}: Plan Section 7 excludes embeddings and all "
            "normalization weights, and nothing else"
        )
    if not bool(optimizer["unique_parameter_enumeration"]):
        raise TrainingRecipeError("the recipe must enumerate unique trainable Parameter objects")

    batch = protocol["batch"]
    target = int(batch["target_loss_tokens_per_update"])
    if target != int(batch["sequences_per_update"]) * int(batch["sequence_length"]):
        raise TrainingRecipeError(
            f"{RECIPE_BATCH_NOT_TARGET}: target_loss_tokens_per_update must equal "
            "sequences_per_update * sequence_length"
        )
    if not bool(batch["exact_division_required"]):
        raise TrainingRecipeError("accumulation must divide the target batch exactly")

    learning_rate = protocol["learning_rate"]
    if str(learning_rate["schedule"]) != "wsd":
        raise TrainingRecipeError(f"{RECIPE_LR_SHAPE_FORBIDDEN}: the frozen schedule is WSD")
    if tuple(str(phase) for phase in learning_rate["phases"]) != LR_PHASES:
        raise TrainingRecipeError(
            f"{RECIPE_LR_PHASE_INVALID}: WSD phases must be {LR_PHASES}, found "
            f"{tuple(learning_rate['phases'])}"
        )
    if bool(learning_rate["cosine_decay_permitted"]):
        raise TrainingRecipeError(f"{RECIPE_LR_SHAPE_FORBIDDEN}: cosine decay is removed from the final campaign")
    if bool(learning_rate["minimum_learning_rate_floor_permitted"]):
        raise TrainingRecipeError(f"{RECIPE_LR_SHAPE_FORBIDDEN}: WSD decays to exactly zero, with no LR floor")
    if float(learning_rate["decay_end_value"]) != 0.0:
        raise TrainingRecipeError(f"{RECIPE_LR_SHAPE_FORBIDDEN}: the decay phase must end at exactly zero")
    if not bool(learning_rate["branch_decay_identical_across_arms"]):
        raise TrainingRecipeError("Plan Section 8.4: branch LR decay is identical across arms")
    if not bool(learning_rate["branch_is_full_horizon_decay_case"]):
        raise TrainingRecipeError("a branch must be the full-horizon case of the same WSD decay")
    if bool(learning_rate["undecayed_release_candidate_permitted"]):
        raise TrainingRecipeError(
            f"{RECIPE_UNDECAYED_RELEASE_CANDIDATE}: Plan Section 15 forbids releasing an "
            "undecayed peak-LR mainline checkpoint"
        )
    if int(learning_rate["minimum_decay_updates"]) < 2:
        raise TrainingRecipeError(
            f"{RECIPE_LR_PHASE_INVALID}: the decay divides by (decay_updates - 1), so a decay "
            "phase needs at least 2 updates"
        )

    precision = protocol["precision"]
    if str(precision["preferred"]) != BF16 or str(precision["fallback"]) != FP16:
        raise TrainingRecipeError("Plan Section 7 freezes BF16 preferred with an FP16 fallback")
    if bool(precision["preferred_uses_grad_scaler"]):
        raise TrainingRecipeError("BF16 does not use a GradScaler")
    if not bool(precision["fallback_uses_grad_scaler"]):
        raise TrainingRecipeError("the FP16 fallback must use a GradScaler")
    if not bool(precision["measured_stability_required_for_final_scope"]):
        raise TrainingRecipeError("a final-scope run requires measured BF16 stability")

    if not bool(protocol["fail_closed"]["never_mutate_a_failed_lineage"]):
        raise TrainingRecipeError("Plan Section 15: never mutate a failed lineage")
    if str(protocol["run_identity"]["on_semantic_change"]) != "issue_a_new_run_id_and_a_new_run_directory":
        raise TrainingRecipeError(
            f"{RECIPE_RUN_ID_SEMANTIC_CHANGE}: a semantic change must issue a new run ID"
        )
    return protocol


def _resolved(protocol: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return protocol if protocol is not None else load_training_recipe()


def recipe_digest(protocol: Mapping[str, Any] | None = None) -> str:
    """The frozen recipe's own digest, so it can be a semantic field of the run ID."""
    return str(_resolved(protocol)["_digest"])


def target_loss_tokens_per_update(protocol: Mapping[str, Any] | None = None) -> int:
    """The 262,144 loss tokens/update of Plan Section 7."""
    return int(_resolved(protocol)["batch"]["target_loss_tokens_per_update"])


def adamw_settings(protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Frozen AdamW hyperparameters, as keyword values for the optimizer constructor."""
    optimizer = _resolved(protocol)["optimizer"]
    return {
        "betas": (float(optimizer["beta1"]), float(optimizer["beta2"])),
        "epsilon": float(optimizer["epsilon"]),
        "weight_decay": float(optimizer["weight_decay"]),
        "gradient_clip_global_norm": float(optimizer["gradient_clip_global_norm"]),
    }


def assert_ready_for_final_run(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: a final campaign needs a measured peak LR, horizon, and BF16 stability."""
    readiness = _resolved(protocol)["readiness"]
    blocked = [
        name
        for name in ("measured_bf16_stability", "selected_peak_lr", "selected_horizon_updates")
        if str(readiness.get(name)) != PASS
    ]
    if blocked:
        raise TrainingRecipeNotReadyError(
            f"the final campaign recipe is not frozen yet: {blocked}. "
            f"blocker={readiness.get('blocker')} owner={readiness.get('owner')} "
            f"next_action={readiness.get('next_action')}"
        )


def readiness_results(protocol: Mapping[str, Any] | None = None) -> tuple[CheckResult, ...]:
    """Reportable readiness statuses. An absent measurement is never PASS."""
    readiness = _resolved(protocol)["readiness"]
    results: list[CheckResult] = []
    for name in (
        "measured_bf16_stability",
        "selected_peak_lr",
        "selected_horizon_updates",
        "measured_safe_microbatch",
    ):
        status = str(readiness.get(name, NOT_RUN))
        results.append(
            CheckResult(
                f"recipe.readiness.{name}",
                "measured evidence",
                status,
                status if status in {PASS, FAIL, DEFERRED, NOT_RUN, BLOCKED} else FAIL,
                str(readiness.get("next_action", "")) if status != PASS else "measured",
            )
        )
    return tuple(results)


# --------------------------------------------------------------------------------------
# Update-aligned horizon arithmetic (Plan Section 7)
# --------------------------------------------------------------------------------------


def _half_up(numerator: int, denominator: int) -> int:
    """``floor(numerator / denominator + 1/2)`` in exact integer arithmetic."""
    if denominator <= 0:
        raise TrainingRecipeError("half-up rounding needs a positive denominator")
    if numerator < 0:
        raise TrainingRecipeError("half-up rounding here is defined for nonnegative numerators")
    return (2 * numerator + denominator) // (2 * denominator)


def updates_for_tokens(
    tokens: int, *, loss_tokens_per_update: int | None = None, protocol: Mapping[str, Any] | None = None
) -> int:
    """Convert a token horizon to optimizer updates, refusing a non-aligned total."""
    resolved = _resolved(protocol)
    per_update = int(
        target_loss_tokens_per_update(resolved) if loss_tokens_per_update is None else loss_tokens_per_update
    )
    if per_update <= 0:
        raise TrainingRecipeError("loss_tokens_per_update must be positive")
    total = int(tokens)
    if total <= 0:
        raise TrainingRecipeError(f"{RECIPE_COUNTER_IMPOSSIBLE}: a token horizon must be positive, got {total}")
    if total % per_update != 0:
        raise TrainingRecipeError(
            f"{RECIPE_HORIZON_NOT_UPDATE_ALIGNED}: {total} tokens is not a multiple of "
            f"{per_update} loss tokens/update"
        )
    return total // per_update


def tokens_for_updates(
    updates: int, *, loss_tokens_per_update: int | None = None, protocol: Mapping[str, Any] | None = None
) -> int:
    """Convert optimizer updates to the exact consumed loss-token total."""
    resolved = _resolved(protocol)
    per_update = int(
        target_loss_tokens_per_update(resolved) if loss_tokens_per_update is None else loss_tokens_per_update
    )
    count = int(updates)
    if count < 0:
        raise TrainingRecipeError(f"{RECIPE_COUNTER_IMPOSSIBLE}: update count must be nonnegative")
    return count * per_update


def warmup_updates_for_horizon(
    total_updates: int,
    *,
    fraction: float | Fraction | str | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> int:
    """Plan Section 7's "warmup approximately 1% of the selected horizon", exactly.

    The fraction is evaluated as an exact rational with nonnegative half-up rounding, so the
    warmup length of a given horizon is identical on every machine, and never zero.
    """
    resolved = _resolved(protocol)
    learning_rate = resolved["learning_rate"]
    declared = learning_rate["warmup_fraction_of_horizon"] if fraction is None else fraction
    ratio = Fraction(str(declared))
    if ratio <= 0 or ratio > 1:
        raise TrainingRecipeError(f"{RECIPE_LR_PHASE_INVALID}: warmup fraction must be in (0, 1], got {ratio}")
    total = int(total_updates)
    if total < 1:
        raise TrainingRecipeError(f"{RECIPE_COUNTER_IMPOSSIBLE}: a horizon needs at least one update")
    minimum = int(learning_rate["minimum_warmup_updates"])
    rounded = _half_up(total * ratio.numerator, ratio.denominator)
    return max(minimum, min(total, rounded))


# --------------------------------------------------------------------------------------
# WSD learning rate (Plan Sections 7 and 8.4)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WSDSchedule:
    """Warmup, stable peak, linear decay to exactly zero, over a fixed update horizon.

    ``decay_updates == 0`` is a legal *mainline* lineage: Plan Section 7 places the decay
    "in each branch", so a parent lineage may hold the peak LR. It is never a release
    candidate -- see :func:`release_candidate_violations` and Plan Section 15.

    A branch is the full-horizon case: :meth:`branch` builds
    ``warmup_updates = 0, decay_updates = K``, which reproduces
    :func:`tinybench_lm.branches.branch_learning_rate` exactly.
    """

    total_updates: int
    warmup_updates: int
    decay_updates: int
    peak_lr: float

    def __post_init__(self) -> None:
        total = int(self.total_updates)
        warmup = int(self.warmup_updates)
        decay = int(self.decay_updates)
        if total < 1:
            raise TrainingRecipeError(
                f"{RECIPE_COUNTER_IMPOSSIBLE}: a horizon needs at least one update, got {total}"
            )
        if warmup < 0 or decay < 0:
            raise TrainingRecipeError(f"{RECIPE_LR_PHASE_INVALID}: phase lengths must be nonnegative")
        if decay == 1:
            raise TrainingRecipeError(
                f"{RECIPE_LR_PHASE_INVALID}: the decay divides by (decay_updates - 1), so a decay "
                "phase needs at least 2 updates"
            )
        if warmup + decay > total:
            raise TrainingRecipeError(
                f"{RECIPE_LR_PHASE_INVALID}: warmup {warmup} + decay {decay} exceeds the "
                f"{total}-update horizon"
            )
        if float(self.peak_lr) <= 0:
            raise TrainingRecipeError("the peak LR must be positive")

    # -- derived shape ------------------------------------------------------------------

    @property
    def stable_updates(self) -> int:
        return int(self.total_updates) - int(self.warmup_updates) - int(self.decay_updates)

    @property
    def decay_start_update(self) -> int:
        """First update index in the decay phase; ``total_updates`` when there is no decay."""
        return int(self.total_updates) - int(self.decay_updates)

    @property
    def decays_to_zero(self) -> bool:
        return int(self.decay_updates) > 0

    def phase(self, k: int) -> str:
        index = self._validate_index(k)
        if index < int(self.warmup_updates):
            return PHASE_WARMUP
        if self.decays_to_zero and index >= self.decay_start_update:
            return PHASE_DECAY
        return PHASE_STABLE

    def learning_rate(self, k: int) -> float:
        """LR for zero-based update index ``k``. A pure function, so resume reproduces it."""
        index = self._validate_index(k)
        peak = float(self.peak_lr)
        warmup = int(self.warmup_updates)
        if index < warmup:
            return peak * (index + 1) / warmup
        if self.decays_to_zero and index >= self.decay_start_update:
            remaining = int(self.total_updates) - 1 - index
            span = int(self.total_updates) - 1 - self.decay_start_update
            if span <= 0:  # pragma: no cover - forbidden by __post_init__
                raise TrainingRecipeError(f"{RECIPE_LR_PHASE_INVALID}: degenerate decay span")
            return peak * remaining / span
        return peak

    def schedule(self) -> tuple[float, ...]:
        return tuple(self.learning_rate(index) for index in range(int(self.total_updates)))

    def phases(self) -> tuple[str, ...]:
        return tuple(self.phase(index) for index in range(int(self.total_updates)))

    def _validate_index(self, k: int) -> int:
        index = int(k)
        if index < 0 or index >= int(self.total_updates):
            raise TrainingRecipeError(
                f"{RECIPE_COUNTER_IMPOSSIBLE}: update index {index} is outside "
                f"[0, {int(self.total_updates) - 1}]"
            )
        return index

    # -- identity -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_updates": int(self.total_updates),
            "warmup_updates": int(self.warmup_updates),
            "stable_updates": self.stable_updates,
            "decay_updates": int(self.decay_updates),
            "decay_start_update": self.decay_start_update,
            "peak_lr": float(self.peak_lr),
            "schedule": "wsd",
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_payload_bytes(self.to_dict())).hexdigest()

    # -- constructors -------------------------------------------------------------------

    @classmethod
    def for_horizon(
        cls,
        total_updates: int,
        peak_lr: float,
        *,
        decay_updates: int = 0,
        warmup_updates: int | None = None,
        warmup_fraction: float | Fraction | str | None = None,
        protocol: Mapping[str, Any] | None = None,
    ) -> "WSDSchedule":
        """Build a schedule whose warmup defaults to the frozen ~1% of the horizon."""
        resolved = _resolved(protocol)
        warmup = (
            warmup_updates_for_horizon(total_updates, fraction=warmup_fraction, protocol=resolved)
            if warmup_updates is None
            else int(warmup_updates)
        )
        return cls(
            total_updates=int(total_updates),
            warmup_updates=warmup,
            decay_updates=int(decay_updates),
            peak_lr=float(peak_lr),
        )

    @classmethod
    def branch(cls, parent_lr: float, update_count: int) -> "WSDSchedule":
        """Plan Section 8.4's arm schedule: linear decay from the parent LR to zero, no warmup."""
        total = int(update_count)
        if total < 2:
            raise TrainingRecipeError(f"{RECIPE_LR_PHASE_INVALID}: branch LR decay needs K >= 2")
        return cls(total_updates=total, warmup_updates=0, decay_updates=total, peak_lr=float(parent_lr))


def release_candidate_violations(schedule: WSDSchedule) -> tuple[str, ...]:
    """Plan Section 15: an undecayed peak-LR lineage is never the release fallback."""
    problems: list[str] = []
    if not schedule.decays_to_zero:
        problems.append(
            f"{RECIPE_UNDECAYED_RELEASE_CANDIDATE}: decay_updates is 0, so this lineage ends at "
            "the peak LR and must not be released as a fallback"
        )
        return tuple(problems)
    final_lr = schedule.learning_rate(int(schedule.total_updates) - 1)
    if final_lr != 0.0:
        problems.append(
            f"{RECIPE_LR_SHAPE_FORBIDDEN}: the decay phase must end at exactly zero, ends at {final_lr}"
        )
    return tuple(problems)


def forbidden_lr_shape_violations(
    source: str, *, markers: Sequence[str] = FORBIDDEN_LR_SHAPE_MARKERS
) -> tuple[str, ...]:
    """Reason-coded report of removed cosine/minimum-LR machinery in a training entry point."""
    return tuple(
        f"{RECIPE_LR_SHAPE_FORBIDDEN}: {marker!r} is a removed final-campaign LR shape"
        for marker in markers
        if marker in source
    )


# --------------------------------------------------------------------------------------
# AdamW parameter groups (Plan Section 7)
# --------------------------------------------------------------------------------------

#: Torch modules whose parameters are normalization gains/shifts.
_TORCH_NORM_TYPES: tuple[type, ...] = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm,
    RMSNorm,
)


def is_normalization_module(module: nn.Module) -> bool:
    """True for our RMSNorm, torch's norm layers, and any module named ``*Norm``."""
    if isinstance(module, _TORCH_NORM_TYPES):
        return True
    if isinstance(module, nn.modules.batchnorm._NormBase):  # noqa: SLF001 - the public base is private
        return True
    return type(module).__name__.endswith("Norm")


def is_embedding_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.Embedding, nn.EmbeddingBag))


@dataclass(frozen=True)
class ClassifiedParameters:
    """Unique trainable parameters split into the two frozen groups, with their names."""

    decay: tuple[tuple[str, nn.Parameter], ...]
    no_decay: tuple[tuple[str, nn.Parameter], ...]

    @property
    def decay_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.decay)

    @property
    def no_decay_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.no_decay)

    @property
    def unique_parameter_count(self) -> int:
        return sum(parameter.numel() for _, parameter in self.decay) + sum(
            parameter.numel() for _, parameter in self.no_decay
        )

    def group_of(self, name: str) -> str:
        if name in self.decay_names:
            return DECAY_GROUP
        if name in self.no_decay_names:
            return NO_DECAY_GROUP
        raise KeyError(name)


def classify_parameters(model: nn.Module) -> ClassifiedParameters:
    """Split unique trainable parameters into ``decay`` and ``no_decay``.

    Plan Section 7 excludes "embeddings and all normalization weights" from weight decay, and
    nothing else. Parameters are enumerated by identity, so the tied embedding/output weight
    -- one shared storage allocation -- is placed once, in ``no_decay``.
    """
    decay: list[tuple[str, nn.Parameter]] = []
    no_decay: list[tuple[str, nn.Parameter]] = []
    seen: dict[int, tuple[str, str]] = {}
    for module_name, module in model.named_modules():
        excluded = is_embedding_module(module) or is_normalization_module(module)
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            group = NO_DECAY_GROUP if excluded else DECAY_GROUP
            key = id(parameter)
            if key in seen:
                previous_name, previous_group = seen[key]
                if previous_group != group:
                    raise TrainingRecipeError(
                        f"{RECIPE_DECAY_GROUP_VIOLATION}: the shared Parameter reached as "
                        f"{previous_name!r} ({previous_group}) and {full_name!r} ({group}) "
                        "cannot belong to two groups"
                    )
                continue
            seen[key] = (full_name, group)
            (no_decay if excluded else decay).append((full_name, parameter))
    return ClassifiedParameters(tuple(decay), tuple(no_decay))


def parameter_group_violations(
    model: nn.Module,
    *,
    classified: ClassifiedParameters | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Reason-coded report on the two groups. Empty means the frozen policy holds."""
    resolved = _resolved(protocol)
    groups = classified if classified is not None else classify_parameters(model)
    required_rank = int(
        {str(group["group_name"]): group for group in resolved["optimizer"]["groups"]}[DECAY_GROUP][
            "required_parameter_rank"
        ]
    )
    problems: list[str] = []

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    placed = [id(parameter) for _, parameter in groups.decay] + [
        id(parameter) for _, parameter in groups.no_decay
    ]
    if len(placed) != len(set(placed)):
        problems.append(f"{RECIPE_DECAY_GROUP_VIOLATION}: a parameter appears in more than one group")
    missing = expected - set(placed)
    if missing:
        problems.append(
            f"{RECIPE_DECAY_GROUP_VIOLATION}: {len(missing)} trainable parameters were placed in no group"
        )
    extra = set(placed) - expected
    if extra:
        problems.append(
            f"{RECIPE_DECAY_GROUP_VIOLATION}: {len(extra)} grouped tensors are not trainable model parameters"
        )
    for name, parameter in groups.decay:
        if parameter.dim() != required_rank:
            problems.append(
                f"{RECIPE_DECAY_GROUP_VIOLATION}: {name!r} has rank {parameter.dim()} in the "
                f"{DECAY_GROUP} group, expected {required_rank}; the frozen architecture decays "
                "only projection matrices"
            )
    if not groups.no_decay:
        problems.append(
            f"{RECIPE_DECAY_GROUP_VIOLATION}: the {NO_DECAY_GROUP} group is empty, so embeddings "
            "and normalization weights are being decayed"
        )
    if not groups.decay:
        problems.append(f"{RECIPE_DECAY_GROUP_VIOLATION}: the {DECAY_GROUP} group is empty")
    return tuple(problems)


def adamw_parameter_groups(
    model: nn.Module,
    *,
    weight_decay: float | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The two frozen AdamW parameter groups, validated before they reach the optimizer.

    This is the ``adamw_groups_frozen_by_task_3_12`` optimizer policy that
    :class:`tinybench_lm.branches.SharedBranchPolicy` names.
    """
    resolved = _resolved(protocol)
    declared = float(resolved["optimizer"]["weight_decay"])
    decay_value = declared if weight_decay is None else float(weight_decay)
    if decay_value < 0:
        raise TrainingRecipeError("weight decay must be nonnegative")
    classified = classify_parameters(model)
    problems = parameter_group_violations(model, classified=classified, protocol=resolved)
    if problems:
        raise TrainingRecipeError("; ".join(problems))
    return [
        {
            "params": [parameter for _, parameter in classified.decay],
            "weight_decay": decay_value,
            "group_name": DECAY_GROUP,
        },
        {
            "params": [parameter for _, parameter in classified.no_decay],
            "weight_decay": 0.0,
            "group_name": NO_DECAY_GROUP,
        },
    ]


def optimizer_violations(
    optimizer: torch.optim.Optimizer, *, protocol: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Reason-coded report that a built optimizer still matches the frozen recipe."""
    resolved = _resolved(protocol)
    settings = adamw_settings(resolved)
    problems: list[str] = []
    if not isinstance(optimizer, torch.optim.AdamW):
        problems.append(f"{RECIPE_DECAY_GROUP_VIOLATION}: Plan Section 7 freezes AdamW, got {type(optimizer).__name__}")
    names = [str(group.get("group_name", "<unnamed>")) for group in optimizer.param_groups]
    if sorted(names) != sorted(PARAMETER_GROUP_NAMES):
        problems.append(
            f"{RECIPE_DECAY_GROUP_VIOLATION}: expected groups {sorted(PARAMETER_GROUP_NAMES)}, found {sorted(names)}"
        )
    for group in optimizer.param_groups:
        label = str(group.get("group_name", "<unnamed>"))
        if tuple(float(value) for value in group["betas"]) != settings["betas"]:
            problems.append(f"{RECIPE_DECAY_GROUP_VIOLATION}: {label} betas are {group['betas']}, expected {settings['betas']}")
        if float(group["eps"]) != settings["epsilon"]:
            problems.append(f"{RECIPE_DECAY_GROUP_VIOLATION}: {label} eps is {group['eps']}, expected {settings['epsilon']}")
        if label == NO_DECAY_GROUP and float(group["weight_decay"]) != 0.0:
            problems.append(
                f"{RECIPE_DECAY_GROUP_VIOLATION}: {NO_DECAY_GROUP} carries weight decay {group['weight_decay']}"
            )
        if label == DECAY_GROUP and float(group["weight_decay"]) <= 0.0:
            problems.append(f"{RECIPE_DECAY_GROUP_VIOLATION}: {DECAY_GROUP} carries no weight decay")
    return tuple(problems)


# --------------------------------------------------------------------------------------
# Precision policy (Plan Section 7)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionPolicy:
    """The selected autocast dtype, whether a GradScaler is required, and the evidence."""

    dtype_name: str
    use_grad_scaler: bool
    status: str
    reason: str
    scope: str
    bf16_supported: bool
    bf16_measured_stable: bool | None

    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype_name == BF16 else torch.float16

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype_name": self.dtype_name,
            "use_grad_scaler": bool(self.use_grad_scaler),
            "status": self.status,
            "reason": self.reason,
            "scope": self.scope,
            "bf16_supported": bool(self.bf16_supported),
            "bf16_measured_stable": self.bf16_measured_stable,
        }


def select_precision_policy(
    *,
    bf16_supported: bool,
    bf16_measured_stable: bool | None,
    scope: str = SCOPE_FINAL,
    protocol: Mapping[str, Any] | None = None,
) -> PrecisionPolicy:
    """Plan Section 7: BF16, with an FP16 + GradScaler fallback when BF16 is unsupported or unstable.

    ``bf16_measured_stable is None`` means the stability run has not happened. A final-scope
    run refuses to start on that basis; a pilot run may proceed with the deviation recorded as
    ``DEFERRED`` rather than promoted to ``PASS``.
    """
    resolved = _resolved(protocol)
    if scope not in RUN_SCOPES:
        raise TrainingRecipeError(f"scope must be one of {RUN_SCOPES}, got {scope!r}")
    supported = bool(bf16_supported)
    if not supported:
        return PrecisionPolicy(
            FP16,
            True,
            PASS,
            "the device does not support BF16, so the frozen FP16 + GradScaler fallback applies",
            scope,
            supported,
            bf16_measured_stable,
        )
    if bf16_measured_stable is False:
        return PrecisionPolicy(
            FP16,
            True,
            PASS,
            "BF16 was measured unstable, so the frozen FP16 + GradScaler fallback applies",
            scope,
            supported,
            False,
        )
    if bf16_measured_stable is True:
        return PrecisionPolicy(
            BF16, False, PASS, "BF16 is supported and was measured stable", scope, supported, True
        )
    readiness = resolved["readiness"]
    if scope == SCOPE_FINAL:
        raise TrainingRecipeNotReadyError(
            f"{RECIPE_PRECISION_UNMEASURED}: BF16 stability is "
            f"{readiness.get('measured_bf16_stability')} and a final-scope run may not assume it. "
            f"blocker={readiness.get('blocker')} owner={readiness.get('owner')} "
            f"next_action={readiness.get('next_action')}"
        )
    return PrecisionPolicy(
        BF16,
        False,
        DEFERRED,
        f"{RECIPE_PRECISION_UNMEASURED}: BF16 stability is "
        f"{readiness.get('measured_bf16_stability')}; provisional for pilot scope only",
        scope,
        supported,
        None,
    )


# --------------------------------------------------------------------------------------
# Batch and token counters (Plan Section 7)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchPlan:
    """Microbatch, sequence length, and accumulation, plus the tokens they actually produce."""

    micro_batch_size: int
    sequence_length: int
    gradient_accumulation: int

    def __post_init__(self) -> None:
        for name in ("micro_batch_size", "sequence_length", "gradient_accumulation"):
            value = int(getattr(self, name))
            if value < 1:
                raise TrainingRecipeError(f"{RECIPE_COUNTER_IMPOSSIBLE}: {name} must be at least 1, got {value}")

    @property
    def loss_tokens_per_update(self) -> int:
        return int(self.micro_batch_size) * int(self.sequence_length) * int(self.gradient_accumulation)

    @property
    def sequences_per_update(self) -> int:
        return int(self.micro_batch_size) * int(self.gradient_accumulation)

    def consumed_loss_tokens(self, updates: int) -> int:
        count = int(updates)
        if count < 0:
            raise TrainingRecipeError(f"{RECIPE_COUNTER_IMPOSSIBLE}: update count must be nonnegative")
        return count * self.loss_tokens_per_update

    def to_dict(self) -> dict[str, Any]:
        return {
            "micro_batch_size": int(self.micro_batch_size),
            "sequence_length": int(self.sequence_length),
            "gradient_accumulation": int(self.gradient_accumulation),
            "loss_tokens_per_update": self.loss_tokens_per_update,
            "sequences_per_update": self.sequences_per_update,
        }


def plan_batch(
    micro_batch_size: int,
    sequence_length: int,
    *,
    target_loss_tokens: int | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> BatchPlan:
    """Derive the accumulation that hits the target batch exactly, or refuse."""
    resolved = _resolved(protocol)
    target = int(
        target_loss_tokens_per_update(resolved) if target_loss_tokens is None else target_loss_tokens
    )
    per_microbatch = int(micro_batch_size) * int(sequence_length)
    if per_microbatch < 1:
        raise TrainingRecipeError(f"{RECIPE_COUNTER_IMPOSSIBLE}: microbatch tokens must be positive")
    if target % per_microbatch != 0:
        raise TrainingRecipeError(
            f"{RECIPE_BATCH_NOT_TARGET}: {micro_batch_size} x {sequence_length} = {per_microbatch} "
            f"loss tokens does not divide the {target}-token target exactly"
        )
    return BatchPlan(int(micro_batch_size), int(sequence_length), target // per_microbatch)


def batch_plan_violations(
    plan: BatchPlan,
    *,
    scope: str = SCOPE_FINAL,
    target_loss_tokens: int | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Final scope must hit the 262,144-token target exactly; pilot scope may deviate."""
    resolved = _resolved(protocol)
    if scope not in RUN_SCOPES:
        raise TrainingRecipeError(f"scope must be one of {RUN_SCOPES}, got {scope!r}")
    target = int(
        target_loss_tokens_per_update(resolved) if target_loss_tokens is None else target_loss_tokens
    )
    observed = plan.loss_tokens_per_update
    if observed == target:
        return ()
    if scope == SCOPE_PILOT:
        return ()
    return (
        f"{RECIPE_BATCH_NOT_TARGET}: the global batch is {observed} loss tokens/update, "
        f"expected the frozen target {target}",
    )


# --------------------------------------------------------------------------------------
# Fail-closed guards (Plan Section 15)
# --------------------------------------------------------------------------------------


def assert_finite(name: str, value: Any) -> float:
    """Fail closed on NaN/Inf. Returns the value as a float so it can be used inline."""
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise TrainingIntegrityError(f"{RECIPE_NON_FINITE}: {name} contains NaN or Inf")
        if value.numel() == 1:
            return float(value.detach().float().item())
        return float(value.detach().float().mean().item())
    number = float(value)
    if not math.isfinite(number):
        raise TrainingIntegrityError(f"{RECIPE_NON_FINITE}: {name} is {value!r}")
    return number


def assert_valid_token_ids(
    tokens: Any,
    vocab_size: int,
    *,
    name: str = "input_ids",
    allow_ignore_index: bool = False,
    ignore_index: int = LOSS_IGNORE_INDEX,
) -> None:
    """Fail closed on a token ID outside ``[0, vocab_size)``.

    Target tensors legitimately carry :data:`tinybench_lm.model.LOSS_IGNORE_INDEX` for padded
    positions, so ``allow_ignore_index`` permits exactly that one sentinel and nothing else.
    """
    limit = int(vocab_size)
    if limit < 1:
        raise TrainingRecipeError("vocab_size must be positive")
    tensor = tokens if isinstance(tokens, torch.Tensor) else torch.as_tensor(tokens)
    if tensor.numel() == 0:
        return
    flat = tensor.reshape(-1).to(torch.int64)
    if allow_ignore_index:
        flat = flat[flat != int(ignore_index)]
        if flat.numel() == 0:
            return
    minimum = int(flat.min().item())
    maximum = int(flat.max().item())
    if minimum < 0 or maximum >= limit:
        raise TrainingIntegrityError(
            f"{RECIPE_TOKEN_ID_OUT_OF_RANGE}: {name} spans [{minimum}, {maximum}], "
            f"outside [0, {limit - 1}]"
        )


def assert_no_hash_drift(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    """Fail closed when any frozen artifact hash changed under a live run."""
    drifted = {
        key: (expected[key], observed.get(key))
        for key in expected
        if str(observed.get(key)) != str(expected[key])
    }
    if drifted:
        details = ", ".join(f"{key}: {was} -> {now}" for key, (was, now) in sorted(drifted.items()))
        raise TrainingIntegrityError(f"{RECIPE_HASH_DRIFT}: {details}")


# --------------------------------------------------------------------------------------
# Run identity (Plan Section 15)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSemantics:
    """Every field whose change makes a run a different run.

    Plan Section 15: "A learning-rate or semantic change creates a new run ID; never mutate a
    failed lineage." The run ID is a hash of exactly these fields, so a changed LR, horizon,
    batch, optimizer setting, precision policy, seed, schedule, or recipe digest produces a
    different ID instead of silently continuing an existing lineage.
    """

    recipe_digest: str
    model_config_hash: str
    peak_lr: float
    total_updates: int
    warmup_updates: int
    decay_updates: int
    loss_tokens_per_update: int
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    gradient_clip_global_norm: float
    precision_dtype: str
    grad_scaler: bool
    seed: int
    train_schedule_content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_digest": str(self.recipe_digest),
            "model_config_hash": str(self.model_config_hash),
            "peak_lr": float(self.peak_lr),
            "total_updates": int(self.total_updates),
            "warmup_updates": int(self.warmup_updates),
            "decay_updates": int(self.decay_updates),
            "loss_tokens_per_update": int(self.loss_tokens_per_update),
            "weight_decay": float(self.weight_decay),
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "epsilon": float(self.epsilon),
            "gradient_clip_global_norm": float(self.gradient_clip_global_norm),
            "precision_dtype": str(self.precision_dtype),
            "grad_scaler": bool(self.grad_scaler),
            "seed": int(self.seed),
            "train_schedule_content_hash": str(self.train_schedule_content_hash),
        }

    def run_id(self, protocol: Mapping[str, Any] | None = None) -> str:
        resolved = _resolved(protocol)
        identity = resolved["run_identity"]
        declared = tuple(str(field) for field in identity["semantic_fields"])
        payload = self.to_dict()
        if tuple(sorted(declared)) != tuple(sorted(payload)):
            raise TrainingRecipeError(
                f"{RECIPE_RUN_ID_SEMANTIC_CHANGE}: the recipe declares semantic fields "
                f"{sorted(declared)} but the run semantics carry {sorted(payload)}"
            )
        digest = hashlib.sha256(canonical_payload_bytes(payload)).hexdigest()
        return f"{identity['id_prefix']}{digest[: int(identity['id_hex_length'])]}"


def model_config_hash(config: Mapping[str, Any]) -> str:
    """Content hash of a model configuration, independent of file formatting."""
    return hashlib.sha256(canonical_payload_bytes(dict(config))).hexdigest()


def build_run_semantics(
    *,
    model_config: Mapping[str, Any],
    schedule: WSDSchedule,
    plan: BatchPlan,
    precision: PrecisionPolicy,
    weight_decay: float,
    gradient_clip_global_norm: float,
    seed: int,
    train_schedule_content_hash: str,
    protocol: Mapping[str, Any] | None = None,
) -> RunSemantics:
    """Collect every semantic field from the objects that already own it."""
    resolved = _resolved(protocol)
    settings = adamw_settings(resolved)
    beta1, beta2 = settings["betas"]
    return RunSemantics(
        recipe_digest=recipe_digest(resolved),
        model_config_hash=model_config_hash(model_config),
        peak_lr=float(schedule.peak_lr),
        total_updates=int(schedule.total_updates),
        warmup_updates=int(schedule.warmup_updates),
        decay_updates=int(schedule.decay_updates),
        loss_tokens_per_update=plan.loss_tokens_per_update,
        weight_decay=float(weight_decay),
        beta1=float(beta1),
        beta2=float(beta2),
        epsilon=float(settings["epsilon"]),
        gradient_clip_global_norm=float(gradient_clip_global_norm),
        precision_dtype=precision.dtype_name,
        grad_scaler=bool(precision.use_grad_scaler),
        seed=int(seed),
        train_schedule_content_hash=str(train_schedule_content_hash),
    )


def semantic_differences(previous: RunSemantics, current: RunSemantics) -> tuple[str, ...]:
    """Names of the semantic fields that changed between two run definitions."""
    was = previous.to_dict()
    now = current.to_dict()
    return tuple(key for key in sorted(was) if was[key] != now[key])


def assert_run_id_unchanged(
    recorded_run_id: str,
    semantics: RunSemantics,
    *,
    recorded_semantics: RunSemantics | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> str:
    """Fail closed when the recorded run ID does not match the current semantics."""
    resolved = _resolved(protocol)
    computed = semantics.run_id(resolved)
    if computed == str(recorded_run_id):
        return computed
    changed = semantic_differences(recorded_semantics, semantics) if recorded_semantics else ()
    detail = f" changed fields: {list(changed)}" if changed else ""
    raise TrainingIntegrityError(
        f"{RECIPE_RUN_ID_SEMANTIC_CHANGE}: run directory records {recorded_run_id!r} but the "
        f"current recipe hashes to {computed!r}.{detail} "
        "Start a new run directory instead of mutating an existing lineage."
    )


# --------------------------------------------------------------------------------------
# Per-update audit record
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UpdateRecord:
    """Auditable optimizer, LR, precision, token, and schedule state for one update."""

    run_id: str
    update_index: int
    phase: str
    learning_rate: float
    loss: float
    grad_norm: float
    consumed_loss_tokens: int
    loss_tokens_per_update: int
    precision_dtype: str
    grad_scaler: bool
    schedule_content_hash: str
    schedule_cursor: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": str(self.run_id),
            "update_index": int(self.update_index),
            "phase": str(self.phase),
            "learning_rate": float(self.learning_rate),
            "loss": float(self.loss),
            "grad_norm": float(self.grad_norm),
            "consumed_loss_tokens": int(self.consumed_loss_tokens),
            "loss_tokens_per_update": int(self.loss_tokens_per_update),
            "precision_dtype": str(self.precision_dtype),
            "grad_scaler": bool(self.grad_scaler),
            "schedule_content_hash": str(self.schedule_content_hash),
            "schedule_cursor": None if self.schedule_cursor is None else int(self.schedule_cursor),
        }


def build_update_record(
    *,
    run_id: str,
    update_index: int,
    schedule: WSDSchedule,
    plan: BatchPlan,
    precision: PrecisionPolicy,
    loss: float,
    grad_norm: float,
    schedule_content_hash: str,
    schedule_cursor: int | None,
) -> UpdateRecord:
    """Build one record from the schedule and plan, so the LR and counters are derived, not typed."""
    index = int(update_index)
    return UpdateRecord(
        run_id=str(run_id),
        update_index=index,
        phase=schedule.phase(index),
        learning_rate=schedule.learning_rate(index),
        loss=assert_finite("loss", loss),
        grad_norm=assert_finite("grad_norm", grad_norm),
        consumed_loss_tokens=plan.consumed_loss_tokens(index + 1),
        loss_tokens_per_update=plan.loss_tokens_per_update,
        precision_dtype=precision.dtype_name,
        grad_scaler=bool(precision.use_grad_scaler),
        schedule_content_hash=str(schedule_content_hash),
        schedule_cursor=None if schedule_cursor is None else int(schedule_cursor),
    )


def update_record_violations(
    record: UpdateRecord,
    *,
    schedule: WSDSchedule,
    plan: BatchPlan,
    previous: UpdateRecord | None = None,
) -> tuple[str, ...]:
    """Reason-coded report on one update record. Empty means the counters reconcile."""
    problems: list[str] = []
    index = int(record.update_index)
    if index < 0 or index >= int(schedule.total_updates):
        problems.append(
            f"{RECIPE_COUNTER_IMPOSSIBLE}: update_index {index} is outside "
            f"[0, {int(schedule.total_updates) - 1}]"
        )
        return tuple(problems)

    for name in ("learning_rate", "loss", "grad_norm"):
        value = float(getattr(record, name))
        if not math.isfinite(value):
            problems.append(f"{RECIPE_NON_FINITE}: {name} is {value}")
    if float(record.grad_norm) < 0:
        problems.append(f"{RECIPE_COUNTER_IMPOSSIBLE}: grad_norm {record.grad_norm} is negative")

    expected_lr = schedule.learning_rate(index)
    if not math.isclose(
        float(record.learning_rate), expected_lr, rel_tol=_RELATIVE_TOLERANCE, abs_tol=_ABSOLUTE_TOLERANCE
    ):
        problems.append(
            f"{RECIPE_LR_PHASE_INVALID}: update {index} records LR {record.learning_rate}, "
            f"the frozen WSD schedule gives {expected_lr}"
        )
    expected_phase = schedule.phase(index)
    if str(record.phase) != expected_phase:
        problems.append(
            f"{RECIPE_LR_PHASE_INVALID}: update {index} records phase {record.phase!r}, expected {expected_phase!r}"
        )

    if int(record.loss_tokens_per_update) != plan.loss_tokens_per_update:
        problems.append(
            f"{RECIPE_BATCH_NOT_TARGET}: record declares {record.loss_tokens_per_update} "
            f"loss tokens/update, the batch plan produces {plan.loss_tokens_per_update}"
        )
    expected_tokens = plan.consumed_loss_tokens(index + 1)
    if int(record.consumed_loss_tokens) != expected_tokens:
        problems.append(
            f"{RECIPE_COUNTER_IMPOSSIBLE}: update {index} records {record.consumed_loss_tokens} "
            f"consumed loss tokens, exact arithmetic gives {expected_tokens}"
        )
    if record.schedule_cursor is not None and int(record.schedule_cursor) < 0:
        problems.append(f"{RECIPE_COUNTER_IMPOSSIBLE}: schedule_cursor {record.schedule_cursor} is negative")

    if previous is not None:
        if str(previous.run_id) != str(record.run_id):
            problems.append(
                f"{RECIPE_RUN_ID_SEMANTIC_CHANGE}: run ID changed from {previous.run_id!r} to {record.run_id!r} "
                "inside one lineage"
            )
        if str(previous.schedule_content_hash) != str(record.schedule_content_hash):
            problems.append(
                f"{RECIPE_HASH_DRIFT}: schedule content hash changed from "
                f"{previous.schedule_content_hash!r} to {record.schedule_content_hash!r}"
            )
        if int(record.update_index) != int(previous.update_index) + 1:
            problems.append(
                f"{RECIPE_COUNTER_IMPOSSIBLE}: update index jumped from {previous.update_index} to "
                f"{record.update_index}"
            )
        if (
            previous.schedule_cursor is not None
            and record.schedule_cursor is not None
            and int(record.schedule_cursor) < int(previous.schedule_cursor)
        ):
            problems.append(
                f"{RECIPE_COUNTER_IMPOSSIBLE}: schedule cursor moved backwards from "
                f"{previous.schedule_cursor} to {record.schedule_cursor}"
            )
        if str(previous.precision_dtype) != str(record.precision_dtype) or bool(
            previous.grad_scaler
        ) != bool(record.grad_scaler):
            problems.append(
                f"{RECIPE_RUN_ID_SEMANTIC_CHANGE}: precision policy changed from "
                f"{previous.precision_dtype}/scaler={previous.grad_scaler} to "
                f"{record.precision_dtype}/scaler={record.grad_scaler} inside one lineage"
            )
    return tuple(problems)


def assert_update_record(
    record: UpdateRecord,
    *,
    schedule: WSDSchedule,
    plan: BatchPlan,
    previous: UpdateRecord | None = None,
) -> UpdateRecord:
    """Fail closed on any counter, LR, precision, hash, or run-ID violation."""
    problems = update_record_violations(record, schedule=schedule, plan=plan, previous=previous)
    if problems:
        raise TrainingIntegrityError("; ".join(problems))
    return record


def format_recipe_report(results: Sequence[CheckResult]) -> str:
    """Human-readable summary of recipe readiness checks."""
    width = max((len(result.check_id) for result in results), default=0)
    lines = [
        f"{result.status:<10} {result.check_id:<{width}}  {result.requirement} -> {result.observed}"
        for result in results
    ]
    lines.append("RESULT: " + ("PASS" if all(result.status == PASS for result in results) else "NOT COMPLETE"))
    return "\n".join(lines)
