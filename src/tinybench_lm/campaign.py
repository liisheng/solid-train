"""Preregistered campaign configurations and decision validators (Plan Sections 6, 8, 12-13).

A comparison is only evidence if its rule was fixed before its outcome was visible. Plan
Section 6.1 states the requirement directly: "Before any proxy result, hash
``PREREGISTRATION-mixture-screen.md``." Everything in this module exists to make that
sequencing checkable rather than merely asserted:

``preregistration_hash``
    the SHA-256 of the frozen mixture-screen document. Recorded before P1/P8 run, then
    carried into every parent binding, so a later edit is visible as a hash change.
``freeze_bundle_hash``
    one G4 hash over all thirteen components of Plan Section 13 G4, with two distinct
    approvers and both required measurements present.
:class:`ParentBindingManifest`
    append-only. Plan Section 8.5 requires the parent hash be bound "**before** any arm runs
    or any arm outcome is observed"; an append-only file is what makes a post-hoc parent
    swap detectable instead of silent.

The contract is backed by one frozen config::

    configs/campaign/preregistration_v1.yaml

Guarantees, mirroring :mod:`tinybench_lm.branches` and :mod:`tinybench_lm.training_recipe`:

1. **Immutable.** The config is verified against a pinned SHA-256 digest
   (:data:`FROZEN_CAMPAIGN_PROTOCOL_SHA256`) on every load. A threshold or a run definition
   cannot drift after a result is observed.
2. **Exact.** Every horizon is validated as a whole number of 262,144-loss-token updates,
   and the proxy parameter count is checked against Plan Section 6.2's
   ``6,291,456 + 8 x 3,097,600 + 512 = 31,072,768``.
3. **Fail closed.** Unequal run lengths, non-comparable seeds or learning rates, an
   incomplete preregistration, an under-approved freeze bundle, and any manifest rewrite
   raise or report ``FAIL`` rather than degrade.
4. **Absence of evidence is never PASS.** No proxy has run, no mixture or peak LR is
   selected, no seed SD is measured, and no parent hash exists. Those stay ``NOT_RUN`` /
   ``BLOCKED``; :func:`assert_ready_for_campaign_runs` refuses to pretend otherwise.

Nothing here runs a proxy, trains an arm, binds a real parent hash, opens ``validation_final``,
or claims a gate pass. The paired-bootstrap analysis that consumes these thresholds is task
3.18's scope; this module only freezes and validates the rules it must obey.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import ModelConfig
from .data_protocols import (
    REPOSITORY_ROOT,
    ProtocolError,
    ProtocolNotReadyError,
    load_protocol,
)
from .environment import CheckResult
from .schedule import canonical_payload_bytes
from .shards import EXPECTED_PROTECTED_SLICES, FAIL, NOT_RUN, PASS

CAMPAIGN_PROTOCOL_DIR = REPOSITORY_ROOT / "configs" / "campaign"
CAMPAIGN_PROTOCOL_PATH = CAMPAIGN_PROTOCOL_DIR / "preregistration_v1.yaml"

#: SHA-256 of the frozen campaign contract, over file bytes with CRLF normalized to LF.
FROZEN_CAMPAIGN_PROTOCOL_SHA256: Mapping[str, str] = {
    "preregistration_v1.yaml": "83f63d8a790be99420c2cd56c4996adbad9939925fc5ded957221525f54b1881",
}

_CAMPAIGN_SCHEMA_VERSION = "campaign_preregistration_v1"

#: A pending organizer/operator field is BLOCKED only when it names its own next step.
BLOCKED = "BLOCKED"

#: Plan Section 6.2 run identities, in their frozen declaration order.
PROXY_RUN_IDS: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P8")
FINAL_RUN_IDS: tuple[str, ...] = ("F1", "F2")
CAMPAIGN_RUN_IDS: tuple[str, ...] = PROXY_RUN_IDS + FINAL_RUN_IDS

MODEL_PROXY = "proxy"
MODEL_FINAL = "final"

MIXTURE_BASE = "M-base"
MIXTURE_EDU = "M-edu"

#: Plan Section 8.5: the parent checkpoint hash does not exist when the campaign is frozen.
PENDING_PARENT_HASH = "PENDING_PARENT_HASH"

#: Plan Section 6.2: F1/F2 peak LR is selected from P1/P4 and does not exist at freeze time.
SELECTED_FROM_P1_P4 = "SELECTED_FROM_P1_P4"

# --------------------------------------------------------------------------------------
# Reason-code vocabulary
# --------------------------------------------------------------------------------------

CAMPAIGN_OK = "CAMPAIGN_OK"
CAMPAIGN_CONFIRMATION_MISSING = "CAMPAIGN_CONFIRMATION_MISSING"
CAMPAIGN_CONTINGENCY_DECIDED_LATE = "CAMPAIGN_CONTINGENCY_DECIDED_LATE"
CAMPAIGN_FREEZE_APPROVAL_INSUFFICIENT = "CAMPAIGN_FREEZE_APPROVAL_INSUFFICIENT"
CAMPAIGN_FREEZE_BUNDLE_INCOMPLETE = "CAMPAIGN_FREEZE_BUNDLE_INCOMPLETE"
CAMPAIGN_FREEZE_EVIDENCE_MISSING = "CAMPAIGN_FREEZE_EVIDENCE_MISSING"
CAMPAIGN_LR_NOT_COMPARABLE = "CAMPAIGN_LR_NOT_COMPARABLE"
CAMPAIGN_MIXTURE_HELD_SOURCE_CHANGED = "CAMPAIGN_MIXTURE_HELD_SOURCE_CHANGED"
CAMPAIGN_MIXTURE_SHARES_INVALID = "CAMPAIGN_MIXTURE_SHARES_INVALID"
CAMPAIGN_PARENT_BOUND_AFTER_OUTCOME = "CAMPAIGN_PARENT_BOUND_AFTER_OUTCOME"
CAMPAIGN_PARENT_HASH_PENDING = "CAMPAIGN_PARENT_HASH_PENDING"
CAMPAIGN_PARENT_MANIFEST_REWRITTEN = "CAMPAIGN_PARENT_MANIFEST_REWRITTEN"
CAMPAIGN_PREREGISTRATION_INCOMPLETE = "CAMPAIGN_PREREGISTRATION_INCOMPLETE"
CAMPAIGN_PREREGISTRATION_LATE = "CAMPAIGN_PREREGISTRATION_LATE"
CAMPAIGN_PROTECTED_SLICE_REGRESSION = "CAMPAIGN_PROTECTED_SLICE_REGRESSION"
CAMPAIGN_PROXY_COUNT_MISMATCH = "CAMPAIGN_PROXY_COUNT_MISMATCH"
CAMPAIGN_SEED_NOT_COMPARABLE = "CAMPAIGN_SEED_NOT_COMPARABLE"
CAMPAIGN_SET_COUNT_INVALID = "CAMPAIGN_SET_COUNT_INVALID"
CAMPAIGN_THRESHOLD_NOT_MET = "CAMPAIGN_THRESHOLD_NOT_MET"
CAMPAIGN_TOKENS_NOT_UPDATE_ALIGNED = "CAMPAIGN_TOKENS_NOT_UPDATE_ALIGNED"
CAMPAIGN_UNEQUAL_RUN_LENGTH = "CAMPAIGN_UNEQUAL_RUN_LENGTH"
CAMPAIGN_VALIDATION_FINAL_OPENED_EARLY = "CAMPAIGN_VALIDATION_FINAL_OPENED_EARLY"

CAMPAIGN_FAIL_CLOSED_REASON_CODES: frozenset[str] = frozenset(
    {
        CAMPAIGN_CONFIRMATION_MISSING,
        CAMPAIGN_CONTINGENCY_DECIDED_LATE,
        CAMPAIGN_FREEZE_APPROVAL_INSUFFICIENT,
        CAMPAIGN_FREEZE_BUNDLE_INCOMPLETE,
        CAMPAIGN_FREEZE_EVIDENCE_MISSING,
        CAMPAIGN_LR_NOT_COMPARABLE,
        CAMPAIGN_MIXTURE_HELD_SOURCE_CHANGED,
        CAMPAIGN_MIXTURE_SHARES_INVALID,
        CAMPAIGN_PARENT_BOUND_AFTER_OUTCOME,
        CAMPAIGN_PARENT_HASH_PENDING,
        CAMPAIGN_PARENT_MANIFEST_REWRITTEN,
        CAMPAIGN_PREREGISTRATION_INCOMPLETE,
        CAMPAIGN_PREREGISTRATION_LATE,
        CAMPAIGN_PROTECTED_SLICE_REGRESSION,
        CAMPAIGN_PROXY_COUNT_MISMATCH,
        CAMPAIGN_SEED_NOT_COMPARABLE,
        CAMPAIGN_SET_COUNT_INVALID,
        CAMPAIGN_THRESHOLD_NOT_MET,
        CAMPAIGN_TOKENS_NOT_UPDATE_ALIGNED,
        CAMPAIGN_UNEQUAL_RUN_LENGTH,
        CAMPAIGN_VALIDATION_FINAL_OPENED_EARLY,
    }
)


class CampaignContractError(ProtocolError):
    """The frozen campaign contract is malformed, or an artifact violates it."""


class CampaignPreregistrationError(CampaignContractError):
    """A preregistration, freeze bundle, or parent binding is incomplete or out of order."""


class CampaignNotReadyError(ProtocolNotReadyError):
    """A campaign decision needs measured evidence that does not exist yet."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def load_campaign_protocol(
    path: Path = CAMPAIGN_PROTOCOL_PATH, *, verify: bool = True
) -> dict[str, Any]:
    """Load the frozen campaign contract, verifying its pinned digest by default."""
    protocol = load_protocol(path, verify=verify, registry=FROZEN_CAMPAIGN_PROTOCOL_SHA256)
    required = (
        "batch_layout",
        "proxy_architecture",
        "final_architecture",
        "mixtures",
        "runs",
        "contingency",
        "comparisons",
        "decision_thresholds",
        "mixture_screen_preregistration",
        "freeze_bundle",
        "parent_binding",
        "branch_sets",
        "validation_final_custody",
        "readiness",
    )
    for section in required:
        if section not in protocol:
            raise CampaignContractError(f"campaign protocol is missing required section {section!r}")

    layout = protocol["batch_layout"]
    sequences = int(layout["sequences_per_update"])
    length = int(layout["sequence_length"])
    loss_tokens = int(layout["loss_tokens_per_update"])
    if loss_tokens != sequences * length:
        raise CampaignContractError(
            f"{CAMPAIGN_TOKENS_NOT_UPDATE_ALIGNED}: loss_tokens_per_update must equal "
            f"{sequences} x {length}, found {loss_tokens}"
        )

    declared = tuple(str(run["run_id"]) for run in protocol["runs"])
    if declared != CAMPAIGN_RUN_IDS:
        raise CampaignContractError(
            f"campaign protocol must declare runs {CAMPAIGN_RUN_IDS}, found {declared}"
        )
    for run in protocol["runs"]:
        run_id = str(run["run_id"])
        tokens = int(run["tokens"])
        updates = int(run["updates"])
        if tokens != updates * loss_tokens:
            raise CampaignContractError(
                f"{CAMPAIGN_TOKENS_NOT_UPDATE_ALIGNED}: run {run_id} declares {tokens} tokens, "
                f"which is not {updates} x {loss_tokens}"
            )
        if str(run["model"]) not in (MODEL_PROXY, MODEL_FINAL):
            raise CampaignContractError(f"run {run_id} names an unknown model {run['model']!r}")

    contingency = protocol["contingency"]
    if int(contingency["tokens"]) != int(contingency["updates"]) * loss_tokens:
        raise CampaignContractError(
            f"{CAMPAIGN_TOKENS_NOT_UPDATE_ALIGNED}: the contingency horizon is not update-aligned"
        )
    if str(contingency["decide_before"]) != "F1":
        raise CampaignContractError(
            f"{CAMPAIGN_CONTINGENCY_DECIDED_LATE}: Plan Section 6.2 freezes the contingency "
            "before F1"
        )

    mixtures = protocol["mixtures"]
    total = int(mixtures["required_share_total"])
    for mixture_id in (MIXTURE_BASE, MIXTURE_EDU):
        shares = mixtures[mixture_id]["shares"]
        if sum(int(value) for value in shares.values()) != total:
            raise CampaignContractError(
                f"{CAMPAIGN_MIXTURE_SHARES_INVALID}: {mixture_id} shares must total {total}"
            )
    held = tuple(str(name) for name in mixtures["held_fixed_sources"])
    for source in held:
        if int(mixtures[MIXTURE_BASE]["shares"][source]) != int(mixtures[MIXTURE_EDU]["shares"][source]):
            raise CampaignContractError(
                f"{CAMPAIGN_MIXTURE_HELD_SOURCE_CHANGED}: {source} must be identical in both mixtures"
            )
    moved = int(mixtures["moved_percentage_points"])
    source_from = str(mixtures["moved_from"])
    source_to = str(mixtures["moved_to"])
    base_shares = mixtures[MIXTURE_BASE]["shares"]
    edu_shares = mixtures[MIXTURE_EDU]["shares"]
    if int(base_shares[source_from]) - int(edu_shares[source_from]) != moved:
        raise CampaignContractError(
            f"{CAMPAIGN_MIXTURE_SHARES_INVALID}: {source_from} must drop by exactly {moved} points"
        )
    if int(edu_shares[source_to]) - int(base_shares[source_to]) != moved:
        raise CampaignContractError(
            f"{CAMPAIGN_MIXTURE_SHARES_INVALID}: {source_to} must rise by exactly {moved} points"
        )

    thresholds = protocol["decision_thresholds"]
    if float(thresholds["bootstrap"]["confidence_level"]) != 0.95:
        raise CampaignContractError("Plan Sections 6.1 and 8.6 freeze a 95% confidence level")
    if float(thresholds["relative_nll_reduction"]["minimum"]) != 0.003:
        raise CampaignContractError("the frozen minimum relative NLL reduction is 0.3%")
    if float(thresholds["seed_sd_multiple"]["minimum"]) != 2.0:
        raise CampaignContractError("the frozen magnitude bar is twice the measured seed SD")
    slice_rule = thresholds["protected_slice_regression"]
    if float(slice_rule["maximum_relative_regression"]) != 0.01:
        raise CampaignContractError("the frozen protected-slice limit is 1% relative regression")
    declared_slices = tuple(str(name) for name in slice_rule["slices"])
    if declared_slices != EXPECTED_PROTECTED_SLICES:
        raise CampaignContractError(
            f"campaign protocol declares protected slices {declared_slices}, "
            f"expected {EXPECTED_PROTECTED_SLICES}"
        )
    if not bool(thresholds["all_conditions_required"]):
        raise CampaignContractError("every decision threshold must hold; none is optional")

    binding = protocol["parent_binding"]
    if str(binding["pending_sentinel"]) != PENDING_PARENT_HASH:
        raise CampaignContractError(
            f"the pending parent sentinel must be {PENDING_PARENT_HASH!r} until the parent exists"
        )
    if bool(binding["hash_exists_at_freeze_time"]):
        raise CampaignContractError("Plan Section 8.5: the parent hash does not exist at freeze time")
    for flag in ("append_only", "rewrite_forbidden", "bind_before_any_arm_outcome_is_observed"):
        if not bool(binding[flag]):
            raise CampaignContractError(f"the parent binding manifest must assert {flag}")

    sets = protocol["branch_sets"]
    if int(sets["primary_sets"]) != 1 or int(sets["confirmation_sets"]) != 1:
        raise CampaignContractError(
            f"{CAMPAIGN_SET_COUNT_INVALID}: Plan Section 8.6 preregisters one primary A/B/C set "
            "and one earlier confirmation"
        )
    if not bool(sets["confirmation_precedes_primary"]):
        raise CampaignContractError("the short confirmation runs before the primary set")

    custody = protocol["validation_final_custody"]
    if not bool(custody["opened_once"]):
        raise CampaignContractError("validation_final is opened exactly once")
    if str(custody["status"]) != "NOT_OPENED":
        raise CampaignContractError(
            f"{CAMPAIGN_VALIDATION_FINAL_OPENED_EARLY}: the frozen contract records "
            "validation_final as NOT_OPENED"
        )
    return protocol


def _resolved(protocol: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return protocol if protocol is not None else load_campaign_protocol()


def campaign_protocol_digest(protocol: Mapping[str, Any] | None = None) -> str:
    """Content hash of the frozen campaign contract, for binding into other manifests."""
    del protocol
    return hashlib.sha256(CAMPAIGN_PROTOCOL_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def loss_tokens_per_update(protocol: Mapping[str, Any] | None = None) -> int:
    """The frozen 262,144 loss tokens every update consumes."""
    return int(_resolved(protocol)["batch_layout"]["loss_tokens_per_update"])


# --------------------------------------------------------------------------------------
# Proxy architecture (Plan Section 6.2)
# --------------------------------------------------------------------------------------


def proxy_model_config(protocol: Mapping[str, Any] | None = None) -> ModelConfig:
    """The final architecture reduced to 8 layers -- the only permitted difference."""
    spec = _resolved(protocol)["proxy_architecture"]
    return ModelConfig(
        vocab_size=int(spec["vocab_size"]),
        max_seq_len=int(spec["max_seq_len"]),
        n_layers=int(spec["n_layers"]),
        d_model=int(spec["d_model"]),
        n_heads=int(spec["n_heads"]),
        n_kv_heads=int(spec["n_kv_heads"]),
        d_ff=int(spec["d_ff"]),
        rope_theta=float(spec["rope_theta"]),
        rms_norm_eps=float(spec["rms_norm_eps"]),
        dropout=float(spec["dropout"]),
        bias=bool(spec["bias"]),
        tie_embeddings=bool(spec["tie_embeddings"]),
    )


def expected_proxy_parameter_count(protocol: Mapping[str, Any] | None = None) -> int:
    """Plan Section 6.2: 6,291,456 + 8 x 3,097,600 + 512 = 31,072,768."""
    return int(_resolved(protocol)["proxy_architecture"]["unique_trainable_parameters"])


def proxy_parameter_count_violations(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Check the declared count against its own frozen formula, without building a model."""
    spec = _resolved(protocol)["proxy_architecture"]
    embedding = int(spec["embedding_parameters"])
    per_layer = int(spec["per_layer_parameters"])
    final_norm = int(spec["final_norm_parameters"])
    layers = int(spec["n_layers"])
    declared = int(spec["unique_trainable_parameters"])
    computed = embedding + layers * per_layer + final_norm
    problems: list[str] = []
    if computed != declared:
        problems.append(
            f"{CAMPAIGN_PROXY_COUNT_MISMATCH}: {embedding} + {layers} x {per_layer} + "
            f"{final_norm} = {computed}, but the contract declares {declared}"
        )
    if embedding != int(spec["vocab_size"]) * int(spec["d_model"]):
        problems.append(
            f"{CAMPAIGN_PROXY_COUNT_MISMATCH}: tied embedding must be "
            f"{spec['vocab_size']} x {spec['d_model']}"
        )
    return tuple(problems)


def assert_proxy_parameter_count(
    counted: int, protocol: Mapping[str, Any] | None = None
) -> None:
    """Fail closed when an enumerated proxy differs from the frozen count."""
    expected = expected_proxy_parameter_count(protocol)
    if int(counted) != expected:
        raise CampaignContractError(
            f"{CAMPAIGN_PROXY_COUNT_MISMATCH}: enumerated {counted} unique trainable "
            f"parameters, frozen contract requires {expected}"
        )


# --------------------------------------------------------------------------------------
# Campaign runs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignRun:
    """One preregistered run. ``learning_rate`` stays a sentinel until P1/P4 select it."""

    run_id: str
    model: str
    mixture: str
    learning_rate: float | str
    seed: int
    tokens: int
    updates: int
    purpose: str
    comparison_group: str

    @property
    def learning_rate_is_pending(self) -> bool:
        return isinstance(self.learning_rate, str)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "mixture": self.mixture,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "tokens": self.tokens,
            "updates": self.updates,
            "purpose": self.purpose,
            "comparison_group": self.comparison_group,
        }


def _coerce_learning_rate(value: Any) -> float | str:
    if isinstance(value, str) and value == SELECTED_FROM_P1_P4:
        return SELECTED_FROM_P1_P4
    return float(value)


def campaign_runs(protocol: Mapping[str, Any] | None = None) -> tuple[CampaignRun, ...]:
    """Every preregistered run in its frozen declaration order."""
    resolved = _resolved(protocol)
    return tuple(
        CampaignRun(
            run_id=str(run["run_id"]),
            model=str(run["model"]),
            mixture=str(run["mixture"]),
            learning_rate=_coerce_learning_rate(run["learning_rate"]),
            seed=int(run["seed"]),
            tokens=int(run["tokens"]),
            updates=int(run["updates"]),
            purpose=str(run["purpose"]),
            comparison_group=str(run["comparison_group"]),
        )
        for run in resolved["runs"]
    )


def run_index(protocol: Mapping[str, Any] | None = None) -> dict[str, CampaignRun]:
    """Run ID -> its preregistered definition."""
    return {run.run_id: run for run in campaign_runs(protocol)}


def apply_contingency(
    runs: Sequence[CampaignRun], protocol: Mapping[str, Any] | None = None
) -> tuple[CampaignRun, ...]:
    """Shorten F1/F2 to the frozen contingency horizon, together and never separately.

    Plan Section 6.2 permits the shorter horizon only as a decision made before F1, applied
    to both runs. Applying it to one run would create exactly the unequal comparison the
    same sentence forbids, so this always rewrites the whole group.
    """
    resolved = _resolved(protocol)
    contingency = resolved["contingency"]
    affected = {str(name) for name in contingency["applies_to"]}
    tokens = int(contingency["tokens"])
    updates = int(contingency["updates"])
    return tuple(
        replace(run, tokens=tokens, updates=updates) if run.run_id in affected else run
        for run in runs
    )


# --------------------------------------------------------------------------------------
# Comparability (Plan Section 6.2: "never compare unequal run lengths")
# --------------------------------------------------------------------------------------


def comparison_definitions(protocol: Mapping[str, Any] | None = None) -> tuple[Mapping[str, Any], ...]:
    return tuple(_resolved(protocol)["comparisons"]["definitions"])


def comparison_violations(
    runs: Sequence[CampaignRun], protocol: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Every preregistered comparison must be between runs that differ only as intended."""
    resolved = _resolved(protocol)
    index = {run.run_id: run for run in runs}
    problems: list[str] = []

    for definition in comparison_definitions(resolved):
        comparison_id = str(definition["comparison_id"])
        if "members" in definition:
            members = [str(name) for name in definition["members"]]
        else:
            members = [str(definition["control"]), str(definition["candidate"])]
        present = [index[name] for name in members if name in index]
        if len(present) != len(members):
            missing = sorted(set(members) - set(index))
            problems.append(f"{CAMPAIGN_UNEQUAL_RUN_LENGTH}: {comparison_id} is missing runs {missing}")
            continue

        if bool(definition.get("equal_length_required")):
            lengths = {(run.tokens, run.updates) for run in present}
            if len(lengths) != 1:
                problems.append(
                    f"{CAMPAIGN_UNEQUAL_RUN_LENGTH}: {comparison_id} compares run lengths "
                    f"{sorted(lengths)}"
                )
        if bool(definition.get("equal_seed_required")):
            seeds = {run.seed for run in present}
            if len(seeds) != 1:
                problems.append(
                    f"{CAMPAIGN_SEED_NOT_COMPARABLE}: {comparison_id} compares seeds {sorted(seeds)}"
                )
        if bool(definition.get("equal_learning_rate_required")):
            rates = {run.learning_rate for run in present}
            if len(rates) != 1:
                problems.append(
                    f"{CAMPAIGN_LR_NOT_COMPARABLE}: {comparison_id} compares learning rates "
                    f"{sorted(str(rate) for rate in rates)}"
                )
    return tuple(problems)


def assert_comparisons_valid(
    runs: Sequence[CampaignRun], protocol: Mapping[str, Any] | None = None
) -> None:
    problems = comparison_violations(runs, protocol)
    if problems:
        raise CampaignContractError("; ".join(problems))


def branch_size_reference_violations(
    protocol: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Tie the campaign to the branch contract it delegates branch sizing to.

    Plan Section 8.1 sizes a branch from a measured throughput, and that table lives in
    ``configs/branches/exposure_v1.yaml``. Naming the file in prose proves nothing, so this
    loads it and confirms every declared band is a whole number of this campaign's updates.
    An imported branch size that is not update-aligned would silently break the update-aligned
    horizon the rest of this contract enforces.
    """
    # Imported lazily so the campaign contract stays loadable on its own.
    from .branches import BRANCH_PROTOCOL_PATH, branch_size_bands, load_branch_protocol

    resolved = _resolved(protocol)
    referenced = str(resolved["branch_sets"]["size_selection_protocol"])
    problems: list[str] = []

    expected_reference = BRANCH_PROTOCOL_PATH.relative_to(REPOSITORY_ROOT).as_posix()
    if referenced != expected_reference:
        problems.append(
            f"{CAMPAIGN_SET_COUNT_INVALID}: branch sizing is delegated to "
            f"{expected_reference!r}, but the contract names {referenced!r}"
        )
        return tuple(problems)

    branch_protocol = load_branch_protocol()
    loss_tokens = loss_tokens_per_update(resolved)
    for band in branch_size_bands(branch_protocol):
        for label, tokens, updates in (
            ("primary", band.primary_tokens_per_arm, band.primary_updates_per_arm),
            ("confirmation", band.confirmation_tokens_per_arm, band.confirmation_updates_per_arm),
        ):
            if tokens != updates * loss_tokens:
                problems.append(
                    f"{CAMPAIGN_TOKENS_NOT_UPDATE_ALIGNED}: band {band.band_id} {label} horizon "
                    f"{tokens} is not {updates} x {loss_tokens}"
                )
    return tuple(problems)


# --------------------------------------------------------------------------------------
# Decision thresholds (Plan Sections 6.1 and 8.6)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionThresholds:
    """The frozen bars a candidate must clear. Every one is required; none is optional."""

    confidence_level: float
    minimum_relative_reduction: float
    seed_sd_multiple: float
    maximum_slice_regression: float
    protected_slices: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_level": self.confidence_level,
            "minimum_relative_reduction": self.minimum_relative_reduction,
            "seed_sd_multiple": self.seed_sd_multiple,
            "maximum_slice_regression": self.maximum_slice_regression,
            "protected_slices": list(self.protected_slices),
        }


def decision_thresholds(protocol: Mapping[str, Any] | None = None) -> DecisionThresholds:
    """Read the frozen thresholds. This module never computes a statistic from them."""
    thresholds = _resolved(protocol)["decision_thresholds"]
    return DecisionThresholds(
        confidence_level=float(thresholds["bootstrap"]["confidence_level"]),
        minimum_relative_reduction=float(thresholds["relative_nll_reduction"]["minimum"]),
        seed_sd_multiple=float(thresholds["seed_sd_multiple"]["minimum"]),
        maximum_slice_regression=float(
            thresholds["protected_slice_regression"]["maximum_relative_regression"]
        ),
        protected_slices=tuple(
            str(name) for name in thresholds["protected_slice_regression"]["slices"]
        ),
    )


# --------------------------------------------------------------------------------------
# Mixture-screen preregistration (Plan Section 6.1)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MixtureScreenPreregistration:
    """The document that must be hashed before any proxy result is observed."""

    document: str
    sections: Mapping[str, str]
    run_ids: tuple[str, ...]
    update_count: int
    primary_endpoint: str
    delta: str
    campaign_protocol_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _CAMPAIGN_SCHEMA_VERSION,
            "document": self.document,
            "sections": {key: self.sections[key] for key in sorted(self.sections)},
            "run_ids": list(self.run_ids),
            "update_count": self.update_count,
            "primary_endpoint": self.primary_endpoint,
            "delta": self.delta,
            "campaign_protocol_digest": self.campaign_protocol_digest,
        }

    @property
    def content_hash(self) -> str:
        """The preregistration hash carried into every later binding."""
        return hashlib.sha256(canonical_payload_bytes(self.to_dict())).hexdigest()


def required_preregistration_sections(
    protocol: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    return tuple(
        str(name)
        for name in _resolved(protocol)["mixture_screen_preregistration"]["required_sections"]
    )


def build_mixture_screen_preregistration(
    sections: Mapping[str, str], protocol: Mapping[str, Any] | None = None
) -> MixtureScreenPreregistration:
    """Assemble the preregistration from supplied prose, failing closed on a missing section."""
    resolved = _resolved(protocol)
    spec = resolved["mixture_screen_preregistration"]
    missing = [name for name in required_preregistration_sections(resolved) if not str(sections.get(name, "")).strip()]
    if missing:
        raise CampaignPreregistrationError(
            f"{CAMPAIGN_PREREGISTRATION_INCOMPLETE}: missing sections {sorted(missing)}"
        )
    index = run_index(resolved)
    screen = [name for name in ("P1", "P8")]
    update_counts = {index[name].updates for name in screen}
    if len(update_counts) != 1:
        raise CampaignPreregistrationError(
            f"{CAMPAIGN_UNEQUAL_RUN_LENGTH}: P1 and P8 declare update counts {sorted(update_counts)}"
        )
    return MixtureScreenPreregistration(
        document=str(spec["document"]),
        sections=dict(sections),
        run_ids=tuple(screen),
        update_count=update_counts.pop(),
        primary_endpoint=str(spec["primary_endpoint"]),
        delta=str(spec["p8_comparison_delta"]),
        campaign_protocol_digest=campaign_protocol_digest(resolved),
    )


def preregistration_violations(
    preregistration: MixtureScreenPreregistration,
    *,
    outcomes_observed: bool,
    hash_recorded: bool,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """A preregistration recorded after an outcome was seen is not a preregistration."""
    resolved = _resolved(protocol)
    problems: list[str] = []
    missing = [
        name
        for name in required_preregistration_sections(resolved)
        if not str(preregistration.sections.get(name, "")).strip()
    ]
    if missing:
        problems.append(f"{CAMPAIGN_PREREGISTRATION_INCOMPLETE}: missing sections {sorted(missing)}")
    if outcomes_observed and not hash_recorded:
        problems.append(
            f"{CAMPAIGN_PREREGISTRATION_LATE}: a proxy outcome was observed before the "
            "preregistration hash was recorded"
        )
    if preregistration.campaign_protocol_digest != campaign_protocol_digest(resolved):
        problems.append(
            f"{CAMPAIGN_PREREGISTRATION_LATE}: the preregistration binds a different campaign "
            "contract than the one loaded"
        )
    return tuple(problems)


# --------------------------------------------------------------------------------------
# G4 freeze bundle (Plan Section 13 G4)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FreezeBundle:
    """One G4 hash over every frozen component, approved by two distinct people."""

    components: Mapping[str, str]
    approvers: tuple[str, ...]
    evidence: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _CAMPAIGN_SCHEMA_VERSION,
            "components": {key: self.components[key] for key in sorted(self.components)},
            "approvers": sorted(self.approvers),
            "evidence": {key: self.evidence[key] for key in sorted(self.evidence)},
        }

    @property
    def bundle_hash(self) -> str:
        """The single hash Plan Section 13 G4 requires two people to approve."""
        return hashlib.sha256(canonical_payload_bytes(self.to_dict())).hexdigest()


def required_freeze_components(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    return tuple(str(name) for name in _resolved(protocol)["freeze_bundle"]["required_components"])


def required_freeze_evidence(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    return tuple(str(name) for name in _resolved(protocol)["freeze_bundle"]["required_evidence"])


def freeze_bundle_violations(
    bundle: FreezeBundle, protocol: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Fail closed on a missing component, too few approvers, or absent measurement."""
    resolved = _resolved(protocol)
    spec = resolved["freeze_bundle"]
    problems: list[str] = []

    missing = [
        name
        for name in required_freeze_components(resolved)
        if not str(bundle.components.get(name, "")).strip()
    ]
    if missing:
        problems.append(f"{CAMPAIGN_FREEZE_BUNDLE_INCOMPLETE}: missing components {sorted(missing)}")

    required_approvals = int(spec["approvals_required"])
    distinct = set(approver.strip() for approver in bundle.approvers if approver.strip())
    if len(distinct) < required_approvals:
        problems.append(
            f"{CAMPAIGN_FREEZE_APPROVAL_INSUFFICIENT}: {len(distinct)} distinct approver(s), "
            f"{required_approvals} required"
        )
    if bool(spec["distinct_approvers_required"]) and len(distinct) != len(
        [approver for approver in bundle.approvers if approver.strip()]
    ):
        problems.append(
            f"{CAMPAIGN_FREEZE_APPROVAL_INSUFFICIENT}: the same approver was counted twice"
        )

    absent = [
        name
        for name in required_freeze_evidence(resolved)
        if str(bundle.evidence.get(name, "")).strip() in ("", NOT_RUN, FAIL, BLOCKED)
    ]
    if absent:
        problems.append(
            f"{CAMPAIGN_FREEZE_EVIDENCE_MISSING}: {sorted(absent)} must pass before G4; "
            "absence of evidence is never a PASS"
        )
    return tuple(problems)


def assert_freeze_bundle_valid(
    bundle: FreezeBundle, protocol: Mapping[str, Any] | None = None
) -> None:
    problems = freeze_bundle_violations(bundle, protocol)
    if problems:
        raise CampaignPreregistrationError("; ".join(problems))


# --------------------------------------------------------------------------------------
# Append-only parent binding (Plan Section 8.5)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParentBindingEntry:
    """One immutable binding of an A/B/C set to the parent checkpoint it branches from."""

    set_id: str
    arm_ids: tuple[str, ...]
    target_parent_tokens: int
    parent_checkpoint_hash: str
    bound_at: str
    bound_by: str
    preregistration_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "arm_ids": list(self.arm_ids),
            "target_parent_tokens": self.target_parent_tokens,
            "parent_checkpoint_hash": self.parent_checkpoint_hash,
            "bound_at": self.bound_at,
            "bound_by": self.bound_by,
            "preregistration_hash": self.preregistration_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParentBindingEntry":
        return cls(
            set_id=str(payload["set_id"]),
            arm_ids=tuple(str(arm) for arm in payload["arm_ids"]),
            target_parent_tokens=int(payload["target_parent_tokens"]),
            parent_checkpoint_hash=str(payload["parent_checkpoint_hash"]),
            bound_at=str(payload["bound_at"]),
            bound_by=str(payload["bound_by"]),
            preregistration_hash=str(payload["preregistration_hash"]),
        )

    @property
    def is_pending(self) -> bool:
        return self.parent_checkpoint_hash == PENDING_PARENT_HASH


@dataclass(frozen=True)
class ParentBindingManifest:
    """An append-only log. Rewriting an entry is the failure Plan Section 8.5 forbids."""

    entries: tuple[ParentBindingEntry, ...] = ()
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParentBindingManifest":
        return cls(
            entries=tuple(ParentBindingEntry.from_dict(item) for item in payload.get("entries", ())),
            schema_version=int(payload.get("schema_version", 1)),
        )

    def appended(self, entry: ParentBindingEntry) -> "ParentBindingManifest":
        """Return a new manifest with one entry added; never mutates an existing entry."""
        if any(existing.set_id == entry.set_id for existing in self.entries):
            raise CampaignPreregistrationError(
                f"{CAMPAIGN_PARENT_MANIFEST_REWRITTEN}: set {entry.set_id!r} is already bound. "
                "A parent binding is append-only and may never be replaced."
            )
        return ParentBindingManifest(self.entries + (entry,), self.schema_version)


def bind_parent(
    manifest: ParentBindingManifest,
    *,
    set_id: str,
    arm_ids: Iterable[str],
    target_parent_tokens: int,
    parent_checkpoint_hash: str,
    bound_at: str,
    bound_by: str,
    preregistration_hash: str,
    arm_outcomes_observed: bool = False,
    protocol: Mapping[str, Any] | None = None,
) -> ParentBindingManifest:
    """Append one binding, refusing a pending hash or a binding made after an outcome."""
    resolved = _resolved(protocol)
    if arm_outcomes_observed:
        raise CampaignPreregistrationError(
            f"{CAMPAIGN_PARENT_BOUND_AFTER_OUTCOME}: Plan Section 8.5 binds the parent hash "
            "before any arm runs or any arm outcome is observed"
        )
    if parent_checkpoint_hash == PENDING_PARENT_HASH or not parent_checkpoint_hash.strip():
        raise CampaignPreregistrationError(
            f"{CAMPAIGN_PARENT_HASH_PENDING}: a real parent checkpoint hash must exist before "
            "it is bound; the sentinel is not a hash"
        )
    expected_arms = tuple(str(arm) for arm in resolved["branch_sets"]["arms_per_set"])
    supplied = tuple(str(arm) for arm in arm_ids)
    if supplied != expected_arms:
        raise CampaignPreregistrationError(
            f"{CAMPAIGN_SET_COUNT_INVALID}: a set binds arms {expected_arms}, got {supplied}"
        )
    entry = ParentBindingEntry(
        set_id=str(set_id),
        arm_ids=supplied,
        target_parent_tokens=int(target_parent_tokens),
        parent_checkpoint_hash=str(parent_checkpoint_hash),
        bound_at=str(bound_at),
        bound_by=str(bound_by),
        preregistration_hash=str(preregistration_hash),
    )
    return manifest.appended(entry)


def append_only_violations(
    earlier: ParentBindingManifest, later: ParentBindingManifest
) -> tuple[str, ...]:
    """Compare two revisions of the manifest; only appends are permitted."""
    problems: list[str] = []
    if len(later.entries) < len(earlier.entries):
        problems.append(
            f"{CAMPAIGN_PARENT_MANIFEST_REWRITTEN}: the manifest shrank from "
            f"{len(earlier.entries)} to {len(later.entries)} entries"
        )
    for index, previous in enumerate(earlier.entries):
        if index >= len(later.entries):
            break
        if later.entries[index] != previous:
            problems.append(
                f"{CAMPAIGN_PARENT_MANIFEST_REWRITTEN}: entry {index} "
                f"({previous.set_id}) was modified after it was written"
            )
    return tuple(problems)


def write_parent_manifest(path: Path, manifest: ParentBindingManifest) -> Path:
    """Persist the manifest as sorted, deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def read_parent_manifest(path: Path) -> ParentBindingManifest:
    """Load a manifest, returning an empty one when nothing has been bound yet."""
    if not path.is_file():
        return ParentBindingManifest()
    return ParentBindingManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------------------
# Readiness. Absence of evidence is never PASS.
# --------------------------------------------------------------------------------------

_BLOCKER_FIELDS = ("blocker", "owner", "next_action")

_READINESS_GATED = (
    "proxy_runs_completed",
    "selected_mixture",
    "selected_peak_learning_rate",
    "measured_seed_sd",
    "freeze_bundle_approved",
    "bound_parent_checkpoint_hash",
)


def assert_ready_for_campaign_runs(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: a campaign decision needs measured proxy evidence that does not exist."""
    readiness = _resolved(protocol)["readiness"]
    blocked = [name for name in _READINESS_GATED if str(readiness.get(name)) != PASS]
    if blocked:
        raise CampaignNotReadyError(
            f"campaign decisions are not ready: {blocked}. "
            f"blocker={readiness.get('blocker')} owner={readiness.get('owner')} "
            f"next_action={readiness.get('next_action')}"
        )


def _verdict(check_id: str, requirement: str, observed: Any, ok: bool, reason: str) -> CheckResult:
    return CheckResult(check_id, requirement, str(observed), PASS if ok else FAIL, reason)


def readiness_results(protocol: Mapping[str, Any] | None = None) -> tuple[CheckResult, ...]:
    """Report each unmeasured prerequisite as NOT_RUN/BLOCKED with its own next action."""
    resolved = _resolved(protocol)
    readiness = resolved["readiness"]
    detail = {name: readiness.get(name) for name in _BLOCKER_FIELDS}
    named = all(str(value or "").strip() for value in detail.values())
    results: list[CheckResult] = []
    for name, value in readiness.items():
        if name in _BLOCKER_FIELDS or not isinstance(value, str):
            continue
        status = str(value)
        if status == PASS:
            results.append(
                _verdict(f"campaign.readiness.{name}", "measured evidence exists", status, True, CAMPAIGN_OK)
            )
            continue
        if not named:
            results.append(
                CheckResult(
                    f"campaign.readiness.{name}",
                    "an unmeasured prerequisite must name its blocker, owner, and next action",
                    status,
                    FAIL,
                    f"{CAMPAIGN_THRESHOLD_NOT_MET}: readiness detail is incomplete",
                )
            )
            continue
        results.append(
            CheckResult(
                f"campaign.readiness.{name}",
                "measured evidence exists",
                status,
                status,
                f"blocker={detail['blocker']} owner={detail['owner']} next_action={detail['next_action']}",
            )
        )
    return tuple(results)


# --------------------------------------------------------------------------------------
# Verification report
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignReport:
    """Every frozen campaign check, plus the honest absences."""

    results: tuple[CheckResult, ...]

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status == FAIL)

    @property
    def not_run(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status == NOT_RUN)

    @property
    def blocked(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status == BLOCKED)

    @property
    def ok(self) -> bool:
        return not self.failures


def verify_campaign(protocol: Mapping[str, Any] | None = None) -> CampaignReport:
    """Run every frozen structural check. Absent evidence is NOT_RUN, never PASS."""
    resolved = _resolved(protocol)
    runs = campaign_runs(resolved)
    loss_tokens = loss_tokens_per_update(resolved)
    results: list[CheckResult] = []

    count_problems = proxy_parameter_count_violations(resolved)
    results.append(
        _verdict(
            "campaign.proxy_parameter_count",
            f"{expected_proxy_parameter_count(resolved)} unique trainable parameters",
            count_problems or expected_proxy_parameter_count(resolved),
            not count_problems,
            CAMPAIGN_OK if not count_problems else "; ".join(count_problems),
        )
    )

    unaligned = [
        run.run_id for run in runs if run.tokens != run.updates * loss_tokens
    ]
    results.append(
        _verdict(
            "campaign.update_alignment",
            f"every horizon is a whole number of {loss_tokens}-token updates",
            unaligned or "all aligned",
            not unaligned,
            CAMPAIGN_OK if not unaligned else f"{CAMPAIGN_TOKENS_NOT_UPDATE_ALIGNED}: {unaligned}",
        )
    )

    comparison_problems = comparison_violations(runs, resolved)
    results.append(
        _verdict(
            "campaign.comparability",
            "every comparison holds run length, seed, and LR fixed except as intended",
            comparison_problems or "comparable",
            not comparison_problems,
            CAMPAIGN_OK if not comparison_problems else "; ".join(comparison_problems),
        )
    )

    contingency_runs = apply_contingency(runs, resolved)
    contingency_problems = comparison_violations(contingency_runs, resolved)
    results.append(
        _verdict(
            "campaign.contingency_comparability",
            "the shortened F1/F2 horizon stays equal-length",
            contingency_problems or "comparable",
            not contingency_problems,
            CAMPAIGN_OK if not contingency_problems else "; ".join(contingency_problems),
        )
    )

    thresholds = decision_thresholds(resolved)
    frozen_thresholds = (
        thresholds.confidence_level == 0.95
        and thresholds.minimum_relative_reduction == 0.003
        and thresholds.seed_sd_multiple == 2.0
        and thresholds.maximum_slice_regression == 0.01
        and thresholds.protected_slices == EXPECTED_PROTECTED_SLICES
    )
    results.append(
        _verdict(
            "campaign.decision_thresholds",
            "95% CI below zero, >=0.3% relative reduction, >2x seed SD, <=1% slice regression",
            thresholds.to_dict(),
            frozen_thresholds,
            CAMPAIGN_OK if frozen_thresholds else CAMPAIGN_THRESHOLD_NOT_MET,
        )
    )

    sets = resolved["branch_sets"]
    set_counts_ok = (
        int(sets["primary_sets"]) == 1
        and int(sets["confirmation_sets"]) == 1
        and bool(sets["confirmation_precedes_primary"])
        and bool(sets["same_direction_confirmation_required"])
    )
    results.append(
        _verdict(
            "campaign.branch_sets",
            "one primary A/B/C set and one earlier same-direction confirmation",
            {
                "primary_sets": sets["primary_sets"],
                "confirmation_sets": sets["confirmation_sets"],
            },
            set_counts_ok,
            CAMPAIGN_OK if set_counts_ok else CAMPAIGN_SET_COUNT_INVALID,
        )
    )

    custody = resolved["validation_final_custody"]
    custody_ok = str(custody["status"]) == "NOT_OPENED" and bool(custody["opened_once"])
    results.append(
        _verdict(
            "campaign.validation_final_custody",
            "validation_final is opened once, after all branch runs",
            custody["status"],
            custody_ok,
            CAMPAIGN_OK if custody_ok else CAMPAIGN_VALIDATION_FINAL_OPENED_EARLY,
        )
    )

    size_problems = branch_size_reference_violations(resolved)
    results.append(
        _verdict(
            "campaign.branch_size_protocol",
            "the referenced branch contract exists and its bands are update-aligned",
            size_problems or str(resolved["branch_sets"]["size_selection_protocol"]),
            not size_problems,
            CAMPAIGN_OK if not size_problems else "; ".join(size_problems),
        )
    )

    binding = resolved["parent_binding"]
    binding_ok = (
        not bool(binding["hash_exists_at_freeze_time"])
        and bool(binding["append_only"])
        and bool(binding["rewrite_forbidden"])
        and str(binding["pending_sentinel"]) == PENDING_PARENT_HASH
    )
    results.append(
        _verdict(
            "campaign.parent_binding_rule",
            "an append-only manifest binds a real parent hash before any arm outcome",
            binding["pending_sentinel"],
            binding_ok,
            CAMPAIGN_OK if binding_ok else CAMPAIGN_PARENT_MANIFEST_REWRITTEN,
        )
    )

    results.extend(readiness_results(resolved))
    return CampaignReport(tuple(results))


def assert_campaign_valid(protocol: Mapping[str, Any] | None = None) -> None:
    """Raise on any FAIL. NOT_RUN and BLOCKED are honest absences, not failures."""
    report = verify_campaign(protocol)
    if not report.ok:
        raise CampaignContractError(
            "; ".join(f"{result.check_id}: {result.reason}" for result in report.failures)
        )


def format_campaign_report(results: Sequence[CheckResult], title: str = "") -> str:
    """Render one aligned, greppable status block."""
    lines = [title] if title else []
    width = max((len(result.check_id) for result in results), default=0)
    for result in results:
        lines.append(f"{result.status:<8} {result.check_id:<{width}}  {result.reason}")
    return "\n".join(lines)


__all__ = [
    "BLOCKED",
    "CAMPAIGN_FAIL_CLOSED_REASON_CODES",
    "CAMPAIGN_OK",
    "CAMPAIGN_PROTOCOL_PATH",
    "CAMPAIGN_RUN_IDS",
    "CAMPAIGN_CONFIRMATION_MISSING",
    "CAMPAIGN_CONTINGENCY_DECIDED_LATE",
    "CAMPAIGN_FREEZE_APPROVAL_INSUFFICIENT",
    "CAMPAIGN_FREEZE_BUNDLE_INCOMPLETE",
    "CAMPAIGN_FREEZE_EVIDENCE_MISSING",
    "CAMPAIGN_LR_NOT_COMPARABLE",
    "CAMPAIGN_MIXTURE_HELD_SOURCE_CHANGED",
    "CAMPAIGN_MIXTURE_SHARES_INVALID",
    "CAMPAIGN_PARENT_BOUND_AFTER_OUTCOME",
    "CAMPAIGN_PARENT_HASH_PENDING",
    "CAMPAIGN_PARENT_MANIFEST_REWRITTEN",
    "CAMPAIGN_PREREGISTRATION_INCOMPLETE",
    "CAMPAIGN_PREREGISTRATION_LATE",
    "CAMPAIGN_PROTECTED_SLICE_REGRESSION",
    "CAMPAIGN_PROXY_COUNT_MISMATCH",
    "CAMPAIGN_SEED_NOT_COMPARABLE",
    "CAMPAIGN_SET_COUNT_INVALID",
    "CAMPAIGN_THRESHOLD_NOT_MET",
    "CAMPAIGN_TOKENS_NOT_UPDATE_ALIGNED",
    "CAMPAIGN_UNEQUAL_RUN_LENGTH",
    "CAMPAIGN_VALIDATION_FINAL_OPENED_EARLY",
    "FINAL_RUN_IDS",
    "FROZEN_CAMPAIGN_PROTOCOL_SHA256",
    "MIXTURE_BASE",
    "MIXTURE_EDU",
    "MODEL_FINAL",
    "MODEL_PROXY",
    "PENDING_PARENT_HASH",
    "PROXY_RUN_IDS",
    "SELECTED_FROM_P1_P4",
    "CampaignContractError",
    "CampaignNotReadyError",
    "CampaignPreregistrationError",
    "CampaignReport",
    "CampaignRun",
    "DecisionThresholds",
    "FreezeBundle",
    "MixtureScreenPreregistration",
    "ParentBindingEntry",
    "ParentBindingManifest",
    "append_only_violations",
    "apply_contingency",
    "assert_campaign_valid",
    "assert_comparisons_valid",
    "assert_freeze_bundle_valid",
    "assert_proxy_parameter_count",
    "assert_ready_for_campaign_runs",
    "bind_parent",
    "branch_size_reference_violations",
    "build_mixture_screen_preregistration",
    "campaign_protocol_digest",
    "campaign_runs",
    "comparison_definitions",
    "comparison_violations",
    "decision_thresholds",
    "expected_proxy_parameter_count",
    "format_campaign_report",
    "freeze_bundle_violations",
    "load_campaign_protocol",
    "loss_tokens_per_update",
    "preregistration_violations",
    "proxy_model_config",
    "proxy_parameter_count_violations",
    "read_parent_manifest",
    "readiness_results",
    "required_freeze_components",
    "required_freeze_evidence",
    "required_preregistration_sections",
    "run_index",
    "verify_campaign",
    "write_parent_manifest",
]
