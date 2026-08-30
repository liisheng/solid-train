"""Preregistered campaign configurations and decision validators (Plan Sections 6, 8, 12-13).

Nothing here runs a proxy, trains an arm, binds a real parent checkpoint, opens
``validation_final``, or claims a gate pass. The tests prove that the campaign's *rules* are
frozen before its outcomes exist, which is the only thing that makes a later comparison
evidence rather than a selection:

- the frozen contract is pinned by digest, and a structurally invalid edit fails before the
  digest is even consulted,
- the proxy is the final architecture reduced to 8 layers and nothing else, and its
  enumerated unique-parameter count is exactly Plan Section 6.2's 31,072,768,
- every preregistered horizon is a whole number of 262,144-loss-token updates,
- every comparison holds run length, seed, and LR fixed except as intended, and the F1/F2
  contingency shortens both runs together or not at all,
- the decision thresholds are exactly 95% CI below zero, >=0.3% relative reduction, >2x the
  measured seed SD, and <=1% protected-slice regression,
- a preregistration missing a section, or hashed after an outcome was observed, fails closed,
- a G4 freeze bundle with a missing component, a duplicated approver, or absent measurement
  fails closed,
- the parent-binding manifest is append-only: a rewrite, a pending sentinel, and a binding
  made after an arm outcome all fail closed,
- absence of evidence is reported NOT_RUN/BLOCKED with an owner and next action, never PASS.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tinybench_lm import TinyBenchLM
from tinybench_lm.campaign import (
    BLOCKED,
    CAMPAIGN_FAIL_CLOSED_REASON_CODES,
    CAMPAIGN_FREEZE_APPROVAL_INSUFFICIENT,
    CAMPAIGN_FREEZE_BUNDLE_INCOMPLETE,
    CAMPAIGN_FREEZE_EVIDENCE_MISSING,
    CAMPAIGN_LR_NOT_COMPARABLE,
    CAMPAIGN_PARENT_MANIFEST_REWRITTEN,
    CAMPAIGN_PREREGISTRATION_INCOMPLETE,
    CAMPAIGN_PREREGISTRATION_LATE,
    CAMPAIGN_PROTOCOL_PATH,
    CAMPAIGN_RUN_IDS,
    CAMPAIGN_SEED_NOT_COMPARABLE,
    CAMPAIGN_UNEQUAL_RUN_LENGTH,
    FROZEN_CAMPAIGN_PROTOCOL_SHA256,
    MIXTURE_BASE,
    MIXTURE_EDU,
    MODEL_FINAL,
    MODEL_PROXY,
    PENDING_PARENT_HASH,
    SELECTED_FROM_P1_P4,
    CampaignContractError,
    CampaignNotReadyError,
    CampaignPreregistrationError,
    FreezeBundle,
    ParentBindingManifest,
    append_only_violations,
    apply_contingency,
    assert_campaign_valid,
    assert_comparisons_valid,
    assert_freeze_bundle_valid,
    assert_proxy_parameter_count,
    assert_ready_for_campaign_runs,
    bind_parent,
    branch_size_reference_violations,
    build_mixture_screen_preregistration,
    campaign_runs,
    comparison_violations,
    decision_thresholds,
    expected_proxy_parameter_count,
    freeze_bundle_violations,
    load_campaign_protocol,
    loss_tokens_per_update,
    preregistration_violations,
    proxy_model_config,
    read_parent_manifest,
    readiness_results,
    required_freeze_components,
    required_freeze_evidence,
    required_preregistration_sections,
    run_index,
    verify_campaign,
    write_parent_manifest,
)
from tinybench_lm.data_protocols import ProtocolMutatedError, protocol_digest
from tinybench_lm.parameters import count_unique_trainable_parameters
from tinybench_lm.shards import EXPECTED_PROTECTED_SLICES, FAIL, NOT_RUN, PASS

#: Plan Section 6.2 verbatim.
PROXY_PARAMETERS = 31_072_768
PROXY_UPDATES = 573
PROXY_TOKENS = 150_208_512
FINAL_SAFETY_UPDATES = 191
FINAL_SAFETY_TOKENS = 50_069_504
CONTINGENCY_UPDATES = 96
CONTINGENCY_TOKENS = 25_165_824
LOSS_TOKENS_PER_UPDATE = 262_144


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_campaign_protocol()


def _sections(protocol: dict) -> dict[str, str]:
    return {name: f"frozen text for {name}" for name in required_preregistration_sections(protocol)}


def _bundle(protocol: dict) -> FreezeBundle:
    return FreezeBundle(
        components={name: f"sha256:{name}" for name in required_freeze_components(protocol)},
        approvers=("teammate_one", "teammate_two"),
        evidence={name: PASS for name in required_freeze_evidence(protocol)},
    )


# --------------------------------------------------------------------------------------
# The frozen contract
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_frozen_campaign_protocol_is_pinned_and_immutable(tmp_path: Path) -> None:
    assert (
        protocol_digest(CAMPAIGN_PROTOCOL_PATH)
        == FROZEN_CAMPAIGN_PROTOCOL_SHA256["preregistration_v1.yaml"]
    )

    text = CAMPAIGN_PROTOCOL_PATH.read_text(encoding="utf-8")

    # A structurally valid edit no other check would notice: the digest is what catches it.
    mutated = tmp_path / "preregistration_v1.yaml"
    mutated.write_text(text.replace("owner: operator", "owner: nobody"), encoding="utf-8")
    with pytest.raises(ProtocolMutatedError):
        load_campaign_protocol(mutated)
    # An unverified load still parses, so a proposed v2 can be reviewed before it is pinned.
    assert str(load_campaign_protocol(mutated, verify=False)["readiness"]["owner"]) == "nobody"

    # A threshold moved after the fact fails even before the digest is consulted.
    loosened = tmp_path / "loosened_v1.yaml"
    loosened.write_text(text.replace("minimum: 0.003", "minimum: 0.001"), encoding="utf-8")
    with pytest.raises(CampaignContractError):
        load_campaign_protocol(loosened, verify=False)

    # So does an unequal comparison smuggled in as a shortened run.
    unequal = tmp_path / "unequal_v1.yaml"
    unequal.write_text(text.replace("    updates: 573\n    purpose: mixture comparison", "    updates: 500\n    purpose: mixture comparison"), encoding="utf-8")
    with pytest.raises(CampaignContractError):
        load_campaign_protocol(unequal, verify=False)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_frozen_contract_declares_the_plan_6_2_runs(protocol: dict) -> None:
    runs = campaign_runs(protocol)
    assert tuple(run.run_id for run in runs) == CAMPAIGN_RUN_IDS
    assert loss_tokens_per_update(protocol) == LOSS_TOKENS_PER_UPDATE

    index = run_index(protocol)
    for run_id in ("P1", "P2", "P3", "P4", "P8"):
        assert index[run_id].model == MODEL_PROXY
        assert index[run_id].updates == PROXY_UPDATES
        assert index[run_id].tokens == PROXY_TOKENS
    for run_id in ("F1", "F2"):
        assert index[run_id].model == MODEL_FINAL
        assert index[run_id].updates == FINAL_SAFETY_UPDATES
        assert index[run_id].tokens == FINAL_SAFETY_TOKENS
        # Plan Section 6.2: the peak LR is selected by P1/P4 and does not exist yet.
        assert index[run_id].learning_rate == SELECTED_FROM_P1_P4
        assert index[run_id].learning_rate_is_pending

    # P1-P3 are three seeds of one configuration; P4 changes only the LR; P8 only the mixture.
    assert {index[name].seed for name in ("P1", "P2", "P3")} == {1001, 1002, 1003}
    assert index["P4"].seed == index["P1"].seed
    assert index["P4"].learning_rate != index["P1"].learning_rate
    assert index["P4"].mixture == index["P1"].mixture == MIXTURE_BASE
    assert index["P8"].seed == index["P1"].seed
    assert index["P8"].learning_rate == index["P1"].learning_rate
    assert index["P8"].mixture == MIXTURE_EDU


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_every_preregistered_horizon_is_update_aligned(protocol: dict) -> None:
    loss_tokens = loss_tokens_per_update(protocol)
    for run in campaign_runs(protocol):
        assert run.tokens == run.updates * loss_tokens
        assert run.tokens % loss_tokens == 0
    contingency = protocol["contingency"]
    assert int(contingency["tokens"]) == CONTINGENCY_TOKENS
    assert int(contingency["updates"]) == CONTINGENCY_UPDATES
    assert CONTINGENCY_TOKENS == CONTINGENCY_UPDATES * loss_tokens


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_mixtures_move_exactly_fifteen_points_and_hold_the_rest_fixed(protocol: dict) -> None:
    mixtures = protocol["mixtures"]
    base = mixtures[MIXTURE_BASE]["shares"]
    edu = mixtures[MIXTURE_EDU]["shares"]
    assert base == {"fineweb_edu": 70, "general_web": 20, "openwebmath": 7, "narrative": 3}
    assert edu == {"fineweb_edu": 85, "general_web": 5, "openwebmath": 7, "narrative": 3}
    assert sum(base.values()) == sum(edu.values()) == 100
    # The manipulation is one transfer, so the screen measures the mixture and nothing else.
    for held in mixtures["held_fixed_sources"]:
        assert base[held] == edu[held]
    assert base["general_web"] - edu["general_web"] == 15
    assert edu["fineweb_edu"] - base["fineweb_edu"] == 15


# --------------------------------------------------------------------------------------
# Proxy architecture (Plan Section 6.2)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1, 3.3**
def test_proxy_is_the_final_architecture_reduced_to_eight_layers(protocol: dict) -> None:
    proxy = proxy_model_config(protocol)
    final = json.loads(Path("configs/final_49m.json").read_text(encoding="utf-8"))

    assert proxy.n_layers == 8
    assert final["n_layers"] == 14
    # n_layers is the ONLY permitted difference; anything else would make the proxy
    # uninformative about the final model.
    for field_name, value in final.items():
        if field_name == "n_layers":
            continue
        assert getattr(proxy, field_name) == value, field_name


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1, 3.3**
def test_proxy_parameter_count_is_exactly_plan_6_2(protocol: dict) -> None:
    assert expected_proxy_parameter_count(protocol) == PROXY_PARAMETERS
    # 6,291,456 + 8 x 3,097,600 + 512
    assert 6_291_456 + 8 * 3_097_600 + 512 == PROXY_PARAMETERS

    model = TinyBenchLM(proxy_model_config(protocol))
    counted = count_unique_trainable_parameters(model)
    assert counted == PROXY_PARAMETERS
    assert_proxy_parameter_count(counted, protocol)

    with pytest.raises(CampaignContractError, match="CAMPAIGN_PROXY_COUNT_MISMATCH"):
        assert_proxy_parameter_count(counted + 1, protocol)


# --------------------------------------------------------------------------------------
# Comparability (Plan Section 6.2: "never compare unequal run lengths")
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
def test_preregistered_comparisons_are_comparable(protocol: dict) -> None:
    runs = campaign_runs(protocol)
    assert comparison_violations(runs, protocol) == ()
    assert_comparisons_valid(runs, protocol)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_unequal_run_length_seed_or_lr_fails_closed(protocol: dict) -> None:
    index = run_index(protocol)

    # A shortened P8 would turn the mixture screen into a length comparison.
    shortened = [replace(run, updates=400, tokens=400 * LOSS_TOKENS_PER_UPDATE) if run.run_id == "P8" else run for run in index.values()]
    problems = comparison_violations(shortened, protocol)
    assert any(CAMPAIGN_UNEQUAL_RUN_LENGTH in problem for problem in problems)
    with pytest.raises(CampaignContractError):
        assert_comparisons_valid(shortened, protocol)

    # A reseeded P8 would confound the mixture effect with seed noise.
    reseeded = [replace(run, seed=9999) if run.run_id == "P8" else run for run in index.values()]
    assert any(CAMPAIGN_SEED_NOT_COMPARABLE in problem for problem in comparison_violations(reseeded, protocol))

    # A retuned P8 would confound the mixture effect with the learning rate.
    retuned = [replace(run, learning_rate=1e-3) if run.run_id == "P8" else run for run in index.values()]
    assert any(CAMPAIGN_LR_NOT_COMPARABLE in problem for problem in comparison_violations(retuned, protocol))


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_contingency_shortens_f1_and_f2_together_or_not_at_all(protocol: dict) -> None:
    runs = campaign_runs(protocol)
    shortened = apply_contingency(runs, protocol)
    index = {run.run_id: run for run in shortened}

    for run_id in ("F1", "F2"):
        assert index[run_id].updates == CONTINGENCY_UPDATES
        assert index[run_id].tokens == CONTINGENCY_TOKENS
    # The proxy runs are untouched by the F1/F2 calendar decision.
    for run_id in ("P1", "P2", "P3", "P4", "P8"):
        assert index[run_id].updates == PROXY_UPDATES

    # Still equal-length, which is the whole constraint Plan 6.2 attaches to the contingency.
    assert comparison_violations(shortened, protocol) == ()

    # Shortening only one of the pair is exactly what "never compare unequal run lengths" forbids.
    half = [replace(run, updates=CONTINGENCY_UPDATES, tokens=CONTINGENCY_TOKENS) if run.run_id == "F1" else run for run in runs]
    lengths = {(run.tokens, run.updates) for run in half if run.run_id in ("F1", "F2")}
    assert len(lengths) == 2


# --------------------------------------------------------------------------------------
# Decision thresholds (Plan Sections 6.1 and 8.6)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
def test_decision_thresholds_are_frozen_exactly(protocol: dict) -> None:
    thresholds = decision_thresholds(protocol)
    assert thresholds.confidence_level == 0.95
    assert thresholds.minimum_relative_reduction == 0.003
    assert thresholds.seed_sd_multiple == 2.0
    assert thresholds.maximum_slice_regression == 0.01
    assert thresholds.protected_slices == EXPECTED_PROTECTED_SLICES

    rules = protocol["decision_thresholds"]
    assert rules["bootstrap"]["method"] == "paired_document_bootstrap"
    assert bool(rules["bootstrap"]["deterministic_seed_required"])
    assert bool(rules["all_conditions_required"])
    # Plan 6.1: "Otherwise select M-base and report the result as null or negative."
    assert "null or negative" in str(rules["otherwise"])
    assert rules["protected_slice_regression"]["control_for_mixture_screen"] == "M-base"
    assert rules["protected_slice_regression"]["control_for_innovation_claim"] == "arm_B"


# --------------------------------------------------------------------------------------
# Mixture-screen preregistration (Plan Section 6.1)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_preregistration_is_complete_and_hash_stable(protocol: dict) -> None:
    prereg = build_mixture_screen_preregistration(_sections(protocol), protocol)
    assert prereg.document == "PREREGISTRATION-mixture-screen.md"
    assert prereg.run_ids == ("P1", "P8")
    assert prereg.update_count == PROXY_UPDATES
    assert prereg.primary_endpoint == "validation_dev mean NLL"

    # The hash is a function of content only, so an unchanged document rehashes identically.
    again = build_mixture_screen_preregistration(_sections(protocol), protocol)
    assert prereg.content_hash == again.content_hash
    assert len(prereg.content_hash) == 64

    # An edited section changes the hash, which is what makes a late edit visible.
    edited = dict(_sections(protocol))
    edited["selection_rule"] = "select whichever mixture wins"
    assert build_mixture_screen_preregistration(edited, protocol).content_hash != prereg.content_hash


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_incomplete_or_late_preregistration_fails_closed(protocol: dict) -> None:
    incomplete = dict(_sections(protocol))
    incomplete.pop("selection_rule")
    with pytest.raises(CampaignPreregistrationError, match=CAMPAIGN_PREREGISTRATION_INCOMPLETE):
        build_mixture_screen_preregistration(incomplete, protocol)

    blank = dict(_sections(protocol))
    blank["primary_endpoint"] = "   "
    with pytest.raises(CampaignPreregistrationError, match=CAMPAIGN_PREREGISTRATION_INCOMPLETE):
        build_mixture_screen_preregistration(blank, protocol)

    prereg = build_mixture_screen_preregistration(_sections(protocol), protocol)
    assert preregistration_violations(prereg, outcomes_observed=False, hash_recorded=True, protocol=protocol) == ()
    assert preregistration_violations(prereg, outcomes_observed=False, hash_recorded=False, protocol=protocol) == ()

    # Plan 6.1: "Before any proxy result, hash PREREGISTRATION-mixture-screen.md."
    late = preregistration_violations(prereg, outcomes_observed=True, hash_recorded=False, protocol=protocol)
    assert any(CAMPAIGN_PREREGISTRATION_LATE in problem for problem in late)


# --------------------------------------------------------------------------------------
# G4 freeze bundle (Plan Section 13 G4)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_freeze_bundle_requires_every_component_and_two_distinct_approvers(protocol: dict) -> None:
    bundle = _bundle(protocol)
    assert freeze_bundle_violations(bundle, protocol) == ()
    assert_freeze_bundle_valid(bundle, protocol)
    assert len(bundle.bundle_hash) == 64

    # All thirteen Plan 13 G4 components are required.
    assert set(required_freeze_components(protocol)) == {
        "code", "environment", "model", "tokenizer", "corpus", "mixture", "optimizer",
        "schedule", "evaluation", "parent_selection_rule", "branch_calendar",
        "recovery_policy", "fallback",
    }

    missing = replace(bundle, components={k: v for k, v in bundle.components.items() if k != "fallback"})
    assert any(CAMPAIGN_FREEZE_BUNDLE_INCOMPLETE in p for p in freeze_bundle_violations(missing, protocol))

    # "Two-person approval" is two people, not one person twice.
    duplicated = replace(bundle, approvers=("teammate_one", "teammate_one"))
    assert any(CAMPAIGN_FREEZE_APPROVAL_INSUFFICIENT in p for p in freeze_bundle_violations(duplicated, protocol))

    solo = replace(bundle, approvers=("teammate_one",))
    assert any(CAMPAIGN_FREEZE_APPROVAL_INSUFFICIENT in p for p in freeze_bundle_violations(solo, protocol))
    with pytest.raises(CampaignPreregistrationError):
        assert_freeze_bundle_valid(solo, protocol)


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
@pytest.mark.parametrize("absent_status", ["", NOT_RUN, FAIL, BLOCKED])
def test_freeze_bundle_never_treats_absent_measurement_as_pass(protocol: dict, absent_status: str) -> None:
    bundle = _bundle(protocol)
    unmeasured = replace(
        bundle,
        evidence={**bundle.evidence, "real_shard_throughput": absent_status},
    )
    problems = freeze_bundle_violations(unmeasured, protocol)
    assert any(CAMPAIGN_FREEZE_EVIDENCE_MISSING in problem for problem in problems)
    # Plan 13 G4 requires both a throughput measurement and a takeover rehearsal.
    assert set(required_freeze_evidence(protocol)) == {"real_shard_throughput", "takeover_rehearsal"}


# --------------------------------------------------------------------------------------
# Append-only parent binding (Plan Section 8.5)
# --------------------------------------------------------------------------------------


def _bind(manifest: ParentBindingManifest, protocol: dict, *, set_id: str, parent_hash: str) -> ParentBindingManifest:
    return bind_parent(
        manifest,
        set_id=set_id,
        arm_ids=("A", "B", "C"),
        target_parent_tokens=8 * LOSS_TOKENS_PER_UPDATE,
        parent_checkpoint_hash=parent_hash,
        bound_at="2026-09-13T00:00:00Z",
        bound_by="operator",
        preregistration_hash="a" * 64,
        protocol=protocol,
    )


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_parent_binding_is_append_only(protocol: dict, tmp_path: Path) -> None:
    empty = ParentBindingManifest()
    first = _bind(empty, protocol, set_id="confirmation", parent_hash="b" * 64)
    second = _bind(first, protocol, set_id="primary", parent_hash="c" * 64)

    assert len(second.entries) == 2
    assert append_only_violations(empty, first) == ()
    assert append_only_violations(first, second) == ()

    # Rebinding the same set is a rewrite wearing an append's clothes.
    with pytest.raises(CampaignPreregistrationError, match=CAMPAIGN_PARENT_MANIFEST_REWRITTEN):
        _bind(second, protocol, set_id="primary", parent_hash="d" * 64)

    # A silently edited earlier entry is detected by comparing revisions.
    tampered = ParentBindingManifest(
        (replace(second.entries[0], parent_checkpoint_hash="e" * 64),) + second.entries[1:]
    )
    assert any(CAMPAIGN_PARENT_MANIFEST_REWRITTEN in p for p in append_only_violations(second, tampered))

    # A truncated manifest is likewise not an append.
    truncated = ParentBindingManifest(second.entries[:1])
    assert any(CAMPAIGN_PARENT_MANIFEST_REWRITTEN in p for p in append_only_violations(second, truncated))

    # Round-trips through disk without changing meaning.
    path = write_parent_manifest(tmp_path / "parents.json", second)
    assert read_parent_manifest(path) == second
    assert read_parent_manifest(tmp_path / "absent.json") == ParentBindingManifest()


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_parent_binding_refuses_a_pending_hash_or_a_post_outcome_binding(protocol: dict) -> None:
    empty = ParentBindingManifest()

    # Plan 8.5: the sentinel marks a hash that does not exist. It is not a hash.
    with pytest.raises(CampaignPreregistrationError, match="CAMPAIGN_PARENT_HASH_PENDING"):
        _bind(empty, protocol, set_id="primary", parent_hash=PENDING_PARENT_HASH)
    with pytest.raises(CampaignPreregistrationError, match="CAMPAIGN_PARENT_HASH_PENDING"):
        _bind(empty, protocol, set_id="primary", parent_hash="   ")

    # Plan 8.5: bound "before any arm runs or any arm outcome is observed".
    with pytest.raises(CampaignPreregistrationError, match="CAMPAIGN_PARENT_BOUND_AFTER_OUTCOME"):
        bind_parent(
            empty,
            set_id="primary",
            arm_ids=("A", "B", "C"),
            target_parent_tokens=8 * LOSS_TOKENS_PER_UPDATE,
            parent_checkpoint_hash="f" * 64,
            bound_at="2026-09-16T00:00:00Z",
            bound_by="operator",
            preregistration_hash="a" * 64,
            arm_outcomes_observed=True,
            protocol=protocol,
        )

    # One set binds all three arms; a partial set is not an A/B/C comparison.
    with pytest.raises(CampaignPreregistrationError, match="CAMPAIGN_SET_COUNT_INVALID"):
        bind_parent(
            empty,
            set_id="primary",
            arm_ids=("A", "B"),
            target_parent_tokens=8 * LOSS_TOKENS_PER_UPDATE,
            parent_checkpoint_hash="f" * 64,
            bound_at="2026-09-13T00:00:00Z",
            bound_by="operator",
            preregistration_hash="a" * 64,
            protocol=protocol,
        )


# **Validates: Requirements 1.1, 2.1, 2.4**
@given(count=st.integers(min_value=1, max_value=6))
@settings(max_examples=6, deadline=None, derandomize=True)
def test_appending_any_number_of_distinct_sets_is_never_a_rewrite(count: int) -> None:
    protocol = load_campaign_protocol()
    manifest = ParentBindingManifest()
    history = [manifest]
    for index in range(count):
        manifest = _bind(manifest, protocol, set_id=f"set_{index}", parent_hash=f"{index:064d}")
        history.append(manifest)
    assert len(manifest.entries) == count
    for earlier, later in zip(history, history[1:]):
        assert append_only_violations(earlier, later) == ()
    # Every earlier revision remains a prefix of the final one.
    assert append_only_violations(history[0], manifest) == ()


# --------------------------------------------------------------------------------------
# Readiness and the whole-contract report
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_campaign_readiness_is_not_run_or_blocked_with_an_owner(protocol: dict) -> None:
    results = readiness_results(protocol)
    assert results
    statuses = {result.status for result in results}
    assert PASS not in statuses, "no campaign prerequisite has been measured yet"
    assert statuses <= {NOT_RUN, BLOCKED}

    for result in results:
        assert "owner=" in result.reason
        assert "next_action=" in result.reason
        assert "blocker=" in result.reason

    # Personal eligibility is an attestation no repository check can supply.
    eligibility = {result.check_id: result for result in results}["campaign.readiness.teammate_eligibility_confirmed"]
    assert eligibility.status == BLOCKED


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_campaign_decisions_refuse_to_run_without_measured_evidence(protocol: dict) -> None:
    with pytest.raises(CampaignNotReadyError) as excinfo:
        assert_ready_for_campaign_runs(protocol)
    message = str(excinfo.value)
    assert "proxy_runs_completed" in message
    assert "next_action=" in message


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.3**
def test_verify_campaign_passes_structurally_and_reports_absences_honestly(protocol: dict) -> None:
    report = verify_campaign(protocol)
    assert report.ok, [result.reason for result in report.failures]
    assert report.failures == ()
    # The structural checks pass; the campaign itself has simply not been run.
    assert report.not_run
    assert report.blocked
    assert_campaign_valid(protocol)

    ids = {result.check_id for result in report.results}
    for expected in (
        "campaign.proxy_parameter_count",
        "campaign.update_alignment",
        "campaign.comparability",
        "campaign.contingency_comparability",
        "campaign.decision_thresholds",
        "campaign.branch_sets",
        "campaign.validation_final_custody",
        "campaign.parent_binding_rule",
    ):
        assert expected in ids


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_validation_final_is_recorded_as_never_opened(protocol: dict) -> None:
    custody = protocol["validation_final_custody"]
    assert custody["status"] == "NOT_OPENED"
    assert bool(custody["opened_once"])
    # It must not feed any decision made before it is opened.
    for forbidden in ("training", "model_selection", "mixture_selection", "learning_rate_selection"):
        assert forbidden in custody["may_not_influence"]


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_branch_size_bands_are_update_aligned_with_this_campaign(protocol: dict) -> None:
    """Plan Section 8.1's branch table must reconcile with Plan Section 7's batch size."""
    from tinybench_lm.branches import BRANCH_PROTOCOL_PATH, branch_size_bands, load_branch_protocol

    assert branch_size_reference_violations(protocol) == ()
    assert protocol["branch_sets"]["size_selection_protocol"] == "configs/branches/exposure_v1.yaml"
    assert BRANCH_PROTOCOL_PATH.is_file()

    # Cross-check the two frozen contracts against each other rather than trusting either.
    bands = branch_size_bands(load_branch_protocol())
    assert bands
    for band in bands:
        assert band.primary_tokens_per_arm == band.primary_updates_per_arm * LOSS_TOKENS_PER_UPDATE
        assert (
            band.confirmation_tokens_per_arm
            == band.confirmation_updates_per_arm * LOSS_TOKENS_PER_UPDATE
        )
        # A confirmation is the shorter run of the pair, by construction.
        assert band.confirmation_updates_per_arm < band.primary_updates_per_arm


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_a_misaligned_branch_band_is_reported(protocol: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bands are clean, so feed the validator a bad one to prove it is not vacuous."""
    from tinybench_lm import branches

    bands = branches.branch_size_bands(branches.load_branch_protocol())
    broken = replace(bands[0], primary_tokens_per_arm=bands[0].primary_tokens_per_arm + 1)
    monkeypatch.setattr(branches, "branch_size_bands", lambda *a, **k: (broken,))

    problems = branch_size_reference_violations(protocol)
    assert problems, "a band whose horizon is not a whole number of updates must be reported"
    assert any("CAMPAIGN_TOKENS_NOT_UPDATE_ALIGNED" in problem for problem in problems)
    assert not verify_campaign(protocol).ok


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_a_redirected_branch_protocol_reference_is_reported(
    protocol: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delegation is only safe if the named target is the one actually verified."""
    redirected = {**protocol, "branch_sets": {**protocol["branch_sets"], "size_selection_protocol": "configs/branches/exposure_v2.yaml"}}
    problems = branch_size_reference_violations(redirected)
    assert problems
    assert any("exposure_v1.yaml" in problem for problem in problems)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_reason_code_vocabulary_matches_the_frozen_contract(protocol: dict) -> None:
    declared = set(protocol["reason_codes"]["fail_closed"])
    assert declared == set(CAMPAIGN_FAIL_CLOSED_REASON_CODES)


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_one_primary_set_and_one_earlier_confirmation(protocol: dict) -> None:
    sets = protocol["branch_sets"]
    assert int(sets["primary_sets"]) == 1
    assert int(sets["confirmation_sets"]) == 1
    assert bool(sets["confirmation_precedes_primary"])
    assert bool(sets["same_direction_confirmation_required"])
    # Plan 8.6: the confirmation reads validation_dev; only the primary opens validation_final.
    assert sets["confirmation_endpoint"] == "validation_dev"
    assert sets["primary_endpoint"] == "validation_final"
    assert bool(sets["confirmation_parent_distinct_from_primary_parent"])
