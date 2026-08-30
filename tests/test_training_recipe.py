"""The frozen training recipe: WSD LR, AdamW decay groups, precision, counters, run identity.

Plan Sections 7, 7.1-7.2, 8.4, and 15. Nothing here starts a training run, touches a GPU,
measures throughput, or measures BF16 stability. What is proven instead:

- the LR is WSD only -- linear warmup to the peak, a stable peak, then linear decay to
  *exactly* zero -- with every phase boundary pinned, and cosine/minimum-LR machinery gone
  from ``train.py``,
- a branch is the full-horizon case of the same function, so
  :meth:`WSDSchedule.branch` reproduces :func:`branches.branch_learning_rate` exactly and
  "identical branch LR decay" is checkable rather than asserted,
- AdamW gets two groups: weight decay 0.1 on projection matrices, zero on embeddings and all
  normalization weights, with the tied embedding/output Parameter enumerated once,
- accumulation and token arithmetic are exact, and the frozen update-aligned campaign token
  totals from Plan Sections 6.2 and 8.1 reconcile to their declared update counts,
- the precision policy falls back to FP16 + GradScaler when BF16 is unsupported or measured
  unstable, and refuses to assume stability for a final-scope run,
- NaN/Inf, out-of-range token IDs, impossible counters, hash drift, and a semantic change
  under an existing run ID all fail closed.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st

from tinybench_lm import ModelConfig, TinyBenchLM
from tinybench_lm.branches import SharedBranchPolicy, branch_learning_rate, load_branch_protocol
from tinybench_lm.data_protocols import ProtocolMutatedError
from tinybench_lm.parameters import count_unique_trainable_parameters
from tinybench_lm.training_recipe import (
    BF16,
    DECAY_GROUP,
    FP16,
    FROZEN_TRAINING_RECIPE_SHA256,
    NO_DECAY_GROUP,
    OPTIMIZER_POLICY_ID,
    PHASE_DECAY,
    PHASE_STABLE,
    PHASE_WARMUP,
    RECIPE_BATCH_NOT_TARGET,
    RECIPE_COUNTER_IMPOSSIBLE,
    RECIPE_DECAY_GROUP_VIOLATION,
    RECIPE_HASH_DRIFT,
    RECIPE_HORIZON_NOT_UPDATE_ALIGNED,
    RECIPE_LR_PHASE_INVALID,
    RECIPE_LR_SHAPE_FORBIDDEN,
    RECIPE_NON_FINITE,
    RECIPE_RUN_ID_SEMANTIC_CHANGE,
    RECIPE_TOKEN_ID_OUT_OF_RANGE,
    RECIPE_UNDECAYED_RELEASE_CANDIDATE,
    SCOPE_FINAL,
    SCOPE_PILOT,
    TRAINING_RECIPE_PATH,
    BatchPlan,
    PrecisionPolicy,
    RunSemantics,
    TrainingIntegrityError,
    TrainingRecipeError,
    TrainingRecipeNotReadyError,
    WSDSchedule,
    adamw_parameter_groups,
    adamw_settings,
    assert_finite,
    assert_no_hash_drift,
    assert_run_id_unchanged,
    assert_update_record,
    assert_valid_token_ids,
    batch_plan_violations,
    build_run_semantics,
    build_update_record,
    classify_parameters,
    forbidden_lr_shape_violations,
    load_training_recipe,
    model_config_hash,
    optimizer_violations,
    parameter_group_violations,
    plan_batch,
    release_candidate_violations,
    select_precision_policy,
    semantic_differences,
    target_loss_tokens_per_update,
    tokens_for_updates,
    update_record_violations,
    updates_for_tokens,
    warmup_updates_for_horizon,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ENTRY_POINT = REPOSITORY_ROOT / "train.py"

RECIPE = load_training_recipe()

TINY_CONFIG = ModelConfig(
    vocab_size=64,
    max_seq_len=32,
    n_layers=2,
    d_model=16,
    n_heads=2,
    n_kv_heads=1,
    d_ff=32,
)


def _tiny_model() -> TinyBenchLM:
    torch.manual_seed(0)
    return TinyBenchLM(TINY_CONFIG)


# --------------------------------------------------------------------------------------
# The frozen config
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_recipe_matches_its_pinned_digest() -> None:
    assert RECIPE["_digest"] == FROZEN_TRAINING_RECIPE_SHA256[TRAINING_RECIPE_PATH.name]
    assert RECIPE["frozen"] is True
    assert adamw_settings(RECIPE) == {
        "betas": (0.9, 0.95),
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "gradient_clip_global_norm": 1.0,
    }
    assert target_loss_tokens_per_update(RECIPE) == 262_144


# **Validates: Requirements 2.4, 2.5**
def test_editing_the_frozen_recipe_fails_closed(tmp_path: Path) -> None:
    mutated = tmp_path / "recipe_v1.yaml"
    mutated.write_bytes(TRAINING_RECIPE_PATH.read_bytes().replace(b"weight_decay: 0.1", b"weight_decay: 0.2"))
    with pytest.raises(ProtocolMutatedError):
        load_training_recipe(mutated)
    # Unverified loading still parses, so a v2 can be reviewed before it is pinned.
    assert load_training_recipe(mutated, verify=False)["optimizer"]["weight_decay"] == 0.2


# **Validates: Requirements 1.1, 2.1, 3.3**
def test_recipe_and_branch_protocol_name_the_same_optimizer_policy() -> None:
    """The branch contract's ``optimizer_policy`` is exactly this module's group policy."""
    assert SharedBranchPolicy(update_count=4, parent_lr=1e-3).optimizer_policy == OPTIMIZER_POLICY_ID
    assert str(load_branch_protocol()["shared_policy"]["optimizer_policy"]) == OPTIMIZER_POLICY_ID


# --------------------------------------------------------------------------------------
# WSD learning rate: phase boundaries
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_wsd_phase_boundaries_are_exact() -> None:
    schedule = WSDSchedule(total_updates=100, warmup_updates=10, decay_updates=20, peak_lr=6e-4)
    assert schedule.stable_updates == 70
    assert schedule.decay_start_update == 80
    assert schedule.decays_to_zero

    # Warmup climbs linearly and reaches the peak on its last update.
    assert schedule.learning_rate(0) == pytest.approx(6e-4 / 10)
    assert schedule.learning_rate(9) == pytest.approx(6e-4)
    assert schedule.phase(0) == PHASE_WARMUP
    assert schedule.phase(9) == PHASE_WARMUP

    # Stable holds the peak exactly, with no drift.
    assert schedule.phase(10) == PHASE_STABLE
    assert schedule.phase(79) == PHASE_STABLE
    assert schedule.learning_rate(10) == 6e-4
    assert schedule.learning_rate(79) == 6e-4

    # Decay starts at the peak and ends at exactly zero, with no LR floor.
    assert schedule.phase(80) == PHASE_DECAY
    assert schedule.phase(99) == PHASE_DECAY
    assert schedule.learning_rate(80) == pytest.approx(6e-4)
    assert schedule.learning_rate(99) == 0.0
    assert schedule.learning_rate(89) == pytest.approx(6e-4 * 10 / 19)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_wsd_rejects_impossible_phase_layouts() -> None:
    with pytest.raises(TrainingRecipeError, match=RECIPE_LR_PHASE_INVALID):
        WSDSchedule(total_updates=10, warmup_updates=0, decay_updates=1, peak_lr=1e-3)
    with pytest.raises(TrainingRecipeError, match=RECIPE_LR_PHASE_INVALID):
        WSDSchedule(total_updates=10, warmup_updates=6, decay_updates=6, peak_lr=1e-3)
    with pytest.raises(TrainingRecipeError, match=RECIPE_COUNTER_IMPOSSIBLE):
        WSDSchedule(total_updates=0, warmup_updates=0, decay_updates=0, peak_lr=1e-3)
    with pytest.raises(TrainingRecipeError):
        WSDSchedule(total_updates=10, warmup_updates=2, decay_updates=2, peak_lr=0.0)
    schedule = WSDSchedule(total_updates=10, warmup_updates=2, decay_updates=2, peak_lr=1e-3)
    for index in (-1, 10):
        with pytest.raises(TrainingRecipeError, match=RECIPE_COUNTER_IMPOSSIBLE):
            schedule.learning_rate(index)


# **Validates: Requirements 1.1, 2.1, 2.4**
@settings(max_examples=80, deadline=None)
@given(
    total=st.integers(min_value=4, max_value=4_000),
    warmup_share=st.integers(min_value=0, max_value=40),
    decay_share=st.integers(min_value=0, max_value=60),
    peak=st.floats(min_value=1e-5, max_value=1e-2, allow_nan=False, allow_infinity=False),
)
def test_wsd_shape_holds_for_every_valid_horizon(
    total: int, warmup_share: int, decay_share: int, peak: float
) -> None:
    """Property: warmup is nondecreasing, stable is constant, decay reaches exactly zero."""
    warmup = total * warmup_share // 100
    decay = total * decay_share // 100
    if decay == 1:
        decay = 2
    if warmup + decay > total:
        decay = total - warmup
        if decay == 1:
            decay = 0
    schedule = WSDSchedule(
        total_updates=total, warmup_updates=warmup, decay_updates=decay, peak_lr=peak
    )
    values = schedule.schedule()
    phases = schedule.phases()

    assert len(values) == total
    assert all(0.0 <= value <= peak + 1e-18 for value in values)
    assert phases.count(PHASE_WARMUP) == warmup
    assert phases.count(PHASE_DECAY) == decay
    assert phases.count(PHASE_STABLE) == schedule.stable_updates

    warmup_values = values[:warmup]
    assert all(b >= a for a, b in zip(warmup_values, warmup_values[1:]))
    if warmup:
        assert warmup_values[-1] == pytest.approx(peak)
    stable_values = values[warmup : schedule.decay_start_update if decay else total]
    assert all(value == peak for value in stable_values)
    if decay:
        decay_values = values[schedule.decay_start_update :]
        assert decay_values[0] == pytest.approx(peak)
        assert all(b <= a for a, b in zip(decay_values, decay_values[1:]))
        assert decay_values[-1] == 0.0
        assert release_candidate_violations(schedule) == ()
    else:
        assert values[-1] == peak
        assert release_candidate_violations(schedule)[0].startswith(RECIPE_UNDECAYED_RELEASE_CANDIDATE)


# **Validates: Requirements 1.1, 2.1, 3.3**
@settings(max_examples=60, deadline=None)
@given(
    update_count=st.integers(min_value=2, max_value=3_000),
    parent_lr=st.floats(min_value=1e-5, max_value=1e-2, allow_nan=False, allow_infinity=False),
)
def test_branch_decay_is_the_full_horizon_case_of_the_same_wsd_function(
    update_count: int, parent_lr: float
) -> None:
    """Plan Section 8.4: branch LR decay is identical across arms, and identical to WSD's tail."""
    schedule = WSDSchedule.branch(parent_lr, update_count)
    assert schedule.warmup_updates == 0
    assert schedule.decay_updates == update_count
    assert schedule.schedule() == tuple(
        branch_learning_rate(parent_lr, index, update_count) for index in range(update_count)
    )
    assert schedule.learning_rate(0) == pytest.approx(parent_lr)
    assert schedule.learning_rate(update_count - 1) == 0.0


# **Validates: Requirements 1.1, 2.1**
def test_warmup_is_the_frozen_one_percent_with_exact_half_up_rounding() -> None:
    assert warmup_updates_for_horizon(1_000, protocol=RECIPE) == 10
    assert warmup_updates_for_horizon(573, protocol=RECIPE) == 6  # 5.73 rounds half-up to 6
    assert warmup_updates_for_horizon(149, protocol=RECIPE) == 1  # 1.49 rounds down
    assert warmup_updates_for_horizon(150, protocol=RECIPE) == 2  # 1.50 rounds half-up
    assert warmup_updates_for_horizon(1, protocol=RECIPE) == 1  # never zero
    assert WSDSchedule.for_horizon(2_289, 6e-4, protocol=RECIPE).warmup_updates == 23
    with pytest.raises(TrainingRecipeError, match=RECIPE_LR_PHASE_INVALID):
        warmup_updates_for_horizon(100, fraction="1.5", protocol=RECIPE)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_train_entry_point_has_no_cosine_or_minimum_learning_rate() -> None:
    """The two shapes Plan Section 7 replaces must be gone from the training entry point."""
    source = TRAIN_ENTRY_POINT.read_text(encoding="utf-8")
    assert forbidden_lr_shape_violations(source) == ()
    assert forbidden_lr_shape_violations("lr = math.cos(x)")[0].startswith(RECIPE_LR_SHAPE_FORBIDDEN)


# --------------------------------------------------------------------------------------
# AdamW parameter groups
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1**
def test_decay_groups_exclude_embeddings_and_every_normalization_weight() -> None:
    model = _tiny_model()
    classified = classify_parameters(model)

    assert "token_embedding.weight" in classified.no_decay_names
    norms = [name for name in classified.no_decay_names if "norm" in name]
    # Two block norms per layer plus the final norm.
    assert len(norms) == 2 * TINY_CONFIG.n_layers + 1
    assert set(classified.no_decay_names) == {"token_embedding.weight", *norms}
    # Seven projection matrices per layer: q, k, v, o, gate, up, down.
    assert len(classified.decay_names) == 7 * TINY_CONFIG.n_layers
    assert all(name.endswith("_proj.weight") for name in classified.decay_names)
    assert parameter_group_violations(model, classified=classified, protocol=RECIPE) == ()


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1**
def test_tied_output_weight_is_enumerated_once_in_the_no_decay_group() -> None:
    model = _tiny_model()
    classified = classify_parameters(model)
    embedding_id = id(model.token_embedding.weight)
    assert id(model.output_weight) == embedding_id
    placements = [name for name, parameter in classified.no_decay if id(parameter) == embedding_id]
    assert placements == ["token_embedding.weight"]
    assert not any(id(parameter) == embedding_id for _, parameter in classified.decay)
    assert classified.unique_parameter_count == count_unique_trainable_parameters(model)


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1**
def test_final_configuration_groups_reconcile_to_the_exact_parameter_count() -> None:
    """The two groups partition the frozen 49,658,368-parameter model with nothing lost."""
    model = TinyBenchLM(ModelConfig.from_json(REPOSITORY_ROOT / "configs" / "final_49m.json"))
    classified = classify_parameters(model)
    assert classified.unique_parameter_count == 49_658_368
    assert classified.unique_parameter_count == count_unique_trainable_parameters(model)
    assert parameter_group_violations(model, classified=classified, protocol=RECIPE) == ()

    groups = adamw_parameter_groups(model, protocol=RECIPE)
    assert [group["group_name"] for group in groups] == [DECAY_GROUP, NO_DECAY_GROUP]
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0
    # Every embedding element sits outside weight decay.
    assert sum(parameter.numel() for parameter in groups[1]["params"]) == 12_288 * 512 + 512 * (2 * 14 + 1)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_built_optimizer_carries_the_frozen_hyperparameters() -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(
        adamw_parameter_groups(model, protocol=RECIPE),
        lr=6e-4,
        betas=adamw_settings(RECIPE)["betas"],
        eps=adamw_settings(RECIPE)["epsilon"],
    )
    assert optimizer_violations(optimizer, protocol=RECIPE) == ()
    for group in optimizer.param_groups:
        assert tuple(group["betas"]) == (0.9, 0.95)
        assert group["eps"] == 1e-8
    decay = next(g for g in optimizer.param_groups if g["group_name"] == DECAY_GROUP)
    no_decay = next(g for g in optimizer.param_groups if g["group_name"] == NO_DECAY_GROUP)
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_undifferentiated_optimizer_is_reported_as_a_group_violation() -> None:
    """The previous behavior -- one AdamW group over parameters() -- must now fail the check."""
    model = _tiny_model()
    undifferentiated = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    problems = optimizer_violations(undifferentiated, protocol=RECIPE)
    assert problems
    assert all(problem.startswith(RECIPE_DECAY_GROUP_VIOLATION) for problem in problems)


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_a_rank_one_decayed_parameter_fails_closed() -> None:
    """A bias-like 1-D parameter must never silently land in the weight-decay group."""

    class WithLooseVector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(4, 4, bias=False)
            self.offset = nn.Parameter(torch.zeros(4))

    model = WithLooseVector()
    classified = classify_parameters(model)
    assert "offset" in classified.decay_names
    problems = parameter_group_violations(model, classified=classified, protocol=RECIPE)
    assert any(RECIPE_DECAY_GROUP_VIOLATION in problem and "offset" in problem for problem in problems)
    with pytest.raises(TrainingRecipeError, match=RECIPE_DECAY_GROUP_VIOLATION):
        adamw_parameter_groups(model, protocol=RECIPE)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_train_entry_point_builds_adamw_from_differentiated_groups() -> None:
    """Structural check on train.py: AdamW is never handed one undifferentiated parameters()."""
    tree = ast.parse(TRAIN_ENTRY_POINT.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "AdamW"
    ]
    assert len(calls) == 1
    first = calls[0].args[0]
    assert not (
        isinstance(first, ast.Call)
        and isinstance(first.func, ast.Attribute)
        and first.func.attr == "parameters"
    )


# --------------------------------------------------------------------------------------
# Accumulation and token arithmetic
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_accumulation_hits_the_frozen_global_batch_exactly() -> None:
    plan = plan_batch(8, 1_024, protocol=RECIPE)
    assert plan.gradient_accumulation == 32
    assert plan.loss_tokens_per_update == 262_144
    assert plan.sequences_per_update == 256
    assert batch_plan_violations(plan, scope=SCOPE_FINAL, protocol=RECIPE) == ()

    with pytest.raises(TrainingRecipeError, match=RECIPE_BATCH_NOT_TARGET):
        plan_batch(3, 1_000, protocol=RECIPE)
    with pytest.raises(TrainingRecipeError, match=RECIPE_COUNTER_IMPOSSIBLE):
        BatchPlan(0, 1_024, 32)


# **Validates: Requirements 1.1, 2.1, 2.4, 3.3**
def test_pilot_batch_may_deviate_while_a_final_batch_may_not() -> None:
    pilot = BatchPlan(8, 512, 4)
    assert pilot.loss_tokens_per_update == 16_384
    assert batch_plan_violations(pilot, scope=SCOPE_PILOT, protocol=RECIPE) == ()
    problems = batch_plan_violations(pilot, scope=SCOPE_FINAL, protocol=RECIPE)
    assert problems and problems[0].startswith(RECIPE_BATCH_NOT_TARGET)


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_frozen_campaign_token_totals_are_update_aligned() -> None:
    """Plan Sections 6.2 and 8.1 token totals must reconcile to their declared update counts."""
    assert updates_for_tokens(150_208_512, protocol=RECIPE) == 573
    assert updates_for_tokens(50_069_504, protocol=RECIPE) == 191
    assert updates_for_tokens(25_165_824, protocol=RECIPE) == 96
    assert updates_for_tokens(200_015_872, protocol=RECIPE) == 763
    assert updates_for_tokens(300_154_880, protocol=RECIPE) == 1_145
    assert updates_for_tokens(100_139_008, protocol=RECIPE) == 382
    assert updates_for_tokens(600_047_616, protocol=RECIPE) == 2_289
    assert tokens_for_updates(2_289, protocol=RECIPE) == 600_047_616
    with pytest.raises(TrainingRecipeError, match=RECIPE_HORIZON_NOT_UPDATE_ALIGNED):
        updates_for_tokens(262_145, protocol=RECIPE)
    with pytest.raises(TrainingRecipeError, match=RECIPE_COUNTER_IMPOSSIBLE):
        updates_for_tokens(0, protocol=RECIPE)


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
@settings(max_examples=80, deadline=None)
@given(
    micro_batch=st.integers(min_value=1, max_value=64),
    sequence_length=st.integers(min_value=1, max_value=2_048),
    accumulation=st.integers(min_value=1, max_value=128),
    updates=st.integers(min_value=0, max_value=5_000),
)
def test_token_counters_are_exact_and_monotone(
    micro_batch: int, sequence_length: int, accumulation: int, updates: int
) -> None:
    """Property: consumed tokens are exactly ``updates x tokens/update`` and never decrease."""
    plan = BatchPlan(micro_batch, sequence_length, accumulation)
    per_update = micro_batch * sequence_length * accumulation
    assert plan.loss_tokens_per_update == per_update
    assert plan.consumed_loss_tokens(updates) == updates * per_update
    assert plan.consumed_loss_tokens(updates + 1) - plan.consumed_loss_tokens(updates) == per_update
    assert updates_for_tokens(per_update * max(1, updates), loss_tokens_per_update=per_update) == max(1, updates)


# --------------------------------------------------------------------------------------
# Precision policy
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_precision_falls_back_to_fp16_with_a_grad_scaler() -> None:
    unsupported = select_precision_policy(
        bf16_supported=False, bf16_measured_stable=None, scope=SCOPE_FINAL, protocol=RECIPE
    )
    assert (unsupported.dtype_name, unsupported.use_grad_scaler, unsupported.status) == (FP16, True, "PASS")
    assert unsupported.torch_dtype() is torch.float16

    unstable = select_precision_policy(
        bf16_supported=True, bf16_measured_stable=False, scope=SCOPE_FINAL, protocol=RECIPE
    )
    assert (unstable.dtype_name, unstable.use_grad_scaler, unstable.status) == (FP16, True, "PASS")

    stable = select_precision_policy(
        bf16_supported=True, bf16_measured_stable=True, scope=SCOPE_FINAL, protocol=RECIPE
    )
    assert (stable.dtype_name, stable.use_grad_scaler, stable.status) == (BF16, False, "PASS")
    assert stable.torch_dtype() is torch.bfloat16


# **Validates: Requirements 1.1, 2.3, 2.4, 2.5**
def test_unmeasured_bf16_stability_is_never_a_final_scope_pass() -> None:
    """Absence of the stability measurement blocks a final run and is DEFERRED for a pilot."""
    with pytest.raises(TrainingRecipeNotReadyError, match="RECIPE_PRECISION_UNMEASURED"):
        select_precision_policy(
            bf16_supported=True, bf16_measured_stable=None, scope=SCOPE_FINAL, protocol=RECIPE
        )
    pilot = select_precision_policy(
        bf16_supported=True, bf16_measured_stable=None, scope=SCOPE_PILOT, protocol=RECIPE
    )
    assert pilot.dtype_name == BF16
    assert pilot.status == "DEFERRED"
    assert pilot.bf16_measured_stable is None
    assert str(RECIPE["readiness"]["measured_bf16_stability"]) == "NOT_RUN"


# --------------------------------------------------------------------------------------
# Fail-closed guards (Plan Section 15)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_non_finite_values_fail_closed() -> None:
    assert assert_finite("loss", 1.25) == 1.25
    assert assert_finite("loss", torch.tensor(2.5)) == pytest.approx(2.5)
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(TrainingIntegrityError, match=RECIPE_NON_FINITE):
            assert_finite("loss", bad)
    with pytest.raises(TrainingIntegrityError, match=RECIPE_NON_FINITE):
        assert_finite("logits", torch.tensor([1.0, float("nan")]))


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_invalid_token_ids_fail_closed() -> None:
    assert_valid_token_ids(torch.tensor([[0, 63]]), 64)
    for bad in (torch.tensor([[0, 64]]), torch.tensor([[-1, 5]])):
        with pytest.raises(TrainingIntegrityError, match=RECIPE_TOKEN_ID_OUT_OF_RANGE):
            assert_valid_token_ids(bad, 64)
    # Padded targets legitimately carry the ignore index, and nothing else.
    assert_valid_token_ids(torch.tensor([[-100, 7]]), 64, allow_ignore_index=True)
    assert_valid_token_ids(torch.tensor([[-100, -100]]), 64, allow_ignore_index=True)
    with pytest.raises(TrainingIntegrityError, match=RECIPE_TOKEN_ID_OUT_OF_RANGE):
        assert_valid_token_ids(torch.tensor([[-99, 7]]), 64, allow_ignore_index=True)


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_hash_drift_fails_closed() -> None:
    frozen = {"schedule": "abc", "model_config": "def"}
    assert_no_hash_drift(frozen, {**frozen, "extra": "ignored"})
    with pytest.raises(TrainingIntegrityError, match=RECIPE_HASH_DRIFT):
        assert_no_hash_drift(frozen, {"schedule": "abc", "model_config": "changed"})
    with pytest.raises(TrainingIntegrityError, match=RECIPE_HASH_DRIFT):
        assert_no_hash_drift(frozen, {"schedule": "abc"})


# --------------------------------------------------------------------------------------
# Run identity (Plan Section 15)
# --------------------------------------------------------------------------------------


def _semantics(**overrides: object) -> RunSemantics:
    base = build_run_semantics(
        model_config=TINY_CONFIG.to_dict(),
        schedule=WSDSchedule(total_updates=100, warmup_updates=1, decay_updates=20, peak_lr=6e-4),
        plan=BatchPlan(8, 1_024, 32),
        precision=select_precision_policy(
            bf16_supported=True, bf16_measured_stable=True, scope=SCOPE_FINAL, protocol=RECIPE
        ),
        weight_decay=0.1,
        gradient_clip_global_norm=1.0,
        seed=1337,
        train_schedule_content_hash="0" * 64,
        protocol=RECIPE,
    )
    return RunSemantics(**{**base.to_dict(), **overrides})


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_run_id_is_stable_and_covers_every_declared_semantic_field() -> None:
    semantics = _semantics()
    run_id = semantics.run_id(RECIPE)
    assert run_id.startswith("run-") and len(run_id) == 4 + 16
    assert _semantics().run_id(RECIPE) == run_id
    assert semantics.model_config_hash == model_config_hash(TINY_CONFIG.to_dict())
    assert sorted(semantics.to_dict()) == sorted(
        str(field) for field in RECIPE["run_identity"]["semantic_fields"]
    )

    replacements = {
        "peak_lr": 1e-3,
        "total_updates": 101,
        "warmup_updates": 2,
        "decay_updates": 21,
        "loss_tokens_per_update": 131_072,
        "weight_decay": 0.05,
        "beta1": 0.95,
        "beta2": 0.99,
        "epsilon": 1e-7,
        "gradient_clip_global_norm": 0.5,
        "precision_dtype": FP16,
        "grad_scaler": True,
        "seed": 1338,
        "train_schedule_content_hash": "1" * 64,
        "model_config_hash": "2" * 64,
        "recipe_digest": "3" * 64,
    }
    for field, value in replacements.items():
        changed = _semantics(**{field: value})
        assert changed.run_id(RECIPE) != run_id, field
        assert semantic_differences(semantics, changed) == (field,)


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_a_semantic_change_under_an_existing_run_id_fails_closed() -> None:
    original = _semantics()
    run_id = original.run_id(RECIPE)
    assert assert_run_id_unchanged(run_id, original, protocol=RECIPE) == run_id
    changed = _semantics(peak_lr=1e-3)
    with pytest.raises(TrainingIntegrityError, match=RECIPE_RUN_ID_SEMANTIC_CHANGE) as error:
        assert_run_id_unchanged(run_id, changed, recorded_semantics=original, protocol=RECIPE)
    assert "peak_lr" in str(error.value)


# --------------------------------------------------------------------------------------
# Per-update audit record
# --------------------------------------------------------------------------------------

_SCHEDULE = WSDSchedule(total_updates=20, warmup_updates=2, decay_updates=4, peak_lr=6e-4)
_PLAN = BatchPlan(8, 1_024, 32)
_PRECISION = PrecisionPolicy(BF16, False, "PASS", "measured stable", SCOPE_FINAL, True, True)
_HASH = "a" * 64


def _record(index: int, **overrides: object):
    record = build_update_record(
        run_id="run-0123456789abcdef",
        update_index=index,
        schedule=_SCHEDULE,
        plan=_PLAN,
        precision=_PRECISION,
        loss=3.5,
        grad_norm=0.75,
        schedule_content_hash=_HASH,
        schedule_cursor=index * _PLAN.sequences_per_update,
    )
    return record if not overrides else type(record)(**{**record.to_dict(), **overrides})


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_every_update_record_carries_auditable_state() -> None:
    previous = None
    for index in range(int(_SCHEDULE.total_updates)):
        record = assert_update_record(_record(index), schedule=_SCHEDULE, plan=_PLAN, previous=previous)
        payload = record.to_dict()
        assert sorted(payload) == sorted(str(field) for field in RECIPE["update_record"]["required_fields"])
        assert payload["learning_rate"] == _SCHEDULE.learning_rate(index)
        assert payload["phase"] == _SCHEDULE.phase(index)
        assert payload["consumed_loss_tokens"] == (index + 1) * 262_144
        assert payload["precision_dtype"] == BF16
        assert payload["schedule_content_hash"] == _HASH
        previous = record
    assert previous is not None
    assert previous.consumed_loss_tokens == 20 * 262_144
    assert previous.learning_rate == 0.0


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_impossible_counters_and_drift_are_reported() -> None:
    good = _record(5)
    assert update_record_violations(good, schedule=_SCHEDULE, plan=_PLAN) == ()

    cases = {
        RECIPE_COUNTER_IMPOSSIBLE: [
            _record(5, consumed_loss_tokens=999),
            _record(5, grad_norm=-1.0),
            _record(5, schedule_cursor=-3),
            _record(5, update_index=25),
        ],
        RECIPE_LR_PHASE_INVALID: [
            _record(5, learning_rate=1e-3),
            _record(5, phase=PHASE_DECAY),
        ],
        RECIPE_NON_FINITE: [_record(5, loss=float("nan"))],
        RECIPE_BATCH_NOT_TARGET: [_record(5, loss_tokens_per_update=1_024)],
    }
    for code, records in cases.items():
        for record in records:
            problems = update_record_violations(record, schedule=_SCHEDULE, plan=_PLAN)
            assert any(problem.startswith(code) for problem in problems), (code, record)
            with pytest.raises(TrainingIntegrityError):
                assert_update_record(record, schedule=_SCHEDULE, plan=_PLAN)

    previous = _record(5)
    sequential_cases = {
        RECIPE_COUNTER_IMPOSSIBLE: [_record(8), _record(6, schedule_cursor=0)],
        RECIPE_RUN_ID_SEMANTIC_CHANGE: [
            _record(6, run_id="run-ffffffffffffffff"),
            _record(6, precision_dtype=FP16, grad_scaler=True),
        ],
        RECIPE_HASH_DRIFT: [_record(6, schedule_content_hash="b" * 64)],
    }
    for code, records in sequential_cases.items():
        for record in records:
            problems = update_record_violations(record, schedule=_SCHEDULE, plan=_PLAN, previous=previous)
            assert any(problem.startswith(code) for problem in problems), (code, record)

    with pytest.raises(TrainingIntegrityError, match=RECIPE_NON_FINITE):
        build_update_record(
            run_id="run-0123456789abcdef",
            update_index=1,
            schedule=_SCHEDULE,
            plan=_PLAN,
            precision=_PRECISION,
            loss=float("inf"),
            grad_norm=0.5,
            schedule_content_hash=_HASH,
            schedule_cursor=0,
        )


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_pilot_sampler_records_a_null_cursor_without_failing() -> None:
    """The pilot flat-stream sampler has no integer cursor; that absence is permitted."""
    record = build_update_record(
        run_id="run-0123456789abcdef",
        update_index=0,
        schedule=_SCHEDULE,
        plan=_PLAN,
        precision=_PRECISION,
        loss=3.5,
        grad_norm=0.5,
        schedule_content_hash="PILOT_ONLY_NO_SCHEDULE",
        schedule_cursor=None,
    )
    assert record.schedule_cursor is None
    assert update_record_violations(record, schedule=_SCHEDULE, plan=_PLAN) == ()
    assert bool(RECIPE["update_record"]["cursor_null_permitted_for_pilot_sampler"]) is True


# --------------------------------------------------------------------------------------
# The training entry point wires the recipe without starting a run
# --------------------------------------------------------------------------------------


def _train_module():
    """Load ``train.py`` by path: it is an entry point, not an installed module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_entry_point_recipe", TRAIN_ENTRY_POINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "shard_root": None,
        "train_manifest": None,
        "train_schedule": None,
        "validation_manifest": None,
        "validation_schedule": None,
        "steps": 1_000,
        "micro_batch_size": 8,
        "sequence_length": 1_024,
        "gradient_accumulation": None,
        "learning_rate": 6e-4,
        "warmup_steps": None,
        "decay_updates": 0,
        "bf16_stability": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _final_args(**overrides: object) -> argparse.Namespace:
    return _args(
        shard_root=Path("a"),
        train_manifest=Path("b"),
        train_schedule=Path("c"),
        validation_manifest=Path("d"),
        validation_schedule=Path("e"),
        **overrides,
    )


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1, 3.3**
def test_entry_point_derives_the_wsd_schedule_and_batch_plan() -> None:
    train = _train_module()
    assert train.PILOT_GRADIENT_ACCUMULATION == 4
    assert train.run_scope(_args()) == SCOPE_PILOT
    assert train.run_scope(_final_args()) == SCOPE_FINAL

    # Warmup defaults to the frozen ~1% of the horizon and is honored when given.
    derived = train.build_lr_schedule(_args(steps=2_289, decay_updates=200), RECIPE)
    assert (derived.total_updates, derived.warmup_updates, derived.decay_updates) == (2_289, 23, 200)
    assert derived.learning_rate(2_288) == 0.0
    explicit = train.build_lr_schedule(_args(steps=1_000, warmup_steps=100, decay_updates=0), RECIPE)
    assert explicit.warmup_updates == 100
    assert release_candidate_violations(explicit)[0].startswith(RECIPE_UNDECAYED_RELEASE_CANDIDATE)

    # A final run derives the accumulation that hits 262,144 loss tokens/update exactly; a
    # pilot run keeps its small preserved default.
    final_plan = train.build_batch_plan(_final_args(), SCOPE_FINAL, RECIPE)
    assert (final_plan.gradient_accumulation, final_plan.loss_tokens_per_update) == (32, 262_144)
    pilot_plan = train.build_batch_plan(_args(sequence_length=512), SCOPE_PILOT, RECIPE)
    assert (pilot_plan.gradient_accumulation, pilot_plan.loss_tokens_per_update) == (4, 16_384)
    with pytest.raises(ValueError, match=RECIPE_BATCH_NOT_TARGET):
        train.build_batch_plan(_final_args(gradient_accumulation=4), SCOPE_FINAL, RECIPE)

    assert train.bf16_measured_stable(_args()) is None
    assert train.bf16_measured_stable(_args(bf16_stability="stable")) is True
    assert train.bf16_measured_stable(_args(bf16_stability="unstable")) is False


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_entry_point_run_directory_refuses_a_semantic_change(tmp_path: Path) -> None:
    """Plan Section 15: a changed LR under an existing run directory fails closed."""
    train = _train_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original = _semantics()
    run_id = train.resolve_run_identity(run_dir, original, RECIPE)

    identity_path = run_dir / train.RUN_IDENTITY_FILENAME
    assert identity_path.is_file()
    recorded = json.loads(identity_path.read_text(encoding="utf-8"))
    assert recorded["run_id"] == run_id
    assert recorded["semantics"] == original.to_dict()

    # Re-running the same recipe in the same directory is a resume, not a mutation.
    assert train.resolve_run_identity(run_dir, original, RECIPE) == run_id

    with pytest.raises(TrainingIntegrityError, match=RECIPE_RUN_ID_SEMANTIC_CHANGE) as error:
        train.resolve_run_identity(run_dir, _semantics(peak_lr=1e-3), RECIPE)
    assert "peak_lr" in str(error.value)
    # The recorded lineage is left exactly as it was.
    assert json.loads(identity_path.read_text(encoding="utf-8")) == recorded
