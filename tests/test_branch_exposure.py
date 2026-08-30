"""A/B/C exposure construction and the exact annealing schedule (Plan Sections 8.1-8.5).

Nothing here downloads a corpus, binds a parent checkpoint hash, trains an arm, or observes
an arm outcome. The tests prove that the three arms are *comparable*, which is the only thing
that makes an annealing claim meaningful:

- ``reserved_in_update(k)`` follows the exact nonnegative half-up cumulative formula, in
  integer arithmetic, for every valid branch length,
- update 0 is all stable, update ``K - 1`` is all reserved, every count is nonnegative and at
  most 256, and the reserved total is exactly ``128 x K``,
- every arm consumes exactly 256 sequences in each of ``K`` updates,
- B and C consume identical stable and reserved multisets (including multiplicity) but have
  different training order hashes,
- A is B with every reserved position replaced by the matching disjoint ``stable_control``
  sequence, and A consumes no reserved data at all,
- parent, LR, batch layout, optimizer, and RNG policy are identical across arms,
- the frozen branch sizes are update aligned, and a branch size cannot be selected without a
  measured throughput,
- disjointness, position matching, multiset identity, order distinctness, supply, and content
  hashes all fail closed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tinybench_lm.branches import (
    ARM_A,
    ARM_B,
    ARM_C,
    ARM_IDS,
    BRANCH_ARM_POLICY_DIVERGED,
    BRANCH_CONTENT_HASH_MISMATCH,
    BRANCH_EXPOSURE_NOT_DISJOINT,
    BRANCH_ORDER_HASH_IDENTICAL,
    BRANCH_POSITION_NOT_MATCHED,
    BRANCH_PROTOCOL_PATH,
    BRANCH_SUPPLY_EXHAUSTED,
    BRANCH_UPDATE_COUNT_INVALID,
    COMMON_STABLE,
    FROZEN_BRANCH_PROTOCOL_SHA256,
    PENDING_PARENT_HASH,
    RESERVED_LIST,
    STABLE_CONTROL,
    ArmSchedule,
    BranchContractError,
    BranchesNotReadyError,
    ExposureLists,
    ExposureSlot,
    SharedBranchPolicy,
    annealed_reserved_schedule,
    assert_branch_arms_valid,
    assert_ready_for_branch_runs,
    branch_learning_rate,
    branch_size_alignment_problems,
    branch_size_bands,
    build_arm_a,
    build_arm_b,
    build_arm_c,
    build_arm_schedules,
    build_exposure_lists,
    constant_reserved_schedule,
    cumulative_reserved,
    disjointness_problems,
    format_branch_report,
    load_arm_schedule,
    load_branch_protocol,
    load_exposure_lists,
    position_matching_problems,
    reserved_in_update,
    reserved_sequences_per_update,
    select_branch_size,
    sequences_per_update,
    total_reserved_sequences,
    verify_branch_arms,
    write_arm_schedule,
    write_exposure_lists,
)
from tinybench_lm.data_protocols import ProtocolMutatedError, protocol_digest
from tinybench_lm.schedule import (
    MaterializedSchedule,
    ScheduleEntry,
    ScheduleResumeError,
    build_materialized_schedule,
    exposure_reference_hash,
)
from tinybench_lm.shards import RESERVED, STABLE_TRAIN, ShardDocument, build_split_manifest
from tinybench_lm.source_manifest import load_source_registry
from tinybench_lm.tokenizer import build_tokenizer, load_tokenizer_protocol

SEQUENCES_PER_UPDATE = 256
STABLE_PER_UPDATE = 128
RESERVED_PER_UPDATE = 128


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_branch_protocol()


# --------------------------------------------------------------------------------------
# Synthetic exposure supply: arm construction needs references, never token files
# --------------------------------------------------------------------------------------


def _synthetic_schedule(
    boundary: str, count: int, *, tag: str, sequence_length: int = 1024, per_shard: int = 64
) -> MaterializedSchedule:
    """A materialized schedule of ``count`` distinct references, no shard files required."""
    stride = sequence_length + 1
    entries: list[ScheduleEntry] = []
    for index in range(count):
        shard_index, slot = divmod(index, per_shard)
        entries.append(
            ScheduleEntry(
                shard_id=f"{tag}/shard-{shard_index:04d}",
                token_offset=slot * stride,
                length=stride,
                source_id=f"{tag}_source_{shard_index % 3}",
                namespace=f"{boundary}/{tag}",
            )
        )
    return MaterializedSchedule(
        schedule_id=f"{boundary}/synthetic/{tag}",
        split_id=boundary,
        boundary=boundary,
        sequence_length=sequence_length,
        label_shift=1,
        seed=0,
        local_shuffle_buffer_sequences=1,
        entries=tuple(entries),
        manifest_content_hash=hashlib.sha256(f"{tag}-manifest".encode("utf-8")).hexdigest(),
        protocol_digest=hashlib.sha256(f"{tag}-protocol".encode("utf-8")).hexdigest(),
    )


def _lists_for(update_count: int, protocol: dict, *, spare: int = 0) -> ExposureLists:
    stable = _synthetic_schedule(
        STABLE_TRAIN, 2 * STABLE_PER_UPDATE * update_count + spare, tag="stable"
    )
    reserved = _synthetic_schedule(
        RESERVED, RESERVED_PER_UPDATE * update_count + spare, tag="reserved"
    )
    return build_exposure_lists(stable, reserved, update_count=update_count, protocol=protocol)


def _policy(update_count: int) -> SharedBranchPolicy:
    return SharedBranchPolicy(update_count=update_count, parent_lr=3e-4)


# --------------------------------------------------------------------------------------
# The frozen contract
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_frozen_branch_protocol_is_pinned_and_immutable(tmp_path: Path) -> None:
    assert protocol_digest(BRANCH_PROTOCOL_PATH) == FROZEN_BRANCH_PROTOCOL_SHA256["exposure_v1.yaml"]

    mutated = tmp_path / "exposure_v1.yaml"
    text = BRANCH_PROTOCOL_PATH.read_text(encoding="utf-8")
    # A structurally valid edit no other check would notice: the digest is what catches it.
    mutated.write_text(text.replace("owner: operator", "owner: nobody"), encoding="utf-8")
    with pytest.raises(ProtocolMutatedError):
        load_branch_protocol(mutated)
    # An unverified load still parses, so a proposed v2 can be reviewed before it is pinned.
    assert str(load_branch_protocol(mutated, verify=False)["readiness"]["owner"]) == "nobody"

    # A structurally invalid edit fails even before the digest is consulted.
    inconsistent = tmp_path / "inconsistent_v1.yaml"
    inconsistent.write_text(text.replace("sequences_per_update: 256", "sequences_per_update: 240"), encoding="utf-8")
    with pytest.raises(BranchContractError):
        load_branch_protocol(inconsistent, verify=False)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_frozen_contract_declares_the_plan_8_2_and_8_4_layout(protocol: dict) -> None:
    assert sequences_per_update(protocol) == SEQUENCES_PER_UPDATE
    assert reserved_sequences_per_update(protocol) == RESERVED_PER_UPDATE
    assert tuple(str(arm["arm_id"]) for arm in protocol["arms"]) == ARM_IDS
    assert int(protocol["batch_layout"]["loss_tokens_per_update"]) == 262_144
    assert str(protocol["parent_binding"]["pending_sentinel"]) == PENDING_PARENT_HASH
    assert protocol["parent_binding"]["hash_exists_at_freeze_time"] is False


# **Validates: Requirements 2.3, 2.5**
def test_arm_runs_are_blocked_without_measurement_or_parent(protocol: dict) -> None:
    readiness = protocol["readiness"]
    assert readiness["measured_3070_throughput"] == "NOT_RUN"
    assert readiness["bound_parent_checkpoint_hash"] == "NOT_RUN"
    with pytest.raises(BranchesNotReadyError) as blocked:
        assert_ready_for_branch_runs(protocol)
    assert "next_action" in str(blocked.value)


# --------------------------------------------------------------------------------------
# The exact annealing schedule (Plan Section 8.4)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_reserved_in_update_matches_hand_computed_values(protocol: dict) -> None:
    # K = 2: cumulative(0) = floor(0 + 0.5) = 0, cumulative(1) = floor(256 * 1 * 2 / 2 + 0.5) = 256.
    assert annealed_reserved_schedule(2, protocol=protocol) == (0, 256)
    # K = 3: cumulative = floor(256 k (k+1) / 4 + 0.5) = 0, 128, 384.
    assert annealed_reserved_schedule(3, protocol=protocol) == (0, 128, 256)
    # K = 5: cumulative = floor(256 k (k+1) / 8 + 0.5) = 0, 64, 192, 384, 640.
    assert annealed_reserved_schedule(5, protocol=protocol) == (0, 64, 128, 192, 256)
    assert cumulative_reserved(-1, 5, protocol=protocol) == 0
    assert cumulative_reserved(4, 5, protocol=protocol) == 640 == STABLE_PER_UPDATE * 5


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_annealing_rejects_branches_shorter_than_two_updates(protocol: dict) -> None:
    for invalid in (-3, 0, 1):
        with pytest.raises(BranchContractError) as error:
            annealed_reserved_schedule(invalid, protocol=protocol)
        assert BRANCH_UPDATE_COUNT_INVALID in str(error.value)
    with pytest.raises(BranchContractError):
        cumulative_reserved(9, 4, protocol=protocol)


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
@settings(max_examples=40, deadline=None)
@given(update_count=st.integers(min_value=2, max_value=4096))
def test_annealing_invariants_hold_for_every_valid_branch_length(update_count: int) -> None:
    """Nonnegative counts, all-stable start, all-reserved end, exact 128 x K reserved total."""
    resolved = load_branch_protocol()
    schedule = annealed_reserved_schedule(update_count, protocol=resolved)

    assert len(schedule) == update_count
    assert min(schedule) >= 0
    assert max(schedule) <= SEQUENCES_PER_UPDATE
    assert schedule[0] == 0
    assert schedule[-1] == SEQUENCES_PER_UPDATE
    assert sum(schedule) == STABLE_PER_UPDATE * update_count
    assert sum(schedule) == total_reserved_sequences(update_count, protocol=resolved)
    assert sum(schedule) * 2 == SEQUENCES_PER_UPDATE * update_count

    cumulative = [cumulative_reserved(k, update_count, protocol=resolved) for k in range(update_count)]
    assert cumulative == sorted(cumulative), "the cumulative anneal never moves backwards"
    assert all(
        reserved_in_update(k, update_count, protocol=resolved) == schedule[k] for k in range(update_count)
    )
    # Arm B's constant split consumes the identical reserved total on a different placement.
    constant = constant_reserved_schedule(update_count, protocol=resolved)
    assert sum(constant) == sum(schedule)
    assert set(constant) == {RESERVED_PER_UPDATE}


# **Validates: Requirements 1.1, 2.1, 2.4**
@settings(max_examples=30, deadline=None)
@given(update_count=st.integers(min_value=2, max_value=512))
def test_integer_half_up_agrees_with_the_written_formula(update_count: int) -> None:
    """The integer evaluation reproduces ``floor(x + 0.5)`` exactly, without float rounding."""
    import math

    resolved = load_branch_protocol()
    for k in range(update_count):
        written = math.floor(SEQUENCES_PER_UPDATE * k * (k + 1) / (2 * (update_count - 1)) + 0.5)
        assert cumulative_reserved(k, update_count, protocol=resolved) == written


# --------------------------------------------------------------------------------------
# Exposure lists (Plan Section 8.3)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
def test_exposure_lists_are_sized_and_disjoint(protocol: dict) -> None:
    lists = _lists_for(3, protocol)
    assert len(lists.common_stable) == STABLE_PER_UPDATE * 3
    assert len(lists.reserved) == RESERVED_PER_UPDATE * 3
    # "a disjoint stable_control exposure list with the same number of sequences as the reserved list"
    assert len(lists.stable_control) == len(lists.reserved)
    assert disjointness_problems(lists) == ()
    assert lists.stable_exposure_hash == exposure_reference_hash(lists.common_stable)
    assert lists.stable_control_exposure_hash != lists.stable_exposure_hash


# **Validates: Requirements 1.1, 2.1, 2.2, 2.4**
def test_exposure_lists_fail_closed_on_supply_boundary_and_overlap(protocol: dict) -> None:
    thin_stable = _synthetic_schedule(STABLE_TRAIN, 2 * STABLE_PER_UPDATE * 2 - 1, tag="stable")
    reserved = _synthetic_schedule(RESERVED, RESERVED_PER_UPDATE * 2, tag="reserved")
    with pytest.raises(BranchContractError) as thin:
        build_exposure_lists(thin_stable, reserved, update_count=2, protocol=protocol)
    assert BRANCH_SUPPLY_EXHAUSTED in str(thin.value)

    # A reserved-boundary schedule may not stand in for the stable exposure supply.
    with pytest.raises(BranchContractError):
        build_exposure_lists(
            _synthetic_schedule(RESERVED, 2 * STABLE_PER_UPDATE * 2, tag="reserved"),
            reserved,
            update_count=2,
            protocol=protocol,
        )

    # An overlapping stable_control would leak B's reserved-position replacement into A's
    # ordinary exposure, so B-vs-A would no longer be a clean substitution.
    lists = _lists_for(2, protocol)
    overlapping = ExposureLists(
        update_count=lists.update_count,
        sequences_per_update=lists.sequences_per_update,
        stable_per_update=lists.stable_per_update,
        reserved_per_update=lists.reserved_per_update,
        common_stable=lists.common_stable,
        reserved=lists.reserved,
        stable_control=lists.common_stable,
        stable_schedule_hash=lists.stable_schedule_hash,
        reserved_schedule_hash=lists.reserved_schedule_hash,
        protocol_digest=lists.protocol_digest,
    )
    problems = disjointness_problems(overlapping)
    assert problems and all(BRANCH_EXPOSURE_NOT_DISJOINT in problem for problem in problems)


# --------------------------------------------------------------------------------------
# Arm identity and comparability (Plan Sections 8.2-8.4)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5, 3.3**
@settings(max_examples=8, deadline=None)
@given(update_count=st.integers(min_value=2, max_value=7))
def test_arm_invariants_hold_for_every_valid_branch_length(update_count: int) -> None:
    """B/C multiset identity, order distinctness, A/B position matching, 256 sequences/update."""
    resolved = load_branch_protocol()
    lists = _lists_for(update_count, resolved)
    arms = build_arm_schedules(lists, _policy(update_count), protocol=resolved)
    arm_a, arm_b, arm_c = arms[ARM_A], arms[ARM_B], arms[ARM_C]
    expected_reserved = STABLE_PER_UPDATE * update_count

    # 256 sequences in every update, for every arm.
    for arm in arms.values():
        assert arm.sequences_per_update_observed == (SEQUENCES_PER_UPDATE,) * update_count
        assert len(arm.slots) == SEQUENCES_PER_UPDATE * update_count

    # B and C: identical multisets including multiplicity, different temporal placement.
    assert arm_b.stable_exposure_hash == arm_c.stable_exposure_hash
    assert arm_b.reserved_exposure_hash == arm_c.reserved_exposure_hash
    assert arm_b.training_order_hash != arm_c.training_order_hash
    assert sorted(entry.reference for entry in arm_b.entries) == sorted(
        entry.reference for entry in arm_c.entries
    )
    assert arm_b.reserved_sequence_count == arm_c.reserved_sequence_count == expected_reserved

    # B is the constant split; C is the exact anneal.
    assert set(arm_b.reserved_per_update) == {RESERVED_PER_UPDATE}
    assert arm_c.reserved_per_update == annealed_reserved_schedule(update_count, protocol=resolved)
    assert arm_c.reserved_per_update[0] == 0
    assert arm_c.reserved_per_update[-1] == SEQUENCES_PER_UPDATE

    # A is B with a position-matched replacement, and consumes no reserved data.
    assert arm_a.reserved_sequence_count == 0
    assert arm_a.stable_control_sequence_count == expected_reserved
    assert arm_a.stable_exposure_hash == arm_b.stable_exposure_hash
    assert arm_a.reserved_positions == () and arm_b.reserved_positions != ()
    assert position_matching_problems(arm_a, arm_b) == ()
    assert arm_a.training_order_hash != arm_b.training_order_hash

    # Preservation: one parent, one LR schedule, one batch layout, one optimizer/RNG policy.
    assert len({arm.policy.fingerprint() for arm in arms.values()}) == 1
    assert len({arm.policy.learning_rate_schedule() for arm in arms.values()}) == 1

    assert_branch_arms_valid(arms, lists, protocol=resolved)


# **Validates: Requirements 2.4, 2.5**
def test_verify_branch_arms_reports_pass_and_explicit_deferrals(protocol: dict) -> None:
    lists = _lists_for(4, protocol)
    arms = build_arm_schedules(lists, _policy(4), protocol=protocol)
    results = verify_branch_arms(arms, lists, protocol=protocol)
    statuses = {result.check_id: result.status for result in results}

    assert not [result for result in results if result.failed]
    assert statuses["branch.bc_training_order_hash_differs"] == "PASS"
    assert statuses["branch.ab_position_matched_replacement"] == "PASS"
    assert statuses["branch.c_follows_exact_anneal"] == "PASS"
    # Unmeasured prerequisites are DEFERRED with a blocker, never a silent PASS.
    for check_id in (
        "branch.measured_throughput",
        "branch.selected_size_band",
        "branch.bound_parent_hash",
        "branch.arm_runs",
    ):
        assert statuses[check_id] == "DEFERRED"
        reason = next(result.reason for result in results if result.check_id == check_id)
        assert "blocker=" in reason and "owner=" in reason

    report = format_branch_report(results)
    assert "RESULT: PASS" in report and "DEFERRED=4" in report


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
def test_identical_bc_order_fails_closed(protocol: dict) -> None:
    """A 'C' that reuses B's constant placement is not a temporal contrast at all."""
    lists = _lists_for(3, protocol)
    policy = _policy(3)
    arm_b = build_arm_b(lists, policy, protocol=protocol)
    fake_c = ArmSchedule(
        arm_id=ARM_C,
        slots=arm_b.slots,
        policy=policy,
        exposure_lists_hash=lists.content_hash(),
        protocol_digest=arm_b.protocol_digest,
    )
    arms = {ARM_A: build_arm_a(arm_b, lists, protocol=protocol), ARM_B: arm_b, ARM_C: fake_c}
    failures = {result.check_id for result in verify_branch_arms(arms, lists, protocol=protocol) if result.failed}
    assert "branch.bc_training_order_hash_differs" in failures
    assert "branch.c_follows_exact_anneal" in failures
    with pytest.raises(BranchContractError) as error:
        assert_branch_arms_valid(arms, lists, protocol=protocol)
    assert BRANCH_ORDER_HASH_IDENTICAL in str(error.value)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
def test_unmatched_ab_position_fails_closed(protocol: dict) -> None:
    """Swapping one replacement breaks the positional contrast, and must be detected."""
    lists = _lists_for(2, protocol)
    policy = _policy(2)
    arms = build_arm_schedules(lists, policy, protocol=protocol)
    arm_a, arm_b = arms[ARM_A], arms[ARM_B]

    slots = list(arm_a.slots)
    first, second = arm_b.reserved_positions[0], arm_b.reserved_positions[1]
    slots[first], slots[second] = (
        ExposureSlot(
            slots[first].update_index,
            slots[first].position_in_update,
            STABLE_CONTROL,
            slots[second].list_index,
            lists.stable_control[slots[second].list_index],
        ),
        ExposureSlot(
            slots[second].update_index,
            slots[second].position_in_update,
            STABLE_CONTROL,
            slots[first].list_index,
            lists.stable_control[slots[first].list_index],
        ),
    )
    shuffled = ArmSchedule(ARM_A, tuple(slots), policy, arm_a.exposure_lists_hash, arm_a.protocol_digest)

    problems = position_matching_problems(shuffled, arm_b)
    assert problems and all(BRANCH_POSITION_NOT_MATCHED in problem for problem in problems)
    # The multiset is unchanged, so only the position check can catch this.
    assert shuffled.stable_control_exposure_hash == arm_a.stable_control_exposure_hash
    with pytest.raises(BranchContractError) as error:
        assert_branch_arms_valid({**arms, ARM_A: shuffled}, lists, protocol=protocol)
    assert BRANCH_POSITION_NOT_MATCHED in str(error.value)


# **Validates: Requirements 1.1, 2.1, 2.4, 3.3**
def test_diverging_shared_policy_fails_closed(protocol: dict) -> None:
    """A and B must share parent, LR, batch layout, optimizer, and RNG policy."""
    lists = _lists_for(2, protocol)
    policy = _policy(2)
    arms = build_arm_schedules(lists, policy, protocol=protocol)
    drifted = SharedBranchPolicy(
        update_count=2,
        parent_lr=3e-4,
        parent_binding_id="some-other-parent",
        rng_policy="per_arm_seed",
    )
    arms = {**arms, ARM_C: ArmSchedule(ARM_C, arms[ARM_C].slots, drifted, arms[ARM_C].exposure_lists_hash, arms[ARM_C].protocol_digest)}
    failures = {
        result.check_id: result.reason
        for result in verify_branch_arms(arms, lists, protocol=protocol)
        if result.failed
    }
    assert "branch.shared_policy_identical" in failures
    assert "branch.common_parent_binding" in failures
    assert BRANCH_ARM_POLICY_DIVERGED in failures["branch.shared_policy_identical"]


# **Validates: Requirements 1.1, 2.1, 2.4, 3.3**
def test_branch_lr_is_identical_across_arms_and_decays_to_zero(protocol: dict) -> None:
    policy = _policy(5)
    schedule = policy.learning_rate_schedule()
    assert schedule[0] == pytest.approx(3e-4)
    assert schedule[-1] == 0.0
    assert list(schedule) == sorted(schedule, reverse=True)
    assert branch_learning_rate(3e-4, 2, 5) == pytest.approx(1.5e-4)
    with pytest.raises(BranchContractError):
        branch_learning_rate(3e-4, 5, 5)

    lists = _lists_for(5, protocol)
    arms = build_arm_schedules(lists, policy, protocol=protocol)
    assert {arm.policy.learning_rate_schedule() for arm in arms.values()} == {schedule}


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_arm_construction_is_deterministic_and_hash_addressed(protocol: dict) -> None:
    lists_one = _lists_for(3, protocol)
    lists_two = _lists_for(3, protocol)
    assert lists_one.content_hash() == lists_two.content_hash()

    arms_one = build_arm_schedules(lists_one, _policy(3), protocol=protocol)
    arms_two = build_arm_schedules(lists_two, _policy(3), protocol=protocol)
    assert {key: arm.content_hash() for key, arm in arms_one.items()} == {
        key: arm.content_hash() for key, arm in arms_two.items()
    }
    # The three arms are distinct artifacts even though they share one exposure-list set.
    assert len({arm.content_hash() for arm in arms_one.values()}) == 3
    assert len({arm.exposure_lists_hash for arm in arms_one.values()}) == 1


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_arm_and_exposure_persistence_round_trip_and_fail_closed(tmp_path: Path, protocol: dict) -> None:
    lists = _lists_for(2, protocol)
    arms = build_arm_schedules(lists, _policy(2), protocol=protocol)

    lists_path = write_exposure_lists(tmp_path / "exposure_lists.json", lists)
    assert load_exposure_lists(lists_path).content_hash() == lists.content_hash()

    arm_path = write_arm_schedule(tmp_path / "arm_c.json", arms[ARM_C])
    reloaded = load_arm_schedule(arm_path)
    assert reloaded.content_hash() == arms[ARM_C].content_hash()
    assert reloaded.training_order_hash == arms[ARM_C].training_order_hash

    payload = json.loads(arm_path.read_text(encoding="utf-8"))
    payload["slots"][0]["entry"]["token_offset"] += 1
    arm_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BranchContractError) as error:
        load_arm_schedule(arm_path)
    assert BRANCH_CONTENT_HASH_MISMATCH in str(error.value)


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1, 3.3**
def test_arm_cursor_is_one_integer_bound_to_the_arm(protocol: dict) -> None:
    """The arm reuses task 3.10's cursor semantics, so resume plumbing does not fork."""
    lists = _lists_for(2, protocol)
    arms = build_arm_schedules(lists, _policy(2), protocol=protocol)
    arm_b, arm_c = arms[ARM_B], arms[ARM_C]

    cursor = arm_b.cursor(0)
    cursor.advance(SEQUENCES_PER_UPDATE)
    state = cursor.state_dict()
    assert sorted(state) == ["format_version", "schedule_content_hash", "schedule_cursor"]

    resumed = arm_b.cursor(0)
    resumed.load_state_dict(state)
    assert resumed.position == SEQUENCES_PER_UPDATE
    # One integer is enough to name the exact next sequence: the first slot of update 1.
    next_slot = arm_b.slots[resumed.position]
    assert (next_slot.update_index, next_slot.position_in_update) == (1, 0)
    assert next_slot.list_id == COMMON_STABLE

    with pytest.raises(ScheduleResumeError):
        arm_c.cursor(0).load_state_dict(state)


# --------------------------------------------------------------------------------------
# Branch sizes (Plan Section 8.1)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.4, 2.5**
def test_frozen_branch_sizes_are_update_aligned(protocol: dict) -> None:
    assert branch_size_alignment_problems(protocol) == ()
    bands = branch_size_bands(protocol)
    assert [band.band_id for band in bands] == [
        "below_12k",
        "12k_to_below_15k",
        "at_least_15k_with_calendar_reserve",
    ]
    assert [band.primary_updates_per_arm for band in bands] == [763, 1145, 2289]
    assert [band.primary_tokens_per_arm for band in bands] == [200_015_872, 300_154_880, 600_047_616]
    assert bands[-1].requires_calendar_reserve is True


# **Validates: Requirements 2.3, 2.5**
def test_branch_size_selection_requires_a_measured_throughput(protocol: dict) -> None:
    with pytest.raises(BranchesNotReadyError) as blocked:
        select_branch_size(None, protocol=protocol)
    assert "measured" in str(blocked.value)

    assert select_branch_size(11_999, protocol=protocol).band_id == "below_12k"
    assert select_branch_size(12_000, protocol=protocol).band_id == "12k_to_below_15k"
    assert select_branch_size(14_999.9, protocol=protocol).band_id == "12k_to_below_15k"
    assert select_branch_size(15_000, protocol=protocol).band_id == "at_least_15k_with_calendar_reserve"
    with pytest.raises(BranchContractError):
        select_branch_size(0, protocol=protocol)


# --------------------------------------------------------------------------------------
# One end-to-end pass over real tiny shards
# --------------------------------------------------------------------------------------


_SENTENCES = (
    "The water cycle moves water between the ocean, the atmosphere, and the land.",
    "Photosynthesis converts light energy into chemical energy stored in sugars.",
    "A prime number has exactly two distinct positive divisors, one and itself.",
    "Sedimentary rock forms when layers of particles are compacted over long periods.",
    "The industrial revolution changed how goods were produced and transported.",
    "An ecosystem includes every organism in an area together with its environment.",
    "Momentum is conserved when no external force acts on a closed system.",
    "A cell membrane controls which substances enter and leave the cell.",
)


def _body(tag: str, index: int) -> str:
    unique = f"{tag}{index:03d}"
    words = " ".join(_SENTENCES[(index * 3 + step) % len(_SENTENCES)] for step in range(8)).split()
    parts: list[str] = []
    for position, word in enumerate(words):
        parts.append(word)
        if position % 3 == 2:
            parts.append(f"[{unique}:{position}]")
    return " ".join(parts) + "\n"


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_arms_build_over_real_source_tagged_shards(tmp_path: Path, protocol: dict) -> None:
    """One bounded end-to-end pass: real shards -> schedules -> exposure lists -> arms.

    A short sequence length keeps the fixture tiny while still supplying the 512 stable and
    256 reserved references a two-update branch needs.
    """
    stable_sources = ("dclm", "fineweb_edu", "narrative", "openwebmath")
    reserved_sources = ("reserved_science", "reserved_textbook", "reserved_wikipedia")
    tokenizer_protocol = load_tokenizer_protocol()
    registry = load_source_registry()

    texts = [_body(source, index) for source in stable_sources + reserved_sources for index in range(10)]
    tokenizer = build_tokenizer(texts, protocol=tokenizer_protocol, vocab_size=900)

    def documents(sources: tuple[str, ...], boundary: str) -> list[ShardDocument]:
        return [
            ShardDocument(
                document_id=f"{source}-{index:03d}",
                source_id=source,
                text=_body(source, index),
                boundary=boundary,
            )
            for source in sources
            for index in range(10)
        ]

    manifests = {}
    for split_id, sources in ((STABLE_TRAIN, stable_sources), (RESERVED, reserved_sources)):
        manifests[split_id] = build_split_manifest(
            tmp_path,
            tokenizer,
            documents(sources, split_id),
            split_id=split_id,
            shard_document_budget=5,
            registry=registry,
            tokenizer_protocol=tokenizer_protocol,
        )

    schedules = {
        split_id: build_materialized_schedule(
            manifest, sequence_length=8, seed=11, local_shuffle_buffer_sequences=16
        )
        for split_id, manifest in manifests.items()
    }
    assert schedules[STABLE_TRAIN].sequence_count >= 2 * STABLE_PER_UPDATE * 2
    assert schedules[RESERVED].sequence_count >= RESERVED_PER_UPDATE * 2

    lists = build_exposure_lists(
        schedules[STABLE_TRAIN], schedules[RESERVED], update_count=2, protocol=protocol
    )
    arms = build_arm_schedules(lists, _policy(2), protocol=protocol)
    assert_branch_arms_valid(arms, lists, protocol=protocol)

    # Reserved exposure really comes from the reserved namespace, and A never touches it.
    reserved_namespaces = {entry.namespace for entry in lists.reserved}
    assert all(namespace.startswith("reserved/") for namespace in reserved_namespaces), reserved_namespaces
    assert arms[ARM_A].entries_from(RESERVED_LIST) == ()
    assert {entry.namespace for entry in arms[ARM_A].entries_from(COMMON_STABLE)} <= {
        entry.namespace for entry in schedules[STABLE_TRAIN].entries
    }
