"""Deterministic materialized mixture schedules and the one integer resume cursor.

Plan Sections 5.4, 7.2, and 8.3. Nothing here downloads a corpus, produces real shards, or
starts training. The tests prove that:

- a schedule is an immutable set of ``(shard_id, token_offset, length)`` references carrying
  source tags and a content hash,
- identical inputs and seed produce a byte-identical schedule, and different seeds change
  the exposure order,
- every reference stays inside its shard, appears once, and never overlaps a neighbour,
- shard order is shuffled while in-shard reads stay sequential inside the bounded local
  shuffle window,
- per-source quotas reconcile exactly and an unmet quota fails closed,
- resume restores one integer ``schedule_cursor`` and yields the exact next sequence, while
  a cursor from a different schedule fails closed,
- the superseded random flat-stream sampler stays available under an explicit pilot label.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from tinybench_lm.data import SAMPLER_SCOPE, PackedTokenDataset
from tinybench_lm.data_protocols import ProtocolMutatedError, protocol_digest
from tinybench_lm.schedule import (
    CURSOR_STATE_KEY,
    FROZEN_SCHEDULE_PROTOCOL_SHA256,
    SCHEDULE_CONTENT_HASH_MISMATCH,
    SCHEDULE_CURSOR_OUT_OF_RANGE,
    SCHEDULE_LOCAL_SHUFFLE_UNBOUNDED,
    SCHEDULE_PROTOCOL_PATH,
    SCHEDULE_REFERENCE_OUT_OF_BOUNDS,
    SCHEDULE_SOURCE_QUOTA_MISMATCH,
    SCHEDULE_SOURCE_SUPPLY_EXHAUSTED,
    MaterializedSchedule,
    ScheduleContractError,
    ScheduleCursor,
    ScheduledTokenStream,
    ScheduleEntry,
    ScheduleResumeError,
    SchedulesNotReadyError,
    assert_ready_for_real_schedules,
    assert_schedule_valid,
    available_sequences_per_source,
    build_materialized_schedule,
    canonical_payload_bytes,
    exposure_reference_hash,
    load_schedule,
    load_schedule_protocol,
    quota_reconciliation,
    reference_bound_violations,
    sequential_read_violations,
    training_order_hash,
    verify_schedule,
    write_schedule,
)
from tinybench_lm.shards import (
    STABLE_TRAIN,
    ShardDocument,
    SplitManifest,
    build_split_manifest,
    write_split_manifest,
)
from tinybench_lm.source_manifest import load_source_registry
from tinybench_lm.tokenizer import build_tokenizer, load_tokenizer_protocol

CPU = torch.device("cpu")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_SENTENCES = (
    "The water cycle moves water between the ocean, the atmosphere, and the land.",
    "Photosynthesis converts light energy into chemical energy stored in sugars.",
    "A prime number has exactly two distinct positive divisors, one and itself.",
    "Sedimentary rock forms when layers of particles are compacted over long periods.",
    "The industrial revolution changed how goods were produced and transported.",
    "An ecosystem includes every organism in an area together with its environment.",
    "Momentum is conserved when no external force acts on a closed system.",
    "A cell membrane controls which substances enter and leave the cell.",
    "She walked to the harbour, counted the boats, and waited for the tide.",
    "Let f(x) = x^2 + 2x + 1. Then f(x) = (x + 1)^2 and f'(x) = 2x + 2.",
)

STABLE_SOURCES = ("dclm", "fineweb_edu", "narrative", "openwebmath")


def _body(tag: str, index: int) -> str:
    """A distinct document body, marked so unrelated fixtures never cluster by accident."""
    unique = f"{tag}{index:03d}"
    words = " ".join(_SENTENCES[(index * 3 + step) % len(_SENTENCES)] for step in range(6)).split()
    parts: list[str] = []
    for position, word in enumerate(words):
        parts.append(word)
        if position % 3 == 2:
            parts.append(f"[{unique}:{position}]")
    return " ".join(parts) + "\n"


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_schedule_protocol()


@pytest.fixture(scope="module")
def shard_root_and_manifest(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, SplitManifest]:
    """A tiny stable-train split: four sources, two shards each, so shuffling is observable."""
    tokenizer_protocol = load_tokenizer_protocol()
    texts = [_body(source_id, index) for source_id in STABLE_SOURCES for index in range(6)]
    tokenizer = build_tokenizer(texts, protocol=tokenizer_protocol, vocab_size=900)
    documents = [
        ShardDocument(
            document_id=f"{source_id}-{index:03d}",
            source_id=source_id,
            text=_body(source_id, index),
            boundary=STABLE_TRAIN,
        )
        for source_id in STABLE_SOURCES
        for index in range(4)
    ]
    root = tmp_path_factory.mktemp("shards")
    manifest = build_split_manifest(
        root,
        tokenizer,
        documents,
        split_id=STABLE_TRAIN,
        shard_document_budget=2,
        registry=load_source_registry(),
        tokenizer_protocol=tokenizer_protocol,
    )
    return root, manifest


@pytest.fixture(scope="module")
def manifest(shard_root_and_manifest: tuple[Path, SplitManifest]) -> SplitManifest:
    return shard_root_and_manifest[1]


@pytest.fixture(scope="module")
def shard_root(shard_root_and_manifest: tuple[Path, SplitManifest]) -> Path:
    return shard_root_and_manifest[0]


# --------------------------------------------------------------------------------------
# The frozen contract
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_frozen_schedule_protocol_is_pinned_and_immutable(tmp_path: Path) -> None:
    assert protocol_digest(SCHEDULE_PROTOCOL_PATH) == FROZEN_SCHEDULE_PROTOCOL_SHA256["schedule_v1.yaml"]

    mutated = tmp_path / "schedule_v1.yaml"
    text = SCHEDULE_PROTOCOL_PATH.read_text(encoding="utf-8")
    mutated.write_text(
        text.replace(
            "default_local_shuffle_buffer_sequences: 1024",
            "default_local_shuffle_buffer_sequences: 999999",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolMutatedError):
        load_schedule_protocol(mutated)


# **Validates: Requirements 2.3, 2.5**
def test_real_scale_schedule_construction_is_explicitly_gated(protocol: dict) -> None:
    with pytest.raises(SchedulesNotReadyError) as error:
        assert_ready_for_real_schedules(protocol)
    message = str(error.value)
    assert "blocker=" in message and "owner=" in message and "next_action=" in message


# **Validates: Requirements 1.1, 3.1, 3.3**
def test_frozen_contract_labels_the_random_flat_stream_sampler_pilot_only(protocol: dict) -> None:
    pilot = protocol["pilot_sampler"]
    assert pilot["status"] == "PILOT_ONLY"
    assert pilot["eligible_for_final_training"] is False
    assert protocol["final_training_reader"]["implementation"] == "tinybench_lm.schedule.ScheduledTokenStream"
    assert protocol["cursor"]["state_key"] == CURSOR_STATE_KEY


# --------------------------------------------------------------------------------------
# Immutable records with source tags and a content hash
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_schedule_records_are_plan_5_4_references_with_source_tags(manifest: SplitManifest) -> None:
    schedule = build_materialized_schedule(
        manifest, sequence_length=8, seed=1337, local_shuffle_buffer_sequences=16
    )
    assert schedule.sequence_count > 0
    records = {record.shard_id: record for record in manifest.shards}

    for entry in schedule.entries:
        shard_id, token_offset, length = entry.reference
        assert shard_id in records
        assert length == schedule.tokens_per_entry == 9
        assert entry.source_id == records[shard_id].source_id
        assert entry.namespace == records[shard_id].namespace
        # Immutable records: a frozen dataclass cannot be edited in place.
        with pytest.raises(Exception):
            entry.token_offset = token_offset + 1  # type: ignore[misc]

    assert len(schedule.content_hash()) == 64
    assert schedule.loss_tokens == 8 * schedule.sequence_count
    # Source identity survives into the schedule, so a mixture is auditable.
    assert set(schedule.sequences_per_source) == set(STABLE_SOURCES)


# **Validates: Requirements 1.1, 2.4, 2.5**
def test_schedule_shuffles_shard_order_and_keeps_all_sources(manifest: SplitManifest) -> None:
    schedule = build_materialized_schedule(
        manifest, sequence_length=8, seed=7, local_shuffle_buffer_sequences=1
    )
    shard_order = schedule.shard_ids
    assert len(shard_order) == len({record.shard_id for record in manifest.shards})
    # Shard-level shuffle: the first-appearance order is not the sorted manifest order.
    assert list(shard_order) != sorted(shard_order)
    # Sequential reads inside shards: with a one-sequence buffer there is no inversion.
    assert sequential_read_violations(schedule) == ()


# --------------------------------------------------------------------------------------
# Property: byte-identical schedules for identical inputs and seed
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    sequence_length=st.integers(min_value=2, max_value=16),
    buffer_sequences=st.integers(min_value=1, max_value=32),
)
@settings(max_examples=12, deadline=None, derandomize=True)
def test_identical_inputs_and_seed_produce_byte_identical_schedules(
    shard_root_and_manifest: tuple[Path, SplitManifest],
    seed: int,
    sequence_length: int,
    buffer_sequences: int,
) -> None:
    manifest = shard_root_and_manifest[1]
    first = build_materialized_schedule(
        manifest,
        sequence_length=sequence_length,
        seed=seed,
        local_shuffle_buffer_sequences=buffer_sequences,
    )
    second = build_materialized_schedule(
        manifest,
        sequence_length=sequence_length,
        seed=seed,
        local_shuffle_buffer_sequences=buffer_sequences,
    )
    assert canonical_payload_bytes(first.to_dict()) == canonical_payload_bytes(second.to_dict())
    assert first.content_hash() == second.content_hash()
    assert training_order_hash(first.entries) == training_order_hash(second.entries)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_different_seeds_change_the_exposure_order(manifest: SplitManifest) -> None:
    orders = {
        training_order_hash(
            build_materialized_schedule(
                manifest, sequence_length=8, seed=seed, local_shuffle_buffer_sequences=8
            ).entries
        )
        for seed in range(8)
    }
    assert len(orders) > 1, "the ordering seed has no effect, so exposure order is not seed-controlled"


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_construction_does_not_depend_on_global_rng_state(manifest: SplitManifest) -> None:
    baseline = build_materialized_schedule(manifest, sequence_length=8, seed=99).content_hash()
    np.random.seed(4)
    np.random.random(1000)
    torch.manual_seed(4)
    assert build_materialized_schedule(manifest, sequence_length=8, seed=99).content_hash() == baseline


# --------------------------------------------------------------------------------------
# Property: every reference stays in bounds
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
@given(
    sequence_length=st.integers(min_value=2, max_value=24),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=12, deadline=None, derandomize=True)
def test_every_reference_stays_inside_its_shard(
    shard_root_and_manifest: tuple[Path, SplitManifest], sequence_length: int, seed: int
) -> None:
    manifest = shard_root_and_manifest[1]
    schedule = build_materialized_schedule(
        manifest, sequence_length=sequence_length, seed=seed, local_shuffle_buffer_sequences=8
    )
    records = {record.shard_id: record for record in manifest.shards}
    stride = sequence_length + 1
    for entry in schedule.entries:
        record = records[entry.shard_id]
        assert 0 <= entry.token_offset
        assert entry.end_offset <= record.token_count
        assert entry.token_offset % stride == 0
    assert reference_bound_violations(manifest, schedule) == ()
    # References never repeat inside one pass, so no sequence is silently seen twice.
    assert len({entry.reference for entry in schedule.entries}) == schedule.sequence_count


# **Validates: Requirements 1.2, 2.2**
def test_out_of_bounds_and_overlapping_references_fail_closed(manifest: SplitManifest) -> None:
    schedule = build_materialized_schedule(manifest, sequence_length=8, seed=3, local_shuffle_buffer_sequences=4)
    first = schedule.entries[0]
    record = {item.shard_id: item for item in manifest.shards}[first.shard_id]

    beyond = MaterializedSchedule(
        **{
            **{key: getattr(schedule, key) for key in ("schedule_id", "split_id", "boundary", "sequence_length", "label_shift", "seed", "local_shuffle_buffer_sequences", "manifest_content_hash", "protocol_digest", "requested_source_quotas")},
            "entries": (
                ScheduleEntry(first.shard_id, record.token_count - 2, first.length, first.source_id, first.namespace),
            ),
        }
    )
    problems = reference_bound_violations(manifest, beyond)
    assert any(SCHEDULE_REFERENCE_OUT_OF_BOUNDS in problem for problem in problems)

    duplicated = MaterializedSchedule(
        **{
            **{key: getattr(schedule, key) for key in ("schedule_id", "split_id", "boundary", "sequence_length", "label_shift", "seed", "local_shuffle_buffer_sequences", "manifest_content_hash", "protocol_digest", "requested_source_quotas")},
            "entries": (first, first),
        }
    )
    assert any("DUPLICATE_REFERENCE" in problem for problem in reference_bound_violations(manifest, duplicated))


# **Validates: Requirements 1.2, 2.2, 2.4**
def test_unbounded_local_shuffle_is_detected(manifest: SplitManifest) -> None:
    schedule = build_materialized_schedule(manifest, sequence_length=4, seed=11, local_shuffle_buffer_sequences=2)
    assert sequential_read_violations(schedule) == ()

    # Move one shard's last reference to the front: displacement now far exceeds the window.
    entries = list(schedule.entries)
    shard_id = entries[0].shard_id
    same_shard = [index for index, entry in enumerate(entries) if entry.shard_id == shard_id]
    entries.insert(0, entries.pop(same_shard[-1]))
    tampered = MaterializedSchedule(
        **{
            **{key: getattr(schedule, key) for key in ("schedule_id", "split_id", "boundary", "sequence_length", "label_shift", "seed", "local_shuffle_buffer_sequences", "manifest_content_hash", "protocol_digest", "requested_source_quotas")},
            "entries": tuple(entries),
        }
    )
    problems = sequential_read_violations(tampered)
    assert problems and all(SCHEDULE_LOCAL_SHUFFLE_UNBOUNDED in problem for problem in problems)


# --------------------------------------------------------------------------------------
# Property: source quotas reconcile
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
@given(
    sequence_length=st.integers(min_value=2, max_value=12),
    seed=st.integers(min_value=0, max_value=10_000),
    share=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=12, deadline=None, derandomize=True)
def test_source_quotas_reconcile_exactly(
    shard_root_and_manifest: tuple[Path, SplitManifest], sequence_length: int, seed: int, share: int
) -> None:
    manifest = shard_root_and_manifest[1]
    available = available_sequences_per_source(manifest, sequence_length=sequence_length)
    quotas = {
        source_id: max(1, min(supply, share * (position + 1)))
        for position, (source_id, supply) in enumerate(sorted(available.items()))
    }
    schedule = build_materialized_schedule(
        manifest,
        sequence_length=sequence_length,
        seed=seed,
        source_sequence_quotas=quotas,
        local_shuffle_buffer_sequences=4,
    )
    assert schedule.sequences_per_source == dict(sorted(quotas.items()))
    assert schedule.sequence_count == sum(quotas.values())
    assert quota_reconciliation(schedule) == ()
    assert schedule.tokens_per_source == {
        source_id: count * (sequence_length + 1) for source_id, count in sorted(quotas.items())
    }


# **Validates: Requirements 1.2, 2.2**
def test_quota_above_supply_and_incomplete_quotas_fail_closed(manifest: SplitManifest) -> None:
    available = available_sequences_per_source(manifest, sequence_length=8)
    impossible = {source_id: supply + 1 for source_id, supply in available.items()}
    with pytest.raises(ScheduleContractError) as error:
        build_materialized_schedule(
            manifest, sequence_length=8, seed=5, source_sequence_quotas=impossible
        )
    assert SCHEDULE_SOURCE_SUPPLY_EXHAUSTED in str(error.value)

    partial = {source_id: 1 for source_id in list(available)[:2]}
    with pytest.raises(ScheduleContractError) as error:
        build_materialized_schedule(manifest, sequence_length=8, seed=5, source_sequence_quotas=partial)
    assert SCHEDULE_SOURCE_QUOTA_MISMATCH in str(error.value)


# --------------------------------------------------------------------------------------
# The one integer cursor
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1, 3.3**
def test_resume_state_is_one_integer_bound_to_the_schedule_hash(
    shard_root: Path, manifest: SplitManifest
) -> None:
    schedule = build_materialized_schedule(manifest, sequence_length=8, seed=21, local_shuffle_buffer_sequences=8)
    stream = ScheduledTokenStream(shard_root, manifest, schedule)
    try:
        stream.get_batch(2, 8, CPU)
        state = stream.state_dict()
        assert sorted(state) == ["format_version", "schedule_content_hash", CURSOR_STATE_KEY]
        assert isinstance(state[CURSOR_STATE_KEY], int)
        assert state[CURSOR_STATE_KEY] == 2
        assert state["schedule_content_hash"] == schedule.content_hash()
        # The state is JSON-serializable: one integer plus one hash, no opaque RNG blob.
        assert json.loads(json.dumps(state))[CURSOR_STATE_KEY] == 2
    finally:
        stream.close()


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5, 3.1, 3.3**
@given(
    consumed_batches=st.integers(min_value=1, max_value=4),
    batch_size=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=10, deadline=None, derandomize=True)
def test_resume_yields_the_exact_next_sequence(
    shard_root_and_manifest: tuple[Path, SplitManifest],
    consumed_batches: int,
    batch_size: int,
    seed: int,
) -> None:
    root, manifest = shard_root_and_manifest
    schedule = build_materialized_schedule(
        manifest, sequence_length=8, seed=seed, local_shuffle_buffer_sequences=8
    )
    stream = ScheduledTokenStream(root, manifest, schedule)
    resumed = ScheduledTokenStream(root, manifest, schedule)
    try:
        for _ in range(consumed_batches):
            stream.get_batch(batch_size, 8, CPU)
        state = stream.state_dict()
        expected_inputs, expected_targets = stream.get_batch(batch_size, 8, CPU)
        # Consume further, so the restored position is not merely where the stream stopped.
        stream.get_batch(batch_size, 8, CPU)

        resumed.load_state_dict(state)
        actual_inputs, actual_targets = resumed.get_batch(batch_size, 8, CPU)
        assert torch.equal(expected_inputs, actual_inputs)
        assert torch.equal(expected_targets, actual_targets)
        assert resumed.position == state[CURSOR_STATE_KEY] + batch_size
    finally:
        stream.close()
        resumed.close()


# **Validates: Requirements 1.2, 2.2, 2.4**
def test_a_cursor_from_a_different_schedule_fails_closed(
    shard_root: Path, manifest: SplitManifest
) -> None:
    first = build_materialized_schedule(manifest, sequence_length=8, seed=1, local_shuffle_buffer_sequences=8)
    other = build_materialized_schedule(manifest, sequence_length=8, seed=2, local_shuffle_buffer_sequences=8)
    assert first.content_hash() != other.content_hash()

    stream = ScheduledTokenStream(shard_root, manifest, first)
    try:
        stream.get_batch(1, 8, CPU)
        with pytest.raises(ScheduleResumeError) as error:
            stream.load_state_dict(other.cursor(3).state_dict())
        assert SCHEDULE_CONTENT_HASH_MISMATCH in str(error.value)

        with pytest.raises(ScheduleResumeError):
            stream.load_state_dict({"schedule_content_hash": first.content_hash()})
        with pytest.raises(ScheduleResumeError) as error:
            stream.load_state_dict(
                {CURSOR_STATE_KEY: first.sequence_count + 1, "schedule_content_hash": first.content_hash()}
            )
        assert SCHEDULE_CURSOR_OUT_OF_RANGE in str(error.value)
    finally:
        stream.close()


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2**
def test_exhausted_schedule_fails_closed_instead_of_resampling(
    shard_root: Path, manifest: SplitManifest
) -> None:
    available = available_sequences_per_source(manifest, sequence_length=8)
    quotas = {source_id: 1 for source_id in available}
    schedule = build_materialized_schedule(
        manifest, sequence_length=8, seed=8, source_sequence_quotas=quotas, local_shuffle_buffer_sequences=1
    )
    stream = ScheduledTokenStream(shard_root, manifest, schedule)
    try:
        stream.get_batch(schedule.sequence_count, 8, CPU)
        with pytest.raises(ScheduleContractError) as error:
            stream.get_batch(1, 8, CPU)
        assert SCHEDULE_CURSOR_OUT_OF_RANGE in str(error.value)
    finally:
        stream.close()


# **Validates: Requirements 1.1, 2.1, 3.3**
def test_reads_are_memory_mapped_and_match_the_referenced_shard_tokens(
    shard_root: Path, manifest: SplitManifest
) -> None:
    schedule = build_materialized_schedule(manifest, sequence_length=6, seed=44, local_shuffle_buffer_sequences=4)
    records = {record.shard_id: record for record in manifest.shards}
    stream = ScheduledTokenStream(shard_root, manifest, schedule)
    try:
        inputs, targets = stream.get_batch(3, 6, CPU)
        assert inputs.shape == (3, 6) and targets.shape == (3, 6)
        assert torch.equal(inputs[:, 1:], targets[:, :-1])

        for position, entry in enumerate(schedule.entries[:3]):
            record = records[entry.shard_id]
            raw = np.memmap(shard_root / record.relative_path, dtype=np.uint16, mode="r")
            expected = np.asarray(raw[entry.token_offset : entry.end_offset]).astype(np.int64)
            assert list(inputs[position].tolist()) == list(expected[:-1])
            assert list(targets[position].tolist()) == list(expected[1:])
            raw._mmap.close()
        # The reader memory-maps each shard once instead of re-opening per sequence.
        assert set(stream._memmaps) <= set(records)
    finally:
        stream.close()


# **Validates: Requirements 1.2, 2.2**
def test_get_batch_rejects_a_sequence_length_the_schedule_was_not_built_for(
    shard_root: Path, manifest: SplitManifest
) -> None:
    schedule = build_materialized_schedule(manifest, sequence_length=8, seed=6, local_shuffle_buffer_sequences=4)
    stream = ScheduledTokenStream(shard_root, manifest, schedule)
    try:
        with pytest.raises(ScheduleContractError):
            stream.get_batch(1, 16, CPU)
    finally:
        stream.close()


# **Validates: Requirements 1.2, 2.2**
def test_cursor_rejects_negative_and_backwards_movement() -> None:
    cursor = ScheduleCursor("a" * 64, 0, 4)
    cursor.advance(4)
    assert cursor.exhausted
    with pytest.raises(ScheduleContractError):
        cursor.advance(1)
    with pytest.raises(ScheduleContractError):
        cursor.advance(-1)
    with pytest.raises(ScheduleContractError):
        ScheduleCursor("a" * 64, -1, 4)


# --------------------------------------------------------------------------------------
# Persistence and verification
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_written_schedule_round_trips_and_tampering_fails_closed(
    tmp_path: Path, manifest: SplitManifest
) -> None:
    schedule = build_materialized_schedule(manifest, sequence_length=8, seed=31, local_shuffle_buffer_sequences=8)
    path = write_schedule(tmp_path / "mainline.schedule.json", schedule)
    loaded = load_schedule(path)
    assert loaded.content_hash() == schedule.content_hash()
    assert loaded.entries == schedule.entries
    # Byte-identical persistence for an identical schedule.
    assert path.read_bytes() == write_schedule(tmp_path / "again.schedule.json", schedule).read_bytes()

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["token_offset"] = int(payload["entries"][0]["token_offset"]) + 9
    tampered = tmp_path / "tampered.schedule.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ScheduleContractError) as error:
        load_schedule(tampered)
    assert SCHEDULE_CONTENT_HASH_MISMATCH in str(error.value)


# **Validates: Requirements 1.1, 2.1, 2.3, 2.4, 2.5**
def test_verify_schedule_passes_and_keeps_real_scale_deferred(manifest: SplitManifest) -> None:
    available = available_sequences_per_source(manifest, sequence_length=8)
    quotas = {source_id: min(3, supply) for source_id, supply in available.items()}
    schedule = build_materialized_schedule(
        manifest, sequence_length=8, seed=17, source_sequence_quotas=quotas, local_shuffle_buffer_sequences=8
    )
    results = verify_schedule(manifest, schedule)
    assert results
    assert [result.check_id for result in results if result.failed] == []
    assert {result.check_id for result in results} >= {
        "schedule.references_in_bounds",
        "schedule.bounded_sequential_reads",
        "schedule.source_quotas_reconcile",
        "schedule.deterministic_rebuild",
        "schedule.cursor_state_is_one_integer",
    }
    deferred = [result for result in results if result.check_id == "schedule.real_scale_construction"]
    assert deferred and deferred[0].status == "DEFERRED"
    assert "blocker=" in deferred[0].reason and "next_action=" in deferred[0].reason
    assert_schedule_valid(manifest, schedule)


# **Validates: Requirements 1.2, 2.2, 2.4**
def test_verify_schedule_rejects_a_schedule_built_over_another_manifest(manifest: SplitManifest) -> None:
    schedule = build_materialized_schedule(manifest, sequence_length=8, seed=12, local_shuffle_buffer_sequences=8)
    relabelled = MaterializedSchedule(
        **{
            **{key: getattr(schedule, key) for key in ("schedule_id", "split_id", "boundary", "sequence_length", "label_shift", "seed", "local_shuffle_buffer_sequences", "protocol_digest", "requested_source_quotas", "entries")},
            "manifest_content_hash": "0" * 64,
        }
    )
    failures = [result.check_id for result in verify_schedule(manifest, relabelled) if result.failed]
    assert "schedule.manifest_binding" in failures
    with pytest.raises(ScheduleContractError):
        assert_schedule_valid(manifest, relabelled)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_exposure_hashes_separate_multiset_identity_from_order(manifest: SplitManifest) -> None:
    """Plan Section 8.3 primitive: same references, different order. Arms land in task 3.11."""
    schedule = build_materialized_schedule(manifest, sequence_length=8, seed=4, local_shuffle_buffer_sequences=8)
    reversed_entries = tuple(reversed(schedule.entries))
    assert exposure_reference_hash(schedule.entries) == exposure_reference_hash(reversed_entries)
    assert training_order_hash(schedule.entries) != training_order_hash(reversed_entries)


# --------------------------------------------------------------------------------------
# Preservation: the pilot random sampler is retained, labeled, and unchanged
# --------------------------------------------------------------------------------------


# **Validates: Requirements 3.1, 3.3**
def test_pilot_random_sampler_is_retained_and_labeled(tmp_path: Path) -> None:
    assert SAMPLER_SCOPE == "PILOT_ONLY"
    assert PackedTokenDataset.scope == "PILOT_ONLY"
    assert "PILOT ONLY" in (PackedTokenDataset.__doc__ or "")

    token_path = tmp_path / "pilot.bin"
    (np.arange(256, dtype=np.uint16) % 61).tofile(token_path)
    dataset = PackedTokenDataset(token_path, seed=5)
    try:
        dataset.get_batch(2, 8, CPU)
        state = dataset.state_dict()
        expected = dataset.get_batch(2, 8, CPU)
        dataset.get_batch(2, 8, CPU)
        dataset.load_state_dict(state)
        resumed = dataset.get_batch(2, 8, CPU)
        assert torch.equal(expected[0], resumed[0])
        assert torch.equal(expected[1], resumed[1])
    finally:
        dataset.tokens._mmap.close()


# --------------------------------------------------------------------------------------
# The training entry point selects the schedule reader, not the pilot sampler
# --------------------------------------------------------------------------------------


def _train_module() -> ModuleType:
    """Load ``train.py`` by path: it is an entry point, not an installed module."""
    spec = importlib.util.spec_from_file_location("train_entry_point", REPOSITORY_ROOT / "train.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schedule_namespace(**overrides: object) -> argparse.Namespace:
    base = {name: None for name in ("shard_root", "train_manifest", "train_schedule", "validation_manifest", "validation_schedule")}
    base.update(overrides)
    return argparse.Namespace(**base)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2**
def test_train_entry_point_refuses_a_half_configured_mixture() -> None:
    train = _train_module()
    assert train.use_materialized_schedule(_schedule_namespace()) is False
    assert (
        train.use_materialized_schedule(
            _schedule_namespace(
                shard_root=Path("a"),
                train_manifest=Path("b"),
                train_schedule=Path("c"),
                validation_manifest=Path("d"),
                validation_schedule=Path("e"),
            )
        )
        is True
    )
    with pytest.raises(ValueError) as error:
        train.use_materialized_schedule(_schedule_namespace(train_schedule=Path("c")))
    assert "Refusing to fall back to pilot random sampling" in str(error.value)


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1, 3.3**
def test_open_batch_sources_uses_the_schedule_cursor_for_final_training(
    tmp_path: Path, shard_root: Path, manifest: SplitManifest
) -> None:
    train = _train_module()
    manifest_path = write_split_manifest(shard_root, manifest)
    schedule = build_materialized_schedule(manifest, sequence_length=8, seed=1337, local_shuffle_buffer_sequences=8)
    schedule_path = write_schedule(tmp_path / "train.schedule.json", schedule)

    args = _schedule_namespace(
        shard_root=shard_root,
        train_manifest=manifest_path,
        train_schedule=schedule_path,
        validation_manifest=manifest_path,
        validation_schedule=schedule_path,
    )
    args.seed = 1337
    args.data_dir = tmp_path / "unused"
    train_data, validation_data, facts = train.open_batch_sources(args)
    try:
        assert isinstance(train_data, ScheduledTokenStream)
        assert facts["batch_source"] == "materialized index schedule"
        assert facts["train_schedule_content_hash"] == schedule.content_hash()
        assert facts["train_sequences_per_source"] == schedule.sequences_per_source
        # The recorded state is the schedule hash plus one integer, which is what a resume
        # needs to reproduce the consumed training input exactly.
        assert train_data.state_dict()[CURSOR_STATE_KEY] == 0
        assert validation_data.wrap is True
    finally:
        train_data.close()
        validation_data.close()
