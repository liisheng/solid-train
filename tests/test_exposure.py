import pytest
import numpy as np
import torch
import json

from tinybench_lm.exposure import CompositeCursor, CompositeExposure, CompositeTokenStream, load_exposure_plan, write_exposure_artifacts
from tinybench_lm.schedule import MaterializedSchedule, ScheduleContractError, ScheduleEntry, ScheduleResumeError
from tinybench_lm.shards import NamespaceManifest, ShardRecord, SplitManifest, STABLE_TRAIN


def _schedule(tag: str, count: int) -> MaterializedSchedule:
    entries = tuple(ScheduleEntry(f"{tag}-shard", i * 9, 9, "fineweb_edu", "stable/fineweb_edu") for i in range(count))
    return MaterializedSchedule(tag, STABLE_TRAIN, STABLE_TRAIN, 8, 1, 1337, 1, entries, "manifest", "protocol")


def _exposure() -> CompositeExposure:
    return CompositeExposure((_schedule("one", 3), _schedule("two", 2)), "contract")


def test_composite_identity_binds_order_and_component_metadata():
    exposure = _exposure()
    assert exposure.sequence_count == 5
    assert exposure.source_counts == {"fineweb_edu": 5}
    swapped = CompositeExposure((exposure.components[1], exposure.components[0]), "contract")
    assert swapped.content_hash != exposure.content_hash


def test_cursor_boundary_resume_and_exhaustion():
    exposure = _exposure()
    for position in (0, 2, 3, 4, 5):
        cursor = exposure.cursor(position)
        restored = exposure.cursor(0)
        restored.load_state_dict(cursor.state_dict())
        assert restored.position == position
    with pytest.raises(ScheduleResumeError):
        exposure.cursor().load_state_dict({"format_version": 1, "schedule_cursor": 3, "schedule_content_hash": "wrong"})
    with pytest.raises(ScheduleContractError):
        exposure.cursor(-1)
    with pytest.raises(ScheduleContractError):
        exposure.cursor(6)


def test_duplicate_reference_inside_component_fails_closed():
    first = _schedule("one", 2)
    duplicate = MaterializedSchedule(first.schedule_id, first.split_id, first.boundary, first.sequence_length,
        first.label_shift, first.seed, first.local_shuffle_buffer_sequences,
        (first.entries[0], first.entries[0]), first.manifest_content_hash, first.protocol_digest)
    with pytest.raises(ScheduleContractError, match="duplicate"):
        CompositeExposure((duplicate, _schedule("two", 1)))


def test_stream_crosses_boundary_and_resumes(tmp_path):
    tokens = np.arange(45, dtype=np.uint16)
    (tmp_path / "stable" / "fineweb_edu").mkdir(parents=True)
    tokens.tofile(tmp_path / "stable" / "fineweb_edu" / "shard_00000.bin")
    record = ShardRecord("stable/fineweb_edu/shard_00000", "stable/fineweb_edu", "fineweb_edu", STABLE_TRAIN,
        "stable/fineweb_edu/shard_00000.bin", "uint16", 45, 1, ("doc",), (0,), (45,), 1, "", "final_token_counter_v1")
    manifest = SplitManifest(STABLE_TRAIN, STABLE_TRAIN,
        (NamespaceManifest("stable/fineweb_edu", "fineweb_edu", STABLE_TRAIN, (record,)),), "final_token_counter_v1")
    def schedule(tag, start):
        entries = tuple(ScheduleEntry(record.shard_id, (start + i) * 9, 9, "fineweb_edu", "stable/fineweb_edu") for i in range(2))
        return MaterializedSchedule(tag, STABLE_TRAIN, STABLE_TRAIN, 8, 1, 1337, 1, entries, manifest.content_hash(), "protocol")
    exposure = CompositeExposure((schedule("one", 0), schedule("two", 2)), "contract")
    stream = CompositeTokenStream(tmp_path, manifest, exposure, validate_components=False)
    inputs, targets = stream.get_batch(3, 8, torch.device("cpu"))
    assert inputs.shape == (3, 8) and targets.shape == (3, 8)
    assert stream.position == 3
    state = stream.state_dict()
    resumed = CompositeTokenStream(tmp_path, manifest, exposure, validate_components=False)
    resumed.load_state_dict(state)
    next_inputs, _ = resumed.get_batch(1, 8, torch.device("cpu"))
    assert next_inputs[0, 0].item() == inputs[-1, 0].item() + 9
    # Recovery immediately before, at, and after the component boundary preserves the
    # exact next reference selected by an uninterrupted reader.
    for position in (1, 2, 3):
        checkpoint = exposure.cursor(position).state_dict()
        probe = CompositeTokenStream(tmp_path, manifest, exposure, validate_components=False)
        probe.load_state_dict(checkpoint)
        expected = exposure.components[0].entries[position:position + 1] if position < 2 else exposure.components[1].entries[position - 2:position - 1]
        assert probe._take_entries(1) == expected
        probe.close()
    final = CompositeTokenStream(tmp_path, manifest, exposure, validate_components=False)
    final.load_state_dict(exposure.cursor(3).state_dict())
    final.get_batch(1, 8, torch.device("cpu"))
    assert final.position == exposure.sequence_count
    with pytest.raises(ScheduleContractError):
        final.get_batch(1, 8, torch.device("cpu"))
    final.close()
    with pytest.raises(ScheduleContractError):
        resumed.get_batch(2, 8, torch.device("cpu"))
    stream.close(); resumed.close()


def test_plan_tampering_fails_closed(tmp_path):
    exposure = _exposure()
    paths = write_exposure_artifacts(tmp_path, exposure)
    payload = json.loads(paths["plan"].read_text())
    for field, value in (("sequence_count", 99), ("content_hash", "wrong"), ("requested_source_quotas", {"wrong": 1})):
        altered = dict(payload)
        altered["components"] = [dict(item) for item in payload["components"]]
        altered["components"][1][field] = value
        paths["plan"].write_text(json.dumps(altered))
        with pytest.raises(ScheduleContractError):
            load_exposure_plan(paths["plan"], (paths["component_1"], paths["component_2"]))
    altered = dict(payload)
    altered["components"] = [dict(item) for item in reversed(payload["components"])]
    paths["plan"].write_text(json.dumps(altered))
    with pytest.raises(ScheduleContractError):
        load_exposure_plan(paths["plan"], (paths["component_1"], paths["component_2"]))
    with pytest.raises(ScheduleContractError):
        load_exposure_plan(tmp_path / "exposure_plan.json", (paths["component_2"], paths["component_1"]))
