"""Finite two-component exposure for the reduced baseline.

The components remain independent materialized schedules.  ``CompositeTokenStream``
adds only an absolute cursor and switches components at the boundary, preserving the
existing schedule reader's lazy mmap behaviour and hot path.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .schedule import (
    CURSOR_STATE_KEY,
    MaterializedSchedule,
    ScheduleContractError,
    ScheduleEntry,
    ScheduleResumeError,
    ScheduledTokenStream,
    assert_schedule_valid,
    canonical_payload_bytes,
    load_schedule,
    training_order_hash,
    write_schedule,
)
from .shards import STABLE_TRAIN, SplitManifest

EXPOSURE_SCHEMA = "baseline_exposure_v1"
DEFAULT_QUOTAS = {"fineweb_edu": 307741, "dclm": 87889, "openwebmath": 30786, "narrative": 13112}
BASE_COMPONENT_COUNTS = {"fineweb_edu": 375907, "dclm": 107439, "openwebmath": 37579, "narrative": 16187}
BASE_COMPONENT_CONTENT_HASH = "820bd7f695643b2df82c5ea8c9816dcd763b375382a9a28f94656ad43529b828"
BASE_COMPONENT_FILE_SHA256 = "ee174527f0a00a23f6d13d88e723dea1988f7df44fd4bb6c3aaf5f1300f4052e"
BASELINE_RECIPE_SHA256 = "1d4901c4c8d83c4c4ffd17e953de2b23f5a762f2af8eb8bb57fbc6fd8e26c541"


def _refs_hash(entries: tuple[ScheduleEntry, ...]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class CompositeExposure:
    """Two ordered schedules with one absolute, hash-bound cursor namespace."""

    components: tuple[MaterializedSchedule, MaterializedSchedule]
    contract_hash: str = ""
    schema_version: str = EXPOSURE_SCHEMA
    recipe_sha256_normalized_lf: str = BASELINE_RECIPE_SHA256
    _component_hashes: tuple[str, str] = ()
    _content_hash: str = ""

    def __post_init__(self) -> None:
        if len(self.components) != 2:
            raise ScheduleContractError("baseline exposure requires exactly two components")
        first, second = self.components
        if first.sequence_count < 1 or second.sequence_count < 1:
            raise ScheduleContractError("baseline exposure components must be nonempty")
        if first.split_id != STABLE_TRAIN or second.split_id != STABLE_TRAIN:
            raise ScheduleContractError("both exposure components must use stable_train")
        for component in self.components:
            seen: set[tuple[str, int, int]] = set()
            for entry in component.entries:
                if entry.reference in seen:
                    raise ScheduleContractError("duplicate reference within exposure component")
                seen.add(entry.reference)
        if not self.contract_hash:
            raise ScheduleContractError("exposure requires a contract identity")
        if self.recipe_sha256_normalized_lf != BASELINE_RECIPE_SHA256:
            raise ScheduleContractError("exposure recipe identity does not match the frozen baseline recipe")
        first, second = self.components
        if (first.manifest_content_hash, first.protocol_digest, first.boundary) != (
            second.manifest_content_hash, second.protocol_digest, second.boundary
        ):
            raise ScheduleContractError("exposure components have incompatible manifest/protocol identity")
        if first.sequence_length != second.sequence_length or first.label_shift != second.label_shift:
            raise ScheduleContractError("exposure components must have identical sequence geometry")
        component_hashes = tuple(component.content_hash() for component in self.components)
        payload = {"schema_version": self.schema_version, "contract_hash": self.contract_hash,
                   "recipe_sha256_normalized_lf": self.recipe_sha256_normalized_lf,
                   "components": [{"content_hash": component_hashes[index],
                                   "sequence_count": component.sequence_count,
                                   "requested_source_quotas": dict(sorted(component.requested_source_quotas.items()))}
                                  for index, component in enumerate(self.components)]}
        object.__setattr__(self, "_component_hashes", component_hashes)
        object.__setattr__(self, "_content_hash", hashlib.sha256(canonical_payload_bytes(payload)).hexdigest())

    @property
    def sequence_count(self) -> int:
        return sum(component.sequence_count for component in self.components)

    @property
    def loss_tokens(self) -> int:
        return sum(component.loss_tokens for component in self.components)

    @property
    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for component in self.components:
            for source, count in component.sequences_per_source.items():
                counts[source] = counts.get(source, 0) + count
        return dict(sorted(counts.items()))

    @property
    def content_hash(self) -> str:
        return self._content_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_hash": self.contract_hash,
            "recipe_sha256_normalized_lf": self.recipe_sha256_normalized_lf,
            "content_hash": self.content_hash,
            "sequence_count": self.sequence_count,
            "loss_tokens": self.loss_tokens,
            "source_counts": self.source_counts,
            "manifest_content_hash": self.components[0].manifest_content_hash,
            "protocol_digest": self.components[0].protocol_digest,
            "sequence_length": self.components[0].sequence_length,
            "components": [
                {"index": index + 1, "schedule_id": component.schedule_id,
                 "content_hash": self._component_hashes[index], "sequence_count": component.sequence_count,
                 "requested_source_quotas": dict(sorted(component.requested_source_quotas.items()))}
                for index, component in enumerate(self.components)
            ],
        }

    def cursor(self, position: int = 0) -> "CompositeCursor":
        return CompositeCursor(self.content_hash, position, self.sequence_count)


@dataclass
class CompositeCursor:
    schedule_content_hash: str
    position: int = 0
    sequence_count: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.position, bool) or int(self.position) != self.position or self.position < 0:
            raise ScheduleContractError("composite cursor must be a nonnegative integer")
        if self.sequence_count is not None and self.position > self.sequence_count:
            raise ScheduleContractError("composite cursor exceeds finite exposure")

    def state_dict(self) -> dict[str, Any]:
        return {"format_version": 1, CURSOR_STATE_KEY: self.position,
                "schedule_content_hash": self.schedule_content_hash}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        position = state.get(CURSOR_STATE_KEY)
        if (state.get("format_version") != 1 or isinstance(position, bool) or not isinstance(position, int)
                or state.get("schedule_content_hash") != self.schedule_content_hash
                or position < 0 or (self.sequence_count is not None and position > self.sequence_count)):
            raise ScheduleResumeError("composite exposure identity or absolute cursor mismatch")
        self.position = position


class CompositeTokenStream:
    """Reader with one lazy mmap cache and O(1) component-boundary dispatch."""

    def __init__(self, root: str | Path, manifest: SplitManifest, exposure: CompositeExposure, *, validate_components: bool = True):
        self.root, self.manifest, self.exposure = Path(root), manifest, exposure
        if any(component.manifest_content_hash != manifest.content_hash() for component in exposure.components):
            raise ScheduleContractError("exposure component is bound to a different manifest")
        if validate_components:
            for component in exposure.components:
                assert_schedule_valid(manifest, component)
        self.content_hash = exposure.content_hash
        self.cursor = exposure.cursor(0)
        self._streams = tuple(ScheduledTokenStream(root, manifest, component) for component in exposure.components)
        # Both components address the same stable shard set. Sharing the mapping cache
        # avoids reopening the same shard when the absolute cursor crosses the boundary.
        self._streams[1]._memmaps = self._streams[0]._memmaps
        self.last_batch_entries: tuple[ScheduleEntry, ...] = ()
        self.last_batch_reference_hash: str | None = None

    @property
    def schedule(self) -> CompositeExposure:
        return self.exposure

    @property
    def position(self) -> int:
        return self.cursor.position

    def state_dict(self) -> dict[str, Any]:
        return self.cursor.state_dict()

    def rewind(self) -> None:
        """Restart a validation replay at the first entry."""
        self.load_state_dict(self.exposure.cursor(0).state_dict())

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.cursor.load_state_dict(state)
        remaining = self.cursor.position
        for stream in self._streams:
            local = min(remaining, stream.schedule.sequence_count)
            stream.cursor.position = local
            remaining -= local

    def _take_entries(self, count: int) -> tuple[ScheduleEntry, ...]:
        if count < 1 or self.position + count > self.exposure.sequence_count:
            raise ScheduleContractError("composite exposure is exhausted")
        position, remaining, entries = self.position, count, []
        for index, stream in enumerate(self._streams):
            start = sum(item.sequence_count for item in self.exposure.components[:index])
            if position >= start + stream.schedule.sequence_count:
                continue
            local = position - start
            take = min(remaining, stream.schedule.sequence_count - local)
            entries.extend(stream.schedule.entries[local : local + take])
            remaining -= take
            position += take
            if not remaining:
                break
        return tuple(entries)

    def get_batch(self, batch_size: int, seq_len: int, device: torch.device):
        entries = self._take_entries(int(batch_size))
        if seq_len != self.exposure.components[0].sequence_length:
            raise ScheduleContractError("sequence length does not match exposure")
        # At most two reads, only when a microbatch straddles the component boundary.
        batches = []
        offset = 0
        while offset < len(entries):
            component_index = 0 if self.position + offset < self.exposure.components[0].sequence_count else 1
            stream = self._streams[component_index]
            start = self.position + offset - (0 if component_index == 0 else self.exposure.components[0].sequence_count)
            take = min(len(entries) - offset, stream.schedule.sequence_count - start)
            stream.cursor.position = start
            batches.append(stream.get_batch(take, seq_len, device))
            offset += take
        inputs = torch.cat([batch[0] for batch in batches], dim=0) if len(batches) > 1 else batches[0][0]
        target = torch.cat([batch[1] for batch in batches], dim=0) if len(batches) > 1 else batches[0][1]
        self.last_batch_entries = entries
        self.last_batch_reference_hash = (
            self._streams[0 if self.position < self.exposure.components[0].sequence_count else 1].last_batch_reference_hash
            if len(batches) == 1 else training_order_hash(entries)
        )
        self.cursor.position += len(entries)
        return inputs, target

    def close(self) -> None:
        for stream in self._streams:
            stream.close()


def verify_exposure(exposure: CompositeExposure, manifest: SplitManifest, *, quotas: Mapping[str, int] = DEFAULT_QUOTAS) -> dict[str, Any]:
    for component in exposure.components:
        assert_schedule_valid(manifest, component)
    if exposure.sequence_count != 976640:
        raise ScheduleContractError(f"expected 976640 sequences, got {exposure.sequence_count}")
    if exposure.components[0].sequence_count != 537112 or exposure.components[1].sequence_count != 439528:
        raise ScheduleContractError("component sequence counts do not match the baseline contract")
    expected_component_1 = BASE_COMPONENT_COUNTS
    if exposure.components[0].content_hash() != BASE_COMPONENT_CONTENT_HASH:
        raise ScheduleContractError("component 1 content hash does not match the frozen G2 base schedule")
    if exposure.components[0].sequences_per_source != expected_component_1:
        raise ScheduleContractError("component 1 source counts do not match the frozen base schedule")
    if exposure.components[1].sequences_per_source != dict(sorted(quotas.items())):
        raise ScheduleContractError("component 2 source quotas do not match the baseline contract")
    if dict(exposure.components[1].requested_source_quotas) != dict(sorted(quotas.items())):
        raise ScheduleContractError("component 2 requested source quotas are not bound to the baseline contract")
    expected_total = {source: expected_component_1[source] + int(quotas[source]) for source in expected_component_1}
    if exposure.source_counts != dict(sorted(expected_total.items())):
        raise ScheduleContractError("aggregate source counts do not match the baseline contract")
    if any(entry.namespace.split("/")[0] != "stable" for component in exposure.components for entry in component.entries):
        raise ScheduleContractError("exposure contains a non-stable namespace")
    if any(component.manifest_content_hash != manifest.content_hash() for component in exposure.components):
        raise ScheduleContractError("exposure manifest identity mismatch")
    return {"sequence_count": exposure.sequence_count, "final_absolute_cursor": exposure.sequence_count,
            "consumed_loss_tokens": exposure.loss_tokens, "loss_tokens": exposure.loss_tokens,
            "source_counts": exposure.source_counts, "content_hash": exposure.content_hash,
            "component_hashes": [component.content_hash() for component in exposure.components],
            "component_reference_hashes": [_refs_hash(component.entries) for component in exposure.components],
            "within_component_duplicates": [False, False]}


def write_exposure_artifacts(root: Path, exposure: CompositeExposure) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {"component_1": root / "component_1.json", "component_2": root / "component_2.json"}
    for path, component in zip(paths.values(), exposure.components):
        write_schedule(path, component)
    plan = root / "exposure_plan.json"
    plan_payload = exposure.to_dict()
    plan_payload["component_file_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
    }
    plan.write_bytes(canonical_payload_bytes(plan_payload) + b"\n")
    return {**paths, "plan": plan}


def load_exposure_plan(path: Path, component_paths: tuple[Path, Path]) -> CompositeExposure:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    exposure = CompositeExposure((load_schedule(component_paths[0]), load_schedule(component_paths[1])), str(payload.get("contract_hash", "")), recipe_sha256_normalized_lf=str(payload.get("recipe_sha256_normalized_lf", "")))
    if exposure.components[0].sequence_count == 537112:
        actual_base_sha = hashlib.sha256(component_paths[0].read_bytes()).hexdigest()
        if actual_base_sha != BASE_COMPONENT_FILE_SHA256:
            raise ScheduleContractError("component 1 file identity does not match the frozen base schedule")
    expected = exposure.to_dict()
    expected["component_file_sha256"] = {
        f"component_{index}": hashlib.sha256(path.read_bytes()).hexdigest()
        for index, path in enumerate(component_paths, 1)
    }
    if payload != expected:
        raise ScheduleContractError("exposure plan content hash mismatch")
    return exposure


__all__ = ["CompositeExposure", "CompositeCursor", "CompositeTokenStream", "verify_exposure", "write_exposure_artifacts", "load_exposure_plan", "DEFAULT_QUOTAS"]
