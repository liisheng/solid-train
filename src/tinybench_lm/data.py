"""Pilot-only flat-stream token sampling.

**Scope label: PILOT_ONLY.** :class:`PackedTokenDataset` draws uniform random offsets from
one monolithic ``uint16`` token file. That is enough for bounded smoke tests, and its
format-v2 RNG/sampler resume is exact and already evidenced, but it cannot reproduce a
source mixture or an exposure order: the flat file has no source identity, and the consumed
mixture is recoverable only through a bit-generator blob.

Final training therefore uses the materialized index schedule in
:mod:`tinybench_lm.schedule` (Plan Section 5.4), whose resume state is one integer
``schedule_cursor`` bound to a schedule content hash. This module is retained, not
superseded in place, so the existing pilot smoke tests and resume evidence stay valid. The
frozen ``pilot_sampler`` block of ``configs/data/schedule_v1.yaml`` records the same label.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

#: This module's scope. Final-campaign training must not use the random flat-stream sampler.
SAMPLER_SCOPE = "PILOT_ONLY"


@runtime_checkable
class TrainingSource(Protocol):
    """What the training loop and checkpoint plumbing need from a batch source.

    Both the pilot random sampler and the schedule-cursor reader in
    :mod:`tinybench_lm.schedule` satisfy this, so checkpointing does not need to learn a
    second protocol to support a reproducible mixture.
    """

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, object]) -> None: ...

    def get_batch(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class PackedTokenDataset:
    """Memory-mapped uint16 token stream with random contiguous sampling.

    PILOT ONLY. Superseded for final training by
    :class:`tinybench_lm.schedule.ScheduledTokenStream`, which reads source-tagged shards
    through deterministic ``(shard_id, token_offset, length)`` references.
    """

    #: Mirrors :data:`SAMPLER_SCOPE` so a caller holding an instance can see the label.
    scope = SAMPLER_SCOPE

    def __init__(self, path: str | Path, seed: int) -> None:
        self.path = Path(path)
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.tokens)

    def state_dict(self) -> dict[str, object]:
        """Return the sampler state needed for an exact training resume."""
        return {"bit_generator_state": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a sampler state produced by :meth:`state_dict`."""
        self.rng.bit_generator.state = copy.deepcopy(state["bit_generator_state"])

    def get_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.tokens) <= seq_len + 1:
            raise ValueError(f"{self.path} has too few tokens for seq_len={seq_len}")
        starts = self.rng.integers(0, len(self.tokens) - seq_len - 1, size=batch_size)
        rows = np.stack([self.tokens[start : start + seq_len + 1] for start in starts]).astype(
            np.int64, copy=False
        )
        batch = torch.from_numpy(rows)
        if device.type == "cuda":
            batch = batch.pin_memory().to(device, non_blocking=True)
        else:
            batch = batch.to(device)
        return batch[:, :-1], batch[:, 1:]


def load_data_metadata(data_dir: str | Path) -> dict[str, object]:
    path = Path(data_dir) / "metadata.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
