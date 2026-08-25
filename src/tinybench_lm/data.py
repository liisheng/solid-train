from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch


class PackedTokenDataset:
    """Memory-mapped uint16 token stream with random contiguous sampling."""

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
