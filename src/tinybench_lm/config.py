from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16_384
    max_seq_len: int = 1_024
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 8
    d_ff: int = 1_536
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.vocab_size > 65_535:
            raise ValueError("The packed-data format uses uint16 token IDs")
        if not self.tie_embeddings:
            raise ValueError("Untied embeddings would exceed the intended parameter budget")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)

