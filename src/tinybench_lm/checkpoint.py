from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .data import TrainingSource
from .model import TinyBenchLM


CHECKPOINT_FORMAT_VERSION = 2


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def save_checkpoint(
    path: Path,
    model: TinyBenchLM,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    args: argparse.Namespace,
    train_data: TrainingSource,
    validation_data: TrainingSource,
    step: int,
    best_validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": config.to_dict(),
            "training_args": vars(args),
            "step": step,
            "best_validation_loss": best_validation_loss,
            "rng_state": capture_rng_state(),
            # Data-source resume state. For the pilot sampler this is a bit-generator blob;
            # for a materialized schedule it is one integer schedule_cursor plus the schedule
            # content hash. The key name is part of the frozen format-v2 contract.
            "data_rng_state": {
                "train": train_data.state_dict(),
                "validation": validation_data.state_dict(),
            },
        },
        temporary_path,
    )
    temporary_path.replace(path)


def restore_checkpoint_state(
    checkpoint: dict[str, object],
    model: TinyBenchLM,
    optimizer: torch.optim.Optimizer,
    train_data: TrainingSource,
    validation_data: TrainingSource,
) -> tuple[int, float, bool]:
    """Restore training state and report whether the resume is reproducible."""
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    reproducible = "rng_state" in checkpoint and "data_rng_state" in checkpoint
    if reproducible:
        data_rng_state = checkpoint["data_rng_state"]
        train_data.load_state_dict(data_rng_state["train"])
        validation_data.load_state_dict(data_rng_state["validation"])
        restore_rng_state(checkpoint["rng_state"])
    return (
        int(checkpoint["step"]) + 1,
        float(checkpoint["best_validation_loss"]),
        reproducible,
    )
