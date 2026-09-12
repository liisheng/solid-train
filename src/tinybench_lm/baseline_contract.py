"""Registered successor baseline contract identity; historical v1 stays valid."""

import hashlib
from pathlib import Path

import yaml

SELECTED_BASELINE_SHA256 = "dcd4d623a8f826f5e007ed33eea4bf2b653414c2088f4354a9b7e6e69a8a3f12"


def validate_selected_identity(identity: dict) -> None:
    """Reject selected-contract and policy drift before the trainer allocates a model."""
    root = Path(__file__).resolve().parents[2]
    content = (root / "configs/training/baseline_reduced_v2.yaml").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(content).hexdigest() != SELECTED_BASELINE_SHA256:
        raise ValueError("selected baseline contract digest mismatch")
    config = yaml.safe_load(content)
    selection = config["selection"]
    settings = (root / selection["path"]).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(settings).hexdigest() != selection["sha256_normalized_lf"]:
        raise ValueError("selected baseline settings digest mismatch")
    expected = {
        "scope": "baseline_reduced_v2",
        "baseline_contract_sha256_normalized_lf": SELECTED_BASELINE_SHA256,
        "selection": selection,
        "seed": config["seed"]["value"],
        "total_updates": config["horizon"]["total_updates"],
        "consumed_loss_tokens": config["horizon"]["consumed_loss_tokens"],
        "loss_tokens_per_update": config["horizon"]["loss_tokens_per_update"],
        "peak_lr": config["learning_rate"]["peak_lr"],
        "warmup_updates": config["learning_rate"]["warmup_updates"],
        "stable_updates": config["learning_rate"]["stable_updates"],
        "decay_updates": config["learning_rate"]["decay_updates"],
        "micro_batch_size": config["batch"]["micro_batch_size"],
        "gradient_accumulation": config["batch"]["gradient_accumulation"],
        "sequence_length": config["batch"]["sequence_length"],
        "weight_decay": config["optimizer"]["weight_decay"],
        "gradient_clip_global_norm": config["optimizer"]["gradient_clip_global_norm"],
        "precision": {"dtype": "bfloat16", "grad_scaler": False},
        "determinism_policy": config["precision"]["determinism_policy"],
        "validation_mode": "full-dev", "eval_interval": 100, "save_interval": 100,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f"selected baseline identity field {key} differs from trusted recipe")
