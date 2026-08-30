"""TinyBench-LM: small, reproducible causal language-model training."""

from .config import FINAL_CONFIG_PATH, ModelConfig
from .model import LOSS_IGNORE_INDEX, TinyBenchLM

__all__ = ["FINAL_CONFIG_PATH", "LOSS_IGNORE_INDEX", "ModelConfig", "TinyBenchLM"]

