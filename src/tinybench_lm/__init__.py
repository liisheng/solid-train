"""TinyBench-LM: small, reproducible causal language-model training."""

from .config import ModelConfig
from .model import TinyBenchLM

__all__ = ["ModelConfig", "TinyBenchLM"]

