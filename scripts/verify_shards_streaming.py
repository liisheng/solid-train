"""CLI entry point for the bounded-memory G1-05 shard aggregate verifier."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tinybench_lm.streaming_verify import _cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(_cli())

