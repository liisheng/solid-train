from __future__ import annotations

import argparse
from pathlib import Path

from tinybench_lm import ModelConfig, TinyBenchLM
from tinybench_lm.config import (
    COMPETITION_PARAMETER_CAP,
    FINAL_CAP_HEADROOM,
    FINAL_CONFIG_PATH,
    FINAL_PARAMETER_COUNT,
)

# The enumeration itself lives in the library so this script, the step-zero provenance
# record, and the release verifier all count the same unique Parameter objects.
from tinybench_lm.parameters import count_unique_trainable_parameters, unique_trainable_parameters

__all__ = [
    "assert_matches_final_parameter_contract",
    "assert_within_competition_cap",
    "count_unique_trainable_parameters",
    "unique_trainable_parameters",
]


def assert_within_competition_cap(count: int) -> int:
    """Fail closed above the competition cap and return the remaining headroom."""
    if count > COMPETITION_PARAMETER_CAP:
        raise SystemExit("Model exceeds the competition parameter cap")
    return COMPETITION_PARAMETER_CAP - count


def assert_matches_final_parameter_contract(count: int) -> None:
    """Fail closed when the final architecture drifts from the frozen contract."""
    headroom = COMPETITION_PARAMETER_CAP - count
    if count != FINAL_PARAMETER_COUNT or headroom != FINAL_CAP_HEADROOM:
        raise SystemExit("Final architecture does not match the frozen parameter contract")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=FINAL_CONFIG_PATH)
    args = parser.parse_args()
    config = ModelConfig.from_json(args.config)
    model = TinyBenchLM(config)
    count = count_unique_trainable_parameters(model)
    print(f"Trainable parameters: {count:,}")
    print(f"Competition cap:      {COMPETITION_PARAMETER_CAP:,}")
    print(f"Safety margin:        {COMPETITION_PARAMETER_CAP - count:,}")
    headroom = assert_within_competition_cap(count)
    print(f"Verified headroom:    {headroom:,}")
    if args.config.resolve() == FINAL_CONFIG_PATH.resolve():
        assert_matches_final_parameter_contract(count)


if __name__ == "__main__":
    main()
