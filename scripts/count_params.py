from __future__ import annotations

import argparse

from tinybench_lm import ModelConfig, TinyBenchLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_49m.json")
    args = parser.parse_args()
    config = ModelConfig.from_json(args.config)
    model = TinyBenchLM(config)
    count = model.count_parameters()
    print(f"Trainable parameters: {count:,}")
    print(f"Competition cap:      {50_000_000:,}")
    print(f"Safety margin:        {50_000_000 - count:,}")
    if count > 50_000_000:
        raise SystemExit("Model exceeds the competition parameter cap")


if __name__ == "__main__":
    main()
