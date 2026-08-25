from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinybench_lm import ModelConfig, TinyBenchLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a TinyBench-LM checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = TinyBenchLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    encoded = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
