from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tinybench_lm import ModelConfig, TinyBenchLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure training throughput without downloading data")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_49m.json"))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    config = ModelConfig.from_json(args.config)
    if args.sequence_length > config.max_seq_len:
        raise ValueError("sequence length exceeds model max_seq_len")
    model = TinyBenchLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokens_per_step = args.micro_batch_size * args.sequence_length * args.gradient_accumulation
    durations: list[float] = []
    torch.cuda.reset_peak_memory_stats()

    for step in range(args.warmup_steps + args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation):
            inputs = torch.randint(
                0,
                config.vocab_size,
                (args.micro_batch_size, args.sequence_length),
                device=device,
            )
            targets = torch.randint(
                0,
                config.vocab_size,
                (args.micro_batch_size, args.sequence_length),
                device=device,
            )
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model(inputs, targets)
                assert loss is not None
                loss = loss / args.gradient_accumulation
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        if step >= args.warmup_steps:
            durations.append(duration)
        print(
            f"step={step + 1}/{args.warmup_steps + args.steps} "
            f"seconds={duration:.3f} tok/s={tokens_per_step / duration:,.0f}"
        )

    median_duration = statistics.median(durations)
    throughput = tokens_per_step / median_duration
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "precision": str(amp_dtype),
        "parameter_count": model.count_parameters(),
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "tokens_per_step": tokens_per_step,
        "median_step_seconds": median_duration,
        "median_tokens_per_second": throughput,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "estimated_hours_per_1b_tokens": 1_000_000_000 / throughput / 3600,
        "estimated_hours_per_3b_tokens": 3_000_000_000 / throughput / 3600,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output:
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")


if __name__ == "__main__":
    main()

