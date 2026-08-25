from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from tinybench_lm import ModelConfig, TinyBenchLM
from tinybench_lm.checkpoint import restore_checkpoint_state, save_checkpoint
from tinybench_lm.data import PackedTokenDataset, load_data_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyBench-LM from random initialization")
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_12m.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/pilot"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=6e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def learning_rate_for_step(args: argparse.Namespace, step: int) -> float:
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return args.minimum_learning_rate + cosine * (args.learning_rate - args.minimum_learning_rate)


@torch.no_grad()
def evaluate(
    model: TinyBenchLM,
    dataset: PackedTokenDataset,
    args: argparse.Namespace,
    device: torch.device,
    autocast_context,
) -> float:
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        inputs, targets = dataset.get_batch(args.micro_batch_size, args.sequence_length, device)
        with autocast_context():
            _, loss = model(inputs, targets)
        assert loss is not None
        losses.append(loss.detach().float())
    model.train()
    return torch.stack(losses).mean().item()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training configuration expects an NVIDIA GPU")
    bf16_supported = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_supported else torch.float16

    config = ModelConfig.from_json(args.config)
    if args.sequence_length > config.max_seq_len:
        raise ValueError("sequence-length exceeds the model configuration")
    metadata = load_data_metadata(args.data_dir)
    if int(metadata["actual_vocab_size"]) > config.vocab_size:
        raise ValueError("Tokenizer vocabulary is larger than the model vocabulary")

    model = TinyBenchLM(config).to(device)
    parameter_count = model.count_parameters()
    if parameter_count > 50_000_000:
        raise RuntimeError(f"Parameter cap exceeded: {parameter_count:,}")
    raw_model = model
    if args.compile:
        model = torch.compile(model)

    train_data = PackedTokenDataset(args.data_dir / "train.bin", args.seed)
    validation_data = PackedTokenDataset(args.data_dir / "validation.bin", args.seed + 1)
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=True,
    )
    first_step = 0
    best_validation_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        first_step, best_validation_loss, reproducible_resume = restore_checkpoint_state(
            checkpoint,
            raw_model,
            optimizer,
            train_data,
            validation_data,
        )
        if not reproducible_resume:
            print(
                "WARNING: legacy checkpoint has no RNG/sampler state; "
                "resume is functional but not exactly reproducible"
            )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "run_config.json").open("w", encoding="utf-8") as output:
        json.dump(
            {
                "model_config": config.to_dict(),
                "training_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "parameter_count": parameter_count,
                "device": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "amp_dtype": str(amp_dtype),
                "data_metadata": metadata,
            },
            output,
            indent=2,
            sort_keys=True,
        )
        output.write("\n")

    log_path = args.run_dir / "metrics.jsonl"
    tokens_per_step = args.micro_batch_size * args.sequence_length * args.gradient_accumulation
    autocast_context = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Precision: {amp_dtype}")
    print(f"Tokens/optimizer step: {tokens_per_step:,}")
    torch.cuda.reset_peak_memory_stats()
    model.train()

    for step in range(first_step, args.steps):
        started = time.perf_counter()
        lr = learning_rate_for_step(args, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation):
            inputs, targets = train_data.get_batch(args.micro_batch_size, args.sequence_length, device)
            with autocast_context():
                _, loss = model(inputs, targets)
                assert loss is not None
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            accumulated_loss += loss.detach().float().item() / args.gradient_accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        record = {
            "step": step,
            "train_loss": accumulated_loss,
            "learning_rate": lr,
            "grad_norm": float(grad_norm),
            "tokens": (step + 1) * tokens_per_step,
            "tokens_per_second": tokens_per_step / elapsed,
            "step_seconds": elapsed,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }

        should_eval = step == first_step or (step + 1) % args.eval_interval == 0 or step + 1 == args.steps
        if should_eval:
            validation_loss = evaluate(model, validation_data, args, device, autocast_context)
            record["validation_loss"] = validation_loss
            record["validation_perplexity"] = math.exp(min(20.0, validation_loss))
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(
                    args.run_dir / "best.pt",
                    raw_model,
                    optimizer,
                    config,
                    args,
                    train_data,
                    validation_data,
                    step,
                    best_validation_loss,
                )
        with log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        if step == first_step or (step + 1) % args.log_interval == 0 or should_eval:
            suffix = f" val={record['validation_loss']:.4f}" if "validation_loss" in record else ""
            print(
                f"step={step + 1}/{args.steps} loss={accumulated_loss:.4f}{suffix} "
                f"tok/s={record['tokens_per_second']:,.0f} vram={record['peak_vram_gib']:.2f}GiB"
            )
        if (step + 1) % args.save_interval == 0 or step + 1 == args.steps:
            save_checkpoint(
                args.run_dir / "latest.pt",
                raw_model,
                optimizer,
                config,
                args,
                train_data,
                validation_data,
                step,
                best_validation_loss,
            )


if __name__ == "__main__":
    main()
