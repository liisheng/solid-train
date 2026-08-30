from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from lm_eval.api.instance import Instance

from tinybench_lm import ModelConfig, TinyBenchLM
from tinybench_lm.checkpoint import restore_checkpoint_state, save_checkpoint
from tinybench_lm.config import FINAL_CONFIG_PATH
from tinybench_lm.data import PackedTokenDataset
from tinybench_lm.lm_eval_adapter import TinyBenchHarnessLM


def state_digest(model: TinyBenchLM) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_breakdown(model: TinyBenchLM) -> dict[str, int]:
    groups: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        if name == "token_embedding.weight":
            group = "tied_embedding_and_output_head"
        elif name == "final_norm.weight":
            group = "final_norm"
        elif name.endswith("norm.weight"):
            group = "block_norms"
        elif ".attn." in name:
            group = "attention"
        elif ".ffn." in name:
            group = "feed_forward"
        else:
            group = "other"
        groups[group] += parameter.numel()
    return dict(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded Phase 1 full-model smoke checks")
    parser.add_argument("--config", type=Path, default=FINAL_CONFIG_PATH)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/fineweb_edu_pilot")
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/processed/fineweb_edu_pilot/tokenizer.json"),
    )
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The Phase 1 full-model smoke check requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    config = ModelConfig.from_json(args.config)
    model = TinyBenchLM(config).to(device)
    breakdown = parameter_breakdown(model)
    parameter_count = sum(breakdown.values())
    if parameter_count != sum(p.numel() for p in model.parameters() if p.requires_grad):
        raise RuntimeError("Independent parameter breakdown disagrees with model enumeration")
    if parameter_count > 50_000_000:
        raise RuntimeError("Competition parameter cap exceeded")

    train_data = PackedTokenDataset(args.data_dir / "train.bin", seed=args.seed)
    validation_data = PackedTokenDataset(args.data_dir / "validation.bin", seed=args.seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    inputs, targets = train_data.get_batch(batch_size=1, seq_len=16, device=device)
    initial_digest = state_digest(model)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(inputs, targets)
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("Forward pass did not produce a finite loss")
    loss.backward()
    trainable_tensors = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradient_tensors = sum(parameter.grad is not None for parameter in trainable_tensors)
    if gradient_tensors != len(trainable_tensors):
        raise RuntimeError("At least one trainable tensor did not receive a gradient")
    optimizer.step()
    post_step_digest = state_digest(model)
    if post_step_digest == initial_digest:
        raise RuntimeError("Optimizer step did not change the model state")

    with tempfile.TemporaryDirectory(prefix="phase1-smoke-", dir=args.data_dir.parent) as temp_dir:
        checkpoint_path = Path(temp_dir) / "smoke.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config,
            args,
            train_data,
            validation_data,
            step=0,
            best_validation_loss=float(loss.detach()),
        )
        checkpoint_size = checkpoint_path.stat().st_size
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        loaded_model = TinyBenchLM(config).to(device)
        loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=1e-4, fused=True)
        loaded_train = PackedTokenDataset(args.data_dir / "train.bin", seed=0)
        loaded_validation = PackedTokenDataset(args.data_dir / "validation.bin", seed=0)
        first_step, best_loss, reproducible_resume = restore_checkpoint_state(
            checkpoint,
            loaded_model,
            loaded_optimizer,
            loaded_train,
            loaded_validation,
        )
        roundtrip_digest = state_digest(loaded_model)
        if roundtrip_digest != post_step_digest:
            raise RuntimeError("Checkpoint model state failed an exact round trip")

        harness_model = TinyBenchHarnessLM(
            checkpoint_path=checkpoint_path,
            tokenizer_path=args.tokenizer,
            batch_size=2,
            device="cuda",
        )
        harness_request = Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("The sky is", " blue"),
            idx=0,
        )
        harness_result = harness_model.loglikelihood([harness_request], disable_tqdm=True)
        if len(harness_result) != 1 or not np.isfinite(harness_result[0][0]):
            raise RuntimeError("lm-evaluation-harness adapter smoke score is invalid")
        rolling_request = Instance(
            request_type="loglikelihood_rolling",
            doc={},
            arguments=("The sky is blue.",),
            idx=0,
        )
        rolling_result = harness_model.loglikelihood_rolling(
            [rolling_request], disable_tqdm=True
        )
        if len(rolling_result) != 1 or not np.isfinite(rolling_result[0]):
            raise RuntimeError("Harness rolling log-likelihood smoke score is invalid")

        loaded_model.eval()
        generated = loaded_model.generate(inputs[:, :4], max_new_tokens=4, temperature=1.0, top_k=8)
        if generated.shape != (1, 8):
            raise RuntimeError("Generation smoke check returned the wrong shape")

    result = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "parameter_count": parameter_count,
        "parameter_breakdown": breakdown,
        "trainable_tensor_count": len(trainable_tensors),
        "gradient_tensor_count": gradient_tensors,
        "forward_logits_shape": list(logits.shape),
        "forward_loss": float(loss.detach()),
        "optimizer_changed_state": post_step_digest != initial_digest,
        "checkpoint_bytes": checkpoint_size,
        "checkpoint_exact_model_roundtrip": roundtrip_digest == post_step_digest,
        "checkpoint_first_resumed_step": first_step,
        "checkpoint_best_loss": best_loss,
        "reproducible_resume_state_present": reproducible_resume,
        "generation_shape": list(generated.shape),
        "generated_token_ids": generated[0].tolist(),
        "harness_loglikelihood": harness_result[0][0],
        "harness_greedy": harness_result[0][1],
        "harness_rolling_loglikelihood": rolling_result[0],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
