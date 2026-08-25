from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        frequencies = torch.outer(positions, inv_freq)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        cos = self.cos[:, :, :seq_len].to(dtype=q.dtype)
        sin = self.sin[:, :, :seq_len].to(dtype=q.dtype)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        if self.n_heads != self.n_kv_heads:
            # PyTorch's native enable_gqa path can fall back to a very slow attention
            # kernel on Windows. Expanding K/V heads preserves GQA parameter savings
            # while allowing the optimized fused SDPA kernel to run.
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            enable_gqa=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class TinyBenchLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.layers:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"Sequence exceeds max_seq_len={self.config.max_seq_len}")
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.token_embedding.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = -float("inf")
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
