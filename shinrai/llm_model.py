"""GPT-style decoder-only transformer — built from scratch with PyTorch.

Architecture decisions (LLaMA-inspired):
  ✦ RMSNorm instead of LayerNorm (faster, no mean subtraction)
  ✦ Rotary Positional Embeddings (RoPE) — no learned position table
  ✦ SwiGLU feed-forward  FFN(x) = (SiLU(xW₁) ⊙ xW₂) W₃
  ✦ Causal (masked) multi-head self-attention
  ✦ Weight tying: token embedding == output projection
  ✦ Pre-norm (norm is applied *before* each sub-layer)
  ✦ Uses torch.nn.functional.scaled_dot_product_attention when available
    (PyTorch ≥ 2.0) for automatic Flash Attention on supported hardware.

Default ~45M-parameter config recommended for RTX 3050 (4 GB):
  vocab_size  : 16000
  d_model     : 512
  n_heads     : 8       head_dim = 64
  n_layers    : 6
  d_ff        : 1536    SwiGLU gate/up projections
  max_seq_len : 512
  dropout     : 0.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size: int = 16000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536          # width of SwiGLU gate/up projections
    max_seq_len: int = 512
    dropout: float = 0.1
    # automatically derived
    head_dim: int = field(init=False)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        self.head_dim = self.d_model // self.n_heads


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean centering)."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ── Rotary Positional Embeddings ─────────────────────────────────────────────

def _precompute_rope_freqs(head_dim: int, max_seq_len: int,
                            theta: float = 10_000.0,
                            device: torch.device = None) -> torch.Tensor:
    """Return (max_seq_len, head_dim/2, 2) complex-valued frequency tensor."""
    assert head_dim % 2 == 0
    # positions
    pos = torch.arange(max_seq_len, device=device).float()
    # frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # outer product → (max_seq_len, head_dim/2)
    angles = torch.outer(pos, freqs)
    # (max_seq_len, head_dim/2) complex
    return torch.polar(torch.ones_like(angles), angles)   # e^{i*angle}


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to query or key tensor.

    x     : (B, n_heads, T, head_dim)
    freqs : (T, head_dim/2)  complex
    """
    B, H, T, D = x.shape
    # view as complex → (B, H, T, D/2)
    x_c = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    # freqs: (T, D/2) → broadcast over batch and heads
    freqs_bc = freqs[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    rotated = torch.view_as_real(x_c * freqs_bc).flatten(-2)  # (B, H, T, D)
    return rotated.to(x.dtype)


# ── SwiGLU Feed-Forward ───────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: FFN(x) = (SiLU(xW₁) ⊙ xW₂) W₃"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)   # value
        self.w3 = nn.Linear(d_ff,    d_model, bias=False) # projection
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ── Multi-Head Causal Self-Attention ─────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Grouped (here: standard) multi-head causal self-attention with RoPE."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model

        # fused QKV projection
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj  = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # causal mask buffer — registered but not a parameter
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        )

        # track whether SDPA (Flash Attention) is available
        self._use_sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        past_kv: Optional[tuple] = None,
    ):
        B, T, C = x.shape

        # QKV
        qkv = self.qkv_proj(x)   # (B, T, 3C)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # reshape to (B, H, T, head_dim)
        def _split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = _split_heads(q), _split_heads(k), _split_heads(v)

        # apply RoPE
        q = _apply_rope(q, rope_freqs)
        k = _apply_rope(k, rope_freqs)

        # KV cache (used during inference)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)

        kv_len = k.shape[2]

        if self._use_sdpa:
            # PyTorch ≥ 2.0 — uses Flash Attention kernel when possible
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=(past_kv is None),   # mask not needed with cache
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, T, kv_len)
            # apply causal mask
            mask = self.causal_mask[:T, :kv_len]
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_w = F.softmax(scores, dim=-1)
            attn_w = self.attn_drop(attn_w)
            attn_out = torch.matmul(attn_w, v)   # (B, H, T, head_dim)

        # merge heads → (B, T, C)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(attn_out)), new_kv


# ── Transformer Block ─────────────────────────────────────────────────────────

class Block(nn.Module):
    """Single transformer decoder block (pre-norm)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn   = SwiGLUFFN(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        past_kv: Optional[tuple] = None,
    ):
        attn_out, new_kv = self.attn(self.norm1(x), rope_freqs, past_kv=past_kv)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


# ── GPT (full model) ─────────────────────────────────────────────────────────

class GPT(nn.Module):
    """Decoder-only GPT model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop  = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)

        # output projection — weight-tied with token_emb
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        # RoPE frequencies (not a parameter — precomputed and cached)
        self.register_buffer(
            "rope_freqs",
            _precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len),
            persistent=False,
        )

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,      # (B, T)
        targets: Optional[torch.Tensor] = None,   # (B, T) shifted labels
        past_kvs: Optional[list] = None,          # list of (k,v) per layer
    ):
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        x = self.emb_drop(self.token_emb(input_ids))   # (B, T, d_model)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv_i = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, self.rope_freqs, past_kv=past_kv_i)
            new_kvs.append(new_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_kvs

    # ── convenience ─────────────────────────────────────────────────────────

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def small(cls, vocab_size: int = 16000) -> "GPT":
        """~45M params — good default for 4 GB VRAM."""
        return cls(GPTConfig(
            vocab_size=vocab_size,
            d_model=512, n_heads=8, n_layers=6,
            d_ff=1536, max_seq_len=512, dropout=0.1,
        ))

    @classmethod
    def medium(cls, vocab_size: int = 16000) -> "GPT":
        """~125M params — for 8+ GB VRAM."""
        return cls(GPTConfig(
            vocab_size=vocab_size,
            d_model=768, n_heads=12, n_layers=12,
            d_ff=2048, max_seq_len=1024, dropout=0.1,
        ))
