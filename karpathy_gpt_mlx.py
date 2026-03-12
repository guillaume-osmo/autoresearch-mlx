#!/usr/bin/env python3
"""
Karpathy "Let's build GPT: from scratch, in code, spelled out" — MLX port.
Based on: https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero (007_GPT/_src/gpt.py)
Tiny Shakespeare, character-level GPT.

Options:
  --modern       RoPE + QK-norm + ReLU² (modded-nanogpt). See https://github.com/KellerJordan/modded-nanogpt
  --mode         base | score | m_score
                   base: standard transformer (n_layer blocks).
                   score: 1 block applied recurrently for `steps` steps: x = x + ode_dt*(block(x)-x).
                   m_score: M blocks, each applied for `steps` steps (total M*steps).

Usage (from mlx-graphs repo root):
  python scripts/karpathy_gpt_mlx.py
  python scripts/karpathy_gpt_mlx.py --modern --max_iters 3000 --block_size 128
  python scripts/karpathy_gpt_mlx.py --mode score --steps 6 --ode_dt 0.5
  python scripts/karpathy_gpt_mlx.py --mode m_score --m 2 --steps 3 --ode_dt 0.5
  python scripts/karpathy_gpt_mlx.py --optimizer muon --mode m_score --m 2 --steps 3 --ode_dt 0.5
  python scripts/karpathy_gpt_mlx.py --optimizer muon_v2 --muon_polar polar_express --muon_normuon
  python scripts/karpathy_gpt_mlx.py --modern --bigram
  python scripts/karpathy_gpt_mlx.py --train_fraction 0.1  # small-data regime: 10% train / 90% val

Optimizers: --optimizer adam | adamw | muon | muon_v2
  adam:    no weight decay (recommended for SCORE/m_score).
  adamw:  optional weight_decay (default 0).
  muon:   Muon (Newton-Schulz) for blocks + Adam for embed/lm_head.
  muon_v2: Polar Express (arxiv 2505.16932) and/or NorMuon (arxiv 2510.05491).
           --muon_polar jordan|polar_express, --muon_polar_safety 1.01, --muon_normuon.
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_data(data_dir: str = "data", train_fraction: float = 0.9) -> Tuple[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.isfile(path):
        print(f"Downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    with open(path, "r") as f:
        text = f.read()
    n = len(text)
    split = int(train_fraction * n)
    return text[:split], text[split:]


def build_vocab(text: str) -> Tuple[list, dict, dict]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


def encode(s: str, stoi: dict) -> list:
    return [stoi[c] for c in s]


def decode(lst: list, itos: dict) -> str:
    return "".join([itos[i] for i in lst])


def get_batch(
    key: mx.random.PRNGKey,
    data: mx.array,
    batch_size: int,
    block_size: int,
    split: str,
    train_data: mx.array,
    val_data: mx.array,
) -> Tuple[mx.array, mx.array]:
    data_arr = train_data if split == "train" else val_data
    n = data_arr.size - block_size
    key, subkey = mx.random.split(key)
    ix = mx.random.randint(0, n, shape=(batch_size,), key=subkey)
    x = mx.array([data_arr[int(ix[i]) : int(ix[i]) + block_size] for i in range(batch_size)])
    y = mx.array([data_arr[int(ix[i]) + 1 : int(ix[i]) + 1 + block_size] for i in range(batch_size)])
    return x, y


# ---------- Model (Karpathy video structure) ----------
# - SDPA: mx.fast.scaled_dot_product_attention(..., mask="causal") — MLX applies causal mask
#   (lower-right, last query sees last key). See mlx.core.fast.scaled_dot_product_attention.
# - Dropout: only when training=True (eval deterministic).
#
# Differences vs nanogpt_shakedzy (nanogpt/mlx_/gpt.py):
# - Mask: we use SDPA mask="causal"; they use manual tril + triu(-1e9). Both causal.
# - Scale: we use 1/sqrt(head_dim); they use 1/sqrt(input_size) — multi-head should use head_dim.
# - Block: we use pre-norm (x + sa(ln1(x))); they use post-norm (ln1(x + mh(x))).
# - Position: we use position_embedding(mx.arange(T)); they use positional_embeddings(indices)
#   (table shape vocab_size × dim, indexed by token ids — not standard position indices).
# - Dropout: we gate on training; they always apply dropout (wrong at eval).
#
# Modern (--modern): RoPE, QK-norm, ReLU² as GELU replacement in MLP (modded-nanogpt).


def _make_rope_cache(block_size: int, head_dim: int, theta: float = 10000.0) -> Tuple[mx.array, mx.array]:
    """Precompute cos/sin for RoPE. Returns (cos, sin) each (block_size, head_dim // 2)."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    t = mx.arange(block_size, dtype=mx.float32)
    freqs = mx.outer(t, inv_freq)
    return mx.cos(freqs), mx.sin(freqs)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply RoPE to x. x: (B, T, H, D), cos/sin: (T, D/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def _qk_norm(q: mx.array, k: mx.array, eps: float = 1e-6) -> Tuple[mx.array, mx.array]:
    """L2-normalize Q and K over last dimension (per head)."""
    q_norm = mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True) + eps)
    k_norm = mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + eps)
    return q * mx.reciprocal(q_norm), k * mx.reciprocal(k_norm)


class RMSNorm(nn.Module):
    """RMSNorm over last dimension, no learnable gamma/beta (nanochat/modded-nanogpt style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * mx.reciprocal(rms)


# Bigram hash embedding (modded-nanogpt PR #201). Constants from ClassicLarry.
BIGRAM_RAND_INT_1 = 36313
BIGRAM_RAND_INT_2 = 27191


def get_bigram_hash(idx: mx.array, bigram_vocab_size: int) -> mx.array:
    """
    Bigram hash per position: [prev_token, curr_token] -> index in [0, bigram_vocab_size-1].
    Position 0 uses reserved index (bigram_vocab_size - 1). Runs on same device as idx.
    """
    B, T = idx.shape
    mod = bigram_vocab_size - 1
    idx = idx.astype(mx.int32)
    # position 0: reserved index
    pos0 = mx.full((B, 1), mod, mx.int32)
    # positions 1..T-1: hash(prev, curr)
    prev = idx[:, :-1]  # (B, T-1)
    curr = idx[:, 1:]
    raw = mx.bitwise_xor(BIGRAM_RAND_INT_1 * curr, BIGRAM_RAND_INT_2 * prev)
    rest = (raw % mod + mod) % mod  # non-negative in [0, mod-1]
    rest = rest.astype(mx.int32)
    return mx.concatenate([pos0, rest], axis=1)


class CausalSelfAttentionSDPA(nn.Module):
    """Multi-head causal self-attention via mx.fast.scaled_dot_product_attention (SDPA).
    Matches mlx-examples / nanolm_mlx style; no list of Heads so all params are registered."""

    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(n_embd, n_embd)
        self.wk = nn.Linear(n_embd, n_embd)
        self.wv = nn.Linear(n_embd, n_embd)
        self.wo = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        B, T, C = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.wo(out)
        out = self.dropout(out) if training else out
        return out


class CausalSelfAttentionSDPA_Modern(nn.Module):
    """SDPA attention with RoPE on Q/K and QK-norm (modded-nanogpt style). No position embed needed."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, rope_theta: float = 10000.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(n_embd, n_embd)
        self.wk = nn.Linear(n_embd, n_embd)
        self.wv = nn.Linear(n_embd, n_embd)
        self.wo = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self._cos, self._sin = _make_rope_cache(block_size, self.head_dim, theta=rope_theta)

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        B, T, C = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = q.reshape(B, T, self.n_head, self.head_dim)
        k = k.reshape(B, T, self.n_head, self.head_dim)
        v = v.reshape(B, T, self.n_head, self.head_dim)
        cos = self._cos[:T]
        sin = self._sin[:T]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        q, k = _qk_norm(q, k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.wo(out)
        out = self.dropout(out) if training else out
        return out


class FeedForward(nn.Module):
    """Linear -> ReLU -> Linear -> Dropout."""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        x = mx.maximum(0, self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x) if training else x
        return x


class FeedForwardReLU2(nn.Module):
    """Linear -> ReLU² -> Linear -> Dropout. ReLU² as GELU replacement (modded-nanogpt)."""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        x = self.linear1(x)
        x = mx.square(mx.maximum(0, x))
        x = self.linear2(x)
        x = self.dropout(x) if training else x
        return x


class Block(nn.Module):
    """Transformer block: communication (SDPA attention) then computation (FF)."""

    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.sa = CausalSelfAttentionSDPA(n_embd, n_head, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        x = x + self.sa(self.ln1(x), training=training)
        x = x + self.ffwd(self.ln2(x), training=training)
        return x


class BlockModern(nn.Module):
    """Block with RoPE + QK-norm attention and ReLU² FF (modded-nanogpt style)."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, rope_theta: float = 10000.0):
        super().__init__()
        self.sa = CausalSelfAttentionSDPA_Modern(n_embd, n_head, block_size, dropout, rope_theta)
        self.ffwd = FeedForwardReLU2(n_embd, dropout)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        x = x + self.sa(self.ln1(x), training=training)
        x = x + self.ffwd(self.ln2(x), training=training)
        return x


class GPTLanguageModel(nn.Module):
    """Karpathy-style GPT: token + position embeddings (or RoPE when modern), blocks, ln_f, lm_head. Optional bigram hash embedding (modded-nanogpt PR #201)."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        use_modern: bool = False,
        rope_theta: float = 10000.0,
        use_bigram: bool = False,
    ):
        super().__init__()
        self.use_modern = use_modern
        self.use_bigram = use_bigram
        self.block_size = block_size
        self.n_layer = n_layer
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if not use_modern:
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = [Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        else:
            self.position_embedding_table = None
            self.blocks = [
                BlockModern(n_embd, n_head, block_size, dropout, rope_theta) for _ in range(n_layer)
            ]
        self.ln_f = RMSNorm(n_embd) if use_modern else nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        if use_bigram:
            self.bigram_vocab_size = 5 * vocab_size
            self.norm_embed = RMSNorm(n_embd) if use_modern else nn.LayerNorm(n_embd)
            self.bigram_embed = nn.Embedding(self.bigram_vocab_size, n_embd)
            self.bigram_embed.weight = mx.zeros((self.bigram_vocab_size, n_embd))
            self.x0_lambdas = mx.zeros((n_layer,))
            self.bigram_lambdas = mx.full((n_layer,), 0.1)

    def __call__(self, idx: mx.array, targets: Optional[mx.array] = None, training: bool = False) -> Tuple[mx.array, Optional[mx.array]]:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        if self.position_embedding_table is not None:
            pos_emb = self.position_embedding_table(mx.arange(T))
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        if self.use_bigram:
            x0 = self.norm_embed(x)
            x0_bigram = self.bigram_embed(get_bigram_hash(idx, self.bigram_vocab_size))
            x = x0
            for i, block in enumerate(self.blocks):
                x = x + self.x0_lambdas[i] * x0 + self.bigram_lambdas[i] * x0_bigram
                x = block(x, training=training)
        else:
            for block in self.blocks:
                x = block(x, training=training)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            loss = mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none"))
        return logits, loss

    def generate(self, key: mx.random.PRNGKey, idx: mx.array, max_new_tokens: int, max_context: Optional[int] = None) -> mx.array:
        ctx_len = min(self.block_size, max_context) if (max_context is not None and max_context > 0) else self.block_size
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -ctx_len:] if idx.shape[1] > ctx_len else idx
            logits, _ = self(idx_cond, training=False)
            logits = logits[:, -1, :]
            key, subkey = mx.random.split(key)
            idx_next = mx.random.categorical(logits, shape=(idx.shape[0], 1), key=subkey)
            idx_next = idx_next.astype(mx.int32)
            idx = mx.concatenate([idx, idx_next], axis=1)
        return idx


class GPTLanguageModel_SCORE(nn.Module):
    """SCORE: 1 block applied recurrently for `steps` steps: x = x + ode_dt*(block(x)-x). Optional bigram hash embedding."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        dropout: float,
        steps: int,
        ode_dt: float,
        use_modern: bool = False,
        rope_theta: float = 10000.0,
        use_bigram: bool = False,
    ):
        super().__init__()
        self.block_size = block_size
        self.steps = steps
        self.ode_dt = ode_dt
        self.use_bigram = use_bigram
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if not use_modern:
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.block = Block(n_embd, n_head, dropout)
        else:
            self.position_embedding_table = None
            self.block = BlockModern(n_embd, n_head, block_size, dropout, rope_theta)
        self.ln_f = RMSNorm(n_embd) if use_modern else nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        if use_bigram:
            self.bigram_vocab_size = 5 * vocab_size
            self.norm_embed = RMSNorm(n_embd) if use_modern else nn.LayerNorm(n_embd)
            self.bigram_embed = nn.Embedding(self.bigram_vocab_size, n_embd)
            self.bigram_embed.weight = mx.zeros((self.bigram_vocab_size, n_embd))
            self.x0_lambdas = mx.zeros((1,))
            self.bigram_lambdas = mx.full((1,), 0.1)

    def __call__(self, idx: mx.array, targets: Optional[mx.array] = None, training: bool = False) -> Tuple[mx.array, Optional[mx.array]]:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        if self.position_embedding_table is not None:
            x = tok_emb + self.position_embedding_table(mx.arange(T))
        else:
            x = tok_emb

        if self.use_bigram:
            x0 = self.norm_embed(x)
            x0_bigram = self.bigram_embed(get_bigram_hash(idx, self.bigram_vocab_size))
            x = x0

        for _ in range(self.steps):
            if self.use_bigram:
                x = x + self.x0_lambdas[0] * x0 + self.bigram_lambdas[0] * x0_bigram
            x_next = self.block(x, training=training)
            x = x + self.ode_dt * (x_next - x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            loss = mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none"))
        return logits, loss

    def generate(self, key: mx.random.PRNGKey, idx: mx.array, max_new_tokens: int, max_context: Optional[int] = None) -> mx.array:
        ctx_len = min(self.block_size, max_context) if (max_context is not None and max_context > 0) else self.block_size
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -ctx_len:] if idx.shape[1] > ctx_len else idx
            logits, _ = self(idx_cond, training=False)
            logits = logits[:, -1, :]
            key, subkey = mx.random.split(key)
            idx_next = mx.random.categorical(logits, shape=(idx.shape[0], 1), key=subkey)
            idx_next = idx_next.astype(mx.int32)
            idx = mx.concatenate([idx, idx_next], axis=1)
        return idx


class GPTLanguageModel_M_SCORE(nn.Module):
    """M_SCORE: M blocks, each applied for `steps` steps (total M*steps). M=1 is SCORE. Optional bigram hash embedding."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        m_blocks: int,
        dropout: float,
        steps: int,
        ode_dt: float,
        use_modern: bool = False,
        rope_theta: float = 10000.0,
        use_bigram: bool = False,
    ):
        super().__init__()
        self.block_size = block_size
        self.steps = steps
        self.ode_dt = ode_dt
        self.use_bigram = use_bigram
        self.m_blocks = m_blocks
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if not use_modern:
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = [Block(n_embd, n_head, dropout) for _ in range(m_blocks)]
        else:
            self.position_embedding_table = None
            self.blocks = [
                BlockModern(n_embd, n_head, block_size, dropout, rope_theta) for _ in range(m_blocks)
            ]
        self.ln_f = RMSNorm(n_embd) if use_modern else nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        if use_bigram:
            self.bigram_vocab_size = 5 * vocab_size
            self.norm_embed = RMSNorm(n_embd) if use_modern else nn.LayerNorm(n_embd)
            self.bigram_embed = nn.Embedding(self.bigram_vocab_size, n_embd)
            self.bigram_embed.weight = mx.zeros((self.bigram_vocab_size, n_embd))
            self.x0_lambdas = mx.zeros((m_blocks,))
            self.bigram_lambdas = mx.full((m_blocks,), 0.1)

    def __call__(self, idx: mx.array, targets: Optional[mx.array] = None, training: bool = False) -> Tuple[mx.array, Optional[mx.array]]:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        if self.position_embedding_table is not None:
            x = tok_emb + self.position_embedding_table(mx.arange(T))
        else:
            x = tok_emb

        if self.use_bigram:
            x0 = self.norm_embed(x)
            x0_bigram = self.bigram_embed(get_bigram_hash(idx, self.bigram_vocab_size))
            x = x0

        for i, block in enumerate(self.blocks):
            for _ in range(self.steps):
                if self.use_bigram:
                    x = x + self.x0_lambdas[i] * x0 + self.bigram_lambdas[i] * x0_bigram
                x_next = block(x, training=training)
                x = x + self.ode_dt * (x_next - x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            loss = mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none"))
        return logits, loss

    def generate(self, key: mx.random.PRNGKey, idx: mx.array, max_new_tokens: int, max_context: Optional[int] = None) -> mx.array:
        ctx_len = min(self.block_size, max_context) if (max_context is not None and max_context > 0) else self.block_size
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -ctx_len:] if idx.shape[1] > ctx_len else idx
            logits, _ = self(idx_cond, training=False)
            logits = logits[:, -1, :]
            key, subkey = mx.random.split(key)
            idx_next = mx.random.categorical(logits, shape=(idx.shape[0], 1), key=subkey)
            idx_next = idx_next.astype(mx.int32)
            idx = mx.concatenate([idx, idx_next], axis=1)
        return idx


def main():
    p = argparse.ArgumentParser(description="Karpathy GPT from scratch (MLX)")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=250, help="Report train/val loss every N iters")
    p.add_argument("--eval_iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--gen_tokens", type=int, default=500)
    p.add_argument("--gen_max_context", type=int, default=0,
                    help="Max context length during generation (0=full block_size). TOVA-style bounded context (arXiv:2401.06104); smaller = faster gen, may reduce quality.")
    p.add_argument("--modern", action="store_true", help="RoPE + QK-norm + ReLU² (modded-nanogpt style)")
    p.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE base (only with --modern)")
    p.add_argument("--bigram", action="store_true", help="Bigram hash embedding (modded-nanogpt PR #201): 5*vocab extra embed, x0 + bigram per layer")
    p.add_argument("--mode", choices=("base", "score", "m_score"), default="base",
                    help="base: n_layer blocks; score: 1 block, steps× recurrence; m_score: m blocks, each steps×")
    p.add_argument("--steps", type=int, default=6, help="SCORE/M_SCORE recurrence steps per block")
    p.add_argument("--ode_dt", type=float, default=0.5, help="SCORE/M_SCORE ODE step size")
    p.add_argument("--m", type=int, default=1, help="M_SCORE: number of blocks (ignored for base/score)")
    p.add_argument("--optimizer", choices=("adam", "adamw", "muon", "muon_v2"), default="adam",
                    help="adam/adamw/muon; muon_v2: Polar Express and/or NorMuon")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for AdamW (default 0; use e.g. 0.01 if desired)")
    p.add_argument("--muon_lr", type=float, default=0.02, help="Learning rate for Muon (when --optimizer muon or muon_v2)")
    p.add_argument("--muon_polar", choices=("jordan", "polar_express"), default="polar_express",
                    help="Muon V2: polar step = jordan (Newton-Schulz) or polar_express (arxiv 2505.16932)")
    p.add_argument("--muon_polar_safety", type=float, default=1.01,
                    help="Polar Express safety factor (default 1.01; try 1.02 for stability)")
    p.add_argument("--muon_normuon", action="store_true", help="Muon V2: enable NorMuon (neuron-wise norm, arxiv 2510.05491)")
    p.add_argument("--train_fraction", type=float, default=0.9,
                    help="Fraction of data for training (default 0.9). Use 0.1 for small-data regime (10%% train / 90%% val).")
    args = p.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading Tiny Shakespeare ...")
    text_train, text_val = load_data(args.data_dir, train_fraction=args.train_fraction)
    _, stoi, itos = build_vocab(text_train + text_val)
    vocab_size = len(stoi)
    train_data = mx.array(encode(text_train, stoi))
    val_data = mx.array(encode(text_val, stoi))
    pct = args.train_fraction * 100
    print(f"Train: {len(text_train):_} chars ({pct:.0f}%)  Val: {len(text_val):_} ({100-pct:.0f}%)  Vocab: {vocab_size}")

    if args.mode == "base":
        model = GPTLanguageModel(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
            use_modern=args.modern,
            rope_theta=args.rope_theta,
            use_bigram=args.bigram,
        )
    elif args.mode == "score":
        model = GPTLanguageModel_SCORE(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            dropout=args.dropout,
            steps=args.steps,
            ode_dt=args.ode_dt,
            use_modern=args.modern,
            rope_theta=args.rope_theta,
            use_bigram=args.bigram,
        )
    else:
        model = GPTLanguageModel_M_SCORE(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            m_blocks=args.m,
            dropout=args.dropout,
            steps=args.steps,
            ode_dt=args.ode_dt,
            use_modern=args.modern,
            rope_theta=args.rope_theta,
            use_bigram=args.bigram,
        )
    n_params = sum(x.size for _, x in tree_flatten(model.parameters()))
    print(f"Mode: {args.mode}  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    if args.bigram:
        print("Bigram hash embedding: on (5*vocab extra embed, x0 + bigram per layer)")
    if args.mode == "score" and args.m != 1:
        print("Note: --m is ignored for mode score (single block). Use --mode m_score for multiple blocks.")

    loss_fn = nn.value_and_grad(model, lambda m, x, y: m(x, targets=y, training=True)[1])
    if args.optimizer == "adam":
        optimizer = optim.Adam(learning_rate=args.lr)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "muon":
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        if _script_dir not in __import__("sys").path:
            __import__("sys").path.insert(0, _script_dir)
        from muon_mlx import muon_with_adam
        optimizer = muon_with_adam(
            muon_lr=args.muon_lr,
            adam_lr=args.lr,
            muon_momentum=0.95,
            muon_ns_steps=5,
            muon_nesterov=True,
        )
    else:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        if _script_dir not in __import__("sys").path:
            __import__("sys").path.insert(0, _script_dir)
        from muon_v2_mlx import muon_v2_with_adam
        optimizer = muon_v2_with_adam(
            muon_lr=args.muon_lr,
            adam_lr=args.lr,
            muon_momentum=0.95,
            muon_ns_steps=5,
            muon_nesterov=True,
            polar_method=args.muon_polar,
            polar_express_safety=args.muon_polar_safety,
            use_normuon=args.muon_normuon,
        )
    key = mx.random.key(args.seed)
    opt_desc = args.optimizer
    if args.optimizer == "adamw":
        opt_desc += f" (weight_decay={args.weight_decay})"
    elif args.optimizer == "muon_v2":
        opt_desc += f" polar={args.muon_polar}" + (" normuon" if args.muon_normuon else "")
    print(f"Optimizer: {opt_desc}")
    if args.optimizer in ("muon", "muon_v2"):
        print(f"  Adam lr: {args.lr}  Muon lr: {args.muon_lr}")

    print(f"Training {args.max_iters} iters ...")
    t0 = time.perf_counter()
    for iter in range(args.max_iters):
        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            train_losses, val_losses = [], []
            for _ in range(args.eval_iters):
                key, sub = mx.random.split(key)
                k1, k2 = mx.random.split(sub)
                xb, yb = get_batch(k1, None, args.batch_size, args.block_size, "train", train_data, val_data)
                _, loss_t = model(xb, targets=yb, training=False)
                train_losses.append(float(loss_t.item()))
                xb, yb = get_batch(k2, None, args.batch_size, args.block_size, "val", train_data, val_data)
                _, loss_v = model(xb, targets=yb, training=False)
                val_losses.append(float(loss_v.item()))
            train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
            print(f"step {iter}: train(loss={train_loss:.4f}, ppl={np.exp(train_loss):.2f}), val(loss={val_loss:.4f}, ppl={np.exp(val_loss):.2f})")

        key, subkey = mx.random.split(key)
        xb, yb = get_batch(subkey, None, args.batch_size, args.block_size, "train", train_data, val_data)
        loss, grads = loss_fn(model, xb, yb)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    print(f"Done. Time: {time.perf_counter() - t0:.1f}s")
    print("\n--- Generated text ---")
    key, subkey = mx.random.split(key)
    context = mx.zeros((1, 1), dtype=mx.int32)
    gen_ctx = args.gen_max_context if args.gen_max_context > 0 else None
    out = model.generate(subkey, context, args.gen_tokens, max_context=gen_ctx)
    print(decode(out[0].tolist(), itos))


if __name__ == "__main__":
    main()
