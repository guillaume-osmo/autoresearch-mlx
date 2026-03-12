"""
trainscorev1: SCORE architecture + PR#4's batched MuonAdamW optimizer.
Merges the SCORE/bigram/ODE model from trainscore.py with the
batched NorMuon+PolarExpress optimizer from BL3IP's PR#4.

Usage: uv run trainscorev1.py
"""

import gc
import math
import os
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from muon_and_beyond_mlx import newton_schulz, polar_express, newton_schulz_jordan
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# PR#4 polar express coefficients (different polynomial fit from muon_and_beyond_mlx)
_PR_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    prefix_layers: int = 0
    score1_m: int = 0
    score1_steps: int = 0
    middle_layers: int = 0
    score2_m: int = 0
    score2_steps: int = 0
    suffix_layers: int = 0
    score_dt: float = 0.5
    single_dt: float = 0.5
    use_bigram: bool = False
    bigram_scale: float = 0.1
    mlp_expansion: float = 4.0
    use_swiglu: bool = False


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def create_additive_causal_mask(seq_len, dtype=mx.float32):
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def create_sliding_window_mask(seq_len, window_size, dtype=mx.float32):
    indices = mx.arange(seq_len)
    causal = indices[None, :] > indices[:, None]
    too_far = (indices[:, None] - indices[None, :]) >= window_size
    blocked = causal | too_far
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def get_peak_memory_mb():
    return mx.get_peak_memory() / 1024 / 1024


BIGRAM_RAND_INT_1 = 36313
BIGRAM_RAND_INT_2 = 27191


def get_bigram_hash(idx: mx.array, bigram_vocab_size: int) -> mx.array:
    """Hash [prev, curr] token pairs into a compact bigram vocabulary."""
    B, T = idx.shape
    mod = bigram_vocab_size - 1
    idx = idx.astype(mx.int32)
    pos0 = mx.full((B, 1), mod, mx.int32)
    prev = idx[:, :-1]
    curr = idx[:, 1:]
    raw = mx.bitwise_xor(BIGRAM_RAND_INT_1 * curr, BIGRAM_RAND_INT_2 * prev)
    rest = (raw % mod + mod) % mod
    rest = rest.astype(mx.int32)
    return mx.concatenate([pos0, rest], axis=1)


# ---------------------------------------------------------------------------
# MLP variants
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Standard ReLU^2 MLP with configurable expansion."""
    def __init__(self, config):
        super().__init__()
        hidden_dim = max(1, int(round(config.mlp_expansion * config.n_embd)))
        self.hidden_dim = hidden_dim
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2
        return self.c_proj(x)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP: gate = silu(W1 x), value = W3 x, out = W2(gate * value)."""
    def __init__(self, config):
        super().__init__()
        hidden_dim = max(1, int(round(config.mlp_expansion * config.n_embd)))
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Attention + Block
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x, ve, mask):
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).reshape(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = norm(self.rope(q))
        k = norm(self.rope(k))

        scale = 1.0 / math.sqrt(self.head_dim)
        if mask is not None and mask.dtype != q.dtype:
            mask = mask.astype(q.dtype)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = SwiGLUMLP(config) if config.use_swiglu else MLP(config)

    def __call__(self, x, ve, mask):
        x = x + self.attn(norm(x), ve, mask)
        x = x + self.mlp(norm(x))
        return x


# ---------------------------------------------------------------------------
# SCORE stages + GPT model (from trainscore.py)
# ---------------------------------------------------------------------------

class ScoreStage(nn.Module):
    def __init__(self, blocks, steps, ode_dt, app_indices, proto_indices):
        super().__init__()
        self.blocks = blocks
        self.steps = steps
        self.ode_dt = ode_dt
        self.app_indices = app_indices
        self.proto_indices = proto_indices


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proto_layer_count = (
            config.prefix_layers
            + config.score1_m
            + config.middle_layers
            + config.score2_m
            + config.suffix_layers
        )
        self.effective_layers = self._compute_effective_layers(config)
        if self.proto_layer_count == 0:
            config.prefix_layers = config.n_layer
            self.proto_layer_count = config.n_layer
            self.effective_layers = config.n_layer

        if self.effective_layers != config.n_layer:
            raise ValueError(
                f"Effective depth mismatch: n_layer={config.n_layer}, but layout expands to {self.effective_layers}. "
                "Set DEPTH equal to singles + recurrent applications."
            )

        self.window_sizes = self._compute_window_sizes(config, self.effective_layers)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.use_bigram = config.use_bigram
        self.bigram_scale = config.bigram_scale
        if self.use_bigram:
            self.bigram_vocab_size = 5 * config.vocab_size
            self.bigram_embed = nn.Embedding(self.bigram_vocab_size, config.n_embd)
            self.bigram_lambdas = mx.full((self.effective_layers,), config.bigram_scale, dtype=mx.float32)
        self.resid_lambdas = mx.ones((self.effective_layers,), dtype=mx.float32)
        self.x0_lambdas = mx.zeros((self.effective_layers,), dtype=mx.float32)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self._mask_cache = {}

        proto_cursor = 0
        app_cursor = 0
        self.prefix_blocks, self.prefix_app_indices, self.prefix_proto_indices, proto_cursor, app_cursor = self._make_blocks(
            config.prefix_layers, proto_cursor, app_cursor
        )
        self.score1_stage, proto_cursor, app_cursor = self._make_score_stage(
            config.score1_m, config.score1_steps, config.score_dt, proto_cursor, app_cursor
        )
        self.middle_blocks, self.middle_app_indices, self.middle_proto_indices, proto_cursor, app_cursor = self._make_blocks(
            config.middle_layers, proto_cursor, app_cursor
        )
        self.score2_stage, proto_cursor, app_cursor = self._make_score_stage(
            config.score2_m, config.score2_steps, config.score_dt, proto_cursor, app_cursor
        )
        self.suffix_blocks, self.suffix_app_indices, self.suffix_proto_indices, proto_cursor, app_cursor = self._make_blocks(
            config.suffix_layers, proto_cursor, app_cursor
        )

        if proto_cursor != self.proto_layer_count:
            raise ValueError(f"Internal proto layer count mismatch: {proto_cursor} != {self.proto_layer_count}")
        if app_cursor != self.effective_layers:
            raise ValueError(f"Internal effective layer count mismatch: {app_cursor} != {self.effective_layers}")

        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(self.proto_layer_count)
            if has_ve(i, self.proto_layer_count)
        }

    def _compute_effective_layers(self, config):
        recurrent_apps = config.score1_m * config.score1_steps + config.score2_m * config.score2_steps
        singles = config.prefix_layers + config.middle_layers + config.suffix_layers
        if recurrent_apps + singles == 0:
            return config.n_layer
        return recurrent_apps + singles

    def _make_blocks(self, count, proto_start_idx, app_start_idx):
        blocks = [Block(self.config, proto_start_idx + i) for i in range(count)]
        app_indices = list(range(app_start_idx, app_start_idx + count))
        proto_indices = list(range(proto_start_idx, proto_start_idx + count))
        return blocks, app_indices, proto_indices, proto_start_idx + count, app_start_idx + count

    def _make_score_stage(self, m_blocks, steps, ode_dt, proto_start_idx, app_start_idx):
        if m_blocks <= 0 or steps <= 0:
            return None, proto_start_idx, app_start_idx
        blocks = [Block(self.config, proto_start_idx + i) for i in range(m_blocks)]
        app_indices = []
        proto_indices = []
        for _ in range(steps):
            app_indices.extend(range(app_start_idx, app_start_idx + m_blocks))
            proto_indices.extend(range(proto_start_idx, proto_start_idx + m_blocks))
            app_start_idx += m_blocks
        stage = ScoreStage(blocks, steps, ode_dt, app_indices, proto_indices)
        return stage, proto_start_idx + m_blocks, app_start_idx

    def init_weights(self):
        n_embd = self.config.n_embd
        scale = 3**0.5 * n_embd**-0.5

        self.wte.weight = (mx.random.normal(self.wte.weight.shape) * 1.0).astype(mx.bfloat16)
        self.lm_head.weight = (mx.random.normal(self.lm_head.weight.shape) * 0.001).astype(mx.bfloat16)
        if self.use_bigram:
            self.bigram_embed.weight = mx.zeros((self.bigram_vocab_size, self.config.n_embd), dtype=mx.bfloat16)

        for block_group in [self.prefix_blocks, self.middle_blocks, self.suffix_blocks]:
            for block in block_group:
                self._init_block(block, scale)
        for stage in [self.score1_stage, self.score2_stage]:
            if stage is not None:
                for block in stage.blocks:
                    self._init_block(block, scale)

        self.resid_lambdas = mx.ones((self.effective_layers,), dtype=mx.float32)
        self.x0_lambdas = mx.full((self.effective_layers,), 0.1, dtype=mx.float32)
        if self.use_bigram:
            self.bigram_lambdas = mx.full((self.effective_layers,), self.bigram_scale, dtype=mx.float32)

        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-scale, scale, ve.weight.shape).astype(mx.bfloat16)

    def _init_block(self, block, scale):
        block.attn.c_q.weight = mx.random.uniform(-scale, scale, block.attn.c_q.weight.shape).astype(mx.bfloat16)
        block.attn.c_k.weight = mx.random.uniform(-scale, scale, block.attn.c_k.weight.shape).astype(mx.bfloat16)
        block.attn.c_v.weight = mx.random.uniform(-scale, scale, block.attn.c_v.weight.shape).astype(mx.bfloat16)
        block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight).astype(mx.bfloat16)

        mlp = block.mlp
        if isinstance(mlp, SwiGLUMLP):
            mlp.w1.weight = mx.random.uniform(-scale, scale, mlp.w1.weight.shape).astype(mx.bfloat16)
            mlp.w2.weight = mx.zeros_like(mlp.w2.weight).astype(mx.bfloat16)
            mlp.w3.weight = mx.random.uniform(-scale, scale, mlp.w3.weight.shape).astype(mx.bfloat16)
        else:
            mlp.c_fc.weight = mx.random.uniform(-scale, scale, mlp.c_fc.weight.shape).astype(mx.bfloat16)
            mlp.c_proj.weight = mx.zeros_like(mlp.c_proj.weight).astype(mx.bfloat16)

        if block.attn.ve_gate is not None:
            block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight).astype(mx.bfloat16)

    def _compute_window_sizes(self, config, n_layers):
        pattern = config.window_pattern.upper()
        assert all(char in "SL" for char in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(n_layers):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = long_window
        return window_sizes

    def _get_masks(self, seq_len):
        unique_windows = set(self.window_sizes)
        for window_size in unique_windows:
            key = (seq_len, window_size)
            if key not in self._mask_cache:
                if window_size >= seq_len:
                    self._mask_cache[key] = create_additive_causal_mask(seq_len)
                else:
                    self._mask_cache[key] = create_sliding_window_mask(seq_len, window_size)
        return [self._mask_cache[(seq_len, window_size)] for window_size in self.window_sizes]

    def _preblock_mix(self, x, x0, x0_bigram, layer_idx):
        resid = self.resid_lambdas[layer_idx].astype(x.dtype)
        x0w = self.x0_lambdas[layer_idx].astype(x.dtype)
        mixed = resid * x + x0w * x0
        if self.use_bigram and x0_bigram is not None:
            bw = self.bigram_lambdas[layer_idx].astype(x.dtype)
            mixed = mixed + bw * x0_bigram
        return mixed

    def _run_standard_blocks(self, x, x0, x0_bigram, idx, blocks, app_indices, proto_indices, masks):
        for block, app_idx, proto_idx in zip(blocks, app_indices, proto_indices):
            x_in = self._preblock_mix(x, x0, x0_bigram, app_idx)
            ve = self.value_embeds[str(proto_idx)](idx) if str(proto_idx) in self.value_embeds else None
            x_next = block(x_in, ve, masks[app_idx])
            x = x_in + self.config.single_dt * (x_next - x_in)
        return x

    def _run_score_stage(self, x, x0, x0_bigram, idx, stage, masks):
        if stage is None:
            return x
        app_ptr = 0
        for _ in range(stage.steps):
            for block in stage.blocks:
                app_idx = stage.app_indices[app_ptr]
                proto_idx = stage.proto_indices[app_ptr]
                app_ptr += 1
                x_in = self._preblock_mix(x, x0, x0_bigram, app_idx)
                ve = self.value_embeds[str(proto_idx)](idx) if str(proto_idx) in self.value_embeds else None
                x_next = block(x_in, ve, masks[app_idx])
                x = x_in + stage.ode_dt * (x_next - x_in)
        return x

    def __call__(self, idx, targets=None, reduction="mean"):
        _, seq_len = idx.shape
        masks = self._get_masks(seq_len)

        x = self.wte(idx)
        x = norm(x)
        x0 = x
        x0_bigram = None
        if self.use_bigram:
            x0_bigram = self.bigram_embed(get_bigram_hash(idx, self.bigram_vocab_size))

        x = self._run_standard_blocks(x, x0, x0_bigram, idx, self.prefix_blocks, self.prefix_app_indices, self.prefix_proto_indices, masks)
        x = self._run_score_stage(x, x0, x0_bigram, idx, self.score1_stage, masks)
        x = self._run_standard_blocks(x, x0, x0_bigram, idx, self.middle_blocks, self.middle_app_indices, self.middle_proto_indices, masks)
        x = self._run_score_stage(x, x0, x0_bigram, idx, self.score2_stage, masks)
        x = self._run_standard_blocks(x, x0, x0_bigram, idx, self.suffix_blocks, self.suffix_app_indices, self.suffix_proto_indices, masks)
        x = norm(x)

        logits = self.lm_head(x).astype(mx.float32)
        logits = 15.0 * mx.tanh(logits / 15.0)

        if targets is None:
            return logits

        valid = targets != -1
        targets_safe = mx.where(valid, targets, mx.zeros_like(targets))
        ce = nn.losses.cross_entropy(logits, targets_safe, reduction="none")
        ce = ce * valid
        if reduction == "none":
            return ce
        denom = mx.maximum(mx.sum(valid), 1)
        return mx.sum(ce) / denom


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

class HybridOptimizer:
    """Original per-param optimizer from trainscore.py (muon/muon_v2/adamw)."""

    def __init__(
        self,
        model,
        unembedding_lr,
        embedding_lr,
        matrix_lr,
        weight_decay,
        adam_betas,
        scalar_lr,
        optimizer_kind="adamw",
        muon_momentum=0.95,
        muon_ns_steps=3,
        muon_ns_order=5,
        muon_nesterov=True,
        muon_polar_method="polar_express",
        muon_polar_safety=1.01,
        muon_use_normuon=False,
    ):
        self.param_config = {}
        self.state_map = {}
        self.optimizer_kind = optimizer_kind

        model_dim = model.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        flat_params = tree_flatten(model.parameters())
        for path, param in flat_params:
            if "blocks" in path and param.ndim == 2:
                self.param_config[path] = {
                    "kind": optimizer_kind,
                    "lr": matrix_lr,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": weight_decay,
                    "momentum": muon_momentum,
                    "ns_steps": muon_ns_steps,
                    "ns_order": muon_ns_order,
                    "nesterov": muon_nesterov,
                    "polar_method": muon_polar_method,
                    "polar_safety": muon_polar_safety,
                    "use_normuon": muon_use_normuon,
                }
            elif "wte" in path or "bigram_embed" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "value_embeds" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "lm_head" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "resid_lambdas" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": scalar_lr * 0.01,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "x0_lambdas" in path or "bigram_lambdas" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": scalar_lr,
                    "betas": (0.96, 0.95),
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            else:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }

        self.initial_lrs = {path: config["lr"] for path, config in self.param_config.items()}

    def _set_path_value(self, model, path, value):
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)

    def _step_adamw(self, path, grad, param, config):
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = config["lr"]
        beta1, beta2 = config["betas"]
        eps = config["eps"]
        weight_decay = config["weight_decay"]

        if path not in self.state_map:
            self.state_map[path] = {
                "m": mx.zeros_like(grad_f32),
                "v": mx.zeros_like(grad_f32),
                "t": 0,
            }

        state = self.state_map[path]
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad_f32
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad_f32 * grad_f32)

        bias1 = 1 - beta1 ** state["t"]
        bias2 = 1 - beta2 ** state["t"]
        denom = mx.sqrt(state["v"] / bias2) + eps
        step_size = lr / bias1

        param_f32 = param_f32 * (1 - lr * weight_decay)
        param_f32 = param_f32 - step_size * (state["m"] / denom)
        return param_f32.astype(param.dtype)

    def _step_muon(self, path, grad, param, config):
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = config["lr"]
        momentum = config["momentum"]
        weight_decay = config["weight_decay"]
        nesterov = config["nesterov"]

        if path not in self.state_map:
            self.state_map[path] = {"momentum": mx.zeros_like(grad_f32)}

        state = self.state_map[path]
        m = state["momentum"]
        m = momentum * m + grad_f32
        state["momentum"] = m

        update = grad_f32 + momentum * m if nesterov else m
        if param.ndim == 2:
            update = newton_schulz(update, steps=config["ns_steps"], order=config["ns_order"])

        param_f32 = param_f32 * (1 - lr * weight_decay)
        param_f32 = param_f32 - lr * update
        return param_f32.astype(param.dtype)

    def _step_muon_v2(self, path, grad, param, config):
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = config["lr"]
        momentum = config["momentum"]
        weight_decay = config["weight_decay"]
        nesterov = config["nesterov"]

        if path not in self.state_map:
            self.state_map[path] = {"momentum": mx.zeros_like(grad_f32)}

        state = self.state_map[path]
        m = state["momentum"]
        m = momentum * m + grad_f32
        state["momentum"] = m
        update = grad_f32 + momentum * m if nesterov else m

        if param.ndim == 2:
            if config["use_normuon"]:
                row_norms = mx.sqrt(mx.sum(update * update, axis=1, keepdims=True) + 1e-8)
                update = update / row_norms
            if config["polar_method"] == "polar_express":
                update = polar_express(update, steps=config["ns_steps"], safety_factor=config["polar_safety"])
            else:
                update = newton_schulz_jordan(update, steps=config["ns_steps"])

        param_f32 = param_f32 - lr * update
        if weight_decay != 0:
            param_f32 = param_f32 - lr * weight_decay * param_f32
        return param_f32.astype(param.dtype)

    def update(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))
        for path, grad in flat_grads.items():
            if path not in self.param_config:
                continue
            config = self.param_config[path]
            param = flat_params[path]
            kind = config.get("kind", "adamw")
            if kind == "muon":
                new_param = self._step_muon(path, grad, param, config)
            elif kind == "muon_v2":
                new_param = self._step_muon_v2(path, grad, param, config)
            else:
                new_param = self._step_adamw(path, grad, param, config)
            self._set_path_value(model, path, new_param)

    def set_lr_multiplier(self, multiplier):
        for path, config in self.param_config.items():
            config["lr"] = self.initial_lrs[path] * multiplier

    @property
    def state(self):
        arrays = []
        for state in self.state_map.values():
            arrays.extend([v for v in state.values() if isinstance(v, mx.array)])
        return arrays


class MuonAdamW:
    """PR#4-style batched optimizer: Muon for 2D block params, AdamW for rest.
    
    Key differences from HybridOptimizer.muon_v2:
    - EMA-style Nesterov momentum: buf = mu*buf + (1-mu)*g; g = (1-mu)*g + mu*buf
    - Batched Newton-Schulz in bf16 (same-shape params stacked)
    - NorMuon with EMA variance tracking (not simple row-norm)
    - Cautious weight decay (only where gradient aligns with param)
    - Shape-aware LR scaling for tall matrices
    """

    def __init__(
        self,
        model,
        unembedding_lr,
        embedding_lr,
        matrix_lr,
        weight_decay,
        adam_betas,
        scalar_lr,
        use_muon=True,
        muon_momentum=0.95,
        ns_steps=3,
        muon_beta2=0.95,
    ):
        self.use_muon = use_muon
        self.ns_steps = ns_steps
        self._muon_momentum = muon_momentum
        self._muon_wd = weight_decay
        self._muon_beta2 = muon_beta2
        self.param_config = {}
        self.adam_state = {}
        self.muon_groups = []

        model_dim = model.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        muon_paths_by_shape = {}
        flat_params = tree_flatten(model.parameters())

        for path, param in flat_params:
            if use_muon and "blocks" in path and param.ndim == 2:
                self.param_config[path] = {
                    "kind": "muon",
                    "lr": matrix_lr,
                    "weight_decay": weight_decay,
                }
                shape = tuple(param.shape)
                if shape not in muon_paths_by_shape:
                    muon_paths_by_shape[shape] = []
                muon_paths_by_shape[shape].append(path)
            elif "blocks" in path and param.ndim == 2:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": matrix_lr,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": weight_decay,
                }
            elif "wte" in path or "bigram_embed" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "value_embeds" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "lm_head" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "resid_lambdas" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": scalar_lr * 0.01,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            elif "x0_lambdas" in path or "bigram_lambdas" in path:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": scalar_lr,
                    "betas": (0.96, 0.95),
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            else:
                self.param_config[path] = {
                    "kind": "adamw",
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }

        for shape in sorted(muon_paths_by_shape.keys()):
            paths = muon_paths_by_shape[shape]
            n = len(paths)
            red_dim = -1 if shape[-2] >= shape[-1] else -2
            v_buf_shape = (n, shape[-2], 1) if shape[-2] >= shape[-1] else (n, 1, shape[-1])
            self.muon_groups.append({
                "paths": paths,
                "shape": shape,
                "lr": matrix_lr,
                "momentum_buf": mx.zeros((n, *shape)),
                "v_buf": mx.zeros(v_buf_shape),
                "red_dim": red_dim,
            })
            print(f"  Muon group: {n} params of shape {shape}")

        self.initial_lrs = {path: config["lr"] for path, config in self.param_config.items()}
        self.initial_muon_lrs = [g["lr"] for g in self.muon_groups]

    def _set_path_value(self, model, path, value):
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)

    def _adamw_step(self, path, grad, param, config):
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = config["lr"]
        beta1, beta2 = config["betas"]
        eps = config["eps"]
        weight_decay = config["weight_decay"]

        if path not in self.adam_state:
            self.adam_state[path] = {
                "m": mx.zeros_like(grad_f32),
                "v": mx.zeros_like(grad_f32),
                "t": 0,
            }

        state = self.adam_state[path]
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad_f32
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad_f32 * grad_f32)

        bias1 = 1 - beta1 ** state["t"]
        bias2 = 1 - beta2 ** state["t"]
        denom = mx.sqrt(state["v"] / bias2) + eps
        step_size = lr / bias1

        param_f32 = param_f32 * (1 - lr * weight_decay)
        param_f32 = param_f32 - step_size * (state["m"] / denom)
        return param_f32.astype(param.dtype)

    def _muon_step(self, stacked_grads, stacked_params, momentum_buf, v_buf, shape, red_dim):
        momentum = self._muon_momentum

        new_buf = momentum * momentum_buf + (1 - momentum) * stacked_grads
        g = (1 - momentum) * stacked_grads + momentum * new_buf

        X = g.astype(mx.bfloat16)
        X_norm = mx.sqrt(mx.sum(X * X, axis=(-2, -1), keepdims=True))
        X = X / (X_norm * 1.02 + 1e-6)

        if shape[-2] > shape[-1]:
            for a, b, c in _PR_POLAR_EXPRESS_COEFFS[:self.ns_steps]:
                A = mx.swapaxes(X, -1, -2) @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:
            for a, b, c in _PR_POLAR_EXPRESS_COEFFS[:self.ns_steps]:
                A = X @ mx.swapaxes(X, -1, -2)
                B = b * A + c * (A @ A)
                X = a * X + B @ X

        g = X.astype(mx.float32)

        v_mean = mx.mean(g * g, axis=red_dim, keepdims=True)
        red_dim_size = shape[red_dim]
        v_norm_sq = mx.sum(v_mean, axis=(-2, -1), keepdims=True) * red_dim_size
        v_norm = mx.sqrt(v_norm_sq)

        beta2 = self._muon_beta2
        new_v_buf = beta2 * v_buf + (1 - beta2) * v_mean

        step_size = mx.rsqrt(mx.maximum(new_v_buf, 1e-10))
        scaled_sq_sum = (v_mean * red_dim_size) * (step_size * step_size)
        v_norm_new = mx.sqrt(mx.sum(scaled_sq_sum, axis=(-2, -1), keepdims=True))
        final_scale = step_size * (v_norm / mx.maximum(v_norm_new, 1e-10))
        g = g * final_scale

        return g, new_buf, new_v_buf

    def update(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))

        for path, grad in flat_grads.items():
            if path not in self.param_config:
                continue
            config = self.param_config[path]
            if config.get("kind") != "adamw":
                continue
            param = flat_params[path]
            new_param = self._adamw_step(path, grad, param, config)
            self._set_path_value(model, path, new_param)

        for group in self.muon_groups:
            paths = group["paths"]
            shape = group["shape"]
            group_grads = []
            group_params = []
            for p in paths:
                if p not in flat_grads:
                    continue
                group_grads.append(flat_grads[p].astype(mx.float32))
                group_params.append(flat_params[p].astype(mx.float32))

            if len(group_grads) != len(paths):
                continue

            stacked_grads = mx.stack(group_grads)
            stacked_params = mx.stack(group_params)

            g, new_buf, new_v_buf = self._muon_step(
                stacked_grads, stacked_params,
                group["momentum_buf"], group["v_buf"],
                shape, group["red_dim"],
            )
            group["momentum_buf"] = new_buf
            group["v_buf"] = new_v_buf

            lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
            wd = self._muon_wd

            mask = (g * stacked_params >= 0).astype(mx.float32)
            new_params = stacked_params - lr * g - lr * wd * stacked_params * mask

            for i, path in enumerate(paths):
                self._set_path_value(model, path, new_params[i].astype(mx.bfloat16))

    def set_lr_multiplier(self, multiplier):
        for path, config in self.param_config.items():
            config["lr"] = self.initial_lrs[path] * multiplier
        for i, group in enumerate(self.muon_groups):
            group["lr"] = self.initial_muon_lrs[i] * multiplier

    def set_muon_momentum(self, momentum):
        self._muon_momentum = momentum

    def set_muon_wd(self, wd):
        self._muon_wd = wd

    @property
    def state(self):
        arrays = []
        for state in self.adam_state.values():
            arrays.extend([state["m"], state["v"]])
        for group in self.muon_groups:
            arrays.extend([group["momentum_buf"], group["v_buf"]])
        return arrays


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"
MLP_EXP = 8.0 / 3.0
USE_SWIGLU = True

# Optimizer selection: "hybrid" (trainscore's muon_v2) or "muon_adamw" (PR#4's batched)
OPTIMIZER_CLASS = "muon_adamw"

# --- Shared LR / schedule ---
TOTAL_BATCH_SIZE = 2**14
EMBEDDING_LR = 0.8
UNEMBEDDING_LR = 0.008
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.05
ADAM_BETAS = (0.85, 0.99)
WARMUP_RATIO = 0.03
WARMDOWN_RATIO = 0.4
FINAL_LR_FRAC = 0.1

# --- MuonAdamW (PR#4) settings ---
USE_MUON = True
NS_STEPS = 5
MUON_MOMENTUM = 0.85
MUON_BETA2 = 0.99

# --- HybridOptimizer (trainscore) settings ---
HYBRID_OPTIMIZER_KIND = "muon_v2"
HYBRID_MUON_MOMENTUM = 0.95
HYBRID_MUON_NS_STEPS = 3
HYBRID_MUON_NS_ORDER = 5
HYBRID_MUON_NESTEROV = True
HYBRID_MUON_POLAR_METHOD = "polar_express"
HYBRID_MUON_POLAR_SAFETY = 1.01
HYBRID_MUON_USE_NORMUON = True

BIGRAM = True
BIGRAM_SCALE = 0.1

DEPTH = 4

# SCORE stage layout
PREFIX_LAYERS = 4
SCORE1_M = 0
SCORE1_STEPS = 0
MIDDLE_LAYERS = 0
SCORE2_M = 0
SCORE2_STEPS = 0
SUFFIX_LAYERS = 0
SCORE_DT = 0.5
SINGLE_DT = 0.5

DEVICE_BATCH_SIZE = 8
FINAL_EVAL_BATCH_SIZE = 256
STARTUP_EXCLUDE_STEPS = 1


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    """Ramp momentum from 0.85 to MUON_MOMENTUM over first 300 steps (PR#4)."""
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * MUON_MOMENTUM


def get_weight_decay(progress):
    """Linearly decay weight decay to zero (PR#4)."""
    return WEIGHT_DECAY * (1 - progress)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)
t_data = time.time()
print(f"Data/tokenizer loaded in {t_data - t_start:.1f}s")

model_dim = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=model_dim // HEAD_DIM,
    n_kv_head=model_dim // HEAD_DIM,
    n_embd=model_dim,
    window_pattern=WINDOW_PATTERN,
    prefix_layers=PREFIX_LAYERS,
    score1_m=SCORE1_M,
    score1_steps=SCORE1_STEPS,
    middle_layers=MIDDLE_LAYERS,
    score2_m=SCORE2_M,
    score2_steps=SCORE2_STEPS,
    suffix_layers=SUFFIX_LAYERS,
    score_dt=SCORE_DT,
    single_dt=SINGLE_DT,
    use_bigram=BIGRAM,
    bigram_scale=BIGRAM_SCALE,
    mlp_expansion=MLP_EXP,
    use_swiglu=USE_SWIGLU,
)

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())
num_params = sum(param.size for _, param in tree_flatten(model.parameters()))

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

if OPTIMIZER_CLASS == "muon_adamw":
    optimizer = MuonAdamW(
        model,
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
        adam_betas=ADAM_BETAS,
        scalar_lr=SCALAR_LR,
        use_muon=USE_MUON,
        muon_momentum=MUON_MOMENTUM,
        ns_steps=NS_STEPS,
        muon_beta2=MUON_BETA2,
    )
else:
    optimizer = HybridOptimizer(
        model,
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
        adam_betas=ADAM_BETAS,
        scalar_lr=SCALAR_LR,
        optimizer_kind=HYBRID_OPTIMIZER_KIND,
        muon_momentum=HYBRID_MUON_MOMENTUM,
        muon_ns_steps=HYBRID_MUON_NS_STEPS,
        muon_ns_order=HYBRID_MUON_NS_ORDER,
        muon_nesterov=HYBRID_MUON_NESTEROV,
        muon_polar_method=HYBRID_MUON_POLAR_METHOD,
        muon_polar_safety=HYBRID_MUON_POLAR_SAFETY,
        muon_use_normuon=HYBRID_MUON_USE_NORMUON,
    )

loss_grad_fn = nn.value_and_grad(model, lambda model, inputs, targets: model(inputs, targets=targets))

opt_name = OPTIMIZER_CLASS
if OPTIMIZER_CLASS == "muon_adamw":
    opt_name += f" (muon={'on' if USE_MUON else 'off'}, ns={NS_STEPS}, beta2={MUON_BETA2})"
else:
    opt_name += f" ({HYBRID_OPTIMIZER_KIND}, normuon={HYBRID_MUON_USE_NORMUON})"

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(
    "Layout: "
    f"prefix={PREFIX_LAYERS}, score1=({SCORE1_M} shared blocks x {SCORE1_STEPS} steps), "
    f"middle={MIDDLE_LAYERS}, score2=({SCORE2_M} shared blocks x {SCORE2_STEPS} steps), "
    f"suffix={SUFFIX_LAYERS}, score_dt={SCORE_DT}, single_dt={SINGLE_DT}"
)
print(f"Optimizer: {opt_name}")
print(f"MLP: {'SwiGLU' if USE_SWIGLU else 'ReLU^2'} | expansion={MLP_EXP:.3f}")
print(f"Bigram: {BIGRAM} | bigram_scale={BIGRAM_SCALE}")

smooth_train_loss = 0.0
total_training_time = 0.0
step = 0
t_compiled = None
progress = 0.0

while True:
    t0 = time.time()
    accum_grads = None
    train_loss = None

    for _ in range(grad_accum_steps):
        loss, grads = loss_grad_fn(model, x, y)
        mx.eval(loss, grads)
        if t_compiled is None:
            t_compiled = time.time()
            print(f"Model compiled in {t_compiled - t_data:.1f}s")
        train_loss = loss
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda lhs, rhs: lhs + rhs, accum_grads, grads)
        x, y, epoch = next(train_loader)

    if grad_accum_steps > 1:
        accum_grads = tree_map(lambda grad: grad * (1.0 / grad_accum_steps), accum_grads)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    optimizer.set_lr_multiplier(lrm)

    if OPTIMIZER_CLASS == "muon_adamw" and USE_MUON:
        optimizer.set_muon_momentum(get_muon_momentum(step))
        optimizer.set_muon_wd(get_weight_decay(progress))

    optimizer.update(model, accum_grads)
    mx.eval(model.parameters(), *optimizer.state)

    train_loss_f = float(train_loss.item())
    if train_loss_f > 100:
        print("FAIL")
        raise SystemExit(1)

    dt = time.time() - t0
    if step >= STARTUP_EXCLUDE_STEPS:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    remaining = max(0.0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if step >= STARTUP_EXCLUDE_STEPS and total_training_time >= TIME_BUDGET:
        break

print()
t_train = time.time()
print(f"Training completed in {t_train - t_compiled:.1f}s")

total_tokens = step * TOTAL_BATCH_SIZE
print("Starting final eval...")
print(f"Final eval batch size: {FINAL_EVAL_BATCH_SIZE}")
val_bpb = evaluate_bpb(model, tokenizer, FINAL_EVAL_BATCH_SIZE)
t_eval = time.time()
print(f"Final eval completed in {t_eval - t_train:.1f}s")

steady_state_mfu = 0.0
peak_vram_mb = get_peak_memory_mb()

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_eval - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"effective_depth:  {model.effective_layers}")
print(f"unique_layers:    {model.proto_layer_count}")
print(f"optimizer:        {opt_name}")
print(f"mlp:              {'SwiGLU' if USE_SWIGLU else 'ReLU^2'}")
