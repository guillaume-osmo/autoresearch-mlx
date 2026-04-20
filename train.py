"""
Autoresearch pretraining script. Single-device, single-file.
Apple Silicon MLX port of karpathy/autoresearch.
Usage: uv run train.py
"""

import gc
import math
import os
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, get_token_bytes, make_dataloader

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

NORM_EPS = 1e-5
FAST_OPS = getattr(mx, "fast", None)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def rms_norm(x):
    if FAST_OPS is not None and hasattr(FAST_OPS, "rms_norm"):
        return FAST_OPS.rms_norm(x, None, NORM_EPS)
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + NORM_EPS)


def relu2(x):
    if FAST_OPS is not None and hasattr(FAST_OPS, "relu2"):
        return FAST_OPS.relu2(x)
    return mx.maximum(x, 0) ** 2


def cached_weight_t(linear):
    weight = linear.weight
    weight_ref = id(weight)
    cached_ref = getattr(linear, "_cached_weight_t_source_id", None)
    if cached_ref != weight_ref:
        linear._cached_weight_t = weight.T
        linear._cached_weight_t_source_id = weight_ref
    return linear._cached_weight_t


def linear_forward(linear, x):
    return linear(x)


def rms_linear(linear, x):
    return linear(rms_norm(x))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    if VE_EVERY_LAYER:
        return True
    return layer_idx % 2 == (n_layer - 1) % 2


def create_additive_causal_mask(seq_len, dtype=mx.bfloat16):
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def create_sliding_window_mask(seq_len, window_size, dtype=mx.bfloat16):
    indices = mx.arange(seq_len)
    causal = indices[None, :] > indices[:, None]
    too_far = (indices[:, None] - indices[None, :]) >= window_size
    blocked = causal | too_far
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def get_peak_memory_mb():
    return mx.get_peak_memory() / 1024 / 1024


class InLoopMemory:
    """
    Lightweight in-training memory feedback:
    - GradMem: token-level reward memory
    - Engram: deterministic hashed n-gram slot memory
    """

    def __init__(self, mode="both", adapt_strength=0.08, adapt_clamp=0.20):
        self.mode = mode
        self.adapt_strength = adapt_strength
        self.adapt_clamp = adapt_clamp
        self.token_stats = {}
        self.engram_slots_2g = {}
        self.engram_slots_3g = {}
        self.seed = 17

    def _hash64(self, text):
        h = 1469598103934665603
        for b in text.encode("utf-8", errors="ignore"):
            h ^= b
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return int(h)

    def _context_tokens(self, probe_delta, train_loss, peak_mem_mb):
        toks = []
        toks.append("probe_up" if probe_delta > 0 else "probe_down")
        toks.append("loss_low" if train_loss < 2.0 else "loss_high")
        toks.append("mem_high" if peak_mem_mb > 24000 else "mem_ok")
        toks.append("delta_big" if abs(probe_delta) > 0.01 else "delta_small")
        return toks

    def _update_gradmem(self, tokens, reward):
        for tok in tokens:
            st = self.token_stats.setdefault(tok, {"score": 0.0, "seen": 0})
            st["score"] += reward
            st["seen"] += 1

    def _engram_slot_id(self, ngram_tokens, mod):
        vals = [self._hash64(tok) for tok in ngram_tokens]
        mix = vals[0] * (10007 + self.seed)
        for idx, val in enumerate(vals[1:], start=1):
            mix ^= val * (10009 + 97 * idx + self.seed)
        return int(mix % mod)

    def _update_engram(self, tokens, reward):
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                ng = tokens[i : i + 2]
                sid = self._engram_slot_id(ng, 10007)
                st = self.engram_slots_2g.setdefault(sid, {"score": 0.0, "seen": 0})
                st["score"] += reward
                st["seen"] += 1
        if len(tokens) >= 3:
            for i in range(len(tokens) - 2):
                ng = tokens[i : i + 3]
                sid = self._engram_slot_id(ng, 20011)
                st = self.engram_slots_3g.setdefault(sid, {"score": 0.0, "seen": 0})
                st["score"] += reward
                st["seen"] += 1

    def _score_gradmem(self, tokens):
        vals = []
        for tok in tokens:
            st = self.token_stats.get(tok)
            if st is None:
                continue
            vals.append(st["score"] / math.sqrt(max(1, st["seen"])))
        return sum(vals) / len(vals) if vals else 0.0

    def _score_engram(self, tokens):
        vals = []
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                sid = self._engram_slot_id(tokens[i : i + 2], 10007)
                st = self.engram_slots_2g.get(sid)
                if st is not None:
                    vals.append(st["score"] / math.sqrt(max(1, st["seen"])))
        if len(tokens) >= 3:
            for i in range(len(tokens) - 2):
                sid = self._engram_slot_id(tokens[i : i + 3], 20011)
                st = self.engram_slots_3g.get(sid)
                if st is not None:
                    vals.append(st["score"] / math.sqrt(max(1, st["seen"])))
        return sum(vals) / len(vals) if vals else 0.0

    def update(self, probe_delta, train_loss, peak_mem_mb):
        if self.mode == "off":
            return 1.0, 0.0
        tokens = self._context_tokens(probe_delta, train_loss, peak_mem_mb)
        reward = probe_delta
        if self.mode in ("both", "gradmem"):
            self._update_gradmem(tokens, reward)
        if self.mode in ("both", "engram"):
            self._update_engram(tokens, reward)
        g_score = self._score_gradmem(tokens) if self.mode in ("both", "gradmem") else 0.0
        e_score = self._score_engram(tokens) if self.mode in ("both", "engram") else 0.0
        score = g_score + e_score
        delta = max(-self.adapt_clamp, min(self.adapt_clamp, self.adapt_strength * math.tanh(score)))
        return 1.0 + delta, score


def evaluate_bpb_probe(model, token_bytes, val_loader, max_batches):
    total_nats = 0.0
    total_bytes = 0
    for _ in range(max_batches):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = mx.take(token_bytes, y_flat, axis=0)
        mask = nbytes > 0
        total_nats += mx.sum(loss_flat * mask).item()
        total_bytes += int(mx.sum(nbytes).item())
    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.use_packed_qkv = self.n_kv_head == self.n_head
        if self.use_packed_qkv:
            qkv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim
            self.c_qkv = nn.Linear(self.n_embd, qkv_dim, bias=False)
        else:
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
        x_norm = rms_norm(x)
        if self.use_packed_qkv:
            qkv = linear_forward(self.c_qkv, x_norm).reshape(
                batch_size,
                seq_len,
                self.n_head + 2 * self.n_kv_head,
                self.head_dim,
            )
            q = qkv[:, :, : self.n_head, :]
            k = qkv[:, :, self.n_head : self.n_head + self.n_kv_head, :]
            v = qkv[:, :, self.n_head + self.n_kv_head :, :]
        else:
            q = linear_forward(self.c_q, x_norm).reshape(batch_size, seq_len, self.n_head, self.head_dim)
            k = linear_forward(self.c_k, x_norm).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
            v = linear_forward(self.c_v, x_norm).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(linear_forward(self.ve_gate, x_norm[..., : self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = rms_norm(self.rope(q))
        k = rms_norm(self.rope(k))

        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return linear_forward(self.c_proj, y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = MLP_RATIO * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def __call__(self, x):
        x = rms_linear(self.c_fc, x)
        x = relu2(x)
        return linear_forward(self.c_proj, x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, mask):
        x = x + self.attn(x, ve, mask)
        x = x + self.mlp(x)
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = [
            nn.Embedding(config.vocab_size, kv_dim) if has_ve(i, config.n_layer) else None
            for i in range(config.n_layer)
        ]
        self._mask_cache = {}

    def init_weights(self):
        n_embd = self.config.n_embd
        scale = 3**0.5 * n_embd**-0.5

        self.wte.weight = (mx.random.normal(self.wte.weight.shape) * 1.0).astype(mx.bfloat16)
        self.lm_head.weight = (mx.random.normal(self.lm_head.weight.shape) * 0.001).astype(mx.bfloat16)

        for block in self.blocks:
            if block.attn.use_packed_qkv:
                block.attn.c_qkv.weight = mx.random.uniform(-scale, scale, block.attn.c_qkv.weight.shape).astype(mx.bfloat16)
            else:
                block.attn.c_q.weight = mx.random.uniform(-scale, scale, block.attn.c_q.weight.shape).astype(mx.bfloat16)
                block.attn.c_k.weight = mx.random.uniform(-scale, scale, block.attn.c_k.weight.shape).astype(mx.bfloat16)
                block.attn.c_v.weight = mx.random.uniform(-scale, scale, block.attn.c_v.weight.shape).astype(mx.bfloat16)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight).astype(mx.bfloat16)
            block.mlp.c_fc.weight = mx.random.uniform(-scale, scale, block.mlp.c_fc.weight.shape).astype(mx.bfloat16)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight).astype(mx.bfloat16)
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight).astype(mx.bfloat16)

        for ve in self.value_embeds:
            if ve is not None:
                ve.weight = mx.random.uniform(-scale, scale, ve.weight.shape).astype(mx.bfloat16)

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(char in "SL" for char in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = long_window
        return window_sizes

    def _get_masks(self, seq_len, dtype):
        unique_windows = set(self.window_sizes)
        for window_size in unique_windows:
            key = (seq_len, window_size, dtype)
            if key not in self._mask_cache:
                if window_size >= seq_len:
                    self._mask_cache[key] = create_additive_causal_mask(seq_len, dtype=dtype)
                else:
                    self._mask_cache[key] = create_sliding_window_mask(
                        seq_len,
                        window_size,
                        dtype=dtype,
                    )
        return [self._mask_cache[(seq_len, window_size, dtype)] for window_size in self.window_sizes]

    def __call__(self, idx, targets=None, reduction="mean"):
        _, seq_len = idx.shape
        x = self.wte(idx)
        masks = self._get_masks(seq_len, x.dtype)
        x = rms_norm(x)
        for i, block in enumerate(self.blocks):
            ve = self.value_embeds[i](idx) if self.value_embeds[i] is not None else None
            x = block(x, ve, masks[i])
        x = rms_norm(x)

        logits = linear_forward(self.lm_head, x).astype(mx.float32)

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


class AdamW:
    def __init__(self, model, unembedding_lr, embedding_lr, matrix_lr, weight_decay, adam_betas, adam_eps):
        self.param_config = {}
        self.adam_state = {}

        model_dim = model.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        flat_params = tree_flatten(model.parameters())
        for path, param in flat_params:
            if "blocks" in path and param.ndim == 2:
                self.param_config[path] = {
                    "lr": matrix_lr,
                    "betas": adam_betas,
                    "eps": adam_eps,
                    "weight_decay": weight_decay,
                }
            elif "wte" in path:
                self.param_config[path] = {
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": adam_eps,
                    "weight_decay": 0.0,
                }
            elif "value_embeds" in path:
                self.param_config[path] = {
                    "lr": embedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": adam_eps,
                    "weight_decay": 0.0,
                }
            elif "lm_head" in path:
                self.param_config[path] = {
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": adam_eps,
                    "weight_decay": 0.0,
                }
            else:
                self.param_config[path] = {
                    "lr": unembedding_lr * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": adam_eps,
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

    def _step(self, path, grad, param, config):
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

    def update(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))
        for path, grad in flat_grads.items():
            if path not in self.param_config:
                continue
            config = self.param_config[path]
            param = flat_params[path]
            new_param = self._step(path, grad, param, config)
            self._set_path_value(model, path, new_param)

    def set_lr_multiplier(self, multiplier):
        for path, config in self.param_config.items():
            config["lr"] = self.initial_lrs[path] * multiplier

    @property
    def state(self):
        arrays = []
        for state in self.adam_state.values():
            arrays.extend([state["m"], state["v"]])
        return arrays


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"
MLP_RATIO = 3
VE_EVERY_LAYER = True

# v0.1: AdamW only. Muon port is future work.
TOTAL_BATCH_SIZE = 2**15
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.005
WEIGHT_DECAY = 0.15
ADAM_BETAS = (0.8, 0.95)
ADAM_EPS = 1e-8
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.4
FINAL_LR_FRAC = 0.0

# Model size
DEPTH = 4
DEVICE_BATCH_SIZE = 16
FINAL_EVAL_BATCH_SIZE = 256
FINAL_EVAL_MICROBATCH_SIZE = DEVICE_BATCH_SIZE
LOG_EVERY_STEPS = 10
STARTUP_EXCLUDE_STEPS = 1

# In-training test iteration + feedback memory
ENABLE_INLOOP_TEST_FEEDBACK = False
MEMORY_MODE = "both"  # off | gradmem | engram | both
EPOCH_TEST_BATCH_SIZE = 32
EPOCH_TEST_BATCHES = 1
MEMORY_ADAPT_STRENGTH = 0.08
MEMORY_ADAPT_CLAMP = 0.20


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


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
)

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())
num_params = sum(param.size for _, param in tree_flatten(model.parameters()))

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = AdamW(
    model,
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    adam_betas=ADAM_BETAS,
    adam_eps=ADAM_EPS,
)

loss_grad_fn = nn.value_and_grad(model, lambda model, inputs, targets: model(inputs, targets=targets))

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

smooth_train_loss = 0.0
total_training_time = 0.0
step = 0
t_compiled = None
prev_epoch = epoch
probe_prev_bpb = None
memory_lr_scale = 1.0

if ENABLE_INLOOP_TEST_FEEDBACK:
    token_bytes = get_token_bytes()
    val_probe_loader = make_dataloader(tokenizer, EPOCH_TEST_BATCH_SIZE, MAX_SEQ_LEN, "val")
    feedback_memory = InLoopMemory(
        mode=MEMORY_MODE,
        adapt_strength=MEMORY_ADAPT_STRENGTH,
        adapt_clamp=MEMORY_ADAPT_CLAMP,
    )
    print(
        f"In-loop feedback enabled | mode={MEMORY_MODE} | "
        f"epoch_test_batches={EPOCH_TEST_BATCHES} | epoch_test_batch_size={EPOCH_TEST_BATCH_SIZE}"
    )

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
    effective_lrm = lrm * memory_lr_scale
    optimizer.set_lr_multiplier(effective_lrm)
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

    if step < 5 or (step + 1) % LOG_EVERY_STEPS == 0 or total_training_time >= TIME_BUDGET:
        print(
            f"step {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
            f"lrm: {effective_lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
            f"epoch: {epoch} | remaining: {remaining:.0f}s",
            flush=True,
        )

    # "Test iteration every epoch": run quick val probe and update feedback memory online.
    if ENABLE_INLOOP_TEST_FEEDBACK and epoch != prev_epoch:
        probe_bpb = evaluate_bpb_probe(model, token_bytes, val_probe_loader, EPOCH_TEST_BATCHES)
        probe_delta = 0.0 if probe_prev_bpb is None else (probe_prev_bpb - probe_bpb)
        peak_mem_mb = get_peak_memory_mb()
        memory_lr_scale, mem_score = feedback_memory.update(
            probe_delta=probe_delta,
            train_loss=debiased_smooth_loss,
            peak_mem_mb=peak_mem_mb,
        )
        probe_prev_bpb = probe_bpb
        print(
            f"\n[epoch-test] epoch={epoch} probe_bpb={probe_bpb:.6f} "
            f"delta={probe_delta:+.6f} mem_score={mem_score:+.4f} "
            f"next_lr_scale={memory_lr_scale:.3f}"
        )
        prev_epoch = epoch

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
print(f"Final eval microbatch size: {FINAL_EVAL_MICROBATCH_SIZE}")
val_bpb = evaluate_bpb(model, tokenizer, FINAL_EVAL_BATCH_SIZE, FINAL_EVAL_MICROBATCH_SIZE)
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
