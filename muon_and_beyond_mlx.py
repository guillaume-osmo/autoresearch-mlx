"""
muon_and_beyond_mlx.py
Unified MLX optimizers for "Muon and beyond":
- Muon (Newton–Schulz) + MuonW (Muon for 2D weights + AdamW for everything else)
- MuonV2 (Polar Express / Jordan + optional NorMuon) + MuonV2W
- PolarGrad / PolarMuon (timlautk/polargrad style) using Polar Express + nuclear-norm scaling
  + PolarGradW / PolarMuonW (2D weights via PolarGrad, rest via AdamW)
  
Best for nanoChat is Muonv2 -> Polar Express & NorMuon
"""

from __future__ import annotations

from typing import Union, Callable, Literal
import mlx.core as mx
import mlx.optimizers as optim

# =============================================================================
# Shared helpers
# =============================================================================

def _neuron_wise_l2_norm(U: mx.array, eps: float = 1e-8) -> mx.array:
    """NorMuon: L2-normalize each row of U (shape [out, in])."""
    row_norms = mx.sqrt(mx.sum(U * U, axis=1, keepdims=True) + eps)
    return U / row_norms


# =============================================================================
# Muon (classic): Newton–Schulz orthogonalization
# =============================================================================

def newton_schulz(G: mx.array, steps: int = 5, order: int = 5) -> mx.array:
    """Newton–Schulz iteration to orthogonalize matrix G (2D).
    Original formulation (matches muon_mlx / bird-of-paradise tutorial).
    """
    if G.ndim != 2:
        raise ValueError(f"newton_schulz expects 2D matrix, got shape {G.shape}")

    G_norm = mx.linalg.norm(G)
    X = G / (G_norm + 1e-7)

    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True

    if order == 3:
        for _ in range(steps):
            XTX = X @ X.T
            I = mx.eye(X.shape[0], dtype=X.dtype)
            X = X @ (1.5 * I - 0.5 * XTX)
    elif order == 5:
        for _ in range(steps):
            A = X.T @ X
            B = A @ A
            I = mx.eye(X.shape[1], dtype=X.dtype)
            X = X @ (3.4445 * I - 4.7750 * A + 2.0315 * B)
    else:
        raise ValueError(f"order must be 3 or 5, got {order}")

    if transpose:
        X = X.T

    return X * mx.sqrt(mx.array(max(G.shape[0], G.shape[1]), dtype=mx.float32))


class Muon(optim.Optimizer):
    """Muon optimizer: momentum + Newton–Schulz orthogonalization for 2D weight matrices."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_order: int = 5,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.ns_order = ns_order

    def init_single(self, parameter: mx.array, state: dict) -> dict:
        state["momentum"] = mx.zeros_like(parameter)
        return state

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        if "momentum" not in state:
            state["momentum"] = mx.zeros_like(parameter)

        m = state["momentum"]
        m = self.momentum * m + gradient
        state["momentum"] = m

        update = (gradient + self.momentum * m) if self.nesterov else m

        if parameter.ndim == 2:
            update = newton_schulz(update, steps=self.ns_steps, order=self.ns_order)

        lr = self.learning_rate.astype(gradient.dtype)
        return parameter - lr * update


def _mlp_2d_weight_filter(key: str, value: mx.array) -> bool:
    # Same filter you used: 2D weights, exclude embeddings
    return value.ndim == 2 and "weight" in key and "embed" not in key


def _mlp_muon_filter(_path, value: mx.array) -> bool:
    """Use MuonV2 for all 2D params (matches muon_v2_mlx)."""
    return value.ndim == 2


class MuonW(optim.MultiOptimizer):
    """
    Muon for 2D weights + AdamW for everything else (biases, LayerNorm, embeddings, ...)
    """

    def __init__(
        self,
        muon_lr: float = 0.02,
        adamw_lr: float = 1e-3,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        muon_ns_order: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.01,
    ):
        muon_opt = Muon(
            learning_rate=muon_lr,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            ns_order=muon_ns_order,
        )
        adamw_opt = optim.AdamW(
            learning_rate=adamw_lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=adamw_weight_decay,
            bias_correction=True,
        )
        super().__init__(optimizers=[muon_opt, adamw_opt], filters=[_mlp_2d_weight_filter])


# =============================================================================
# MuonV2: Polar Express / Jordan + optional NorMuon
# =============================================================================

_POLAR_EXPRESS_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]


def polar_express(
    G: mx.array,
    steps: int = 5,
    safety_factor: float = 1.01,
    eps: float = 1e-7,
    use_bf16: bool = False,
) -> mx.array:
    """
    Polar Express: matmul-only approximation of the polar factor.
    use_bf16=False (default) matches muon_v2_mlx for best results.
    """
    if G.ndim != 2:
        raise ValueError(f"polar_express expects 2D matrix, got shape {G.shape}")

    X = G.astype(mx.bfloat16) if use_bf16 else G
    norm_G = mx.linalg.norm(X) + eps
    X = X / (norm_G * safety_factor + eps)

    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True

    # Safety-adjusted coeffs except last
    coeffs = [
        (a / safety_factor, b / (safety_factor**3), c / (safety_factor**5))
        for (a, b, c) in _POLAR_EXPRESS_COEFFS[:-1]
    ] + [_POLAR_EXPRESS_COEFFS[-1]]

    while len(coeffs) < steps:
        coeffs.append(coeffs[-1])
    coeffs = coeffs[:steps]

    for (a, b, c) in coeffs:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transpose:
        X = X.T

    return X.astype(G.dtype)


def newton_schulz_jordan(G: mx.array, steps: int = 5, eps: float = 1e-7) -> mx.array:
    """Jordan-style iteration (same polynomial form)."""
    if G.ndim != 2:
        raise ValueError(f"newton_schulz_jordan expects 2D matrix, got shape {G.shape}")

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (mx.linalg.norm(G) + eps)

    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transpose:
        X = X.T
    return X


def _apply_polar(
    update: mx.array,
    polar_method: Literal["jordan", "polar_express"],
    steps: int,
    polar_express_safety: float,
) -> mx.array:
    if polar_method == "polar_express":
        return polar_express(update, steps=steps, safety_factor=polar_express_safety)
    return newton_schulz_jordan(update, steps=steps)


class MuonV2(optim.Optimizer):
    """
    MuonV2:
    - Momentum (optionally Nesterov)
    - Optional NorMuon (row-wise normalization)
    - Polar step via Polar Express or Jordan
    - Optional coupled weight decay (kept for parity with your existing muon_v2_mlx.py)
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        polar_method: Literal["jordan", "polar_express"] = "polar_express",
        polar_express_safety: float = 1.01,
        use_normuon: bool = False,
        normuon_eps: float = 1e-8,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.polar_method = polar_method
        self.polar_express_safety = polar_express_safety
        self.use_normuon = use_normuon
        self.normuon_eps = normuon_eps

    def init_single(self, parameter: mx.array, state: dict) -> dict:
        state["momentum"] = mx.zeros_like(parameter)
        return state

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        if "momentum" not in state:
            state["momentum"] = mx.zeros_like(parameter)

        m = state["momentum"]
        m = self.momentum * m + gradient
        state["momentum"] = m

        update = (gradient + self.momentum * m) if self.nesterov else m

        if parameter.ndim == 2:
            if self.use_normuon:
                update = _neuron_wise_l2_norm(update, eps=self.normuon_eps)
            update = _apply_polar(
                update,
                polar_method=self.polar_method,
                steps=self.ns_steps,
                polar_express_safety=self.polar_express_safety,
            )

        lr = self.learning_rate.astype(gradient.dtype)
        p = parameter - lr * update

        # Coupled weight decay (parity with your v2 file)
        if self.weight_decay != 0:
            p = p - lr * self.weight_decay * parameter

        return p


def MuonV2W(
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-3,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
    adamw_weight_decay: float = 0.01,
    muon_ns_steps: int = 5,
    muon_nesterov: bool = True,
    polar_method: Literal["jordan", "polar_express"] = "polar_express",
    polar_express_safety: float = 1.01,
    use_normuon: bool = False,
    adamw_betas: tuple[float, float] = (0.9, 0.999),
    adamw_eps: float = 1e-8,
) -> optim.MultiOptimizer:
    """
    MuonV2 for all 2D weights + AdamW for biases/LayerNorm.
    """
    muon_opt = MuonV2(
        learning_rate=muon_lr,
        momentum=muon_momentum,
        weight_decay=muon_weight_decay,
        nesterov=muon_nesterov,
        ns_steps=muon_ns_steps,
        polar_method=polar_method,
        polar_express_safety=polar_express_safety,
        use_normuon=use_normuon,
    )
    adamw_opt = optim.AdamW(
        learning_rate=adamw_lr,
        betas=adamw_betas,
        eps=adamw_eps,
        weight_decay=adamw_weight_decay,
    )
    return optim.MultiOptimizer(
        optimizers=[muon_opt, adamw_opt],
        filters=[_mlp_muon_filter],
    )


# =============================================================================
# PolarGrad / PolarMuon (Optimizer API; Option A)
# =============================================================================

def nuclear_norm_via_polar(X: mx.array, U: mx.array) -> mx.array:
    """Dual-norm identity: ||X||_* = <X, U> when U is the polar factor of X."""
    return mx.sum(X * U)


class PolarGrad(optim.Optimizer):
    """
    PolarGrad / PolarMuon (timlautk/polargrad style) implemented as an MLX Optimizer.

    Variants:
      polar_first=False (Muon-like):
        m <- beta*m + (1-beta)*g
        U <- polar(m)
        nuc <- <m, U>
        update <- nuc * U
      polar_first=True (polar-first):
        U <- polar(g)
        nuc <- <g, U>
        m <- beta*m + (1-beta)*U
        update <- nuc * m

    Notes:
    - For use as a drop-in optimizer in MLX, you typically apply it only to 2D weights
      via MultiOptimizer filters (see PolarGradW/PolarMuonW).
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        momentum: float = 0.95,
        weight_decay: float = 0.0,  # decoupled
        polar_first: bool = False,
        polar_steps: int = 2,
        polar_express_safety: float = 1.01,
        use_bf16: bool = True,
        eps: float = 1e-7,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.polar_first = polar_first
        self.polar_steps = polar_steps
        self.polar_express_safety = polar_express_safety
        self.use_bf16 = use_bf16
        self.eps = eps

    def init_single(self, parameter: mx.array, state: dict) -> dict:
        state["momentum"] = mx.zeros_like(parameter)
        return state

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        if "momentum" not in state:
            state["momentum"] = mx.zeros_like(parameter)

        m = state["momentum"]
        lr = self.learning_rate.astype(gradient.dtype)

        if self.polar_first:
            # U = polar(g); nuc = <g, U>; m <- ema(m, U); update = nuc * m
            U = polar_express(
                gradient,
                steps=self.polar_steps,
                safety_factor=self.polar_express_safety,
                eps=self.eps,
                use_bf16=self.use_bf16,
            ) if gradient.ndim == 2 else gradient
            nuc = nuclear_norm_via_polar(gradient.astype(U.dtype), U) if gradient.ndim == 2 else mx.array(1.0, dtype=gradient.dtype)
            m = self.momentum * m + (1.0 - self.momentum) * U
            update = nuc * m
        else:
            # m <- ema(m, g); U = polar(m); nuc = <m, U>; update = nuc * U
            m = self.momentum * m + (1.0 - self.momentum) * gradient
            U = polar_express(
                m,
                steps=self.polar_steps,
                safety_factor=self.polar_express_safety,
                eps=self.eps,
                use_bf16=self.use_bf16,
            ) if m.ndim == 2 else m
            nuc = nuclear_norm_via_polar(m.astype(U.dtype), U) if m.ndim == 2 else mx.array(1.0, dtype=m.dtype)
            update = nuc * U

        state["momentum"] = m

        # decoupled weight decay (applied to parameter directly)
        if self.weight_decay != 0:
            parameter = parameter * (1.0 - lr * self.weight_decay)

        return parameter - lr * update


def PolarGradW(
    polar_lr: float = 0.02,
    adamw_lr: float = 1e-3,
    polar_momentum: float = 0.95,
    polar_weight_decay: float = 0.0,     # decoupled
    adamw_weight_decay: float = 0.01,
    polar_first: bool = False,
    polar_steps: int = 2,
    polar_express_safety: float = 1.01,
    use_bf16: bool = True,
    adamw_betas: tuple[float, float] = (0.9, 0.999),
    adamw_eps: float = 1e-8,
) -> optim.MultiOptimizer:
    """
    PolarGrad on 2D weights + AdamW on everything else.
    """
    polar_opt = PolarGrad(
        learning_rate=polar_lr,
        momentum=polar_momentum,
        weight_decay=polar_weight_decay,
        polar_first=polar_first,
        polar_steps=polar_steps,
        polar_express_safety=polar_express_safety,
        use_bf16=use_bf16,
    )
    adamw_opt = optim.AdamW(
        learning_rate=adamw_lr,
        betas=adamw_betas,
        eps=adamw_eps,
        weight_decay=adamw_weight_decay,
    )
    return optim.MultiOptimizer(
        optimizers=[polar_opt, adamw_opt],
        filters=[_mlp_2d_weight_filter],
    )


def PolarMuonW(**kwargs) -> optim.MultiOptimizer:
    """
    Alias for PolarGradW with polar_first=False (Muon-like), matching common naming.
    """
    kwargs = dict(kwargs)
    kwargs["polar_first"] = False
    return PolarGradW(**kwargs)
