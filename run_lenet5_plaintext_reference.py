"""
Plaintext reference for LeNet-5 with fixed-point arithmetic and MPC-matching approximations.

Validates correctness and numerical stability independently of MPC:
- Same Q-format (SCALE=1000), same scaling, same truncation (round-to-nearest).
- Same softmax/exp/log approximations as secure_softmax.py (Taylor exp, atanh log, mean-subtract softmax).

If this converges but the MPC version does not → problem is likely in MPC protocol/sync/communication.
If this already fails in plaintext → issue is numerical (scaling, truncation, or approximation error).

Defaults (lr, temperature, init-gain) match run_lenet5_training_full.py for fair MPC vs plaintext comparison.

Usage:
  # 80%%+ accuracy: float32 defaults to 150x150 RGB (Kather-native); use --grayscale or --input-size 32 to change
  python run_lenet5_plaintext_reference.py --float32 --dataset Kather_texture_2016_image_tiles_5000 --epochs 50
  # Faster: install scipy; use --batch-size 128 --acc-every 2
  python run_lenet5_plaintext_reference.py --float32 --dataset Kather_texture_2016_image_tiles_5000 --epochs 50 --batch-size 128 --acc-every 2
  python run_lenet5_plaintext_reference.py --fast --dataset Kather_texture_2016_image_tiles_5000 --epochs 20   # fixed-point (MPC comparison)
  python run_lenet5_plaintext_reference.py --fc3-only --epochs 5   # legacy: only last layer (low acc)
  python run_lenet5_plaintext_reference.py --quick-test --epochs 5   # 16 samples, fast check
  python run_lenet5_plaintext_reference.py --investigate   # print logit/prob/grad stats
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_training.image_loader import load_kather_dataset, split_dataset

# ---------------------------------------------------------------------------
# Fixed-point constants (match MPC: beaver_triples.py, secure_softmax.py)
# ---------------------------------------------------------------------------
SCALE = 1000
FIELD_SIZE = 2**32 - 5  # Not used for modulo in plaintext; only for value range reference.
# Clamp weight updates so params stay in int64 range (avoid OverflowError in --fast mode).
MAX_DELTA = 2**62
# Minimum FC1/FC2 bias (SCALE units) so ReLU pre-activations stay positive after updates (avoid dead h1/h2).
MIN_FC_BIAS = 30


def round_div(value: int, divisor: int) -> int:
    """Integer division with round-to-nearest (matches MPC opener truncation)."""
    if divisor == 0:
        raise ValueError("divisor must be non-zero")
    adj = (divisor // 2) if value >= 0 else -(divisor // 2)
    return (value + adj) // divisor


def fp_mul(a: int, b: int) -> int:
    """Fixed-point multiply: (a * b) / SCALE, round-to-nearest. Uses Python int to avoid overflow."""
    a, b = int(a), int(b)
    prod = a * b
    adj = (SCALE // 2) if prod >= 0 else -(SCALE // 2)
    return (prod + adj) // SCALE


def _round_scaled(prod: np.ndarray) -> np.ndarray:
    """Round product array by SCALE (for vectorized path). prod in int64."""
    adj = np.where(prod >= 0, SCALE // 2, -(SCALE // 2))
    return ((prod + adj) // SCALE).astype(np.int64)


# ---------------------------------------------------------------------------
# Exp approximation (same as secure_softmax._exp_poly: Taylor 5th order)
# ---------------------------------------------------------------------------
def exp_poly(x: int) -> int:
    """exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120. Input and output SCALE-scaled."""
    one = SCALE
    x2 = fp_mul(x, x)
    x3 = fp_mul(x2, x)
    x4 = fp_mul(x3, x)
    x5 = fp_mul(x4, x)
    t2 = round_div(x2, 2)
    t3 = round_div(x3, 6)
    t4 = round_div(x4, 24)
    t5 = round_div(x5, 120)
    out = one + x + t2 + t3 + t4 + t5
    return out


# ---------------------------------------------------------------------------
# Log approximation (same as secure_softmax: ln(x) = 2*atanh((x-1)/(x+1)), atanh series)
# ---------------------------------------------------------------------------
def atanh_series(y: int, terms: int = 6) -> int:
    """atanh(y) ≈ y + y³/3 + y⁵/5 + ... (y is SCALE-scaled, output SCALE-scaled)."""
    out = y
    y2 = fp_mul(y, y)
    ypow = y
    for k in range(1, terms):
        ypow = fp_mul(ypow, y2)
        denom = 2 * k + 1
        out += round_div(ypow, denom)
    return out


def log_fp(x: int) -> int:
    """ln(x) for x SCALE-scaled (x in (0, SCALE]). Uses 2*atanh((x-1)/(x+1))."""
    if x <= 0:
        return -10 * SCALE  # clamp for safety
    one = SCALE
    num = x - one
    den = x + one
    if den == 0:
        return -10 * SCALE
    # y = (x-1)/(x+1) in fixed-point: y_int = round(num * SCALE / den)
    y = round_div(num * SCALE, den)
    at = atanh_series(y, terms=6)
    return 2 * at


# ---------------------------------------------------------------------------
# Softmax (same as secure_softmax: temperature, subtract mean, exp, normalize)
# ---------------------------------------------------------------------------
def softmax_fp(logits: List[int], temperature: int) -> List[int]:
    """Softmax with temperature. Logits and output SCALE-scaled."""
    if not logits:
        return []
    # logits := logits / T
    scaled = [round_div(li, temperature) for li in logits]
    # Subtract mean for stability
    mean = sum(scaled) // len(scaled)
    centered = [s - mean for s in scaled]
    # exp and sum
    exps = [exp_poly(c) for c in centered]
    sumexp = sum(exps)
    if sumexp == 0:
        return [SCALE // len(logits)] * len(logits)
    # probs = exp / sumexp (output SCALE-scaled)
    probs = [round_div(e * SCALE, sumexp) for e in exps]
    return probs


def cross_entropy_fp(logits: List[int], target_onehot: List[int], temperature: int) -> Tuple[int, List[int]]:
    """CE = -sum_i y_i*log(p_i). Returns (loss_scaled_int, probs). Gradient = (p - y)/T."""
    probs = softmax_fp(logits, temperature)
    loss_int = 0
    for yi, pi in zip(target_onehot, probs):
        log_pi = log_fp(pi)
        loss_int += fp_mul(yi, log_pi)
    ce_scaled = -loss_int  # SCALE²-scaled for display: ce_real = ce_scaled / SCALE²
    return ce_scaled, probs


# ---------------------------------------------------------------------------
# LeNet-5 architecture (same as lenet5.py)
# ---------------------------------------------------------------------------
INPUT_H, INPUT_W, INPUT_C = 32, 32, 1
CONV1_SHAPE = (5, 5, 1, 6)
CONV2_SHAPE = (5, 5, 6, 16)
FC1_SHAPE = (120, 400)
FC2_SHAPE = (84, 120)
FC3_SHAPE = (8, 84)
NUM_CLASSES = 8


def conv2d_fp(x: np.ndarray, w: np.ndarray, fast: bool = False) -> np.ndarray:
    """x (H,W,Cin), w (Kh,Kw,Cin,Cout). Output (H-Kh+1, W-Kw+1, Cout). All int, SCALE-scaled."""
    from numpy.lib.stride_tricks import as_strided
    H, W, Cin = x.shape
    Kh, Kw, _, Cout = w.shape
    outH, outW = H - Kh + 1, W - Kw + 1
    patches = as_strided(
        x, shape=(outH, outW, Kh, Kw, Cin),
        strides=(x.strides[0], x.strides[1], x.strides[0], x.strides[1], x.strides[2]),
        writeable=False,
    )
    if fast:
        # Vectorized: one round per output (faster; numerics differ slightly from per-term truncation)
        P = patches.reshape(outH * outW, Kh * Kw * Cin).astype(np.int64)
        Wflat = w.reshape(Kh * Kw * Cin, Cout).astype(np.int64)
        out = _round_scaled(P @ Wflat).reshape(outH, outW, Cout)
        return out
    out = np.zeros((outH, outW, Cout), dtype=np.int64)
    for oh in range(outH):
        for ow in range(outW):
            for co in range(Cout):
                s = 0
                for kh in range(Kh):
                    for kw in range(Kw):
                        for ci in range(Cin):
                            s += fp_mul(x[oh + kh, ow + kw, ci], w[kh, kw, ci, co])
                out[oh, ow, co] = s
    return out


def max_pool2x2_fp(x: np.ndarray) -> np.ndarray:
    """x (H,W,C), H,W even. Output (H/2, W/2, C)."""
    H, W, C = x.shape
    return x.reshape(H // 2, 2, W // 2, 2, C).max(axis=(1, 3))


def max_pool2x2_backward_fp(grad_out: np.ndarray, input_pre_pool: np.ndarray) -> np.ndarray:
    """grad_out (H/2, W/2, C), input_pre_pool (H, W, C). Gradient goes only to max position in each 2x2 window."""
    H, W, C = input_pre_pool.shape
    grad_in = np.zeros_like(input_pre_pool, dtype=np.int64)
    view = input_pre_pool.reshape(H // 2, 2, W // 2, 2, C)
    for i in range(H // 2):
        for j in range(W // 2):
            for c in range(C):
                window = view[i, :, j, :, c]
                flat_idx = int(np.argmax(window))
                hi, wi = flat_idx // 2, flat_idx % 2
                grad_in[2 * i + hi, 2 * j + wi, c] = grad_out[i, j, c]
    return grad_in


# int64 range: avoid np.clip (float64 can't represent 2**63-1 exactly, so use int comparison).
_INT64_MAX = 2**63 - 1
_INT64_MIN = -(2**63)
# Safe float64 bounds for clipping before .astype(np.int64): float(2**63-1) rounds to 2**63 and overflows.
_F64_SAFE_MAX = float(2**63 - 2**10)
_F64_SAFE_MIN = float(-(2**63))


def _clip_int64(val: int) -> int:
    """Clamp to int64 range so numpy assignment doesn't overflow."""
    val = int(val)
    if val > _INT64_MAX:
        return _INT64_MAX
    if val < _INT64_MIN:
        return _INT64_MIN
    return val


def _conv2d_backward_fast(
    grad_out: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized conv backward: grad_w via patches.T @ grad_out; grad_in via scatter. Float64 then round."""
    from numpy.lib.stride_tricks import as_strided
    H, W, Cin = x.shape
    Kh, Kw, _, Cout = w.shape
    outH, outW = H - Kh + 1, W - Kw + 1
    patches = as_strided(
        x, shape=(outH, outW, Kh, Kw, Cin),
        strides=(x.strides[0], x.strides[1], x.strides[0], x.strides[1], x.strides[2]),
        writeable=False,
    )
    P = patches.reshape(outH * outW, Kh * Kw * Cin).astype(np.float64)
    G = grad_out.reshape(outH * outW, Cout).astype(np.float64)
    raw_w = np.nan_to_num((P.T @ G) / SCALE, nan=0.0, posinf=_F64_SAFE_MAX, neginf=_F64_SAFE_MIN)
    raw_w = np.clip(raw_w, _F64_SAFE_MIN, _F64_SAFE_MAX)
    grad_w = np.round(raw_w).astype(np.int64).reshape(Kh, Kw, Cin, Cout)
    grad_w = np.minimum(np.maximum(grad_w, _INT64_MIN), _INT64_MAX)
    grad_in = np.zeros((H, W, Cin), dtype=np.float64)
    for oh in range(outH):
        for ow in range(outW):
            for co in range(Cout):
                grad_in[oh : oh + Kh, ow : ow + Kw, :] += (
                    grad_out[oh, ow, co].astype(np.float64) * w[:, :, :, co].astype(np.float64) / SCALE
                )
    grad_in = np.nan_to_num(grad_in, nan=0.0, posinf=_F64_SAFE_MAX, neginf=_F64_SAFE_MIN)
    grad_in = np.clip(grad_in, _F64_SAFE_MIN, _F64_SAFE_MAX)
    grad_in = np.round(grad_in).astype(np.int64)
    grad_in = np.minimum(np.maximum(grad_in, _INT64_MIN), _INT64_MAX)
    return grad_in, grad_w


def conv2d_backward_fp(
    grad_out: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    fast: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """grad_out (outH, outW, Cout), x (H, W, Cin), w (Kh, Kw, Cin, Cout). Returns (grad_input, grad_kernel)."""
    if fast:
        return _conv2d_backward_fast(grad_out, x, w)
    H, W, Cin = x.shape
    Kh, Kw, _, Cout = w.shape
    outH, outW = H - Kh + 1, W - Kw + 1
    grad_in = np.zeros((H, W, Cin), dtype=np.int64)
    grad_w = np.zeros_like(w, dtype=np.int64)
    for oh in range(outH):
        for ow in range(outW):
            for co in range(Cout):
                g = int(grad_out[oh, ow, co])
                for kh in range(Kh):
                    for kw in range(Kw):
                        for ci in range(Cin):
                            v_in = _clip_int64(fp_mul(g, int(w[kh, kw, ci, co])))
                            v_w = _clip_int64(fp_mul(g, int(x[oh + kh, ow + kw, ci])))
                            grad_in[oh + kh, ow + kw, ci] = _clip_int64(
                                int(grad_in[oh + kh, ow + kw, ci]) + v_in
                            )
                            grad_w[kh, kw, ci, co] = _clip_int64(
                                int(grad_w[kh, kw, ci, co]) + v_w
                            )
    return grad_in.astype(np.int64), grad_w.astype(np.int64)


def relu_fp(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0).astype(x.dtype)


def forward_lenet5_fp(
    image: np.ndarray,
    conv1_w: np.ndarray,
    conv2_w: np.ndarray,
    fc1_w: np.ndarray,
    fc2_w: np.ndarray,
    fc3_w: np.ndarray,
    fc1_b: np.ndarray,
    fc2_b: np.ndarray,
    fc3_b: np.ndarray,
    fast: bool = False,
) -> Tuple[List[int], Dict]:
    """Single sample forward. image float [0,1] (H,W,C). Weights int SCALE-scaled. Returns logits [8] int, intermediates."""
    x = (image * SCALE).astype(np.int64)
    if x.ndim == 2:
        x = x[:, :, np.newaxis]
    c1 = conv2d_fp(x, conv1_w, fast=fast)
    c1 = relu_fp(c1)
    p1 = max_pool2x2_fp(c1)
    c2 = conv2d_fp(p1, conv2_w, fast=fast)
    c2 = relu_fp(c2)
    p2 = max_pool2x2_fp(c2)
    flat = p2.reshape(-1)
    if fast:
        h1 = _round_scaled(fc1_w.astype(np.int64) @ flat.astype(np.int64)) + fc1_b
        h1 = relu_fp(h1)
        h2 = _round_scaled(fc2_w.astype(np.int64) @ h1.astype(np.int64)) + fc2_b
        h2 = relu_fp(h2)
        logits = _round_scaled(fc3_w.astype(np.int64) @ h2.astype(np.int64)) + fc3_b
    else:
        h1 = np.zeros(120, dtype=np.int64)
        for i in range(120):
            for j in range(400):
                h1[i] += fp_mul(int(fc1_w[i, j]), int(flat[j]))
            h1[i] += fc1_b[i]
        h1 = relu_fp(h1)
        h2 = np.zeros(84, dtype=np.int64)
        for i in range(84):
            for j in range(120):
                h2[i] += fp_mul(int(fc2_w[i, j]), int(h1[j]))
            h2[i] += fc2_b[i]
        h2 = relu_fp(h2)
        logits = np.zeros(8, dtype=np.int64)
        for i in range(8):
            for j in range(84):
                logits[i] += fp_mul(int(fc3_w[i, j]), int(h2[j]))
            logits[i] += fc3_b[i]
    logits_list = [int(logits[i]) for i in range(NUM_CLASSES)]
    inter = {"flat": flat, "h1": h1, "h2": h2, "c1": c1, "p1": p1, "c2": c2, "p2": p2, "x": x}
    return logits_list, inter


def backward_ce_fp(
    probs: List[int],
    target_onehot: List[int],
    temperature: int,
) -> List[int]:
    """Gradient of CE w.r.t. logits: (p - y) / T. Returns grad_logits [8] SCALE-scaled."""
    grad = [round_div(probs[i] - target_onehot[i], temperature) for i in range(NUM_CLASSES)]
    return grad


def ce_loss_display(ce_scaled: int) -> float:
    """Convert scaled CE to display value. ce_scaled = -sum(y*log_p); CE_real = -sum(y*ln(p)) = ce_scaled/SCALE."""
    return ce_scaled / SCALE


# ---------------------------------------------------------------------------
# Weight init (same as lenet5.initialize_weights: He scaled by init_gain, then * SCALE)
# ---------------------------------------------------------------------------
def init_weights(seed: int, init_gain: float) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params = {}

    # Conv1: 5,5,1,6
    fan_in = 5 * 5 * 1
    std = np.sqrt(4.0 / fan_in) * init_gain
    w = (rng.standard_normal((5, 5, 1, 6)) * std * SCALE).astype(np.int64)
    params["conv1_w"] = w
    params["accum_conv1_w"] = np.zeros((5, 5, 1, 6), dtype=np.float64)

    # Conv2: 5,5,6,16
    fan_in = 5 * 5 * 6
    std = np.sqrt(4.0 / fan_in) * init_gain
    w = (rng.standard_normal((5, 5, 6, 16)) * std * SCALE).astype(np.int64)
    params["conv2_w"] = w
    params["accum_conv2_w"] = np.zeros((5, 5, 6, 16), dtype=np.float64)

    # FC1: 120, 400. Small positive bias so ReLU doesn't kill all activations (avoids zero h1/h2 -> zero gradients).
    fan_in = 400
    std = np.sqrt(6.0 / fan_in) * init_gain
    params["fc1_w"] = (rng.standard_normal((120, 400)) * std * SCALE).astype(np.int64)
    params["fc1_b"] = np.full(120, 150, dtype=np.int64)  # 0.15 in real; keep pre-activation positive after updates

    # FC2: 84, 120
    fan_in = 120
    std = np.sqrt(4.0 / fan_in) * init_gain
    params["fc2_w"] = (rng.standard_normal((84, 120)) * std * SCALE).astype(np.int64)
    params["fc2_b"] = np.full(84, 150, dtype=np.int64)  # 0.15 in real; keep h2 > 0 after epoch updates

    # FC3: 8, 84
    fan_in = 84
    std = np.sqrt(2.0 / fan_in) * init_gain
    params["fc3_w"] = (rng.standard_normal((8, 84)) * std * SCALE).astype(np.int64)
    params["fc3_b"] = np.zeros(8, dtype=np.int64)
    # Fractional accumulators so small updates aren't lost to rounding
    params["accum_fc3_w"] = np.zeros((8, 84), dtype=np.float64)
    params["accum_fc3_b"] = np.zeros(8, dtype=np.float64)
    params["accum_fc2_w"] = np.zeros((84, 120), dtype=np.float64)
    params["accum_fc2_b"] = np.zeros(84, dtype=np.float64)
    params["accum_fc1_w"] = np.zeros((120, 400), dtype=np.float64)
    params["accum_fc1_b"] = np.zeros(120, dtype=np.float64)

    return params


# ---------------------------------------------------------------------------
# Backprop through conv+pool: grad_flat -> pool2 -> ReLU -> conv2 -> pool1 -> ReLU -> conv1.
# ---------------------------------------------------------------------------
def backward_conv_fp(
    grad_flat: np.ndarray,
    inter: Dict,
    params: Dict[str, np.ndarray],
    fast: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute grad w.r.t. conv1_w and conv2_w from grad_flat. Returns (grad_conv1_w, grad_conv2_w)."""
    c1 = inter["c1"]
    c2 = inter["c2"]
    p1 = inter["p1"]
    x = inter["x"]
    conv1_w = params["conv1_w"]
    conv2_w = params["conv2_w"]
    grad_p2 = grad_flat.reshape(5, 5, 16).astype(np.int64)
    grad_c2 = max_pool2x2_backward_fp(grad_p2, c2)
    grad_c2_pre = np.where(c2 > 0, grad_c2, 0).astype(np.int64)
    grad_p1, grad_conv2_w = conv2d_backward_fp(grad_c2_pre, p1, conv2_w, fast=fast)
    grad_c1 = max_pool2x2_backward_fp(grad_p1, c1)
    grad_c1_pre = np.where(c1 > 0, grad_c1, 0).astype(np.int64)
    _, grad_conv1_w = conv2d_backward_fp(grad_c1_pre, x, conv1_w, fast=fast)
    return grad_conv1_w, grad_conv2_w


# ---------------------------------------------------------------------------
# Backprop through FC layers (grad_logits -> FC3 -> ReLU -> FC2 -> ReLU -> FC1).
# Vectorized with float64 matmul + round for speed (used when fast=True).
# ---------------------------------------------------------------------------
def _fc_backward_vectorized(
    grad_logits: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    fc1_w: np.ndarray,
    fc2_w: np.ndarray,
    fc3_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized FC backward: grad = W.T @ g, round( / SCALE ), clip. Returns (grad_h2_pre, grad_h1_pre, grad_flat)."""
    g = np.asarray(grad_logits, dtype=np.float64)
    # Use safe float bounds so .astype(np.int64) never overflows (float64 can't represent 2**63-1).
    fmin, fmax = _F64_SAFE_MIN, _F64_SAFE_MAX
    raw_h2 = np.nan_to_num((fc3_w.astype(np.float64).T @ g) / SCALE, nan=0.0, posinf=fmax, neginf=fmin)
    raw_h2 = np.clip(raw_h2, fmin, fmax)
    grad_h2 = np.round(raw_h2).astype(np.int64)
    grad_h2 = np.minimum(np.maximum(grad_h2, _INT64_MIN), _INT64_MAX)
    grad_h2_pre = np.where(h2 > 0, grad_h2, 0).astype(np.int64)
    raw_h1 = np.nan_to_num((fc2_w.astype(np.float64).T @ grad_h2_pre.astype(np.float64)) / SCALE, nan=0.0, posinf=fmax, neginf=fmin)
    raw_h1 = np.clip(raw_h1, fmin, fmax)
    grad_h1 = np.round(raw_h1).astype(np.int64)
    grad_h1 = np.minimum(np.maximum(grad_h1, _INT64_MIN), _INT64_MAX)
    grad_h1_pre = np.where(h1 > 0, grad_h1, 0).astype(np.int64)
    raw_flat = np.nan_to_num((fc1_w.astype(np.float64).T @ grad_h1_pre.astype(np.float64)) / SCALE, nan=0.0, posinf=fmax, neginf=fmin)
    raw_flat = np.clip(raw_flat, fmin, fmax)
    grad_flat = np.round(raw_flat).astype(np.int64)
    grad_flat = np.minimum(np.maximum(grad_flat, _INT64_MIN), _INT64_MAX)
    return grad_h2_pre, grad_h1_pre, grad_flat


def backward_fc_fp(
    grad_logits: List[int],
    inter: Dict,
    params: Dict[str, np.ndarray],
    fast: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute grad w.r.t. h2, h1, flat from grad_logits. Returns (grad_h2_pre_relu, grad_h1_pre_relu, grad_flat)."""
    h1 = inter["h1"]
    h2 = inter["h2"]
    fc1_w = params["fc1_w"]
    fc2_w = params["fc2_w"]
    fc3_w = params["fc3_w"]
    if fast:
        return _fc_backward_vectorized(
            np.array(grad_logits, dtype=np.float64), h1, h2, fc1_w, fc2_w, fc3_w
        )
    grad_h2 = np.zeros(84, dtype=np.int64)
    for j in range(84):
        s = 0
        for i in range(8):
            s += round_div(int(grad_logits[i]) * int(fc3_w[i, j]), SCALE)
        grad_h2[j] = _clip_int64(s)
    grad_h2_pre = np.where(h2 > 0, grad_h2, 0).astype(np.int64)
    grad_h1 = np.zeros(120, dtype=np.int64)
    for j in range(120):
        s = 0
        for i in range(84):
            s += round_div(int(grad_h2_pre[i]) * int(fc2_w[i, j]), SCALE)
        grad_h1[j] = _clip_int64(s)
    grad_h1_pre = np.where(h1 > 0, grad_h1, 0).astype(np.int64)
    grad_flat = np.zeros(400, dtype=np.int64)
    for j in range(400):
        s = 0
        for i in range(120):
            s += round_div(int(grad_h1_pre[i]) * int(fc1_w[i, j]), SCALE)
        grad_flat[j] = _clip_int64(s)
    return grad_h2_pre, grad_h1_pre, grad_flat


# ---------------------------------------------------------------------------
# Training step: full backprop (conv1, conv2, FC1, FC2, FC3) by default. Use --fc3-only for last layer only.
# ---------------------------------------------------------------------------
def _apply_accum_updates(
    params: Dict[str, np.ndarray],
    fc3_only: bool,
    conv_lr_scale: float,
    fc12_lr_scale: float,
    fast: bool,
    verbose: bool = False,
) -> None:
    """Apply accumulated float updates as integer deltas (round, clip, subtract)."""
    delta_fc3 = np.clip(np.round(params["accum_fc3_w"]), -MAX_DELTA, MAX_DELTA).astype(np.int64)
    if verbose:
        print(f"    [diagnostic] |fc3_w update| max={np.max(np.abs(delta_fc3))}  non-zero={np.count_nonzero(delta_fc3)}/{delta_fc3.size}", flush=True)
    delta = delta_fc3
    params["fc3_w"] -= delta
    params["accum_fc3_w"] -= delta
    delta_b = np.clip(np.round(params["accum_fc3_b"]), -MAX_DELTA, MAX_DELTA).astype(np.int64)
    params["fc3_b"] -= delta_b
    params["accum_fc3_b"] -= delta_b
    if not fc3_only:
        delta = np.clip(np.round(params["accum_fc2_w"]), -MAX_DELTA, MAX_DELTA).astype(np.int64)
        params["fc2_w"] -= delta
        params["accum_fc2_w"] -= delta
        delta_b = np.clip(np.round(params["accum_fc2_b"]), -MAX_DELTA, MAX_DELTA).astype(np.int64)
        params["fc2_b"] -= delta_b
        params["fc2_b"] = np.maximum(params["fc2_b"], MIN_FC_BIAS)  # keep h2 non-dead
        params["accum_fc2_b"] -= delta_b
        delta = np.clip(np.round(params["accum_fc1_w"]), -MAX_DELTA, MAX_DELTA).astype(np.int64)
        params["fc1_w"] -= delta
        params["accum_fc1_w"] -= delta
        delta_b = np.clip(np.round(params["accum_fc1_b"]), -MAX_DELTA, MAX_DELTA).astype(np.int64)
        params["fc1_b"] -= delta_b
        params["fc1_b"] = np.maximum(params["fc1_b"], MIN_FC_BIAS)  # keep h1 non-dead
        params["accum_fc1_b"] -= delta_b
        delta = np.clip(np.round(params["accum_conv1_w"]), -MAX_DELTA, MAX_DELTA).astype(np.int64)
        params["conv1_w"] -= delta
        params["accum_conv1_w"] -= delta
        delta = np.clip(np.round(params["accum_conv2_w"]), -MAX_DELTA, MAX_DELTA).astype(np.int64)
        params["conv2_w"] -= delta
        params["accum_conv2_w"] -= delta


def train_step_fp(
    images: List[np.ndarray],
    labels: List[int],
    params: Dict[str, np.ndarray],
    temperature: int,
    lr: float,
    progress_interval: int = 500,
    fc3_only: bool = False,
    conv_lr_scale: float = 20.0,
    fc12_lr_scale: float = 10.0,
    fast: bool = False,
    batch_size: int = 1,
    grad_scale: float = 1.0,
    verbose: bool = False,
) -> float:
    """One pass: forward, CE, full backprop. If batch_size>1, accumulate gradients then update every batch_size samples."""
    total_ce = 0
    n = 0
    first_apply = True
    # Diagnostic for first batch when verbose: why are updates zero?
    diag_h2_sum, diag_h2_nnz, diag_g_sum, diag_add_sum, diag_n = 0.0, 0, 0.0, 0.0, 0
    for idx, (img, label) in enumerate(zip(images, labels)):
        if progress_interval and n > 0 and n % progress_interval == 0:
            print(f"    train step {n}/{len(images)}", flush=True)
        onehot = [SCALE if i == label else 0 for i in range(NUM_CLASSES)]
        logits, inter = forward_lenet5_fp(
            img,
            params["conv1_w"], params["conv2_w"],
            params["fc1_w"], params["fc2_w"], params["fc3_w"],
            params["fc1_b"], params["fc2_b"], params["fc3_b"],
            fast=fast,
        )
        ce_scaled, probs = cross_entropy_fp(logits, onehot, temperature)
        total_ce += ce_scaled
        n += 1
        grad_logits = backward_ce_fp(probs, onehot, temperature)
        h2 = inter["h2"]
        h1 = inter["h1"]
        flat = inter["flat"]

        g = np.asarray(grad_logits, dtype=np.float64)
        if verbose and idx < batch_size:
            h2_f = h2.astype(np.float64)
            add_ij = (grad_scale * lr * np.outer(g, h2_f) / SCALE)
            diag_h2_sum += np.sum(np.abs(h2_f))
            diag_h2_nnz += int(np.count_nonzero(h2))
            diag_g_sum += np.sum(np.abs(g))
            diag_add_sum += np.sum(np.abs(add_ij))
            diag_n += 1
        if fast:
            # Accumulate gradients (apply integer update every batch_size steps)
            params["accum_fc3_w"] += grad_scale * lr * np.outer(g, h2) / SCALE
            params["accum_fc3_b"] += grad_scale * lr * g
        else:
            for i in range(8):
                for j in range(84):
                    params["accum_fc3_w"][i, j] += grad_scale * lr * grad_logits[i] * h2[j] / SCALE
                params["accum_fc3_b"][i] += grad_scale * lr * grad_logits[i]

        if not fc3_only:
            grad_h2_pre, grad_h1_pre, grad_flat = backward_fc_fp(
                [int(g) for g in grad_logits], inter, params, fast=fast
            )
            lr_fc12 = grad_scale * lr * fc12_lr_scale
            if fast:
                params["accum_fc2_w"] += lr_fc12 * np.outer(grad_h2_pre, h1) / SCALE
                params["accum_fc2_b"] += lr_fc12 * grad_h2_pre
                params["accum_fc1_w"] += lr_fc12 * np.outer(grad_h1_pre, flat) / SCALE
                params["accum_fc1_b"] += lr_fc12 * grad_h1_pre
                grad_conv1_w, grad_conv2_w = backward_conv_fp(grad_flat, inter, params, fast=fast)
                lr_conv = grad_scale * lr * conv_lr_scale
                params["accum_conv1_w"] += lr_conv * grad_conv1_w
                params["accum_conv2_w"] += lr_conv * grad_conv2_w
            else:
                for i in range(84):
                    for j in range(120):
                        params["accum_fc2_w"][i, j] += lr_fc12 * grad_h2_pre[i] * h1[j] / SCALE
                    params["accum_fc2_b"][i] += lr_fc12 * grad_h2_pre[i]
                for i in range(120):
                    for j in range(400):
                        params["accum_fc1_w"][i, j] += lr_fc12 * grad_h1_pre[i] * flat[j] / SCALE
                    params["accum_fc1_b"][i] += lr_fc12 * grad_h1_pre[i]
                grad_conv1_w, grad_conv2_w = backward_conv_fp(grad_flat, inter, params, fast=fast)
                lr_conv = grad_scale * lr * conv_lr_scale
                for kh in range(5):
                    for kw in range(5):
                        for ci in range(1):
                            for co in range(6):
                                params["accum_conv1_w"][kh, kw, ci, co] += lr_conv * grad_conv1_w[kh, kw, ci, co]
                for kh in range(5):
                    for kw in range(5):
                        for ci in range(6):
                            for co in range(16):
                                params["accum_conv2_w"][kh, kw, ci, co] += lr_conv * grad_conv2_w[kh, kw, ci, co]

        if (idx + 1) % batch_size == 0:
            if verbose and first_apply and diag_n > 0:
                print(f"    [diagnostic] first batch: h2 sum(|.|)={diag_h2_sum:.0f} nnz={diag_h2_nnz} (of {84*diag_n})  |g| sum={diag_g_sum:.0f}  |add| sum={diag_add_sum:.2f}  => round(add) nonzero needs |add|/672 >= 0.5", flush=True)
            _apply_accum_updates(
                params, fc3_only, conv_lr_scale, fc12_lr_scale, fast,
                verbose=(verbose and first_apply),
            )
            first_apply = False
    if n > 0 and n % batch_size != 0:
        _apply_accum_updates(params, fc3_only, conv_lr_scale, fc12_lr_scale, fast, verbose=False)
    return ce_loss_display(total_ce // n) if n else 0.0


def _run_investigate(
    images: List[np.ndarray],
    labels: List[int],
    params: Dict[str, np.ndarray],
    temperature: int,
) -> None:
    """Print logit/prob/grad stats and summarize why accuracy can be low."""
    print("\n" + "=" * 70)
    print("ACCURACY INVESTIGATION (sample stats)")
    print("=" * 70)
    logits_min, logits_max = [], []
    max_prob_list, correct_prob_list = [], []
    grad_norms = []
    for idx in range(min(50, len(images))):
        img, label = images[idx], labels[idx]
        onehot = [SCALE if i == label else 0 for i in range(NUM_CLASSES)]
        logits, inter = forward_lenet5_fp(
            img,
            params["conv1_w"], params["conv2_w"],
            params["fc1_w"], params["fc2_w"], params["fc3_w"],
            params["fc1_b"], params["fc2_b"], params["fc3_b"],
            fast=False,
        )
        ce_scaled, probs = cross_entropy_fp(logits, onehot, temperature)
        grad = backward_ce_fp(probs, onehot, temperature)
        logits_min.append(min(logits))
        logits_max.append(max(logits))
        probs_real = [p / SCALE for p in probs]
        max_prob_list.append(max(probs_real))
        correct_prob_list.append(probs_real[label])
        grad_norms.append(sum(g * g for g in grad) ** 0.5 / SCALE)
    print("Logits (SCALE=1000): min={:.0f} max={:.0f} (per sample)".format(
        np.mean(logits_min), np.mean(logits_max)))
    print("Max prob (true): {:.4f}  Prob of correct class: {:.4f}".format(
        np.mean(max_prob_list), np.mean(correct_prob_list)))
    print("Grad magnitude (approx): {:.4f}".format(np.mean(grad_norms)))
    print()
    print("Why accuracy can be low:")
    print("  1. FC3-ONLY MODE (--fc3-only): Only the last layer is trained; all other")
    print("     layers frozen -> barely above random (12.5% for 8 classes).")
    print("  2. FULL BACKPROP (default): All layers (conv1, conv2, FC1, FC2, FC3) update.")
    print("     Use this for good accuracy.")
    print("  3. Fixed-point rounding can zero out small updates; we use fractional")
    print("     accumulators to reduce this.")
    print("  4. Taylor exp / atanh log are approximations; large logits can hurt.")
    print("=" * 70 + "\n")


def accuracy_fp(
    images: List[np.ndarray],
    labels: List[int],
    params: Dict[str, np.ndarray],
    max_n: int = None,
    fast: bool = False,
) -> float:
    correct = 0
    total = len(images) if max_n is None else min(len(images), max_n)
    for i in range(total):
        logits, _ = forward_lenet5_fp(
            images[i],
            params["conv1_w"], params["conv2_w"],
            params["fc1_w"], params["fc2_w"], params["fc3_w"],
            params["fc1_b"], params["fc2_b"], params["fc3_b"],
            fast=fast,
        )
        pred = int(np.argmax(logits))
        if pred == labels[i]:
            correct += 1
    return correct / total if total else 0.0


# ---------------------------------------------------------------------------
# Float32 training path: same LeNet-5, standard arithmetic, no fixed-point.
# Use --float32 to train with normal SGD and reach 80%+ accuracy.
# ---------------------------------------------------------------------------
def _conv2d_f32(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x (H,W,Cin), w (Kh,Kw,Cin,Cout). Output (outH, outW, Cout). float32."""
    from numpy.lib.stride_tricks import as_strided
    H, W, Cin = x.shape
    Kh, Kw, _, Cout = w.shape
    outH, outW = H - Kh + 1, W - Kw + 1
    patches = as_strided(
        x, shape=(outH, outW, Kh, Kw, Cin),
        strides=(x.strides[0], x.strides[1], x.strides[0], x.strides[1], x.strides[2]),
        writeable=False,
    )
    P = patches.reshape(outH * outW, Kh * Kw * Cin).astype(np.float32)
    Wflat = w.reshape(Kh * Kw * Cin, Cout).astype(np.float32)
    return (P @ Wflat).reshape(outH, outW, Cout)


def _conv2d_batch_f32(x_batch: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x_batch (B,H,W,Cin), w (Kh,Kw,Cin,Cout). Output (B, outH, outW, Cout). float32. Much faster than B separate convs."""
    from numpy.lib.stride_tricks import as_strided
    B, H, W, Cin = x_batch.shape
    Kh, Kw, _, Cout = w.shape
    outH, outW = H - Kh + 1, W - Kw + 1
    K = Kh * Kw * Cin
    Wflat = w.reshape(K, Cout).astype(np.float32)
    patches_list = []
    for b in range(B):
        patches = as_strided(
            x_batch[b], shape=(outH, outW, Kh, Kw, Cin),
            strides=(x_batch.strides[1], x_batch.strides[2], x_batch.strides[1], x_batch.strides[2], x_batch.strides[3]),
            writeable=False,
        )
        patches_list.append(patches.reshape(outH * outW, K))
    P = np.concatenate(patches_list, axis=0).astype(np.float32)  # (B*outH*outW, K)
    out = (P @ Wflat).reshape(B, outH, outW, Cout)
    return out


def _pad_to_even_batch_f32(x_batch: np.ndarray) -> np.ndarray:
    """x_batch (B,H,W,C). Pad H,W to even. Returns (B, H', W', C)."""
    B, H, W, C = x_batch.shape
    pad_h = (0, 1) if H % 2 == 1 else (0, 0)
    pad_w = (0, 1) if W % 2 == 1 else (0, 0)
    if pad_h == (0, 0) and pad_w == (0, 0):
        return x_batch
    return np.pad(x_batch, ((0, 0), pad_h, pad_w, (0, 0)), mode="constant", constant_values=0.0).astype(np.float32)


def _max_pool2x2_batch_f32(x_batch: np.ndarray) -> np.ndarray:
    """x_batch (B,H,W,C). H,W must be even. Output (B, H/2, W/2, C). float32."""
    B, H, W, C = x_batch.shape
    return x_batch.reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))


def _pad_to_even_f32(x: np.ndarray) -> np.ndarray:
    """Pad H,W to even (0 or 1 row/col) so 2x2 pool works. x (H,W,C). Returns padded (H',W',C)."""
    H, W, C = x.shape
    pad_h = (0, 1) if H % 2 == 1 else (0, 0)
    pad_w = (0, 1) if W % 2 == 1 else (0, 0)
    if pad_h == (0, 0) and pad_w == (0, 0):
        return x
    return np.pad(x, (pad_h, pad_w, (0, 0)), mode="constant", constant_values=0.0).astype(np.float32)


def _max_pool2x2_f32(x: np.ndarray) -> np.ndarray:
    """x (H,W,C). H,W must be even. Output (H/2, W/2, C). float32."""
    H, W, C = x.shape
    return x.reshape(H // 2, 2, W // 2, 2, C).max(axis=(1, 3))


def flat_size_after_conv2_pool(input_h: int, input_w: int) -> int:
    """After two conv 5x5 + pool 2x2 (with pad-to-even before each pool): flat length (oh*ow*16)."""
    c1_h, c1_w = input_h - 4, input_w - 4
    p1_h = (c1_h + (1 if c1_h % 2 else 0)) // 2
    p1_w = (c1_w + (1 if c1_w % 2 else 0)) // 2
    c2_h, c2_w = p1_h - 4, p1_w - 4
    p2_h = (c2_h + (1 if c2_h % 2 else 0)) // 2
    p2_w = (c2_w + (1 if c2_w % 2 else 0)) // 2
    return p2_h * p2_w * 16


def init_weights_float32(seed: int, init_gain: float, flat_size: int = 400, in_channels: int = 1) -> Dict[str, np.ndarray]:
    """Same architecture as init_weights but float32 (no SCALE). flat_size = FC1 input dim (400 for 32x32). in_channels = 1 or 3 (RGB)."""
    rng = np.random.default_rng(seed)
    params = {}
    # Conv1
    fan_in = 5 * 5 * in_channels
    std = np.sqrt(4.0 / fan_in) * init_gain
    params["conv1_w"] = (rng.standard_normal((5, 5, in_channels, 6)) * std).astype(np.float32)
    # Conv2
    fan_in = 5 * 5 * 6
    std = np.sqrt(4.0 / fan_in) * init_gain
    params["conv2_w"] = (rng.standard_normal((5, 5, 6, 16)) * std).astype(np.float32)
    # FC1
    std = np.sqrt(6.0 / flat_size) * init_gain
    params["fc1_w"] = (rng.standard_normal((120, flat_size)) * std).astype(np.float32)
    params["fc1_b"] = np.zeros(120, dtype=np.float32)
    # FC2
    std = np.sqrt(4.0 / 120) * init_gain
    params["fc2_w"] = (rng.standard_normal((84, 120)) * std).astype(np.float32)
    params["fc2_b"] = np.zeros(84, dtype=np.float32)
    # FC3
    std = np.sqrt(2.0 / 84) * init_gain
    params["fc3_w"] = (rng.standard_normal((8, 84)) * std).astype(np.float32)
    params["fc3_b"] = np.zeros(8, dtype=np.float32)
    return params


def copy_params_float32(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Deep copy of float32 LeNet-5 params (for best-model checkpointing)."""
    return {k: v.copy() for k, v in params.items()}


def forward_lenet5_float32(image: np.ndarray, params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict]:
    """Single sample forward. image float32 [0,1]. Returns logits (8,) float32, inter dict. Pads to even before pool so any input size works (e.g. 150x150)."""
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, :, np.newaxis]
    c1 = _conv2d_f32(x, params["conv1_w"])
    c1 = np.maximum(c1, 0.0)
    c1_padded = _pad_to_even_f32(c1)
    p1 = _max_pool2x2_f32(c1_padded)
    c2 = _conv2d_f32(p1, params["conv2_w"])
    c2 = np.maximum(c2, 0.0)
    c2_padded = _pad_to_even_f32(c2)
    p2 = _max_pool2x2_f32(c2_padded)
    flat = p2.reshape(-1)
    h1 = params["fc1_w"] @ flat + params["fc1_b"]
    h1 = np.maximum(h1, 0.0)
    h2 = params["fc2_w"] @ h1 + params["fc2_b"]
    h2 = np.maximum(h2, 0.0)
    logits = params["fc3_w"] @ h2 + params["fc3_b"]
    inter = {"flat": flat, "h1": h1, "h2": h2, "c1": c1, "c1_padded": c1_padded, "p1": p1, "c2": c2, "c2_padded": c2_padded, "p2": p2, "x": x}
    return logits, inter


def forward_lenet5_float32_batch(
    batch_imgs: List[np.ndarray], params: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Dict]:
    """Batched forward. batch_imgs list of (H,W,C). Returns logits (B, 8), inter with batched arrays. Much faster than B single forwards."""
    arrs = [np.asarray(img, dtype=np.float32) for img in batch_imgs]
    arrs = [a[:, :, np.newaxis] if a.ndim == 2 else a for a in arrs]
    x_batch = np.stack(arrs, axis=0)
    B = x_batch.shape[0]
    c1 = _conv2d_batch_f32(x_batch, params["conv1_w"])
    c1 = np.maximum(c1, 0.0)
    c1_padded = _pad_to_even_batch_f32(c1)
    p1 = _max_pool2x2_batch_f32(c1_padded)
    c2 = _conv2d_batch_f32(p1, params["conv2_w"])
    c2 = np.maximum(c2, 0.0)
    c2_padded = _pad_to_even_batch_f32(c2)
    p2 = _max_pool2x2_batch_f32(c2_padded)
    flat = p2.reshape(B, -1)
    h1 = (flat @ params["fc1_w"].T) + params["fc1_b"]
    h1 = np.maximum(h1, 0.0)
    h2 = (h1 @ params["fc2_w"].T) + params["fc2_b"]
    h2 = np.maximum(h2, 0.0)
    logits = (h2 @ params["fc3_w"].T) + params["fc3_b"]
    inter = {"flat": flat, "h1": h1, "h2": h2, "c1": c1, "c1_padded": c1_padded, "p1": p1, "c2": c2, "c2_padded": c2_padded, "p2": p2, "x": x_batch}
    return logits, inter


def _max_pool2x2_backward_f32(grad_out: np.ndarray, input_pre_pool: np.ndarray) -> np.ndarray:
    """Vectorized: no Python loops over spatial dims (was ~32K iters per sample)."""
    H, W, C = input_pre_pool.shape
    oh, ow = H // 2, W // 2
    view = input_pre_pool.reshape(oh, 2, ow, 2, C)
    flat_idx = np.argmax(view.reshape(oh, ow, 4, C), axis=2)  # (oh, ow, C), values 0-3
    hi, wi = flat_idx // 2, flat_idx % 2
    i_idx = np.arange(oh, dtype=np.intp)[:, None, None]
    j_idx = np.arange(ow, dtype=np.intp)[None, :, None]
    row_idx = 2 * i_idx + hi
    col_idx = 2 * j_idx + wi
    c_idx = np.arange(C, dtype=np.intp)[None, None, :]
    grad_in = np.zeros_like(input_pre_pool, dtype=np.float32)
    grad_in[row_idx, col_idx, c_idx] = grad_out
    return grad_in


def _max_pool2x2_backward_batch_f32(grad_out_batch: np.ndarray, input_pre_pool_batch: np.ndarray) -> np.ndarray:
    """Batched vectorized max_pool backward. grad_out (B,oh,ow,C), input (B,H,W,C) -> grad_in (B,H,W,C)."""
    B, H, W, C = input_pre_pool_batch.shape
    oh, ow = H // 2, W // 2
    view = input_pre_pool_batch.reshape(B, oh, 2, ow, 2, C)
    flat_idx = np.argmax(view.reshape(B, oh, ow, 4, C), axis=3)  # (B, oh, ow, C)
    hi, wi = flat_idx // 2, flat_idx % 2
    i_idx = np.arange(oh, dtype=np.intp)[None, :, None, None]
    j_idx = np.arange(ow, dtype=np.intp)[None, None, :, None]
    row_idx = 2 * i_idx + hi
    col_idx = 2 * j_idx + wi
    b_idx = np.arange(B, dtype=np.intp)[:, None, None, None]
    c_idx = np.arange(C, dtype=np.intp)[None, None, None, :]
    grad_in = np.zeros_like(input_pre_pool_batch, dtype=np.float32)
    grad_in[b_idx, row_idx, col_idx, c_idx] = grad_out_batch
    return grad_in


def _conv2d_backward_f32(grad_out: np.ndarray, x: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from numpy.lib.stride_tricks import as_strided
    try:
        from scipy.signal import convolve2d
    except ImportError:
        convolve2d = None
    H, W, Cin = x.shape
    Kh, Kw, _, Cout = w.shape
    outH, outW = H - Kh + 1, W - Kw + 1
    patches = as_strided(
        x, shape=(outH, outW, Kh, Kw, Cin),
        strides=(x.strides[0], x.strides[1], x.strides[0], x.strides[1], x.strides[2]),
        writeable=False,
    )
    P = patches.reshape(outH * outW, Kh * Kw * Cin).astype(np.float32)
    G = grad_out.reshape(outH * outW, Cout).astype(np.float32)
    grad_w = (P.T @ G).reshape(Kh, Kw, Cin, Cout)
    grad_in = np.zeros((H, W, Cin), dtype=np.float32)
    if convolve2d is not None:
        for co in range(Cout):
            for cin in range(Cin):
                grad_in[:, :, cin] += convolve2d(
                    grad_out[:, :, co], w[::-1, ::-1, cin, co], mode="full"
                ).astype(np.float32)
    else:
        for oh in range(outH):
            for ow in range(outW):
                for co in range(Cout):
                    grad_in[oh : oh + Kh, ow : ow + Kw, :] += grad_out[oh, ow, co] * w[:, :, :, co]
    return grad_in, grad_w


def _conv2d_backward_w_batch(grad_out_batch: np.ndarray, x_batch: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Batched conv backward for weights only. grad_out_batch (B,outH,outW,Cout), x_batch (B,H,W,Cin). Returns grad_w (Kh,Kw,Cin,Cout)."""
    from numpy.lib.stride_tricks import as_strided
    B, H, W, Cin = x_batch.shape
    Kh, Kw, _, Cout = w.shape
    outH, outW = H - Kh + 1, W - Kw + 1
    K = Kh * Kw * Cin
    patches_list = []
    for b in range(B):
        patches = as_strided(
            x_batch[b], shape=(outH, outW, Kh, Kw, Cin),
            strides=(x_batch.strides[1], x_batch.strides[2], x_batch.strides[1], x_batch.strides[2], x_batch.strides[3]),
            writeable=False,
        )
        patches_list.append(patches.reshape(outH * outW, K))
    P_all = np.concatenate(patches_list, axis=0).astype(np.float32)
    G_all = grad_out_batch.reshape(B * outH * outW, Cout).astype(np.float32)
    grad_w = (P_all.T @ G_all).reshape(Kh, Kw, Cin, Cout)
    return grad_w


def train_step_float32(
    images: List[np.ndarray],
    labels: List[int],
    params: Dict[str, np.ndarray],
    lr: float,
    batch_size: int = 64,
    progress_interval: int = 500,
) -> float:
    """One epoch: batched float32 forward/backward, softmax+CE, SGD. Returns average loss. Much faster than per-sample loops."""
    n_train = len(images)
    total_loss = 0.0
    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        batch_imgs = [images[i] for i in range(start, end)]
        batch_lbls = [labels[i] for i in range(start, end)]
        if progress_interval and start > 0 and start % progress_interval < batch_size:
            print(f"    train step {start}/{n_train}", flush=True)
        # Batched forward
        logits_batch, inter = forward_lenet5_float32_batch(batch_imgs, params)
        B = len(batch_lbls)
        # Softmax + CE
        logits_max = logits_batch.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_batch - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        onehot = np.zeros((B, NUM_CLASSES), dtype=np.float32)
        for i, lbl in enumerate(batch_lbls):
            onehot[i, lbl] = 1.0
        loss = -np.sum(onehot * np.log(probs + 1e-12)) / B
        total_loss += loss * B
        grad_logits = (probs - onehot) / B  # (B, 8)
        # Batched FC backward (no loop)
        h2, h1, flat = inter["h2"], inter["h1"], inter["flat"]
        grad_fc3_w = grad_logits.T @ h2
        grad_fc3_b = grad_logits.sum(axis=0)
        grad_h2 = (grad_logits @ params["fc3_w"]) * (h2 > 0)
        grad_fc2_w = grad_h2.T @ h1
        grad_fc2_b = grad_h2.sum(axis=0)
        grad_h1 = (grad_h2 @ params["fc2_w"]) * (h1 > 0)
        grad_fc1_w = grad_h1.T @ flat
        grad_fc1_b = grad_h1.sum(axis=0)
        grad_flat = grad_h1 @ params["fc1_w"]
        # Pool backward: batched (no loop over B). Conv backward: batched grad_w, loop for grad_in.
        p2_shape = inter["p2"].shape
        c2_h, c2_w = inter["c2"].shape[1], inter["c2"].shape[2]
        c1_h, c1_w = inter["c1"].shape[1], inter["c1"].shape[2]
        grad_p2_batch = grad_flat.reshape(B, p2_shape[1], p2_shape[2], p2_shape[3])
        grad_c2_padded = _max_pool2x2_backward_batch_f32(grad_p2_batch, inter["c2_padded"])
        grad_c2 = grad_c2_padded[:, :c2_h, :c2_w, :]
        grad_c2_pre_sum = grad_c2 * (inter["c2"] > 0)
        grad_conv2_w = _conv2d_backward_w_batch(grad_c2_pre_sum, inter["p1"], params["conv2_w"])
        grad_p1_batch = np.zeros_like(inter["p1"], dtype=np.float32)
        for b in range(B):
            grad_p1_batch[b], _ = _conv2d_backward_f32(grad_c2_pre_sum[b], inter["p1"][b], params["conv2_w"])
        grad_c1_padded = _max_pool2x2_backward_batch_f32(grad_p1_batch, inter["c1_padded"])
        grad_c1 = grad_c1_padded[:, :c1_h, :c1_w, :]
        grad_c1_pre_batch = grad_c1 * (inter["c1"] > 0)
        grad_conv1_w = _conv2d_backward_w_batch(grad_c1_pre_batch, inter["x"], params["conv1_w"])
        params["fc3_w"] -= lr * grad_fc3_w
        params["fc3_b"] -= lr * grad_fc3_b
        params["fc2_w"] -= lr * grad_fc2_w
        params["fc2_b"] -= lr * grad_fc2_b
        params["fc1_w"] -= lr * grad_fc1_w
        params["fc1_b"] -= lr * grad_fc1_b
        params["conv2_w"] -= lr * grad_conv2_w
        params["conv1_w"] -= lr * grad_conv1_w
    return total_loss / n_train if n_train else 0.0


def accuracy_float32(
    images: List[np.ndarray],
    labels: List[int],
    params: Dict[str, np.ndarray],
    max_n: int = None,
    batch_size: int = 128,
) -> float:
    """Accuracy = fraction of samples where argmax(logits) equals true label. Returns in [0, 1]. Uses batched forward for speed."""
    total = len(images) if max_n is None else min(len(images), max_n)
    correct = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_imgs = [images[i] for i in range(start, end)]
        batch_lbls = [labels[i] for i in range(start, end)]
        logits_batch, _ = forward_lenet5_float32_batch(batch_imgs, params)
        preds = np.argmax(logits_batch.astype(np.float64), axis=1)
        correct += np.sum(preds == np.asarray(batch_lbls, dtype=preds.dtype))
    return correct / total if total else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LeNet-5 plaintext reference (fixed-point + MPC-matching approximations)")
    ap.add_argument("--dataset", type=str, default="Kather_texture_2016_image_tiles_5000", help="Dataset path")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.05, help="Learning rate (0.05 helps full-dataset; use 0.01 for MPC comparison)")
    ap.add_argument("--temperature", type=float, default=2.0, help="Match MPC softmax temperature (run_lenet5_training_full.py default)")
    ap.add_argument("--init-gain", type=float, default=0.1, help="Match MPC weight init (run_lenet5_training_full.py default)")
    ap.add_argument("--scale", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--quick-test", action="store_true", help="Use 16 samples only")
    ap.add_argument("--max-train", type=int, default=None, help="Max training samples per epoch (default: all). Use e.g. 500 for faster epochs.")
    ap.add_argument("--fc3-only", action="store_true", help="Train only FC3 (legacy: linear on frozen features)")
    ap.add_argument("--investigate", action="store_true", help="Print logit/prob/grad stats and accuracy causes")
    ap.add_argument("--conv-lr-scale", type=float, default=40.0,
                    help="Scale LR for conv layers (default 40); full-dataset needs larger steps to learn")
    ap.add_argument("--fc12-lr-scale", type=float, default=20.0,
                    help="Scale LR for FC1/FC2 (default 20); full-dataset needs larger steps to learn")
    ap.add_argument("--fast", action="store_true",
                    help="Use vectorized conv/FC forward (much faster; numerics differ slightly)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Gradient accumulation: update every N samples (default 64; use 1 for per-sample)")
    ap.add_argument("--grad-scale", type=float, default=1.0,
                    help="Multiply gradients by this before accum (default 1; try 2 or 4 if accuracy stuck at 12%%)")
    ap.add_argument("--verbose", action="store_true", help="Print first-batch update diagnostic (max |delta|, non-zero count)")
    ap.add_argument("--float32", action="store_true",
                    help="Train in float32 (standard SGD, no fixed-point). Use for 80%%+ accuracy; fixed-point path for MPC comparison.")
    ap.add_argument("--lr-decay", type=float, default=None,
                    help="Per-epoch LR decay for float32: lr = lr * (lr_decay ** (epoch-1)). Default 0.98 when --float32.")
    ap.add_argument("--input-size", type=int, default=150,
                    help="Input image size (H=W) for float32 path. Default 150 (native Kather). Use 32 for LeNet-5 default.")
    ap.add_argument("--grayscale", action="store_true",
                    help="Use grayscale for float32 (default is RGB, conforming to Kather dataset).")
    ap.add_argument("--rgb", action="store_true",
                    help="Use RGB for float32 (same as default; use --grayscale to disable).")
    ap.add_argument("--acc-every", type=int, default=1,
                    help="Compute train/val accuracy every N epochs (float32). Default 1. Use 2 or 5 to speed up.")
    args = ap.parse_args()

    global SCALE
    SCALE = int(args.scale)
    temp_int = int(round(args.temperature))
    if abs(args.temperature - temp_int) > 1e-9:
        print("Warning: temperature must be integer for fixed-point; using", temp_int)

    print("=" * 70)
    print("LeNet-5 Plaintext Reference (fixed-point + MPC-matching approximations)")
    print("=" * 70)
    print("SCALE:", SCALE, "| Temperature:", temp_int, "| Init gain:", args.init_gain)
    print("LR:", args.lr, "| Epochs:", args.epochs)
    if not args.fc3_only:
        print("Conv LR scale:", args.conv_lr_scale, "| FC1/FC2 LR scale:", args.fc12_lr_scale)
    if args.fast:
        print("Fast mode: vectorized conv/FC forward (faster, numerics differ slightly)")
    if args.batch_size > 1:
        print("Batch size:", args.batch_size, "(update every", args.batch_size, "samples)")
    if args.grad_scale != 1.0:
        print("Grad scale:", args.grad_scale)
    image_size = (args.input_size, args.input_size) if args.float32 else (32, 32)
    grayscale = (args.grayscale and not args.rgb) if args.float32 else True
    if args.float32:
        print("Mode: FLOAT32 (standard SGD, target 80%%+ accuracy)")
        print("Input: {}x{} {}".format(args.input_size, args.input_size, "grayscale" if grayscale else "RGB"))
    print("=" * 70)
    images, labels = load_kather_dataset(
        args.dataset,
        image_size=image_size,
        grayscale=grayscale,
        normalize=True,
    )
    train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls = split_dataset(
        images, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=args.seed
    )

    if args.quick_test:
        train_imgs = train_imgs[:16]
        train_lbls = train_lbls[:16]
        print("[Quick test] Using 16 training samples")
    elif args.max_train is not None:
        train_imgs = train_imgs[: args.max_train]
        train_lbls = train_lbls[: args.max_train]
        print(f"[Max train] Using {len(train_imgs)} training samples per epoch")

    if args.float32:
        print("[Float32] Full backprop, standard SGD (same LeNet-5 architecture)")
        try:
            __import__("scipy.signal")
            print("  scipy found: conv backward uses fast vectorized ops")
        except ImportError:
            print("  Tip: install scipy (pip install scipy) for faster conv backward")
        if getattr(args, "acc_every", 1) > 1:
            print("  Accuracy every {} epochs (use --acc-every 1 for every epoch)".format(args.acc_every))
        # Stronger init (0.5) so float32 learns quickly; fixed-point keeps 0.1 for MPC match.
        init_gain_f32 = 0.5 if args.init_gain == 0.1 else args.init_gain
        print(f"  Init gain (float32): {init_gain_f32} (use --init-gain to override)")
        lr_decay = args.lr_decay if args.lr_decay is not None else 0.98
        print(f"  LR decay per epoch: {lr_decay} (use --lr-decay 1.0 for no decay)")
        # Stable defaults for 80%+ target: use 0.001 when user didn't pass --lr (default 0.05)
        if args.lr == 0.05:
            print("  LR (float32): 0.001 (default for stability; use --lr to override)")
            args.lr = 0.001
        flat_size = flat_size_after_conv2_pool(args.input_size, args.input_size)
        in_channels = 1 if grayscale else 3
        params = init_weights_float32(args.seed, init_gain_f32, flat_size=flat_size, in_channels=in_channels)
        loss_hist = []
        n_train = len(train_imgs)
        best_val = -1.0
        best_params = None
        for epoch in range(1, args.epochs + 1):
            # Shuffle training set each epoch so batches mix classes (critical for learning).
            rng = np.random.default_rng(args.seed + epoch)
            perm = rng.permutation(n_train)
            epoch_imgs = [train_imgs[i] for i in perm]
            epoch_lbls = [train_lbls[i] for i in perm]
            lr_epoch = args.lr * (lr_decay ** (epoch - 1))
            print(f"Epoch {epoch}/{args.epochs}  Training on {n_train} samples (lr={lr_epoch:.4f})...", flush=True)
            loss_display = train_step_float32(
                epoch_imgs, epoch_lbls, params, lr_epoch,
                batch_size=args.batch_size, progress_interval=500,
            )
            loss_hist.append(loss_display)
            acc_every = getattr(args, "acc_every", 1)
            if (epoch - 1) % acc_every == 0:
                print(f"  Computing train accuracy ({n_train} samples)...", flush=True)
                acc_train = accuracy_float32(train_imgs, train_lbls, params)
                print(f"  Computing val accuracy (500 samples)...", flush=True)
                acc_val = accuracy_float32(val_imgs, val_lbls, params, max_n=500)
                if acc_val > best_val:
                    best_val = acc_val
                    best_params = copy_params_float32(params)
                n_correct_train = int(round(acc_train * n_train))
                n_correct_val = int(round(acc_val * min(500, len(val_imgs))))
                print(f"Epoch {epoch}/{args.epochs}  Loss ≈ {loss_display:.6f}  Train acc: {n_correct_train}/{n_train} ({acc_train*100:.2f}%)  Val acc: {n_correct_val}/500 ({acc_val*100:.2f}%)")
            else:
                print(f"  Computing val accuracy (500 samples)...", flush=True)
                acc_val = accuracy_float32(val_imgs, val_lbls, params, max_n=500)
                if acc_val > best_val:
                    best_val = acc_val
                    best_params = copy_params_float32(params)
                n_correct_val = int(round(acc_val * min(500, len(val_imgs))))
                print(f"Epoch {epoch}/{args.epochs}  Loss ≈ {loss_display:.6f}  Val acc: {n_correct_val}/500 ({acc_val*100:.2f}%)")
        if best_params is not None:
            for k in params:
                params[k][...] = best_params[k]
            print("Restored best model (val acc {:.2f}%) for final metrics.".format(best_val * 100))
        print()
        print("Final:")
        acc_t = accuracy_float32(train_imgs, train_lbls, params)
        acc_v = accuracy_float32(val_imgs, val_lbls, params, max_n=500)
        acc_te = accuracy_float32(test_imgs, test_lbls, params, max_n=500)
        print(f"  Train accuracy: {int(round(acc_t*n_train))}/{n_train} ({acc_t*100:.2f}%)")
        print(f"  Val accuracy:   {int(round(acc_v*500))}/500 ({acc_v*100:.2f}%)")
        print(f"  Test accuracy: {int(round(acc_te*500))}/500 ({acc_te*100:.2f}%)")
        print("  Random baseline (8 classes): 12.5%")
        if loss_hist:
            print("  Loss history:", [f"{l:.4f}" for l in loss_hist])
        print("=" * 70)
        return
    if args.fc3_only:
        print("[FC3 only] Training last layer only (frozen conv + FC1/FC2)")
    else:
        print("[Full backprop] Training all layers (conv1, conv2, FC1, FC2, FC3)")

    params = init_weights(args.seed, args.init_gain)

    if args.investigate:
        _run_investigate(train_imgs[:50], train_lbls[:50], params, temp_int)

    loss_hist = []
    n_train = len(train_imgs)
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}  Training on {n_train} samples...", flush=True)
        loss_display = train_step_fp(
            train_imgs, train_lbls, params, temp_int, args.lr,
            fc3_only=args.fc3_only, conv_lr_scale=args.conv_lr_scale,
            fc12_lr_scale=args.fc12_lr_scale, fast=args.fast,
            batch_size=args.batch_size, grad_scale=args.grad_scale, verbose=args.verbose,
        )
        loss_hist.append(loss_display)
        print(f"  Computing train accuracy ({n_train} samples)...", flush=True)
        acc_train = accuracy_fp(train_imgs, train_lbls, params, fast=args.fast)
        print(f"  Computing val accuracy (500 samples)...", flush=True)
        acc_val = accuracy_fp(val_imgs, val_lbls, params, max_n=500, fast=args.fast)
        print(f"Epoch {epoch}/{args.epochs}  Loss ≈ {loss_display:.6f}  Train acc: {acc_train*100:.2f}%  Val acc: {acc_val*100:.2f}%")

    print()
    print("Final:")
    print("  Train accuracy:", accuracy_fp(train_imgs, train_lbls, params, fast=args.fast) * 100, "%")
    print("  Val accuracy:  ", accuracy_fp(val_imgs, val_lbls, params, max_n=500, fast=args.fast) * 100, "%")
    print("  Test accuracy: ", accuracy_fp(test_imgs, test_lbls, params, max_n=500, fast=args.fast) * 100, "%")
    print("  Random baseline (8 classes): 12.5%")
    print()
    if loss_hist:
        print("  Loss history:", [f"{l:.4f}" for l in loss_hist])
    print("=" * 70)
    print("If this converges (loss down, accuracy above random) but MPC does not,")
    print("the problem is likely in the MPC protocol/sync/communication.")
    print("If this already fails, the issue is numerical (scaling/truncation/approximation).")
    print("=" * 70)


if __name__ == "__main__":
    main()
