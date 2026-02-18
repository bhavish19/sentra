"""
Full LeNet-5 Training on Kather Texture Dataset
Integrated with SENTRA training pipeline
"""

import sys
import os
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import random
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.image_loader import load_kather_dataset, prepare_dataset_for_training, split_dataset
from ml_training.kvs import KVSCluster
from ml_training.lenet5 import LeNet5
from ml_training.beaver_triples import BeaverTripleGenerator, BeaverTriplePool, SecureMultiplier
from ml_training.secure_comparison import SecureComparator
from ml_training.secure_division import SecureDivider
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_matrix_ops import SecureMatrixOperations
from ml_training.secure_softmax import SecureSoftmax
from ml_training.secure_comm import create_mpc_network
from ml_training.reconstruction import create_reconstruction_manager
from ml_training.beaver_triples import BeaverTripleDealerService


def _field_u64_to_float(arr_u64: np.ndarray, field_size: int, scale_factor: int) -> np.ndarray:
    """Convert field elements (0..p-1) into signed float values."""
    if arr_u64.size == 0:
        return arr_u64.astype(np.float32)
    p = int(field_size)
    half = p // 2
    signed = arr_u64.astype(np.int64, copy=False)
    # Values live in [0, p); interpret upper half as negative
    signed = np.where(signed > half, signed - p, signed)
    return (signed.astype(np.float32)) / float(scale_factor)


def _flatten_model_weights_to_u32(weights: List, field_size: int) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int, ...], int]]]:
    """
    Flatten weights (list-of-layers of Share objects) into a single uint32 vector.
    Returns (vec_u32, layout) where layout entries are (name, shape, count).
    """
    conv1_w, conv2_w, fc1_w, fc2_w, fc3_w, fc1_b, fc2_b, fc3_b = weights

    layout: List[Tuple[str, Tuple[int, ...], int]] = []
    layout.append(("conv1_w", (5, 5, 1, 6), 5 * 5 * 1 * 6))
    layout.append(("conv2_w", (5, 5, 6, 16), 5 * 5 * 6 * 16))
    layout.append(("fc1_w", (120, 400), 120 * 400))
    layout.append(("fc2_w", (84, 120), 84 * 120))
    layout.append(("fc3_w", (8, 84), 8 * 84))
    layout.append(("fc1_b", (120,), 120))
    layout.append(("fc2_b", (84,), 84))
    layout.append(("fc3_b", (8,), 8))

    total = sum(c for _, _, c in layout)
    out = np.empty((total,), dtype=np.uint32)
    idx = 0

    # conv1
    for kh in range(5):
        for kw in range(5):
            for ci in range(1):
                for co in range(6):
                    out[idx] = np.uint32(int(conv1_w[kh][kw][ci][co].y) & 0xFFFFFFFF)
                    idx += 1
    # conv2
    for kh in range(5):
        for kw in range(5):
            for ci in range(6):
                for co in range(16):
                    out[idx] = np.uint32(int(conv2_w[kh][kw][ci][co].y) & 0xFFFFFFFF)
                    idx += 1
    # fc1
    for i in range(120):
        for j in range(400):
            out[idx] = np.uint32(int(fc1_w[i][j].y) & 0xFFFFFFFF)
            idx += 1
    # fc2
    for i in range(84):
        for j in range(120):
            out[idx] = np.uint32(int(fc2_w[i][j].y) & 0xFFFFFFFF)
            idx += 1
    # fc3
    for i in range(8):
        for j in range(84):
            out[idx] = np.uint32(int(fc3_w[i][j].y) & 0xFFFFFFFF)
            idx += 1
    # biases
    for i in range(120):
        out[idx] = np.uint32(int(fc1_b[i].y) & 0xFFFFFFFF)
        idx += 1
    for i in range(84):
        out[idx] = np.uint32(int(fc2_b[i].y) & 0xFFFFFFFF)
        idx += 1
    for i in range(8):
        out[idx] = np.uint32(int(fc3_b[i].y) & 0xFFFFFFFF)
        idx += 1

    if idx != total:
        raise RuntimeError(f"flatten mismatch: wrote {idx} vs expected {total}")
    return out, layout


def _plain_max_pool2x2(x: np.ndarray) -> np.ndarray:
    # x: (H, W, C), H and W even
    H, W, C = x.shape
    return x.reshape(H // 2, 2, W // 2, 2, C).max(axis=(1, 3))


def _plain_conv2d_valid(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    # x: (H, W, Cin), w: (Kh, Kw, Cin, Cout)
    x = np.asarray(x, dtype=np.float32, order="C")
    w = np.asarray(w, dtype=np.float32, order="C")
    H, W, Cin = x.shape
    Kh, Kw, Cin2, Cout = w.shape
    if Cin != Cin2:
        raise ValueError("Cin mismatch")
    outH = H - Kh + 1
    outW = W - Kw + 1
    # im2col via strided view, then GEMM
    from numpy.lib.stride_tricks import as_strided
    patches = as_strided(
        x,
        shape=(outH, outW, Kh, Kw, Cin),
        strides=(x.strides[0], x.strides[1], x.strides[0], x.strides[1], x.strides[2]),
        writeable=False,
    )
    patches_mat = patches.reshape(outH * outW, Kh * Kw * Cin)
    w_mat = w.reshape(Kh * Kw * Cin, Cout)
    out = patches_mat @ w_mat
    return out.reshape(outH, outW, Cout)


def _plain_lenet5_predict(image: np.ndarray, params: Dict[str, np.ndarray]) -> int:
    # Ensure shape (32,32,1)
    if image.ndim == 2:
        x = image[:, :, None].astype(np.float32)
    elif image.ndim == 3 and image.shape[2] == 1:
        x = image.astype(np.float32)
    else:
        x = image[:, :, :1].astype(np.float32)

    x = _plain_conv2d_valid(x, params["conv1_w"])
    x = np.maximum(x, 0.0, dtype=np.float32)
    x = _plain_max_pool2x2(x)

    x = _plain_conv2d_valid(x, params["conv2_w"])
    x = np.maximum(x, 0.0, dtype=np.float32)
    x = _plain_max_pool2x2(x)

    flat = x.reshape(-1).astype(np.float32)
    h1 = params["fc1_w"].astype(np.float32) @ flat + params["fc1_b"].astype(np.float32)
    h1 = np.maximum(h1, 0.0, dtype=np.float32)
    h2 = params["fc2_w"].astype(np.float32) @ h1 + params["fc2_b"].astype(np.float32)
    h2 = np.maximum(h2, 0.0, dtype=np.float32)
    logits = params["fc3_w"].astype(np.float32) @ h2 + params["fc3_b"].astype(np.float32)
    return int(np.argmax(logits))

def _plain_eval_accuracy(
    images: List[np.ndarray],
    labels: List[int],
    params: Dict[str, np.ndarray],
    *,
    max_n: Optional[int] = None,
) -> float:
    n = len(images)
    if max_n is not None:
        n = min(n, int(max_n))
    if n <= 0:
        return 0.0
    correct = 0
    for i in range(n):
        pred = _plain_lenet5_predict(images[i], params)
        if pred == int(labels[i]):
            correct += 1
    return correct / n


def create_node_configs(n_nodes: int, base_port: int = 8000, host: str = 'localhost'):
    """Create node configurations for multi-node setup"""
    configs = {}
    for i in range(1, n_nodes + 1):
        configs[i] = {
            'host': host,
            'port': base_port + i
        }
    return configs


def image_to_shares(image: np.ndarray, n_nodes: int, t: int, node_id: int,
                   field_size: int, shamir: ShamirSecretSharing, scale_factor: int = 10_000_000) -> List[List[List[Share]]]:
    """
    Convert image to secret shares
    
    Args:
        image: Image array [H x W x C] or [H x W]
        n_nodes: Number of nodes
        t: Privacy threshold
        node_id: This node's ID
        field_size: Prime field size
        shamir: ShamirSecretSharing instance
    
    Returns:
        Image as shares [H x W x C]
    """
    if len(image.shape) == 2:
        # Grayscale: add channel dimension
        image = image[:, :, np.newaxis]
    
    H, W, C = image.shape
    image_shares = []
    
    for h in range(H):
        row_shares = []
        for w in range(W):
            channel_shares = []
            for c in range(C):
                # Convert pixel to integer
                pixel_value = float(image[h, w, c])
                pixel_int = int(pixel_value * scale_factor) % field_size
                
                # Secret share
                shares = shamir.share(pixel_int, n_nodes, t)
                node_share = next(s for s in shares if s.node_id == node_id)
                channel_shares.append(node_share)
            row_shares.append(channel_shares)
        image_shares.append(row_shares)
    
    return image_shares


def label_to_shares(label: np.ndarray, n_nodes: int, t: int, node_id: int,
                   field_size: int, shamir: ShamirSecretSharing, scale_factor: int = 10_000_000) -> List[Share]:
    """
    Convert label (one-hot) to secret shares
    
    Args:
        label: One-hot label array [num_classes]
        n_nodes: Number of nodes
        t: Privacy threshold
        node_id: This node's ID
        field_size: Prime field size
        shamir: ShamirSecretSharing instance
    
    Returns:
        Label as shares [num_classes]
    """
    label_shares = []
    
    for i, val in enumerate(label):
        val_int = int(val * scale_factor) % field_size
        shares = shamir.share(val_int, n_nodes, t)
        node_share = next(s for s in shares if s.node_id == node_id)
        label_shares.append(node_share)
    
    return label_shares


def compute_accuracy(lenet5: LeNet5, image_shares: List[List[List[List[Share]]]],
                    true_labels: List[int], weights: List, node_id: int,
                    max_samples: int = None, softmax_op=None, debug: bool = False) -> float:
    """
    Compute accuracy on a dataset
    
    Args:
        lenet5: LeNet-5 model instance
        image_shares: Images as shares [num_images x H x W x C]
        true_labels: True class labels [num_images] (integer class indices)
        weights: Model weights
        node_id: Node ID
        max_samples: Maximum number of samples to evaluate (for speed)
        softmax_op: Optional SecureSoftmax instance for computing probabilities (for debug)
        debug: If True, print detailed debug information
    
    Returns:
        Accuracy as a float (0.0 to 1.0)
    """
    if max_samples and max_samples < len(image_shares):
        image_shares = image_shares[:max_samples]
        true_labels = true_labels[:max_samples]
    
    correct = 0
    total = len(image_shares)
    # IMPORTANT FIX:
    # With a ~32-bit prime field, SCALE_FACTOR=10,000,000 causes wraparound in dot-products.
    # Use a smaller fixed-point scale to keep products and accumulations within the field.
    SCALE_FACTOR = 1000
    
    # Debug statistics
    logit_ranges = []
    uniform_predictions = 0  # Count predictions where logits are too uniform
    
    for i, img_share in enumerate(image_shares):
        # Forward pass
        output = lenet5.forward_pass(img_share, weights, node_id=node_id, context=f"eval_{i}")
        
        # Extract logit values
        output_values = [out.y % lenet5.field_size for out in output]
        output_values = [v if v < lenet5.field_size // 2 else v - lenet5.field_size for v in output_values]
        output_scaled = [v / SCALE_FACTOR for v in output_values]
        
        # Compute logit range (discrimination measure)
        logit_range = max(output_scaled) - min(output_scaled)
        logit_ranges.append(logit_range)
        
        # Check if logits are too uniform (low discrimination)
        if logit_range < 0.5:  # Very close logits
            uniform_predictions += 1
        
        # Find predicted class (argmax of logits)
        predicted_class = output_scaled.index(max(output_scaled))
        
        # Debug output for first few samples
        if debug and i < 3:
            # Compute softmax probabilities if available
            if softmax_op:
                probs = softmax_op.softmax(output, node_id, f"acc_debug_{i}")
                prob_values = []
                for prob in probs:
                    val = prob.y % lenet5.field_size
                    if val > lenet5.field_size // 2:
                        val = val - lenet5.field_size
                    prob_values.append(val / SCALE_FACTOR)
                
                true_prob = prob_values[true_labels[i]]
                pred_prob = prob_values[predicted_class]
                
                print(f"  [ACC DEBUG] Sample {i}: True={true_labels[i]}, Pred={predicted_class}, "
                      f"Logit range={logit_range:.4f}, True prob={true_prob:.4f}, "
                      f"Pred prob={pred_prob:.4f}, Correct={'✓' if predicted_class == true_labels[i] else '✗'}")
            else:
                print(f"  [ACC DEBUG] Sample {i}: True={true_labels[i]}, Pred={predicted_class}, "
                      f"Logit range={logit_range:.4f}, Logits={[f'{l:.3f}' for l in output_scaled[:4]]}..., "
                      f"Correct={'✓' if predicted_class == true_labels[i] else '✗'}")
        
        # Check if correct
        if predicted_class == true_labels[i]:
            correct += 1
    
    # Print summary statistics
    if debug and logit_ranges:
        avg_logit_range = sum(logit_ranges) / len(logit_ranges)
        min_logit_range = min(logit_ranges)
        max_logit_range = max(logit_ranges)
        uniform_pct = (uniform_predictions / total) * 100
        
        print(f"  [ACC STATS] Avg logit range: {avg_logit_range:.4f} (min={min_logit_range:.4f}, max={max_logit_range:.4f})")
        print(f"  [ACC STATS] Uniform predictions (range<0.5): {uniform_predictions}/{total} ({uniform_pct:.1f}%)")
        if avg_logit_range < 1.0:
            print(f"  [WARNING] Logits are too uniform (avg range={avg_logit_range:.4f}) - model lacks discrimination!")
    
    return correct / total if total > 0 else 0.0


def train_lenet5_batch(
    lenet5: LeNet5,
    softmax_op: SecureSoftmax,
    image_shares: List[List[List[List[Share]]]],
    label_shares: List[List[Share]],
    label_plain: List[np.ndarray],
    weights: List,
    learning_rate: float,
    node_id: int,
    *,
    batch_context: str,
    reconstruction_manager=None,
    open_softmax: bool = True,
    softmax_opener_node: int = 1,
    open_relu: bool = True,
    relu_opener_node: int = 1,
    fc_batch_simd: bool = False,
    packed_pss: bool = False,
    n_nodes: int = 1,
    t: int = 0,
    quiet: bool = False,
    profile: bool = False,
    debug_loss: bool = False,
) -> Tuple[List, float]:
    """
    Train LeNet-5 on a batch of images
    
    Args:
        lenet5: LeNet-5 model instance
        image_shares: Batch of images as shares [batch_size x H x W x C]
        label_shares: Batch of labels as shares [batch_size x num_classes]
        weights: Current weights
        learning_rate: Learning rate
        node_id: Node ID
    
    Returns:
        Tuple of (updated_weights, loss_value)
    """
    batch_size = len(image_shares)
    predictions = []
    losses = []
    
    p = int(lenet5.field_size)
    SCALE_FACTOR = int(getattr(lenet5, "scale_factor", 1000))

    # ----------------------------
    # Lightweight profiling helpers
    # ----------------------------
    prof_enabled = bool(profile) and (not quiet)
    prof_t0 = time.time()
    prof: dict[str, float] = {}

    def _pt_add(k: str, dt: float):
        prof[k] = float(prof.get(k, 0.0)) + float(dt)

    if bool(fc_batch_simd):
        # Conv stack per-sample (still expensive), FC stack batched/SIMD across the mini-batch.
        flattened_batch: List[List[Share]] = []
        conv_intermediates_batch: List[dict] = []

        for i in range(batch_size):
            image_share = image_shares[i]
            print(f"      → Conv stack {i+1}/{batch_size} (this may take 1-5 minutes per image)...", end='\r', flush=True)
            _t0 = time.time()
            _, conv_intermediates = lenet5.forward_pass(
                image_share,
                weights,
                node_id=node_id,
                context=f"{batch_context}_s{i}",
                return_intermediates=True,
                reconstruction_manager=reconstruction_manager if open_relu else None,
                open_relu=bool(open_relu),
                relu_opener_node=int(relu_opener_node),
                stop_after_flatten=True,
            )
            if prof_enabled:
                _pt_add("conv_fwd", time.time() - _t0)
            conv_intermediates_batch.append(conv_intermediates)
            flattened_batch.append(conv_intermediates["flattened"])
            print(f"      → Conv stack {i+1}/{batch_size} completed ✓", flush=True)

        print(f"      → FC stack (batched/SIMD) for {batch_size} sample(s)...", end='', flush=True)
        _t0 = time.time()
        logits_batch, fc_intermediates_batch = lenet5.forward_fc_stack_batch(
            flattened_batch=flattened_batch,
            weights=weights,
            node_id=int(node_id),
            context=f"{batch_context}_fc_batch",
            reconstruction_manager=reconstruction_manager if (open_relu or packed_pss) else None,
            open_relu=bool(open_relu),
            relu_opener_node=int(relu_opener_node),
            packed_pss=bool(packed_pss),
        )
        if prof_enabled:
            _pt_add("fc_fwd", time.time() - _t0)
        print(" ✓", flush=True)

        for i in range(batch_size):
            output = logits_batch[i]
            fwd = {}
            fwd.update(conv_intermediates_batch[i])
            fwd.update(fc_intermediates_batch[i])
            predictions.append((output, fwd))

            if not (open_softmax and reconstruction_manager is not None):
                _t0 = time.time()
                sample_loss = softmax_op.cross_entropy_loss(
                    output, label_shares[i], node_id, context=f"ce_loss_{i}"
                )
                if prof_enabled:
                    _pt_add("secure_ce_loss", time.time() - _t0)
                losses.append(sample_loss)
    else:
        # Forward pass for each image in batch (full model per-sample)
        for i in range(batch_size):
            image_share = image_shares[i]
            label_share = label_shares[i]
            
            # Forward pass with progress indicator
            print(f"      → Forward pass {i+1}/{batch_size} (this may take 1-5 minutes per image)...", end='\r', flush=True)
            output, forward_intermediates = lenet5.forward_pass(
                image_share,
                weights,
                node_id=node_id,
                context=f"{batch_context}_s{i}",
                return_intermediates=True,
                reconstruction_manager=reconstruction_manager if open_relu else None,
                open_relu=bool(open_relu),
                relu_opener_node=int(relu_opener_node),
            )
            predictions.append((output, forward_intermediates))
            print(f"      → Forward pass {i+1}/{batch_size} completed ✓", flush=True)
            
            # Compute Cross-Entropy Loss per sample.
            # In opened-softmax mode we compute loss in plaintext on the opener (below),
            # so skip the expensive MPC softmax/log approximation here.
            if not (open_softmax and reconstruction_manager is not None):
                sample_loss = softmax_op.cross_entropy_loss(
                    output, label_share, node_id, context=f"ce_loss_{i}"
                )
                # Store per-sample loss (before averaging)
                losses.append(sample_loss)
    
    # Sum all per-sample losses (still SCALE-scaled)
    total_loss_share = None
    if losses:
        total_loss_share = Share(x=losses[0].x, y=0, node_id=node_id)
        for loss in losses:
            total_loss_share = Share(
                x=total_loss_share.x,
                y=(total_loss_share.y + loss.y) % lenet5.field_size,
                node_id=node_id
            )

    # If we're using opened softmax, we'll compute the batch loss in plaintext on the opener.
    # Otherwise, keep the MPC loss share path.
    loss_display = float("nan")
    if open_softmax and reconstruction_manager is not None:
        # loss_display will be computed during gradient generation below (on opener) and broadcasted as an opened share
        pass
    else:
        # Average loss share (SCALE-scaled)
        if total_loss_share is None:
            avg_loss_share = Share(x=0, y=0, node_id=node_id)
        elif lenet5.divider and batch_size > 0:
            avg_loss_share = lenet5.divider.secure_scalar_divide(
                total_loss_share,
                batch_size,
                node_id,
                context=f"{batch_context}_avg_loss",
            )
        else:
            avg_loss_share = total_loss_share

        # Open loss for monitoring (requires all nodes to participate with same context)
        if reconstruction_manager is not None:
            opened = int(reconstruction_manager.get_reconstructed_value([avg_loss_share], context=f"{batch_context}_avg_loss")) % p
            # Loss is conceptually non-negative. In privacy_mode, secure softmax/log
            # approximations can yield values > p/2 (wrap), so display as unsigned mod-p.
            privacy_mode = bool(getattr(getattr(lenet5, "matrix_ops", None), "multiplier", None) and getattr(lenet5.matrix_ops.multiplier, "privacy_mode", False))
            if not privacy_mode:
                # For non-privacy runs, keep the historical signed interpretation
                if opened > (p // 2):
                    opened = opened - p
            loss_display = float(opened) / float(SCALE_FACTOR)
    
    # Enable backward pass - set to False to disable for testing
    ENABLE_BACKWARD = True
    
    if not ENABLE_BACKWARD:
        print(f"      [NOTE] Backward pass disabled for testing", flush=True)
        return weights, loss_display
    
    # Compute gradients and update weights
    print(f"      → Computing gradients...", end='', flush=True)
    
    # Compute output gradients (dL/dlogits) for each sample.
    # Two modes:
    # - open_softmax=True: opener reconstructs logits, computes softmax/CE in plaintext, then re-shares gradients.
    # - open_softmax=False: use MPC softmax/log approximation path (may be unstable).
    all_output_grads: List[List[Share]] = []
    if open_softmax and reconstruction_manager is not None:
        net = reconstruction_manager.network
        opener = int(softmax_opener_node)
        p_mod = int(lenet5.field_size)
        shamir = ShamirSecretSharing(p_mod)

        # For reporting: opener computes average CE loss in plaintext and shares it as a field element
        batch_loss_scaled_int = 0
        debug_ce_per_sample: List[float] = []
        debug_scale_used: List[str] = []

        for i, (output, _) in enumerate(predictions):
            # Broadcast our local logits share vector for this sample
            ctx = f"{batch_context}_logits_{i}"
            x_local = int(output[0].x)
            local_u32 = np.asarray([int(s.y) & 0xFFFFFFFF for s in output], dtype=np.uint32)
            net.broadcast_vector(ctx, x=x_local, values=local_u32)

            if int(node_id) == opener:
                _t0 = time.time()
                opened_u64 = reconstruction_manager.reconstruct_opened_vector_values(
                    context=ctx, values_local=local_u32, x=x_local, timeout=120.0
                )
                if prof_enabled:
                    _pt_add("open_softmax_open_logits", time.time() - _t0)
                # Convert to signed floats
                opened = opened_u64.astype(np.int64)
                opened = np.where(opened > (p_mod // 2), opened - p_mod, opened)
                T = float(softmax_op.temperature)
                y = np.asarray(label_plain[i], dtype=np.float64)
                y_idx = int(np.argmax(y))

                # FC output is SCALE-scaled (secure_matrix_matrix_multiply_fixed_point does (A*B)/SCALE).
                # Try single-scale first; fall back to scale² only when single-scale gives insane CE.
                CE_SANE_MAX = 50.0
                logits = (opened.astype(np.float64)) / float(SCALE_FACTOR) / T
                m = float(np.max(logits))
                exps = np.exp(logits - m)
                sumexp = float(np.sum(exps))
                probs = exps / sumexp
                ce = (m + float(np.log(sumexp))) - float(logits[y_idx])

                if ce < CE_SANE_MAX:
                    logits_scaled = logits
                    ce_used = ce
                    scale_used = "scale"
                else:
                    # Fallback: FC might be in scale² in some path; try scale²
                    logits_alt = (opened.astype(np.float64)) / (float(SCALE_FACTOR) ** 2) / T
                    m_alt = float(np.max(logits_alt))
                    exps_alt = np.exp(logits_alt - m_alt)
                    sumexp_alt = float(np.sum(exps_alt))
                    probs_alt = exps_alt / sumexp_alt
                    ce_alt = (m_alt + float(np.log(sumexp_alt))) - float(logits_alt[y_idx])
                    if ce_alt < CE_SANE_MAX:
                        logits_scaled = logits_alt
                        probs = probs_alt
                        ce_used = ce_alt
                        scale_used = "scale2"
                    else:
                        logits_scaled = logits
                        ce_used = ce
                        scale_used = "scale"
                ce = ce_used
                batch_loss_scaled_int += int(round(ce * SCALE_FACTOR))
                if debug_loss and int(node_id) == opener:
                    debug_ce_per_sample.append(ce)
                    debug_scale_used.append(scale_used)

                # One-time debug print to sanity-check that opened-softmax loss is sane
                if (not quiet) and (int(node_id) == opener) and (not hasattr(train_lenet5_batch, "_open_softmax_dbg_printed")):
                    setattr(train_lenet5_batch, "_open_softmax_dbg_printed", True)
                    p_true = float(probs[y_idx])
                    log_min = float(np.min(logits_scaled))
                    log_max = float(np.max(logits_scaled))
                    raw_min, raw_max = int(np.min(opened)), int(np.max(opened))
                    print(
                        f"      [open_softmax debug] sample0: y={y_idx}  p_true={p_true:.6f}  ce={ce:.6f}  "
                        f"logits(min,max)=({log_min:.3f},{log_max:.3f})  raw_opened=({raw_min},{raw_max})  T={T:.3f}",
                        flush=True,
                    )

                # Gradient:
                # if p = softmax(z/T), then dL/dz = (p - y)/T.
                grad = (probs - y) / T
                grad_int = np.asarray(np.round(grad * float(SCALE_FACTOR)), dtype=np.int64)
                grad_int = np.mod(grad_int, p_mod).astype(np.int64)

                # Shamir-share each grad component and send per-node vectors
                _t0 = time.time()
                per_node_vec: dict[int, np.ndarray] = {nid: np.empty((len(grad_int),), dtype=np.uint32) for nid in range(1, n_nodes + 1)}
                for j in range(len(grad_int)):
                    shares = shamir.share(int(grad_int[j]) % p_mod, n_nodes, t)
                    for s in shares:
                        per_node_vec[s.node_id][j] = np.uint32(int(s.y) & 0xFFFFFFFF)
                if prof_enabled:
                    _pt_add("open_softmax_share_grads", time.time() - _t0)

                # Send vectors to each node (including self via local assignment)
                for nid in range(1, n_nodes + 1):
                    if nid == opener:
                        continue
                    net.channel.send_vector(nid, f"{batch_context}_grad_{i}_to_{nid}", x=nid, values=per_node_vec[nid])

                # Opener's own gradient shares
                grads_shares = [Share(x=opener, y=int(per_node_vec[opener][j]) % p_mod, node_id=opener) for j in range(len(grad_int))]
                all_output_grads.append(grads_shares)
            else:
                # Non-opener waits for its gradient vector from opener
                ctxg = f"{batch_context}_grad_{i}_to_{node_id}"
                start = time.time()
                got = None
                while time.time() - start < 120.0:
                    recv = net.channel.get_received_vector(ctxg)
                    if opener in recv:
                        values = recv[opener].get("values")
                        arr = np.asarray(values, dtype=np.uint32)
                        got = arr
                        break
                    time.sleep(0.01)
                if got is None:
                    raise RuntimeError(f"Timed out waiting for opener gradients for sample {i}")
                try:
                    net.channel.clear_vector(ctxg)
                except Exception:
                    pass
                grads_shares = [Share(x=int(node_id), y=int(got[j]) % p_mod, node_id=int(node_id)) for j in range(int(got.size))]
                all_output_grads.append(grads_shares)

        if debug_loss and int(node_id) == opener and debug_ce_per_sample:
            avg_ce = sum(debug_ce_per_sample) / len(debug_ce_per_sample)
            avg_scaled = int(round(avg_ce * SCALE_FACTOR))
            print(
                f"      [debug_loss] batch_loss_scaled_int={batch_loss_scaled_int} avg_ce={avg_ce:.4f} "
                f"avg_scaled={avg_scaled} scale_used={debug_scale_used} ce_per_sample={[f'{c:.3f}' for c in debug_ce_per_sample]}",
                flush=True,
            )

        # Compute average loss and publish it as a share opened via reconstruction (so all nodes see the same number)
        if int(node_id) == opener:
            avg_loss_scaled_int = int(round(batch_loss_scaled_int / max(1, batch_size))) % p_mod
            shares = shamir.share(avg_loss_scaled_int, n_nodes, t)
            per_node_loss = {s.node_id: s.y for s in shares}
            for nid in range(1, n_nodes + 1):
                if nid == opener:
                    continue
                # Send as a single-element vector
                net.channel.send_vector(nid, f"{batch_context}_loss_to_{nid}", x=nid, values=np.asarray([per_node_loss[nid] & 0xFFFFFFFF], dtype=np.uint32))

            my_loss_share = Share(x=opener, y=int(per_node_loss[opener]) % p_mod, node_id=opener)
        else:
            ctxl = f"{batch_context}_loss_to_{node_id}"
            start = time.time()
            val = None
            while time.time() - start < 120.0:
                recv = net.channel.get_received_vector(ctxl)
                if opener in recv:
                    values = recv[opener].get("values")
                    arr = np.asarray(values, dtype=np.uint32)
                    if arr.size == 1:
                        val = int(arr[0])
                        break
                time.sleep(0.01)
            if val is None:
                raise RuntimeError("Timed out waiting for opener loss share")
            try:
                net.channel.clear_vector(ctxl)
            except Exception:
                pass
            my_loss_share = Share(x=int(node_id), y=val % p_mod, node_id=int(node_id))

        # Open the average loss share (all nodes participate with same context)
        _t0 = time.time()
        opened = int(reconstruction_manager.get_reconstructed_value([my_loss_share], context=f"{batch_context}_avg_loss"))
        if prof_enabled:
            _pt_add("open_softmax_open_loss", time.time() - _t0)
        if opened > (p_mod // 2):
            opened = opened - p_mod
        loss_display = float(opened) / float(SCALE_FACTOR)
    else:
        for i, (output, _) in enumerate(predictions):
            _t0 = time.time()
            output_grads = softmax_op.cross_entropy_loss_gradient(
                output, label_shares[i], node_id, context=f"ce_grad_{i}"
            )
            if prof_enabled:
                _pt_add("secure_ce_grad", time.time() - _t0)
            all_output_grads.append(output_grads)

        # IMPORTANT (privacy_mode / non-opened softmax):
        # Dealer node can finish secure softmax/CE gradient significantly earlier than other nodes
        # (non-dealer nodes must request triples over the network). If the dealer enters the
        # backward conv kernels early, its first batched opening can time out waiting for peers
        # that are still computing softmax grads. A cheap barrier here keeps all nodes aligned.
        if reconstruction_manager is not None and n_nodes > 1:
            try:
                reconstruction_manager.network.barrier(tag=f"{batch_context}_after_ce_grad", timeout=600.0)
            except Exception as e:
                # If barrier fails (e.g. a node died), continue and let safety mechanisms handle it.
                if not quiet:
                    print(f"\n      [WARNING] Barrier after CE grad failed: {e}", flush=True)
        
        # Debug output (disabled to reduce verbosity)
        # if not quiet and i == 0:
        #     output_grad_vals = []
        #     for grad in output_grads:
        #         g_val = grad.y % lenet5.field_size
        #         if g_val > lenet5.field_size // 2:
        #             g_val = g_val - lenet5.field_size
        #         output_grad_vals.append(g_val / SCALE_FACTOR)
        #     avg_out_grad = sum(abs(g) for g in output_grad_vals) / len(output_grad_vals)
        #     max_out_grad = max(abs(g) for g in output_grad_vals)
        #     print(f"\n      [OUTPUT GRAD DEBUG] Avg: {avg_out_grad:.6f}, Max: {max_out_grad:.6f}, Values: {[f'{g:.4f}' for g in output_grad_vals[:4]]}...", flush=True)
    
    # CRITICAL FIX: Compute weight gradients for EACH image separately, then average
    # The previous code was averaging output gradients but using only the first image's
    # forward intermediates, which is mathematically incorrect and prevents learning.
    all_weight_grads = []
    
    for i in range(batch_size):
        output_grad = all_output_grads[i]
        image_share = image_shares[i]
        output, intermediates = predictions[i]
        
        # Compute weight gradients for this image
        _t0 = time.time()
        weight_grads_i = lenet5.backward_pass(
            output_grad, image_share, weights,
            forward_outputs=intermediates,
            node_id=node_id, context=f"{batch_context}_bwd_s{i}"
        )
        if prof_enabled:
            _pt_add("backward", time.time() - _t0)
        all_weight_grads.append(weight_grads_i)
    
    # Average weight gradients across batch
    # Structure: all_weight_grads[batch_idx][layer_idx][...]
    num_layers = len(all_weight_grads[0])
    weight_grads = []

    def _avg_shares_vector_opened(
        shares_flat: List[Share],
        *,
        divisor: int,
        ctx: str,
    ) -> List[Share]:
        """
        Average a flat list of shares by a public divisor.

        In privacy_mode we use the opener-based vector truncation (fast: 1 round per layer),
        instead of calling secure_scalar_divide per element (which is extremely slow).
        """
        if not shares_flat:
            return []
        if int(divisor) == 1:
            return shares_flat

        mult = getattr(getattr(lenet5, "matrix_ops", None), "multiplier", None)
        p_mod = int(lenet5.field_size)
        x_local = int(shares_flat[0].x)

        # If we can do opened vector truncation, use it
        if mult is not None and getattr(mult, "privacy_mode", False) and getattr(mult, "reconstruction_manager", None) is not None and int(n_nodes) > 1:
            y_local = np.asarray([int(s.y) % p_mod for s in shares_flat], dtype=np.uint64)
            y_out = mult._opened_divide_and_reshare_vector(  # type: ignore[attr-defined]
                values_local_u64=y_local,
                divisor=int(divisor),
                node_id=int(node_id),
                x=int(x_local),
                context_prefix=str(ctx),
                timeout=600.0,
            )
            return [Share(x=x_local, y=int(y_out[i]) % p_mod, node_id=int(node_id)) for i in range(int(y_out.size))]

        # Fallback: field division by modular inverse (fast, but not integer-truncating)
        inv = pow(int(divisor) % p_mod, p_mod - 2, p_mod)
        return [Share(x=s.x, y=(int(s.y) * inv) % p_mod, node_id=int(node_id)) for s in shares_flat]
    
    for layer_idx in range(num_layers):
        _t0 = time.time()
        layer_grads_batch = [all_weight_grads[i][layer_idx] for i in range(batch_size)]
        
        # Determine layer structure
        if isinstance(layer_grads_batch[0][0], Share):
            # 1D layer (bias): layer_grads_batch[i] is List[Share]
            layer_size = len(layer_grads_batch[0])
            # Sum across batch (elementwise), then divide ONCE as a vector (opened truncation).
            sums: List[Share] = []
            for elem_idx in range(layer_size):
                y_sum = 0
                x0 = layer_grads_batch[0][elem_idx].x
                for b in range(batch_size):
                    y_sum = (y_sum + int(layer_grads_batch[b][elem_idx].y)) % int(lenet5.field_size)
                sums.append(Share(x=int(x0), y=int(y_sum), node_id=int(node_id)))
            if batch_size > 1:
                sums = _avg_shares_vector_opened(sums, divisor=int(batch_size), ctx=f"{batch_context}_avg_layer{layer_idx}_1d")
            weight_grads.append(sums)
        elif isinstance(layer_grads_batch[0][0][0], Share):
            # 2D layer (FC weights): layer_grads_batch[i] is List[List[Share]]
            num_rows = len(layer_grads_batch[0])
            num_cols = len(layer_grads_batch[0][0])
            # Sum then divide as a single flat vector (opened truncation), then reshape back.
            flat: List[Share] = []
            xs: List[int] = []
            for row_idx in range(num_rows):
                for col_idx in range(num_cols):
                    x0 = int(layer_grads_batch[0][row_idx][col_idx].x)
                    y_sum = 0
                    for b in range(batch_size):
                        y_sum = (y_sum + int(layer_grads_batch[b][row_idx][col_idx].y)) % int(lenet5.field_size)
                    flat.append(Share(x=x0, y=int(y_sum), node_id=int(node_id)))
                    xs.append(x0)
            if batch_size > 1:
                flat = _avg_shares_vector_opened(flat, divisor=int(batch_size), ctx=f"{batch_context}_avg_layer{layer_idx}_2d")
            # Reshape
            avg_layer_grad: List[List[Share]] = []
            idx = 0
            for row_idx in range(num_rows):
                row: List[Share] = []
                for col_idx in range(num_cols):
                    row.append(flat[idx])
                    idx += 1
                avg_layer_grad.append(row)
            weight_grads.append(avg_layer_grad)
        else:
            # 4D layer (conv weights): layer_grads_batch[i] is List[List[List[List[Share]]]]
            k_h_size = len(layer_grads_batch[0])
            k_w_size = len(layer_grads_batch[0][0])
            c_in_size = len(layer_grads_batch[0][0][0])
            c_out_size = len(layer_grads_batch[0][0][0][0])
            flat: List[Share] = []
            for k_h in range(k_h_size):
                for k_w in range(k_w_size):
                    for c_in in range(c_in_size):
                        for c_out in range(c_out_size):
                            x0 = int(layer_grads_batch[0][k_h][k_w][c_in][c_out].x)
                            y_sum = 0
                            for b in range(batch_size):
                                y_sum = (y_sum + int(layer_grads_batch[b][k_h][k_w][c_in][c_out].y)) % int(lenet5.field_size)
                            flat.append(Share(x=x0, y=int(y_sum), node_id=int(node_id)))
            if batch_size > 1:
                flat = _avg_shares_vector_opened(flat, divisor=int(batch_size), ctx=f"{batch_context}_avg_layer{layer_idx}_4d")
            # Reshape back
            avg_layer_grad_4d: List[List[List[List[Share]]]] = []
            idx = 0
            for k_h in range(k_h_size):
                avg_k_h: List[List[List[Share]]] = []
                for k_w in range(k_w_size):
                    avg_k_w: List[List[Share]] = []
                    for c_in in range(c_in_size):
                        avg_c_in: List[Share] = []
                        for c_out in range(c_out_size):
                            avg_c_in.append(flat[idx])
                            idx += 1
                        avg_k_w.append(avg_c_in)
                    avg_k_h.append(avg_k_w)
                avg_layer_grad_4d.append(avg_k_h)
            weight_grads.append(avg_layer_grad_4d)
        if prof_enabled:
            _pt_add("grad_avg", time.time() - _t0)
    
    print(f" ✓", flush=True)
    print(f"      → Updating weights (lr={learning_rate})...", end='', flush=True)
    
    try:
        _t0 = time.time()
        # Update weights: W_new = W_old - lr * grad   (all in the field, no plaintext peeking)
        # grad is SCALE-scaled. lr is public float; approximate lr as lr_int/lr_scale.
        lr_scale = 1_000_000
        lr_int = int(round(float(learning_rate) * lr_scale)) % p
        inv_lr_scale = pow(lr_scale % p, p - 2, p)
        lr_mul = (lr_int * inv_lr_scale) % p
        updated_weights = []
        
        for layer_idx, (layer_weights, layer_grads) in enumerate(zip(weights, weight_grads)):
            updated_layer = []
            
            # Determine layer type: 4D conv, 2D FC weights, or 1D bias
            # Bias arrays (1D): layer_weights[0] is a Share, not a list
            # FC weights (2D): layer_weights[0] is a list of Shares
            # Conv weights (4D): layer_weights[0][0] is a list
            
            is_bias = hasattr(layer_weights[0], 'y')  # Biases are 1D: list of Shares directly
            is_conv = not is_bias and isinstance(layer_weights[0][0], list) and isinstance(layer_weights[0][0][0], list)
            
            if is_bias:
                # Bias layer: 1D array of Shares (fc1_bias, fc2_bias, fc3_bias)
                for i in range(len(layer_weights)):
                    w = layer_weights[i]
                    g = layer_grads[i]
                    
                    lr_grad_y = (int(g.y) * lr_mul) % p
                    
                    # b - lr * grad
                    updated_b = Share(
                        x=w.x,
                        y=(w.y - lr_grad_y) % lenet5.field_size,
                        node_id=node_id
                    )
                    updated_layer.append(updated_b)
            elif is_conv:
                # Convolutional layer: [K_h x K_w x C_in x C_out]
                for k_h in range(len(layer_weights)):
                    updated_row = []
                    for k_w in range(len(layer_weights[k_h])):
                        updated_ch_in = []
                        for c_in in range(len(layer_weights[k_h][k_w])):
                            updated_ch_out = []
                            for c_out in range(len(layer_weights[k_h][k_w][c_in])):
                                w = layer_weights[k_h][k_w][c_in][c_out]
                                g = layer_grads[k_h][k_w][c_in][c_out]
                                
                                lr_grad_y = (int(g.y) * lr_mul) % p
                                
                                # W - lr * grad
                                updated_w = Share(
                                    x=w.x,
                                    y=(w.y - lr_grad_y) % lenet5.field_size,
                                    node_id=node_id
                                )
                                updated_ch_out.append(updated_w)
                            updated_ch_in.append(updated_ch_out)
                        updated_row.append(updated_ch_in)
                    updated_layer.append(updated_row)
            else:
                # Fully connected layer: [output x input]
                for i in range(len(layer_weights)):
                    updated_row = []
                    for j in range(len(layer_weights[i])):
                        w = layer_weights[i][j]
                        g = layer_grads[i][j]
                        lr_grad_y = (int(g.y) * lr_mul) % p
                        
                        # W - lr * grad
                        updated_w = Share(
                            x=w.x,
                            y=(w.y - lr_grad_y) % lenet5.field_size,
                            node_id=node_id
                        )
                        updated_row.append(updated_w)
                    updated_layer.append(updated_row)
            
            updated_weights.append(updated_layer)
        
        print(f" ✓", flush=True)

        if prof_enabled:
            _pt_add("weight_update", time.time() - _t0)
            _pt_add("batch_total", time.time() - prof_t0)

            mult = getattr(getattr(lenet5, "matrix_ops", None), "multiplier", None)
            stats = None
            try:
                if mult is not None and hasattr(mult, "profile_snapshot_and_reset"):
                    stats = mult.profile_snapshot_and_reset()
            except Exception:
                stats = None

            if int(node_id) == 1:
                msg = (
                    f"[PROFILE] {batch_context} "
                    f"conv={prof.get('conv_fwd', 0.0):.1f}s "
                    f"fc={prof.get('fc_fwd', 0.0):.1f}s "
                    f"ce_loss={prof.get('secure_ce_loss', 0.0):.1f}s "
                    f"ce_grad={prof.get('secure_ce_grad', 0.0):.1f}s "
                    f"open_logits={prof.get('open_softmax_open_logits', 0.0):.1f}s "
                    f"share_grads={prof.get('open_softmax_share_grads', 0.0):.1f}s "
                    f"bwd={prof.get('backward', 0.0):.1f}s "
                    f"avg={prof.get('grad_avg', 0.0):.1f}s "
                    f"upd={prof.get('weight_update', 0.0):.1f}s "
                    f"total={prof.get('batch_total', 0.0):.1f}s"
                )
                if stats:
                    msg += (
                        f" | mult={stats.get('multiply_calls', 0)} "
                        f"mult_batch={stats.get('multiply_batch_calls', 0)} "
                        f"mbv={stats.get('multiply_batch_values_calls', 0)}/{stats.get('multiply_batch_values_elems', 0)} "
                        f"mbv_fp={stats.get('multiply_batch_values_fp_calls', 0)}/{stats.get('multiply_batch_values_fp_elems', 0)} "
                        f"open_div={stats.get('opened_div_vectors', 0)}/{stats.get('opened_div_elems', 0)} "
                        f"dealer_req={stats.get('dealer_triple_vector_requests', 0)}/{stats.get('dealer_triple_vector_elems', 0)}"
                    )
                print(f"\n      {msg}", flush=True)
        
        return updated_weights, loss_display
    except Exception as e:
        print(f" ✗ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Return original weights on error
        return weights, loss_display


def main():
    parser = argparse.ArgumentParser(description='Train LeNet-5 on Kather Texture Dataset')
    parser.add_argument('--node-id', type=int, required=True,
                       help='This node\'s ID (1, 2, 3, ...)')
    parser.add_argument('--n-nodes', type=int, default=5,
                       help='Total number of nodes (default: 5)')
    parser.add_argument('--base-port', type=int, default=8000,
                       help='Base port number (default: 8000)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host address (default: localhost)')
    parser.add_argument('--data-dir', type=str, default='Kather_texture_2016_image_tiles_5000',
                       help='Path to Kather dataset folder')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Mini-batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--temperature', type=float, default=2.0,
                       help='Temperature scaling for softmax (default: 2.0). Higher = softer probabilities, easier learning')
    parser.add_argument(
        '--init-gain',
        type=float,
        default=0.1,
        help='Global multiplier for weight initialization stddev (default: 0.1). '
             'Use 1.0 to restore the previous (much larger) init; if you see huge logits/CE, keep this small.',
    )
    parser.add_argument(
        '--triple-dealer-node',
        type=int,
        default=1,
        help='Node ID to act as trusted Beaver-triple dealer (default: 1). '
             'Required when --enable-network and n-nodes>1 to avoid shared-PRSS triple secrets.'
    )
    parser.add_argument(
        '--privacy-mode',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Strict privacy mode: forces dealer-backed Beaver triples (no PRSS/pool fallbacks) and disables opened-softmax. '
             'Use this when you want end-to-end MPC semantics for softmax/CE/gradients.',
    )
    parser.add_argument(
        '--open-softmax',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Compute softmax/cross-entropy by reconstructing logits on one opener node (inside enclave), '
             'then re-share gradients back to all nodes. This avoids brittle MPC log/exp approximations. '
             'Default: enabled.',
    )
    parser.add_argument(
        '--softmax-opener-node',
        type=int,
        default=1,
        help='Node ID that reconstructs logits and computes softmax/CE in plaintext (default: 1).',
    )
    parser.add_argument(
        '--open-relu',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Compute ReLU masks by reconstructing pre-activations on one opener node, then applying the mask locally on each node. '
             'This avoids the incorrect local-share sign check in secure comparisons and stabilizes training. Default: enabled.',
    )
    parser.add_argument(
        '--relu-opener-node',
        type=int,
        default=1,
        help='Node ID that reconstructs pre-activations and broadcasts ReLU masks (default: 1).',
    )
    parser.add_argument(
        '--fc-batch-simd',
        action='store_true',
        help='Experimental: compute FC1/FC2/FC3 for the whole mini-batch using SIMD-style '
             'secure matrix×matrix multiplication (batched Beaver openings). '
             'Conv/pool is still computed per-sample.',
    )
    parser.add_argument(
        '--packed-pss',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Experimental: exercise true Packed Shamir secret sharing (PSS) across the mini-batch dimension '
             '(k lanes; for n=3,t=1 => k=2). This currently pack+unpacks FC activations/logits (no opening) '
             'to validate packed protocols inside the LeNet-5 training pipeline.',
    )
    parser.add_argument(
        '--profile',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Print per-batch timing breakdown and protocol counters (node 1 only).',
    )
    parser.add_argument(
        '--debug-loss',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Print per-batch CE, scale used, and avg_loss_scaled (opener only) to investigate high loss.',
    )
    parser.add_argument('--num-epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--t', type=int, default=1,
                       help='Privacy threshold (default: 1)')
    parser.add_argument('--s', type=int, default=1,
                       help='Adversarial share limit (default: 1)')
    parser.add_argument('--image-size', type=int, default=32,
                       help='Image size (default: 32 for LeNet-5)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of training samples to use (for quick testing)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce verbose output (disable debug prints)')
    parser.add_argument('--val-frequency', type=int, default=1,
                       help='Compute validation accuracy every N epochs (default: 1, every epoch)')
    parser.add_argument('--enable-network', action='store_true',
                       help='Enable true multi-node MPC networking (required when n-nodes > 1 for secure reconstruction)')
    parser.add_argument('--shared-seed', type=int, default=1337,
                       help='Shared RNG seed for multi-process multi-node runs (ensures identical share/triple generation across nodes).')
    parser.add_argument(
        '--open-final-model',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='After training, reconstruct/open the final model parameters and run a plaintext test. '
             'Defaults to enabled when using true multi-node networking.',
    )
    parser.add_argument(
        '--open-final-model-node',
        type=int,
        default=1,
        help='Node ID that will reconstruct/open the final model and print plaintext test results.',
    )
    parser.add_argument(
        '--secure-test',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Run the (slow) MPC-based test accuracy at the end. Defaults to disabled when opening final model.',
    )
    parser.add_argument(
        '--secure-val',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Run the (slow) MPC-based validation accuracy during training. '
             'Defaults to disabled when opening final model (use plaintext eval instead).',
    )
    parser.add_argument(
        '--open-test-samples',
        type=int,
        default=None,
        help='Max number of test samples for plaintext test on opened model (default: all test samples).',
    )
    
    args = parser.parse_args()

    # Privacy mode: force fully secure softmax path (no opener reconstruction).
    # Note: open_relu is not forced off here because fully secure comparisons/ReLU are
    # still experimental in this codebase; you can toggle it manually.
    if bool(getattr(args, "privacy_mode", False)):
        args.open_softmax = False
    
    # Validate node_id
    if args.node_id < 1 or args.node_id > args.n_nodes:
        print(f"Error: node-id must be between 1 and {args.n_nodes}")
        sys.exit(1)
    
    # Adjust threshold for single-node mode
    # For n_nodes=1, we need t=0 (no privacy requirement with single node)
    if args.n_nodes == 1:
        if args.t >= 1:
            print(f"⚠️  WARNING: Threshold t={args.t} is invalid for single-node mode (n_nodes=1)")
            print(f"   Setting t=0 for single-node mode")
            args.t = 0
    elif args.t >= args.n_nodes:
        print(f"Error: Threshold t={args.t} must be less than n_nodes={args.n_nodes}")
        sys.exit(1)

    if args.open_final_model is None:
        args.open_final_model = (args.n_nodes > 1 and args.enable_network)
    if args.secure_test is None:
        # If we open the model and test in plaintext, skip secure test by default (saves minutes).
        args.secure_test = (not args.open_final_model)
    if args.secure_val is None:
        # If we open the model and can eval in plaintext quickly, skip MPC validation by default.
        args.secure_val = (not args.open_final_model)
    
    print("=" * 70)
    print("LeNet-5 Training on Kather Texture Dataset")
    print("=" * 70)
    print(f"Node ID: {args.node_id}")
    print(f"Total nodes: {args.n_nodes}")
    print(f"Dataset: {args.data_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print(f"Init gain: {args.init_gain}")
    print(f"FC batch SIMD: {bool(getattr(args, 'fc_batch_simd', False))}")
    print(f"Packed PSS (FC activations): {bool(getattr(args, 'packed_pss', False))}")
    print(f"Profiling: {bool(getattr(args, 'profile', False))}")
    print(f"Privacy mode: {bool(getattr(args, 'privacy_mode', False))}")
    print(f"Epochs: {args.num_epochs}")
    print("=" * 70)
    
    # Load dataset
    print("\n[Loading Dataset]")
    try:
        images, labels = load_kather_dataset(
            data_dir=args.data_dir,
            image_size=(args.image_size, args.image_size),
            grayscale=True,
            normalize=True
        )
        
        if len(images) == 0:
            print(f"Error: No images found in {args.data_dir}")
            sys.exit(1)
        
        print(f"Loaded {len(images)} images")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Prepare dataset
    print("\n[Preparing Dataset]")
    processed_images, one_hot_labels_all = prepare_dataset_for_training(
        images, labels, flatten=False
    )
    
    # Split dataset
    train_images, train_labels, val_images, val_labels, test_images, test_labels = split_dataset(
        processed_images, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        shuffle=True, seed=42
    )
    
    # Create one-hot labels for training set from train_labels
    num_classes = len(set(labels))
    one_hot_labels = []
    for label in train_labels:
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[label] = 1.0
        one_hot_labels.append(one_hot)
    
    # Limit samples for quick testing
    if args.max_samples and args.max_samples < len(train_images):
        print(f"\n[Quick Test Mode] Limiting to {args.max_samples} samples")
        # Use stratified sampling (as much as possible) so tiny quick-tests don't
        # accidentally pick a single class and produce misleading results.
        import random
        random.seed(42)  # For reproducibility

        # Build per-class index lists
        class_to_indices: Dict[int, List[int]] = {}
        for i, lab in enumerate(train_labels):
            class_to_indices.setdefault(int(lab), []).append(i)
        for cls in class_to_indices:
            random.shuffle(class_to_indices[cls])

        # Desired counts per class: spread max_samples across classes
        base = int(args.max_samples) // int(num_classes)
        extra = int(args.max_samples) % int(num_classes)
        selected_indices: List[int] = []
        for cls in range(int(num_classes)):
            want = base + (1 if cls < extra else 0)
            if want <= 0:
                continue
            idxs = class_to_indices.get(cls, [])
            take = min(want, len(idxs))
            selected_indices.extend(idxs[:take])

        # If some classes were missing / short, fill remainder from leftover pool
        if len(selected_indices) < int(args.max_samples):
            selected_set = set(selected_indices)
            leftovers = [i for i in range(len(train_images)) if i not in selected_set]
            random.shuffle(leftovers)
            need = int(args.max_samples) - len(selected_indices)
            selected_indices.extend(leftovers[:need])

        # Final shuffle so we don't always group classes
        random.shuffle(selected_indices)
        train_images = [train_images[i] for i in selected_indices]
        train_labels = [train_labels[i] for i in selected_indices]
        one_hot_labels = [one_hot_labels[i] for i in selected_indices]
        
        # Show class distribution of selected samples
        class_counts = {}
        for label in train_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        print(f"  Selected samples class distribution:")
        for cls in sorted(class_counts.keys()):
            print(f"    Class {cls}: {class_counts[cls]} samples")
    
    print(f"Training set: {len(train_images)} samples")
    print(f"Validation set: {len(val_images)} samples")
    print(f"Test set: {len(test_images)} samples")
    
    if len(train_images) > 10:
        print(f"\n⚠️  WARNING: Training {len(train_images)} samples will take a VERY long time!")
        print(f"   Estimated time: ~{len(train_images) * 2} minutes minimum")
        if args.max_samples is None:
            print(f"   Consider using --max-samples 8 or --max-samples 16 for quick testing")
        else:
            # If the user already chose quick-test mode, suggest smaller knobs without contradicting them.
            if int(args.max_samples) > 16:
                print(f"   Tip: for faster iteration, try --max-samples 8 or --max-samples 16")
    
    # Initialize secure operations
    print("\n[Initializing Secure Operations]")
    # Use 2^32 - 5 (nearest prime to 2^32) for larger range and better precision
    field_size = 2**32 - 5  # 4,294,967,291 (prime)
    SCALE_FACTOR = 1000
    shamir = ShamirSecretSharing(field_size)

    # IMPORTANT for multi-process multi-node testing:
    # Each node process must generate *consistent* Shamir shares and Beaver triples.
    # This uses a shared seed so all nodes follow the same RNG stream.
    # (This is a testing convenience; a production SENTRA setup would distribute shares/triples via KVS/offline protocols.)
    if args.n_nodes > 1 and args.enable_network:
        random.seed(args.shared_seed)
        np.random.seed(args.shared_seed)
    
    triple_gen = BeaverTripleGenerator(field_size)
    # Increase pool size significantly for CNN operations (convolution needs many multiplications)
    # For LeNet-5: ~322K multiplications per image, so we need a large pool
    # Estimate: batch_size * 322K multiplications per batch
    # For batch_size=2: ~644K multiplications, so pool should be at least 50K-100K
    # But generating that many upfront is slow, so we use 50K and replenish aggressively
    estimated_mults_per_batch = args.batch_size * 322000  # Conservative estimate
    pool_size = max(50000, min(estimated_mults_per_batch // 10, 200000))  # 10% of batch needs, max 200K
    triple_pool = BeaverTriplePool(triple_gen, initial_size=pool_size)
    print(f"  Beaver triple pool size: {pool_size:,} (estimated {estimated_mults_per_batch:,} mults/batch)")
    reconstruction_manager = None
    if args.n_nodes > 1:
        if args.enable_network:
            node_configs = create_node_configs(args.n_nodes, args.base_port, args.host)
            # Use our configured port so other nodes can connect correctly
            network = create_mpc_network(
                node_id=args.node_id,
                node_configs=node_configs,
                port=node_configs[args.node_id]['port'],
                use_tls=False
            )
            reconstruction_manager = create_reconstruction_manager(network, args.t, field_size=field_size)
            # Barrier so all nodes start training/eval in sync (prevents batched reconstruction timeouts)
            print(f"  [SYNC] Waiting for all nodes at startup barrier (seed={args.shared_seed})...", flush=True)
            network.barrier(tag=f"startup_seed_{args.shared_seed}", timeout=300.0)
            print(f"  [SYNC] Startup barrier complete ✓", flush=True)
            print("  Multi-node network: enabled (batched reconstruction active)")
        else:
            print("  ⚠ Multi-node requested but --enable-network not set.")
            print("    Running in local/simplified mode (NOT secure, NOT representative).")
            print("    Use --enable-network and start one process per node to enable true MPC.")
            print("  Multi-node network: disabled")

    # IMPORTANT:
    # - We intentionally DO NOT use prss_seed for Beaver triples in true multi-node mode.
    #   Instead, we use a trusted dealer inside enclaves to serve deterministic triple shares
    #   keyed by (context_prefix, idx). This prevents any single node from deriving triple secrets.
    if args.n_nodes > 1 and args.enable_network and reconstruction_manager is not None:
        dealer_id = int(args.triple_dealer_node)
        if args.node_id == dealer_id:
            dealer = BeaverTripleDealerService(
                network=reconstruction_manager.network,
                dealer_node_id=dealer_id,
                n_nodes=args.n_nodes,
                t=args.t,
                field_size=field_size,
            )
            dealer.register()
            print(f"  Beaver triple source: trusted dealer (node {dealer_id})")
        else:
            print(f"  Beaver triple source: trusted dealer (node {dealer_id})")
        multiplier = SecureMultiplier(
            triple_pool, args.n_nodes, args.t, field_size,
            reconstruction_manager=reconstruction_manager,
            prss_seed=None,
            triple_dealer_id=dealer_id,
            privacy_mode=bool(getattr(args, "privacy_mode", False)),
        )
    else:
        # Single-node or local/simplified mode
        multiplier = SecureMultiplier(
            triple_pool, args.n_nodes, args.t, field_size,
            reconstruction_manager=reconstruction_manager,
            prss_seed=None,
            triple_dealer_id=None,
            privacy_mode=bool(getattr(args, "privacy_mode", False)),
        )
    comparator = SecureComparator(multiplier, field_size)
    divider = SecureDivider(multiplier, field_size, SCALE_FACTOR)
    from ml_training.secure_softmax import SecureSoftmax
    # Use temperature scaling to reduce overconfidence and make learning easier
    # Temperature > 1.0 makes probabilities softer (less extreme)
    # This helps the model learn from the start instead of being overconfident
    # Default temperature=2.0 means logits are divided by 2 before softmax
    softmax_op = SecureSoftmax(multiplier, divider, field_size, SCALE_FACTOR, temperature=args.temperature)
    
    # Initialize LeNet-5
    print("\n[Initializing LeNet-5 Model]")
    lenet5 = LeNet5(
        n_nodes=args.n_nodes,
        t=args.t,
        multiplier=multiplier,
        field_size=field_size,
        comparator=comparator,
        divider=divider,
        scale_factor=SCALE_FACTOR,
        init_gain=float(args.init_gain),
    )
    
    # Initialize weights
    weights = lenet5.initialize_weights(node_id=args.node_id)
    print("LeNet-5 weights initialized")
    
    # Training loop
    print("\n[Starting Training]")
    print("=" * 70)
    
    # Track training history
    train_loss_history = []  # List of average losses per epoch
    batch_losses = []  # All batch losses for current epoch
    
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")
        batch_losses = []  # Reset for new epoch
        epoch_start_time = time.time()
        
        # Process batches
        num_batches = (len(train_images) + args.batch_size - 1) // args.batch_size
        epoch_batch_times = []
        
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(train_images))
            
            print(f"  [Epoch {epoch + 1}/{args.num_epochs}] Batch {batch_idx + 1}/{num_batches}: Converting to shares...", end='', flush=True)
            
            batch_images = train_images[start_idx:end_idx]
            batch_labels = [one_hot_labels[start_idx + i] for i in range(len(batch_images))]
            
            # Convert to shares
            batch_image_shares = []
            batch_label_shares = []
            
            for img_idx, (img, label) in enumerate(zip(batch_images, batch_labels)):
                if args.batch_size > 1:
                    print(f"\r  [Epoch {epoch + 1}/{args.num_epochs}] Batch {batch_idx + 1}/{num_batches}: Converting image {img_idx+1}/{len(batch_images)}...", end='', flush=True)
                img_shares = image_to_shares(img, args.n_nodes, args.t, args.node_id, field_size, shamir, SCALE_FACTOR)
                label_shares = label_to_shares(label, args.n_nodes, args.t, args.node_id, field_size, shamir, SCALE_FACTOR)
                
                batch_image_shares.append(img_shares)
                batch_label_shares.append(label_shares)
            
            print(f"\r  [Epoch {epoch + 1}/{args.num_epochs}] Batch {batch_idx + 1}/{num_batches}: Running forward pass...", end='', flush=True)
            
            # Train batch
            try:
                batch_context = f"train_e{epoch+1}_b{batch_idx+1}_seed{args.shared_seed}"
                updated_weights, loss = train_lenet5_batch(
                    lenet5, softmax_op, batch_image_shares, batch_label_shares,
                    batch_labels,  # plaintext one-hot labels
                    weights,
                    args.learning_rate, args.node_id,
                    batch_context=batch_context,
                    reconstruction_manager=reconstruction_manager,
                    open_softmax=args.open_softmax,
                    softmax_opener_node=args.softmax_opener_node,
                    open_relu=args.open_relu,
                    relu_opener_node=args.relu_opener_node,
                    fc_batch_simd=bool(args.fc_batch_simd),
                    packed_pss=bool(getattr(args, "packed_pss", False)),
                    n_nodes=args.n_nodes,
                    t=args.t,
                    quiet=args.quiet,
                    profile=bool(getattr(args, "profile", False)),
                    debug_loss=bool(getattr(args, "debug_loss", False)),
                )
                weights = updated_weights
                batch_losses.append(loss)
                batch_time_s = time.time() - batch_start_time
                epoch_batch_times.append(batch_time_s)
                imgs = len(batch_images)
                ips = (imgs / batch_time_s) if batch_time_s > 0 else 0.0
                eta_s = (num_batches - (batch_idx + 1)) * (sum(epoch_batch_times) / len(epoch_batch_times)) if epoch_batch_times else 0.0
                print(
                    f"\r  [Epoch {epoch + 1}/{args.num_epochs}] Batch {batch_idx + 1}/{num_batches}: "
                    f"Loss ≈ {loss:.6f} ✓  time={batch_time_s:.1f}s  ({ips:.3f} img/s)  ETA≈{eta_s/60:.1f}m"
                )
            except Exception as e:
                print(f"\r  [Epoch {epoch + 1}/{args.num_epochs}] Batch {batch_idx + 1}/{num_batches}: Error - {e}")
                import traceback
                traceback.print_exc()
                # Continue with next batch
                continue
        
        # Compute epoch average loss
        if batch_losses:
            avg_loss = sum(batch_losses) / len(batch_losses)
            train_loss_history.append(avg_loss)
            print(f"\n  Epoch {epoch + 1} Summary:")
            print(f"    Average Training Loss: {avg_loss:.6f}")
            epoch_time_s = time.time() - epoch_start_time
            avg_batch_s = (sum(epoch_batch_times) / len(epoch_batch_times)) if epoch_batch_times else 0.0
            print(f"    Epoch time: {epoch_time_s/60:.1f} minutes (avg batch {avg_batch_s:.1f}s)")
            if epoch > 0:
                prev_loss = train_loss_history[epoch - 1]
                loss_change = prev_loss - avg_loss
                change_pct = (loss_change / prev_loss * 100) if prev_loss > 0 else 0
                if loss_change > 0:
                    print(f"    Loss decreased by {loss_change:.6f} ({change_pct:.2f}%) ✓")
                elif loss_change < 0:
                    print(f"    Loss increased by {abs(loss_change):.6f} ({abs(change_pct):.2f}%) ⚠")
                else:
                    print(f"    Loss unchanged")
            
            # Compute validation accuracy (based on frequency setting)
            if args.secure_val and len(val_images) > 0 and (epoch + 1) % args.val_frequency == 0:
                val_start_time = time.time()
                print(f"    Computing validation accuracy...", end='', flush=True)
                try:
                    # Convert validation images to shares (limit to 20 for speed)
                    val_sample_size = min(20, len(val_images))
                    val_image_shares = []
                    val_true_labels = []
                    for i in range(val_sample_size):
                        img_shares = image_to_shares(val_images[i], args.n_nodes, args.t, args.node_id, field_size, shamir, SCALE_FACTOR)
                        val_image_shares.append(img_shares)
                        # val_labels[i] is already an integer label (not one-hot), so use it directly
                        val_true_labels.append(val_labels[i])
                    
                    # Ensure all nodes enter validation together (avoid batched reconstruction mismatches)
                    if args.n_nodes > 1 and args.enable_network and reconstruction_manager is not None:
                        try:
                            network.barrier(tag=f"val_epoch_{epoch+1}_seed_{args.shared_seed}", timeout=300.0)
                        except Exception as e:
                            print(f"\n  [WARNING] Validation barrier failed: {e}")

                    val_accuracy = compute_accuracy(lenet5, val_image_shares, val_true_labels, weights, args.node_id, 
                                                    softmax_op=softmax_op, debug=not args.quiet)
                    val_time_s = time.time() - val_start_time
                    print(f"\r    Validation Accuracy: {val_accuracy*100:.2f}% ({val_sample_size} samples)  time={val_time_s:.1f}s")
                except Exception as e:
                    print(f"\r    Validation Accuracy: Error - {e}")
        
        print(f"Epoch {epoch + 1} completed")
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    
    # Print training summary
    if train_loss_history:
        print("\n[Training Summary]")
        print(f"  Initial Loss (Epoch 1): {train_loss_history[0]:.6f}")
        print(f"  Final Loss (Epoch {args.num_epochs}): {train_loss_history[-1]:.6f}")
        total_improvement = train_loss_history[0] - train_loss_history[-1]
        improvement_pct = (total_improvement / train_loss_history[0] * 100) if train_loss_history[0] > 0 else 0
        print(f"  Total Improvement: {total_improvement:.6f} ({improvement_pct:.2f}%)")
        
        # Show loss trend
        if len(train_loss_history) > 1:
            print("\n  Loss History:")
            for i, loss in enumerate(train_loss_history, 1):
                print(f"    Epoch {i}: {loss:.6f}")
    
    # Compute final test accuracy
    if len(test_images) > 0 and args.open_final_model:
        # Collectively open the final model (shares -> reconstructed parameters), then test in plaintext.
        # All nodes MUST broadcast their parameter-share vector under the same context.
        try:
            open_ctx = f"final_model_open_seed_{args.shared_seed}"
            local_vec_u32, layout = _flatten_model_weights_to_u32(weights, field_size)
            # Use the actual Shamir x-coordinate from our shares (usually equals node_id)
            try:
                x_point = int(weights[0][0][0][0].x)
            except Exception:
                x_point = int(args.node_id)

            if args.n_nodes > 1 and args.enable_network and reconstruction_manager is not None:
                print("\n[Opening Final Model] Entering final open stage...", flush=True)
                # Ensure all nodes arrive here together
                try:
                    network.barrier(tag=f"final_open_enter_seed_{args.shared_seed}", timeout=600.0)
                except Exception as e:
                    print(f"  [WARNING] Final open enter barrier failed: {e}")

                # Everyone broadcasts their local parameter shares
                print(f"  [Opening Final Model] Node {args.node_id}: broadcasting {len(local_vec_u32):,} parameter shares...", flush=True)
                network.broadcast_vector(open_ctx, x=x_point, values=local_vec_u32)
                print(f"  [Opening Final Model] Node {args.node_id}: broadcast done.", flush=True)

                opened_u64 = None
                try:
                    if args.node_id == args.open_final_model_node:
                        print("\n[Opening Final Model Parameters]")
                        opened_u64 = reconstruction_manager.reconstruct_opened_vector_values(
                            context=open_ctx,
                            values_local=local_vec_u32,
                            x=x_point,
                            timeout=600.0,
                        )
                finally:
                    # Always try to release peers even if reconstruction fails on opener
                    try:
                        network.barrier(tag=f"final_open_done_seed_{args.shared_seed}", timeout=600.0)
                    except Exception as e:
                        print(f"  [WARNING] Final open done barrier failed: {e}")

                # Clear vector buffers on all nodes (avoid lingering memory)
                try:
                    network.channel.clear_vector(open_ctx)
                except Exception:
                    pass
            else:
                # Single-node / no-network: local share is the secret (t should be 0 in single-node)
                opened_u64 = local_vec_u32.astype(np.uint64)

            if args.node_id == args.open_final_model_node and opened_u64 is not None:
                # Unpack
                offset = 0
                params_u64: Dict[str, np.ndarray] = {}
                for name, shape, count in layout:
                    params_u64[name] = opened_u64[offset: offset + count].reshape(shape)
                    offset += count

                # Convert to float params
                params: Dict[str, np.ndarray] = {
                    "conv1_w": _field_u64_to_float(params_u64["conv1_w"], field_size, SCALE_FACTOR),
                    "conv2_w": _field_u64_to_float(params_u64["conv2_w"], field_size, SCALE_FACTOR),
                    "fc1_w": _field_u64_to_float(params_u64["fc1_w"], field_size, SCALE_FACTOR),
                    "fc2_w": _field_u64_to_float(params_u64["fc2_w"], field_size, SCALE_FACTOR),
                    "fc3_w": _field_u64_to_float(params_u64["fc3_w"], field_size, SCALE_FACTOR),
                    "fc1_b": _field_u64_to_float(params_u64["fc1_b"], field_size, SCALE_FACTOR),
                    "fc2_b": _field_u64_to_float(params_u64["fc2_b"], field_size, SCALE_FACTOR),
                    "fc3_b": _field_u64_to_float(params_u64["fc3_b"], field_size, SCALE_FACTOR),
                }

                # Plaintext eval on opened model
                print("\n[Plaintext Eval on Opened Model]")
                t0 = time.time()

                # Train accuracy on the actual training subset used in this run (e.g., 20 samples in quick-test mode)
                train_acc = _plain_eval_accuracy(train_images, train_labels, params, max_n=None)

                # Validation accuracy (optionally limit via --open-test-samples to keep it fast on huge runs)
                val_n = args.open_test_samples if args.open_test_samples is not None else min(500, len(val_images))
                val_acc = _plain_eval_accuracy(val_images, val_labels, params, max_n=val_n)

                # Test accuracy (optionally limit)
                test_n = args.open_test_samples if args.open_test_samples is not None else len(test_images)
                test_acc = _plain_eval_accuracy(test_images, test_labels, params, max_n=test_n)

                dt = time.time() - t0
                print(f"  Opened-model Train Accuracy: {train_acc*100:.2f}% ({len(train_images)} samples)")
                print(f"  Opened-model Val Accuracy:   {val_acc*100:.2f}% ({min(int(val_n), len(val_images))} samples)")
                print(f"  Opened-model Test Accuracy:  {test_acc*100:.2f}% ({min(int(test_n), len(test_images))} samples)")
                print(f"  Random Baseline: {100.0/8:.2f}% (8 classes)")
                print(f"  Plaintext eval time: {dt:.1f}s")
        except Exception as e:
            print(f"\n[Opening Final Model] Error - {e}")
            import traceback
            traceback.print_exc()

    # Optional: MPC-based test (slow)
    if len(test_images) > 0 and args.secure_test:
        print("\n[Computing Test Accuracy]")
        try:
            test_start_time = time.time()
            # Convert test images to shares (limit to 50 for speed)
            test_sample_size = min(50, len(test_images))
            test_image_shares = []
            test_true_labels = []
            for i in range(test_sample_size):
                img_shares = image_to_shares(test_images[i], args.n_nodes, args.t, args.node_id, field_size, shamir, SCALE_FACTOR)
                test_image_shares.append(img_shares)
                # test_labels[i] is already an integer label (not one-hot), so use it directly
                test_true_labels.append(test_labels[i])
            
            if args.n_nodes > 1 and args.enable_network and reconstruction_manager is not None:
                try:
                    network.barrier(tag=f"test_seed_{args.shared_seed}", timeout=300.0)
                except Exception as e:
                    print(f"  [WARNING] Test barrier failed: {e}")

            test_accuracy = compute_accuracy(lenet5, test_image_shares, test_true_labels, weights, args.node_id,
                                            softmax_op=softmax_op, debug=not args.quiet)
            test_time_s = time.time() - test_start_time
            print(f"  Test Accuracy: {test_accuracy*100:.2f}% ({test_sample_size} samples)  time={test_time_s:.1f}s")
            print(f"  Random Baseline: {100.0/8:.2f}% (8 classes)")
        except Exception as e:
            print(f"  Test Accuracy: Error - {e}")
    
    print("\n" + "=" * 70)
    
    print("\nPress Enter to close...")
    try:
        input()
    except:
        pass


if __name__ == "__main__":
    main()
