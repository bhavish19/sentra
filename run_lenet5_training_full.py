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
    SCALE_FACTOR = 10_000_000
    
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


def train_lenet5_batch(lenet5: LeNet5, softmax_op: SecureSoftmax, image_shares: List[List[List[List[Share]]]],
                      label_shares: List[List[Share]], weights: List,
                      learning_rate: float, node_id: int, quiet: bool = False) -> Tuple[List, float]:
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
    
    # Forward pass for each image in batch
    for i in range(batch_size):
        image_share = image_shares[i]
        label_share = label_shares[i]
        
        # Forward pass with progress indicator
        print(f"      → Forward pass {i+1}/{batch_size} (this may take 1-5 minutes per image)...", end='\r', flush=True)
        output, forward_intermediates = lenet5.forward_pass(
            image_share, weights, node_id=node_id, 
            context=f"batch_{i}", return_intermediates=True
        )
        predictions.append((output, forward_intermediates))
        print(f"      → Forward pass {i+1}/{batch_size} completed ✓", flush=True)
        
        # Debug: Show output values (approximate, from shares)
        if not quiet:
            SCALE_FACTOR = 10_000_000  # Increased from 1M to 10M for better precision
            output_values = [out.y % lenet5.field_size for out in output]
            # Handle potential wraparound
            output_values = [v if v < lenet5.field_size // 2 else v - lenet5.field_size for v in output_values]
            output_scaled = [v / SCALE_FACTOR for v in output_values]
            # FIXED: Use max() not max(key=abs) - we want the largest logit value, not largest absolute value
            max_class = output_values.index(max(output_values))
            # Debug output (disabled to reduce verbosity)
            # print(f"      [DEBUG] Output range: [{min(output_scaled):.4f}, {max(output_scaled):.4f}], Max class: {max_class}")
        
        # Extract output and intermediates
        output, forward_intermediates = predictions[i]
        
        # Debug: Check softmax probabilities before computing loss
        SCALE_FACTOR = 10_000_000
        probs_debug = softmax_op.softmax(output, node_id, f"debug_softmax_{i}")
        prob_values_debug = []
        for prob in probs_debug:
            val = prob.y % lenet5.field_size
            if val > lenet5.field_size // 2:
                val = val - lenet5.field_size
            prob_actual = val / SCALE_FACTOR
            prob_values_debug.append(prob_actual)
        
        # Find true class for debug
        # Since labels are secret-shared, find the share with maximum value
        target_values_debug = []
        target_raw_values = []
        for j, t in enumerate(label_share):
            t_val = t.y % lenet5.field_size
            if t_val > lenet5.field_size // 2:
                t_val = t_val - lenet5.field_size
            target_raw_values.append(t_val)
            t_actual = t_val / SCALE_FACTOR
            target_values_debug.append(t_actual)
        
        # Find the index with maximum raw value (this should be the true class)
        true_class_debug = max(range(len(target_raw_values)), key=lambda i: target_raw_values[i])
        max_target_val = target_values_debug[true_class_debug]
        
        # Debug output (reduced to summary only)
        if not quiet and i == 0:  # Only show for first sample
            prob_sum = sum(prob_values_debug)
            if true_class_debug is not None:
                true_prob_debug = prob_values_debug[true_class_debug]
                # Find predicted class using LOGITS (more precise than softmax probabilities)
                predicted_class_debug = output_scaled.index(max(output_scaled))
                pred_prob_debug = prob_values_debug[predicted_class_debug]
                # Show all probabilities for debugging
                prob_str = ", ".join([f"{p:.4f}" for p in prob_values_debug])
                print(f"      [SOFTMAX] True class: {true_class_debug}, Pred class: {predicted_class_debug} (from logits), True prob: {true_prob_debug:.6f}, Sum: {prob_sum:.4f}", flush=True)
                print(f"      [SOFTMAX] All probs: [{prob_str}]", flush=True)
                # Check if probabilities are too uniform
                prob_range = max(prob_values_debug) - min(prob_values_debug)
                if prob_range < 0.1:
                    print(f"      [WARNING] Probabilities are very uniform (range={prob_range:.4f}) - model lacks discrimination!", flush=True)
                if true_prob_debug < 0.01:
                    print(f"      [WARNING] True class probability is very low ({true_prob_debug:.6f}) - model is predicting wrong class!", flush=True)
            else:
                print(f"      [SOFTMAX] WARNING: Could not find true class!", flush=True)
        
        # Compute Cross-Entropy Loss per sample
        # Cross-entropy: -log(softmax(logits)[true_class])
        sample_loss = softmax_op.cross_entropy_loss(
            output, label_share, node_id, context=f"ce_loss_{i}"
        )
        
        # Store per-sample loss (before averaging)
        losses.append(sample_loss)
    
    # Compute average loss: sum per-sample losses, then divide
    # For Cross-Entropy, loss is already per-sample (not per-element)
    SCALE_FACTOR = 10_000_000  # Scaling factor used when converting to integers (increased from 1M to 10M)
    loss_scale = SCALE_FACTOR  # For cross-entropy, loss is scaled by SCALE_FACTOR (not squared)
    num_elements = batch_size  # For cross-entropy, we average over samples, not elements
    
    # Sum all per-sample losses
    total_loss_share = Share(x=losses[0].x, y=0, node_id=node_id)
    for loss in losses:
        total_loss_share = Share(
            x=total_loss_share.x,
            y=(total_loss_share.y + loss.y) % lenet5.field_size,
            node_id=node_id
        )
    
    # Get the raw loss value and handle wraparound properly
    total_loss_raw = total_loss_share.y
    field_size = lenet5.field_size
    field_half = field_size // 2
    
    # Handle field wraparound more carefully
    # For loss, we always expect positive values (sum of squares)
    # If the value appears negative (wrapped), we need to unwrap it
    if total_loss_raw > field_half:
        # Value is in the upper half of the field
        # This could mean:
        # 1. It's a large positive value (valid)
        # 2. It wrapped around from a negative value (invalid for loss)
        # 
        # For loss (sum of squares), we expect positive values
        # If it's > 3/4 of field_size, it's likely wrapped multiple times
        # Estimate: if > 0.75 * field_size, likely wrapped
        if total_loss_raw > (field_size * 3) // 4:
            # Likely wrapped - unwrap it
            # But loss should be positive, so this suggests the sum is HUGE
            # In this case, we can't accurately compute the loss from a single share
            # For now, use the wrapped value but note it's approximate
            total_loss_value = total_loss_raw - field_size
            # Since loss should be positive, if we get negative, the sum wrapped
            # We'll use abs() but this is an approximation
            total_loss_value = abs(total_loss_value)
        else:
            # Large but valid positive value
            total_loss_value = total_loss_raw
    else:
        # Small to medium positive value (most common case)
        total_loss_value = total_loss_raw
    
    # Compute average loss per sample
    # total_loss_value is sum of cross-entropy losses (scaled by SCALE_FACTOR)
    # Divide by batch_size to get average loss per sample, then by loss_scale to get actual loss
    if num_elements > 0:
        # Average the sum: divide by number of samples
        # Note: This is for display only - the actual secure computation uses shares
        # For display, we approximate by dividing in plaintext
        avg_loss_scaled = total_loss_value / num_elements
        loss_display = avg_loss_scaled / loss_scale
        
        # Additional debug: show if loss might be wrapped
        if not quiet and total_loss_raw > (field_size * 3) // 4:
            print(f"      [WARNING] Loss value may have wrapped (raw={total_loss_raw}, field_size={field_size})", flush=True)
    else:
        loss_display = 0.0
    
    # Debug: Show loss summary (reduced verbosity)
    if not quiet:
        # Only show warnings for unusual loss values
        if loss_display < 0.1 and loss_display > 0:
            print(f"      [WARNING] Loss is very small ({loss_display:.6f}) - may indicate computation issue", flush=True)
        elif loss_display > 10.0:
            print(f"      [WARNING] Loss is very large ({loss_display:.6f}) - may indicate training instability", flush=True)
    
    # Enable backward pass - set to False to disable for testing
    ENABLE_BACKWARD = True
    
    if not ENABLE_BACKWARD:
        print(f"      [NOTE] Backward pass disabled for testing", flush=True)
        return weights, loss_display
    
    # Compute gradients and update weights
    print(f"      → Computing gradients...", end='', flush=True)
    
    # Compute output gradients (loss derivative w.r.t. predictions)
    # For Cross-Entropy with Softmax: dL/dlogits = softmax(logits) - target
    # This is much simpler than MSE!
    all_output_grads = []
    for i, (output, _) in enumerate(predictions):
        # Compute gradient using softmax: gradient = softmax(logits) - target
        output_grads = softmax_op.cross_entropy_loss_gradient(
            output, label_shares[i], node_id, context=f"ce_grad_{i}"
        )
        all_output_grads.append(output_grads)
        
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
        weight_grads_i = lenet5.backward_pass(
            output_grad, image_share, weights,
            forward_outputs=intermediates,
            node_id=node_id, context=f"batch_backward_{i}"
        )
        all_weight_grads.append(weight_grads_i)
    
    # Average weight gradients across batch
    # Structure: all_weight_grads[batch_idx][layer_idx][...]
    num_layers = len(all_weight_grads[0])
    weight_grads = []
    
    for layer_idx in range(num_layers):
        layer_grads_batch = [all_weight_grads[i][layer_idx] for i in range(batch_size)]
        
        # Determine layer structure
        if isinstance(layer_grads_batch[0][0], Share):
            # 1D layer (bias): layer_grads_batch[i] is List[Share]
            # Average each element across batch
            layer_size = len(layer_grads_batch[0])
            avg_layer_grad = []
            for elem_idx in range(layer_size):
                grad_sum = Share(x=layer_grads_batch[0][elem_idx].x, y=0, node_id=node_id)
                for batch_idx in range(batch_size):
                    grad_sum = Share(
                        x=grad_sum.x,
                        y=(grad_sum.y + layer_grads_batch[batch_idx][elem_idx].y) % lenet5.field_size,
                        node_id=node_id
                    )
                # Average
                if lenet5.divider and batch_size > 1:
                    avg_grad = lenet5.divider.secure_scalar_divide(grad_sum, batch_size, node_id)
                else:
                    avg_grad = grad_sum
                avg_layer_grad.append(avg_grad)
            weight_grads.append(avg_layer_grad)
        elif isinstance(layer_grads_batch[0][0][0], Share):
            # 2D layer (FC weights): layer_grads_batch[i] is List[List[Share]]
            num_rows = len(layer_grads_batch[0])
            num_cols = len(layer_grads_batch[0][0])
            avg_layer_grad = []
            for row_idx in range(num_rows):
                avg_row = []
                for col_idx in range(num_cols):
                    grad_sum = Share(x=layer_grads_batch[0][row_idx][col_idx].x, y=0, node_id=node_id)
                    for batch_idx in range(batch_size):
                        grad_sum = Share(
                            x=grad_sum.x,
                            y=(grad_sum.y + layer_grads_batch[batch_idx][row_idx][col_idx].y) % lenet5.field_size,
                            node_id=node_id
                        )
                    # Average
                    if lenet5.divider and batch_size > 1:
                        avg_grad = lenet5.divider.secure_scalar_divide(grad_sum, batch_size, node_id)
                    else:
                        avg_grad = grad_sum
                    avg_row.append(avg_grad)
                avg_layer_grad.append(avg_row)
            weight_grads.append(avg_layer_grad)
        else:
            # 4D layer (conv weights): layer_grads_batch[i] is List[List[List[List[Share]]]]
            # Average each element across batch
            k_h_size = len(layer_grads_batch[0])
            k_w_size = len(layer_grads_batch[0][0])
            c_in_size = len(layer_grads_batch[0][0][0])
            c_out_size = len(layer_grads_batch[0][0][0][0])
            avg_layer_grad = []
            for k_h in range(k_h_size):
                avg_k_h = []
                for k_w in range(k_w_size):
                    avg_k_w = []
                    for c_in in range(c_in_size):
                        avg_c_in = []
                        for c_out in range(c_out_size):
                            grad_sum = Share(x=layer_grads_batch[0][k_h][k_w][c_in][c_out].x, y=0, node_id=node_id)
                            for batch_idx in range(batch_size):
                                grad_sum = Share(
                                    x=grad_sum.x,
                                    y=(grad_sum.y + layer_grads_batch[batch_idx][k_h][k_w][c_in][c_out].y) % lenet5.field_size,
                                    node_id=node_id
                                )
                            # Average
                            if lenet5.divider and batch_size > 1:
                                avg_grad = lenet5.divider.secure_scalar_divide(grad_sum, batch_size, node_id)
                            else:
                                avg_grad = grad_sum
                            avg_c_in.append(avg_grad)
                        avg_k_w.append(avg_c_in)
                    avg_k_h.append(avg_k_w)
                avg_layer_grad.append(avg_k_h)
            weight_grads.append(avg_layer_grad)
    
    # Debug: Check gradient magnitudes (for FC3 layer - output layer)
    # Check ALL gradients, not just a sample, to get accurate statistics
    if not quiet and len(weight_grads) >= 5:  # FC3 is the 5th layer (index 4)
        fc3_grads = weight_grads[4]  # FC3 layer
        grad_magnitudes = []
        for i in range(len(fc3_grads)):  # Check all output neurons
            for j in range(len(fc3_grads[i])):  # Check all input connections
                g_val = fc3_grads[i][j].y % lenet5.field_size
                if g_val > lenet5.field_size // 2:
                    g_val = g_val - lenet5.field_size
                grad_magnitudes.append(g_val / SCALE_FACTOR)
        if grad_magnitudes:
            abs_grads = [abs(g) for g in grad_magnitudes]
            avg_grad_mag = sum(abs_grads) / len(abs_grads)
            max_grad_mag = max(abs_grads)
            min_grad_mag = min(abs_grads)
            non_zero_count = sum(1 for g in abs_grads if g > 1e-10)
            # Compute std dev
            if len(abs_grads) > 1:
                variance = sum((g - avg_grad_mag) ** 2 for g in abs_grads) / len(abs_grads)
                std_grad_mag = variance ** 0.5
            else:
                std_grad_mag = 0.0
            print(f"\n      [GRAD DEBUG] FC3: avg={avg_grad_mag:.6f}, max={max_grad_mag:.6f}, min={min_grad_mag:.6f}, std={std_grad_mag:.6f}, non-zero={non_zero_count}/{len(grad_magnitudes)}", flush=True)
            if avg_grad_mag < 0.0001:
                print(f"      [WARNING] Gradients are very small - weight updates may be negligible", flush=True)
            elif non_zero_count == 0:
                print(f"      [ERROR] All gradients are zero - no learning will occur!", flush=True)
    
    print(f" ✓", flush=True)
    print(f"      → Updating weights (lr={learning_rate})...", end='', flush=True)
    
    try:
        # Update weights: W_new = W_old - lr * grad
        # Note: Learning rate needs to be scaled appropriately
        # CRITICAL FIX: Use higher precision to avoid truncation errors
        # Instead of: lr_grad = (g.y * lr_scaled // SCALE_FACTOR)
        # We compute: lr_grad = round(g.y * learning_rate) in scaled units
        # This preserves precision better than integer division
        updated_weights = []
        
        # Debug: Track weight changes for FC3 layer (track ALL weights, not just a sample)
        weight_changes = []
        
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
                    
                    # Extract gradient value (handling wraparound)
                    g_val = g.y % lenet5.field_size
                    if g_val > lenet5.field_size // 2:
                        g_val = g_val - lenet5.field_size
                    
                    # Convert to actual value, multiply by LR, then scale back
                    g_actual = g_val / SCALE_FACTOR
                    lr_grad_actual = g_actual * learning_rate
                    lr_grad_y = int(round(lr_grad_actual * SCALE_FACTOR))
                    
                    # Handle field wraparound
                    lr_grad_y = lr_grad_y % lenet5.field_size
                    
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
                                
                                # Extract gradient value (handling wraparound)
                                g_val = g.y % lenet5.field_size
                                if g_val > lenet5.field_size // 2:
                                    g_val = g_val - lenet5.field_size
                                
                                # Convert to actual value, multiply by LR, then scale back
                                g_actual = g_val / SCALE_FACTOR
                                lr_grad_actual = g_actual * learning_rate
                                lr_grad_y = int(round(lr_grad_actual * SCALE_FACTOR))
                                
                                # Handle field wraparound
                                lr_grad_y = lr_grad_y % lenet5.field_size
                                
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
                        
                        # Extract gradient value (handling wraparound)
                        g_val = g.y % lenet5.field_size
                        if g_val > lenet5.field_size // 2:
                            g_val = g_val - lenet5.field_size
                        
                        # Convert to actual value, multiply by LR, then scale back
                        # This preserves precision better than integer division
                        g_actual = g_val / SCALE_FACTOR
                        lr_grad_actual = g_actual * learning_rate
                        lr_grad_y = int(round(lr_grad_actual * SCALE_FACTOR))
                        
                        # Handle field wraparound
                        lr_grad_y = lr_grad_y % lenet5.field_size
                        
                        # Track weight changes for FC3 (last layer, index 4) - ALL weights
                        if layer_idx == 4:  # FC3 layer - track all weights
                            w_val = w.y % lenet5.field_size
                            if w_val > lenet5.field_size // 2:
                                w_val = w_val - lenet5.field_size
                            w_actual = w_val / SCALE_FACTOR
                            
                            # lr_grad_actual is already computed above
                            weight_changes.append((w_actual, lr_grad_actual))
                        
                        # W - lr * grad
                        updated_w = Share(
                            x=w.x,
                            y=(w.y - lr_grad_y) % lenet5.field_size,
                            node_id=node_id
                        )
                        updated_row.append(updated_w)
                    updated_layer.append(updated_row)
            
            updated_weights.append(updated_layer)
        
        # Debug: Show weight changes (always show, even in quiet mode, as it's critical)
        if weight_changes:
            abs_changes = [abs(lr_g) for _, lr_g in weight_changes]
            avg_w_change = sum(abs_changes) / len(abs_changes)
            max_w_change = max(abs_changes)
            min_w_change = min(abs_changes)
            num_changed = sum(1 for c in abs_changes if c > 1e-10)
            # Compute std dev
            if len(abs_changes) > 1:
                variance = sum((c - avg_w_change) ** 2 for c in abs_changes) / len(abs_changes)
                std_w_change = variance ** 0.5
            else:
                std_w_change = 0.0
            print(f"\n      [WEIGHT UPDATE] FC3: avg={avg_w_change:.8f}, max={max_w_change:.8f}, min={min_w_change:.8f}, std={std_w_change:.8f}, changed={num_changed}/{len(weight_changes)}", flush=True)
            if avg_w_change < 0.0000001:
                print(f"      [WARNING] Weight changes are extremely small - learning may be too slow", flush=True)
            elif num_changed == 0:
                print(f"      [ERROR] No weights changed - no learning occurred!", flush=True)
            elif avg_w_change > 0.0001:
                print(f"      [INFO] Weight changes look reasonable", flush=True)
        
        print(f" ✓", flush=True)
        
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
        # Use random sampling to ensure class diversity (not just first N samples)
        import random
        random.seed(42)  # For reproducibility
        indices = list(range(len(train_images)))
        random.shuffle(indices)
        selected_indices = indices[:args.max_samples]
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
        print(f"   Consider using --max-samples 2 for quick testing")
    
    # Initialize secure operations
    print("\n[Initializing Secure Operations]")
    # Use 2^32 - 5 (nearest prime to 2^32) for larger range and better precision
    field_size = 2**32 - 5  # 4,294,967,291 (prime)
    SCALE_FACTOR = 10_000_000  # Scaling factor used when converting to integers (increased from 1M to 10M)
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

    multiplier = SecureMultiplier(
        triple_pool, args.n_nodes, args.t, field_size,
        reconstruction_manager=reconstruction_manager,
        prss_seed=(args.shared_seed if (args.n_nodes > 1 and args.enable_network) else None)
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
        divider=divider
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
                updated_weights, loss = train_lenet5_batch(
                    lenet5, softmax_op, batch_image_shares, batch_label_shares, weights,
                    args.learning_rate, args.node_id, quiet=args.quiet
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
