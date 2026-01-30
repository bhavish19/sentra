"""
Diagnostic script to verify weight updates and model learning
Checks if weights are changing, gradients are non-zero, and loss is decreasing
"""

import sys
import os
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.lenet5 import LeNet5
from ml_training.beaver_triples import BeaverTripleGenerator, BeaverTriplePool, SecureMultiplier
from ml_training.secure_comparison import SecureComparator
from ml_training.secure_division import SecureDivider
from ml_training.secret_sharing import ShamirSecretSharing


def extract_weight_values(weights, field_size, scale_factor):
    """Extract actual weight values from shares"""
    weight_values = []
    
    for layer_idx, layer_weights in enumerate(weights):
        layer_values = []
        
        # Check if convolutional layer (4D) or FC layer (2D)
        if isinstance(layer_weights[0][0], list) and isinstance(layer_weights[0][0][0], list):
            # Convolutional layer: [K_h x K_w x C_in x C_out]
            for k_h in range(len(layer_weights)):
                for k_w in range(len(layer_weights[k_h])):
                    for c_in in range(len(layer_weights[k_h][k_w])):
                        for c_out in range(len(layer_weights[k_h][k_w][c_in])):
                            w = layer_weights[k_h][k_w][c_in][c_out]
                            w_val = w.y % field_size
                            if w_val > field_size // 2:
                                w_val = w_val - field_size
                            w_actual = w_val / scale_factor
                            layer_values.append(w_actual)
        else:
            # Fully connected layer: [output x input]
            for i in range(len(layer_weights)):
                for j in range(len(layer_weights[i])):
                    w = layer_weights[i][j]
                    w_val = w.y % field_size
                    if w_val > field_size // 2:
                        w_val = w_val - field_size
                    w_actual = w_val / scale_factor
                    layer_values.append(w_actual)
        
        weight_values.append(layer_values)
    
    return weight_values


def compare_weights(weights_before, weights_after, field_size, scale_factor, layer_names):
    """Compare weights before and after update"""
    print("\n" + "=" * 70)
    print("WEIGHT UPDATE DIAGNOSTICS")
    print("=" * 70)
    
    total_changes = 0
    total_weights = 0
    significant_changes = 0
    
    for layer_idx, (w_before, w_after, layer_name) in enumerate(zip(weights_before, weights_after, layer_names)):
        changes = []
        abs_changes = []
        
        for w_b, w_a in zip(w_before, w_after):
            change = w_a - w_b
            abs_change = abs(change)
            changes.append(change)
            abs_changes.append(abs_change)
            total_weights += 1
            if abs_change > 1e-8:  # Significant change threshold
                significant_changes += 1
                total_changes += 1
        
        if changes:
            mean_change = np.mean(changes)
            std_change = np.std(changes)
            mean_abs_change = np.mean(abs_changes)
            max_abs_change = np.max(abs_changes)
            min_abs_change = np.min(abs_changes)
            
            print(f"\n{layer_name}:")
            print(f"  Total weights: {len(changes)}")
            print(f"  Weights changed: {sum(1 for c in abs_changes if c > 1e-8)} ({100*sum(1 for c in abs_changes if c > 1e-8)/len(changes):.1f}%)")
            print(f"  Mean change: {mean_change:.10f}")
            print(f"  Std change: {std_change:.10f}")
            print(f"  Mean |change|: {mean_abs_change:.10f}")
            print(f"  Max |change|: {max_abs_change:.10f}")
            print(f"  Min |change|: {min_abs_change:.10f}")
            
            # Check if weights are actually changing
            if max_abs_change < 1e-10:
                print(f"  ⚠️  WARNING: No significant weight changes detected!")
            elif max_abs_change < 1e-6:
                print(f"  ⚠️  WARNING: Weight changes are very small (may indicate slow learning)")
            else:
                print(f"  ✓ Weights are updating")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY:")
    print(f"  Total weights: {total_weights}")
    print(f"  Weights changed: {total_changes} ({100*total_changes/total_weights:.1f}%)")
    print(f"  Significant changes: {significant_changes} ({100*significant_changes/total_weights:.1f}%)")
    
    if total_changes == 0:
        print(f"\n  ❌ CRITICAL: No weights are being updated!")
        print(f"     Possible causes:")
        print(f"     1. Gradients are zero")
        print(f"     2. Learning rate is too small")
        print(f"     3. Weight update code has a bug")
    elif significant_changes < total_weights * 0.1:
        print(f"\n  ⚠️  WARNING: Very few weights are changing")
        print(f"     Possible causes:")
        print(f"     1. Learning rate too small")
        print(f"     2. Gradients are very small")
        print(f"     3. Model is stuck in local minimum")
    else:
        print(f"\n  ✓ Weights are updating correctly")
    
    print("=" * 70)


def check_gradient_magnitudes(gradients, field_size, scale_factor, layer_names):
    """Check if gradients are non-zero and reasonable"""
    print("\n" + "=" * 70)
    print("GRADIENT DIAGNOSTICS")
    print("=" * 70)
    
    total_grads = 0
    non_zero_grads = 0
    large_grads = 0
    
    for layer_idx, (layer_grads, layer_name) in enumerate(zip(gradients, layer_names)):
        grad_values = []
        
        # Check if convolutional layer (4D) or FC layer (2D)
        if isinstance(layer_grads[0][0], list) and isinstance(layer_grads[0][0][0], list):
            # Convolutional layer
            for k_h in range(len(layer_grads)):
                for k_w in range(len(layer_grads[k_h])):
                    for c_in in range(len(layer_grads[k_h][k_w])):
                        for c_out in range(len(layer_grads[k_h][k_w][c_in])):
                            g = layer_grads[k_h][k_w][c_in][c_out]
                            g_val = g.y % field_size
                            if g_val > field_size // 2:
                                g_val = g_val - field_size
                            g_actual = g_val / scale_factor
                            grad_values.append(g_actual)
        else:
            # Fully connected layer
            for i in range(len(layer_grads)):
                for j in range(len(layer_grads[i])):
                    g = layer_grads[i][j]
                    g_val = g.y % field_size
                    if g_val > field_size // 2:
                        g_val = g_val - field_size
                    g_actual = g_val / scale_factor
                    grad_values.append(g_actual)
        
        if grad_values:
            mean_grad = np.mean(grad_values)
            std_grad = np.std(grad_values)
            mean_abs_grad = np.mean([abs(g) for g in grad_values])
            max_abs_grad = np.max([abs(g) for g in grad_values])
            min_abs_grad = np.min([abs(g) for g in grad_values])
            non_zero = sum(1 for g in grad_values if abs(g) > 1e-10)
            large = sum(1 for g in grad_values if abs(g) > 1e-3)
            
            total_grads += len(grad_values)
            non_zero_grads += non_zero
            large_grads += large
            
            print(f"\n{layer_name}:")
            print(f"  Total gradients: {len(grad_values)}")
            print(f"  Non-zero gradients: {non_zero} ({100*non_zero/len(grad_values):.1f}%)")
            print(f"  Large gradients (|g| > 1e-3): {large} ({100*large/len(grad_values):.1f}%)")
            print(f"  Mean gradient: {mean_grad:.10f}")
            print(f"  Std gradient: {std_grad:.10f}")
            print(f"  Mean |gradient|: {mean_abs_grad:.10f}")
            print(f"  Max |gradient|: {max_abs_grad:.10f}")
            print(f"  Min |gradient|: {min_abs_grad:.10f}")
            
            if non_zero == 0:
                print(f"  ❌ CRITICAL: All gradients are zero!")
            elif non_zero < len(grad_values) * 0.1:
                print(f"  ⚠️  WARNING: Most gradients are zero")
            elif mean_abs_grad < 1e-6:
                print(f"  ⚠️  WARNING: Gradients are very small")
            else:
                print(f"  ✓ Gradients look reasonable")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY:")
    print(f"  Total gradients: {total_grads}")
    print(f"  Non-zero gradients: {non_zero_grads} ({100*non_zero_grads/total_grads:.1f}%)")
    print(f"  Large gradients: {large_grads} ({100*large_grads/total_grads:.1f}%)")
    
    if non_zero_grads == 0:
        print(f"\n  ❌ CRITICAL: No non-zero gradients!")
        print(f"     Model cannot learn without gradients")
    elif non_zero_grads < total_grads * 0.1:
        print(f"\n  ⚠️  WARNING: Very few non-zero gradients")
        print(f"     Model learning will be very slow")
    else:
        print(f"\n  ✓ Gradients are present and non-zero")
    
    print("=" * 70)


def check_loss_trend(loss_history):
    """Check if loss is decreasing over epochs"""
    print("\n" + "=" * 70)
    print("LOSS TREND DIAGNOSTICS")
    print("=" * 70)
    
    if len(loss_history) < 2:
        print("  ⚠️  Need at least 2 epochs to check loss trend")
        return
    
    print(f"  Loss history: {[f'{l:.6f}' for l in loss_history]}")
    
    # Check if loss is decreasing
    decreasing = sum(1 for i in range(1, len(loss_history)) if loss_history[i] < loss_history[i-1])
    increasing = sum(1 for i in range(1, len(loss_history)) if loss_history[i] > loss_history[i-1])
    
    total_change = loss_history[0] - loss_history[-1]
    change_pct = (total_change / loss_history[0] * 100) if loss_history[0] > 0 else 0
    
    print(f"\n  Epochs with decreasing loss: {decreasing}/{len(loss_history)-1}")
    print(f"  Epochs with increasing loss: {increasing}/{len(loss_history)-1}")
    print(f"  Total change: {total_change:.6f} ({change_pct:.2f}%)")
    
    if total_change > 0:
        print(f"  ✓ Loss is decreasing overall")
    elif total_change < 0:
        print(f"  ⚠️  WARNING: Loss is increasing overall")
        print(f"     Possible causes:")
        print(f"     1. Learning rate too high")
        print(f"     2. Model is diverging")
        print(f"     3. Data issue")
    else:
        print(f"  ⚠️  WARNING: Loss is not changing")
        print(f"     Possible causes:")
        print(f"     1. Learning rate too small")
        print(f"     2. Model is stuck")
        print(f"     3. Weights not updating")
    
    # Check for smooth decrease
    if len(loss_history) >= 3:
        recent_trend = loss_history[-3:]
        if recent_trend[0] > recent_trend[1] > recent_trend[2]:
            print(f"  ✓ Loss is decreasing smoothly in recent epochs")
        elif recent_trend[0] < recent_trend[1] < recent_trend[2]:
            print(f"  ⚠️  WARNING: Loss is increasing in recent epochs")
        else:
            print(f"  ⚠️  Loss trend is unstable")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Check weight updates and model learning')
    parser.add_argument('--loss-history', type=float, nargs='+',
                       help='Loss values from each epoch (space-separated)')
    parser.add_argument('--field-size', type=int, default=2**32 - 5,
                       help='Field size (default: 2^32 - 5)')
    parser.add_argument('--scale-factor', type=int, default=10_000_000,
                       help='Scale factor (default: 10,000,000)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MODEL LEARNING DIAGNOSTICS")
    print("=" * 70)
    print("\nThis script helps diagnose if the model is learning properly.")
    print("It checks:")
    print("  1. If weights are changing between updates")
    print("  2. If gradients are non-zero and reasonable")
    print("  3. If loss is decreasing over epochs")
    print("\nTo use this script:")
    print("  1. Run training with --quiet flag disabled")
    print("  2. Look for [WEIGHT DEBUG] and [GRAD DEBUG] messages")
    print("  3. Check loss history from training output")
    print("  4. Use --loss-history to analyze loss trend")
    
    if args.loss_history:
        check_loss_trend(args.loss_history)
    
    print("\n" + "=" * 70)
    print("MANUAL DIAGNOSTICS")
    print("=" * 70)
    print("\nTo check weight updates during training:")
    print("  1. Run training WITHOUT --quiet flag")
    print("  2. Look for these messages:")
    print("     - [WEIGHT DEBUG] FC3 avg weight change: ...")
    print("     - [GRAD DEBUG] FC3 avg grad mag: ...")
    print("  3. Check if weight changes are > 1e-8")
    print("  4. Check if gradients are > 1e-6")
    print("\nTo check loss trend:")
    print("  1. Look at 'Loss History' in training summary")
    print("  2. Loss should decrease over epochs")
    print("  3. Use --loss-history flag with this script")
    print("\nExpected values:")
    print("  - Weight changes: 1e-6 to 1e-3 (depends on learning rate)")
    print("  - Gradients: 1e-5 to 1e-2 (depends on layer)")
    print("  - Loss: Should decrease by 10-50% over 10 epochs")
    print("=" * 70)


if __name__ == "__main__":
    main()
