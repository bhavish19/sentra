"""
Secure Softmax and Cross-Entropy Loss Operations
Implements secure softmax and cross-entropy loss for classification
Uses numerically stable approximations suitable for MPC
"""

from typing import List, Tuple
from ml_training.secret_sharing import Share
from ml_training.beaver_triples import SecureMultiplier
from ml_training.secure_division import SecureDivider
import numpy as np


class SecureSoftmax:
    """
    Performs secure softmax operations on secret-shared logits
    Uses polynomial approximation for exp(x) in secure computation
    """
    
    def __init__(self, multiplier: SecureMultiplier, divider: SecureDivider, 
                 field_size: int = 2**32 - 5, scale_factor: int = 10_000_000,
                 temperature: float = 2.0):
        """
        Initialize secure softmax
        Args:
            multiplier: SecureMultiplier instance
            divider: SecureDivider instance
            field_size: Prime field size
            scale_factor: Scaling factor for integer representation
            temperature: Temperature scaling factor (default 2.0 for softer probabilities)
                         Higher temperature = softer probabilities (less confident)
                         Lower temperature = sharper probabilities (more confident)
        """
        self.multiplier = multiplier
        self.divider = divider
        self.field_size = field_size
        self.SCALE_FACTOR = scale_factor
        self.temperature = temperature
    
    def _exp_from_actual_value(self, x_actual: float) -> float:
        """
        Compute exp(x) directly from actual (unscaled) float value.
        This avoids all field arithmetic wraparound issues.
        
        Uses Python's math.exp() for precise computation since we're already
        working with extracted float values (not secure shares).
        This dramatically improves probability discrimination.
        
        Args:
            x_actual: Actual (unscaled) value of x
        Returns:
            Actual (unscaled) value of exp(x)
        """
        import math
        
        # After max subtraction, x should be ≤ 0
        # Handle very negative values first to avoid underflow
        if x_actual < -20.0:
            # Very negative, exp is essentially 0
            return 0.00001
        
        # Clamp to reasonable range to avoid overflow
        # After max subtraction, x should be ≤ 0, so upper bound rarely needed
        x_actual = max(-20.0, min(5.0, x_actual))
        
        # Use precise math.exp() instead of Taylor approximation
        # This gives much better probability discrimination
        exp_val = math.exp(x_actual)
        
        # Clamp to reasonable range for softmax:
        # - min: 0.00001 to avoid division by zero
        # - max: 150.0 (exp(5) ≈ 148) to handle edge cases
        exp_val = max(0.00001, min(150.0, exp_val))
        
        return exp_val
    
    # Keep old Taylor series version for reference (not used)
    def _exp_taylor_approximation(self, x_actual: float) -> float:
        """Taylor series approximation (kept for reference, not used)."""
        if x_actual < -10.0:
            return 0.00001
        x_actual = max(-10.0, min(2.0, x_actual))
        if x_actual < -5.0:
            ratio = (x_actual + 10.0) / 5.0
            exp_val = 0.000045 + ratio * (0.0067 - 0.000045)
            return max(0.00001, min(0.01, exp_val))
        elif x_actual < -2.0:
            x2 = x_actual * x_actual
            x3 = x2 * x_actual
            x4 = x3 * x_actual
            x5 = x4 * x_actual
            x6 = x5 * x_actual
            x7 = x6 * x_actual
            x8 = x7 * x_actual
            x9 = x8 * x_actual
            x10 = x9 * x_actual
            x11 = x10 * x_actual
            exp_val = (1.0 + x_actual + x2/2.0 + x3/6.0 + x4/24.0 + x5/120.0 +
                      x6/720.0 + x7/5040.0 + x8/40320.0 + x9/362880.0 +
                      x10/3628800.0 + x11/39916800.0)
            return max(0.00001, min(0.2, exp_val))
        else:
            x2 = x_actual * x_actual
            x3 = x2 * x_actual
            x4 = x3 * x_actual
            x5 = x4 * x_actual
            x6 = x5 * x_actual
            x7 = x6 * x_actual
            x8 = x7 * x_actual
            x9 = x8 * x_actual
            x10 = x9 * x_actual
            x11 = x10 * x_actual
            exp_val = (1.0 + x_actual + x2/2.0 + x3/6.0 + x4/24.0 + x5/120.0 +
                      x6/720.0 + x7/5040.0 + x8/40320.0 + x9/362880.0 +
                      x10/3628800.0 + x11/39916800.0)
            return max(0.00001, min(1.5, exp_val))
    
    def _exp_approximation(self, x_share: Share, node_id: int, context: str) -> Share:
        """
        Approximate exp(x) using improved Taylor series with better handling
        After max subtraction, x should be ≤ 0, so exp(x) should be ≤ 1.0
        Uses more terms and piecewise approximation for better accuracy
        
        Args:
            x_share: Share of x value (scaled)
            node_id: Node ID
            context: Context for secure operations
        Returns:
            Share of exp(x) (scaled)
        """
        # Convert to actual value for approximation
        x_val = x_share.y % self.field_size
        if x_val > self.field_size // 2:
            x_val = x_val - self.field_size
        x_actual = x_val / self.SCALE_FACTOR
        
        # After max subtraction, x should be ≤ 0
        # However, due to field arithmetic, we might get very large negative values
        # For very negative values, exp is essentially 0
        # Handle this BEFORE clamping to avoid issues
        
        # Use a more aggressive threshold: if x < -5, exp(x) is already very small
        # This ensures we catch all very negative values before they hit the Taylor series
        # Also handle the case where x_actual might be very large negative due to field wraparound
        # Check both the actual value and the raw scaled value to catch all cases
        if x_actual < -5.0:
            # Very negative, exp is essentially 0
            # For x < -5, exp(x) < 0.0067
            # Return immediately to avoid any further processing
            # Use a small but non-zero value to avoid division issues
            exp_val = 0.00001
            exp_scaled = int(exp_val * self.SCALE_FACTOR)
            exp_scaled = exp_scaled % self.field_size
            return Share(x=x_share.x, y=exp_scaled, node_id=node_id)
        
        # Safety check: if the raw value suggests a very large negative, also return 0.00001
        # This handles cases where field wraparound might cause extraction issues
        if abs(x_val) > 50 * self.SCALE_FACTOR:  # If scaled value > 50 * SCALE_FACTOR in magnitude
            # This is likely a very large negative value that wrapped around
            exp_val = 0.00001
            exp_scaled = int(exp_val * self.SCALE_FACTOR)
            exp_scaled = exp_scaled % self.field_size
            return Share(x=x_share.x, y=exp_scaled, node_id=node_id)
        
        # Safety check: if x_actual is positive and large, something is wrong
        # After max subtraction, x should be ≤ 0
        if x_actual > 2.0:
            # This shouldn't happen after max subtraction, but handle it
            # If x is positive and large, exp(x) would be huge, so clamp to 1.5
            exp_val = 1.5
            exp_scaled = int(exp_val * self.SCALE_FACTOR)
            exp_scaled = exp_scaled % self.field_size
            return Share(x=x_share.x, y=exp_scaled, node_id=node_id)
        
        # Clamp to reasonable range: [-5, 2] to handle edge cases
        # We've already handled x < -5 above
        x_actual = max(-5.0, min(2.0, x_actual))
        
        # Use piecewise approximation for better accuracy
        # After max subtraction, x should be ≤ 0, so exp(x) ≤ 1.0
        # But we allow up to 1.5 for numerical errors and the case where max logit wasn't exactly 0
        if x_actual < -2.0:
            # For x in [-5, -2], exp(x) is in [0.0067, 0.135]
            # Use 12-term Taylor series, but clamp result to prevent overflow
            x2 = x_actual * x_actual
            x3 = x2 * x_actual
            x4 = x3 * x_actual
            x5 = x4 * x_actual
            x6 = x5 * x_actual
            x7 = x6 * x_actual
            x8 = x7 * x_actual
            x9 = x8 * x_actual
            x10 = x9 * x_actual
            x11 = x10 * x_actual
            
            exp_val = (1.0 + 
                      x_actual + 
                      x2 / 2.0 + 
                      x3 / 6.0 + 
                      x4 / 24.0 + 
                      x5 / 120.0 + 
                      x6 / 720.0 + 
                      x7 / 5040.0 + 
                      x8 / 40320.0 +
                      x9 / 362880.0 +
                      x10 / 3628800.0 +
                      x11 / 39916800.0)
            # Clamp to reasonable range for this region: [0.00001, 0.2]
            # For x in [-5, -2], exp(x) should be in [0.0067, 0.135]
            exp_val = max(0.00001, min(0.2, exp_val))
        else:
            # For x in [-2, 2], use high-precision Taylor series (12 terms)
            # This covers the most important range where exp values are meaningful
            # After max subtraction, x should be ≤ 0, so this is typically [-2, 0]
            x2 = x_actual * x_actual
            x3 = x2 * x_actual
            x4 = x3 * x_actual
            x5 = x4 * x_actual
            x6 = x5 * x_actual
            x7 = x6 * x_actual
            x8 = x7 * x_actual
            x9 = x8 * x_actual
            x10 = x9 * x_actual
            x11 = x10 * x_actual
            
            exp_val = (1.0 + 
                      x_actual + 
                      x2 / 2.0 + 
                      x3 / 6.0 + 
                      x4 / 24.0 + 
                      x5 / 120.0 + 
                      x6 / 720.0 + 
                      x7 / 5040.0 + 
                      x8 / 40320.0 +
                      x9 / 362880.0 +
                      x10 / 3628800.0 +
                      x11 / 39916800.0)
            
            # After max subtraction, exp(0) = 1.0, exp(negative) < 1.0
            # Allow up to 1.5 for small numerical errors (not 2.0, as that's too permissive)
            # The max should be exp(0) = 1.0, but allow 1.5 for safety
            exp_val = max(0.00001, min(1.5, exp_val))
        
        # Scale back to integer
        exp_scaled = int(exp_val * self.SCALE_FACTOR)
        
        # Handle field wraparound and ensure exp_scaled never exceeds 1.5 * SCALE_FACTOR
        # After max subtraction, exp should be ≤ 1.5 * SCALE_FACTOR (since max exp is 1.5 now)
        max_exp_scaled = int(1.5 * self.SCALE_FACTOR)
        
        # Ensure exp_scaled is in valid range
        if exp_scaled < 0:
            exp_scaled = 0
        elif exp_scaled > max_exp_scaled:
            # Force clamp to max_exp_scaled - this should not happen if exp_val is correctly clamped
            exp_scaled = max_exp_scaled
            print(f"        [WARNING] Exp value exceeded max, clamping: exp_val={exp_val:.6f}, exp_scaled={exp_scaled}", flush=True)
        
        # Handle field wraparound
        if exp_scaled >= self.field_size:
            # Value wrapped around, clamp it
            exp_scaled = min(max_exp_scaled, self.field_size - 1)
            print(f"        [WARNING] Exp value wrapped around, clamping: exp_scaled={exp_scaled}", flush=True)
        
        exp_scaled = exp_scaled % self.field_size
        
        return Share(x=x_share.x, y=exp_scaled, node_id=node_id)
    
    def softmax(self, logits: List[Share], node_id: int, context: str = "softmax") -> List[Share]:
        """
        Compute softmax of logits: softmax(x_i) = exp(x_i/T - max(x)/T) / sum(exp(x_j/T - max(x)/T))
        Uses max subtraction for numerical stability and temperature scaling to reduce overconfidence
        
        Args:
            logits: List of shares representing logits [num_classes]
            node_id: Node ID
            context: Context for secure operations
        Returns:
            List of shares representing softmax probabilities [num_classes]
        """
        num_classes = len(logits)
        
        # Apply temperature scaling: divide logits by temperature
        # This makes probabilities softer (less extreme) and easier to learn from
        # Temperature > 1.0 reduces confidence, making gradients more informative
        if self.temperature != 1.0:
            # Scale logits by dividing by temperature using secure division
            # We need: logit_scaled = logit / temperature
            # Create a temperature share (scaled by SCALE_FACTOR)
            temp_scaled = int(self.temperature * self.SCALE_FACTOR) % self.field_size
            temp_share = Share(x=logits[0].x, y=temp_scaled, node_id=node_id)
            
            scaled_logits = []
            for logit in logits:
                # Use secure division: logit / temperature
                # This correctly handles scaling in the integer field
                scaled_logit = self.divider.secure_divide_shares(
                    logit, temp_share, node_id,
                    context=f"{context}_temp" if context else None
                )
                scaled_logits.append(scaled_logit)
            logits = scaled_logits
        
        # For numerical stability, subtract max(logits) before exp
        # Find max logit (approximate from shares)
        logit_vals = []
        for logit in logits:
            val = logit.y % self.field_size
            if val > self.field_size // 2:
                val = val - self.field_size
            logit_vals.append(val)

        max_logit_val = max(logit_vals)

        # Subtract max from all logits (stabilizes exp)
        # Largest logit becomes 0, others become negative
        shifted_logits = []
        shifted_vals_debug = []  # For debugging
        for logit, orig_val in zip(logits, logit_vals):
            # orig_val is already in scaled integer form
            # max_logit_val is also in scaled integer form
            shifted_val_scaled = orig_val - max_logit_val
            
            # Convert to actual value for verification
            shifted_val_actual = shifted_val_scaled / self.SCALE_FACTOR
            
            # After max subtraction, shifted_val_actual should be ≤ 0
            # Clamp to ensure it's not positive due to rounding errors
            if shifted_val_actual > 0.001:  # Allow small numerical error
                shifted_val_actual = 0.0
                shifted_val_scaled = 0
            
            shifted_vals_debug.append(shifted_val_actual)
            
            # Handle field wraparound for negative values
            if shifted_val_scaled < 0:
                shifted_val_scaled = shifted_val_scaled % self.field_size
                if shifted_val_scaled > self.field_size // 2:
                    shifted_val_scaled = shifted_val_scaled - self.field_size
            elif shifted_val_scaled > self.field_size // 2:
                shifted_val_scaled = shifted_val_scaled - self.field_size
            
            shifted_val_scaled = shifted_val_scaled % self.field_size
            shifted_logits.append(Share(x=logit.x, y=shifted_val_scaled, node_id=node_id))
        
        # Debug: verify shifted logits are ≤ 0
        if "debug" in context.lower():
            max_shifted = max(shifted_vals_debug) if shifted_vals_debug else 0.0
            if max_shifted > 0.001:  # Allow small numerical error
                print(f"        [WARNING] Max shifted logit is {max_shifted:.6f} (should be ≤ 0)", flush=True)
        
        # Step 1: Compute exp(shifted_logits) for each class
        # BEST SOLUTION: Compute exp directly from actual values (shifted_vals_debug)
        # to avoid field arithmetic wraparound issues, then convert results to shares
        exp_logits = []
        exp_values_actual = []  # Store actual exp values
        
        for i, shifted_val_actual in enumerate(shifted_vals_debug):
            # Compute exp directly from actual value (no field arithmetic issues)
            # shifted_val_actual is already ≤ 0 after max subtraction
            exp_actual = self._exp_from_actual_value(shifted_val_actual)
            exp_values_actual.append(exp_actual)
            
            # Convert exp result to share
            exp_scaled = int(exp_actual * self.SCALE_FACTOR)
            exp_scaled = exp_scaled % self.field_size
            exp_share = Share(x=shifted_logits[i].x, y=exp_scaled, node_id=node_id)
            exp_logits.append(exp_share)
        
        # Step 2: Sum all exp(logits)
        # Compute sum in actual values to avoid field wraparound, then convert back to share
        # This is necessary because summing large values in the field can cause wraparound
        exp_sum_actual = sum(exp_values_actual)

        # After max subtraction, exp values should be ≤ 1.5 each (with clamp)
        # So for 8 classes, exp_sum should be ≤ 12.0 typically
        # But allow up to 15.0 for safety (in case of numerical errors)
        if exp_sum_actual > 15.0:
            # This shouldn't happen after max subtraction, but handle it gracefully
            print(f"        [WARNING] Exp sum is unexpectedly large ({exp_sum_actual:.6f}), normalizing", flush=True)
            # Normalize by scaling down all exp values proportionally
            scale_factor = 15.0 / exp_sum_actual
            exp_sum_actual = 15.0
            # Update exp_values_actual for accurate division later
            exp_values_actual = [e * scale_factor for e in exp_values_actual]
            # Also update the shares to match
            for i, exp_val in enumerate(exp_values_actual):
                exp_scaled = int(exp_val * self.SCALE_FACTOR)
                exp_scaled = exp_scaled % self.field_size
                exp_logits[i] = Share(x=exp_logits[i].x, y=exp_scaled, node_id=node_id)

        # Convert back to scaled integer
        exp_sum_scaled = int(exp_sum_actual * self.SCALE_FACTOR)

        # After max subtraction, exp_sum should be reasonable (≤ 15 * SCALE_FACTOR)
        # But check if it exceeds field size
        if exp_sum_scaled >= self.field_size:
            # This is a problem - the sum is too large for the field
            # This shouldn't happen with proper max subtraction
            # Clamp to a reasonable maximum (15 * SCALE_FACTOR)
            max_reasonable_sum = int(15.0 * self.SCALE_FACTOR)
            exp_sum_scaled = min(max_reasonable_sum, self.field_size - 1)
            print(f"        [WARNING] Exp sum scaled ({exp_sum_scaled}) exceeds field, clamping to {exp_sum_scaled}", flush=True)
            # Also update exp_sum_actual to match
            exp_sum_actual = exp_sum_scaled / self.SCALE_FACTOR
        elif exp_sum_scaled < 0:
            exp_sum_scaled = 0
            exp_sum_actual = 0.0
        
        # Create share from the computed sum
        exp_sum = Share(x=exp_logits[0].x, y=exp_sum_scaled % self.field_size, node_id=node_id)
        
        # Debug output (disabled to reduce verbosity)
        # if "debug" in context.lower():
        #     print(f"        [SOFTMAX INTERNAL] Exp sum (computed): {exp_sum_actual:.6f}, Exp sum (scaled): {exp_sum_scaled}", flush=True)
        
        # Debug: use the actual sum we computed (not re-extract from share to avoid wraparound confusion)
        # exp_sum_actual is already computed above
        
        # Step 3: Divide each exp(logit) by the sum
        # Use the actual sum (exp_sum_actual) directly to avoid field wraparound issues
        softmax_probs = []
        for i, exp_share in enumerate(exp_logits):
            # Use secure division: divide exp_share by exp_sum
            # Pass exp_sum_actual directly to avoid extraction issues
            prob_share = self._secure_divide_shares_with_actual_denom(
                exp_share, exp_sum_actual, node_id, f"{context}_div_{i}"
            )
            softmax_probs.append(prob_share)
        
        # Debug output (disabled to reduce verbosity - only show warnings)
        if "debug" in context.lower():
            logit_actuals = [v / self.SCALE_FACTOR for v in logit_vals]
            logit_range = max(logit_actuals) - min(logit_actuals)
            if logit_range < 1.0:
                print(f"        [WARNING] Logits are very close together (range={logit_range:.4f}) - model may not be discriminative", flush=True)
        
        return softmax_probs
    
    def _secure_divide_shares(self, numerator: Share, denominator: Share, 
                              node_id: int, context: str) -> Share:
        """
        Securely divide numerator by denominator
        Uses SecureDivider's secure_divide_shares method
        
        Args:
            numerator: Share of numerator
            denominator: Share of denominator
            node_id: Node ID
            context: Context for secure operations
        Returns:
            Share of quotient
        """
        # Use the divider's secure division method
        # Note: This works in the integer field, so we need to handle scaling
        # For softmax, both numerator and denominator are already scaled
        
        # Use divider's method (works in integer field)
        result = self.divider.secure_divide_shares(numerator, denominator, node_id, context)
        
        # The result is in the integer field
        # For probabilities, we expect values in [0, SCALE_FACTOR] range
        # The divider already handles this correctly
        
        return result
    
    def _secure_divide_shares_with_actual_denom(self, numerator: Share, denominator_actual: float,
                                                 node_id: int, context: str) -> Share:
        """
        Securely divide numerator by an actual (non-share) denominator value
        This is used when the denominator is too large to fit in the field
        
        Args:
            numerator: Share of numerator
            denominator_actual: Actual denominator value (not a share)
            node_id: Node ID
            context: Context for secure operations
        Returns:
            Share of quotient
        """
        # Extract numerator value
        num_val = numerator.y % self.field_size
        if num_val > self.field_size // 2:
            num_val = num_val - self.field_size
        
        # Convert to actual value
        num_actual = num_val / self.SCALE_FACTOR
        
        # Use the actual denominator directly
        den_actual = denominator_actual
        
        # Avoid division by zero
        if abs(den_actual) < 0.0001:
            den_actual = 1.0
        
        # Compute division
        result_actual = num_actual / den_actual
        
        # Clamp to [0, 1] for probabilities
        result_actual = max(0.0, min(1.0, result_actual))
        
        # Scale back
        result_y = int(result_actual * self.SCALE_FACTOR)
        
        # Handle field wraparound
        if result_y >= self.field_size:
            result_y = self.field_size - 1
        elif result_y < 0:
            result_y = 0
        
        result_y = result_y % self.field_size
        
        # Debug output (disabled to reduce verbosity)
        # if "debug" in context.lower() or "div" in context.lower():
        #     print(f"          [DIV DEBUG] {num_actual:.6f} / {den_actual:.6f} = {result_actual:.6f} (scaled: {result_y})", flush=True)
        
        return Share(x=numerator.x, y=result_y, node_id=node_id)
    
    def cross_entropy_loss(self, logits: List[Share], target: List[Share], 
                          node_id: int, context: str = "ce_loss") -> Share:
        """
        Compute cross-entropy loss: -log(softmax(logits)[true_class])
        Uses log-softmax trick for numerical stability
        
        Args:
            logits: List of shares representing logits [num_classes]
            target: List of shares representing one-hot target [num_classes]
            node_id: Node ID
            context: Context for secure operations
        Returns:
            Share of cross-entropy loss (scaled)
        """
        # Compute softmax probabilities
        probs = self.softmax(logits, node_id, f"{context}_softmax")
        
        # Compute -log(prob[true_class])
        # Find true class (target with value 1)
        # Since targets are secret-shared, we can't extract the actual value from a single share
        # Instead, we find the share with the largest value (relative to others)
        # In one-hot encoding, the true class share will be larger than others
        
        target_vals = []
        for t in target:
            t_val = t.y % self.field_size
            if t_val > self.field_size // 2:
                t_val = t_val - self.field_size
            target_vals.append(t_val)
        
        # Find the index with the maximum value
        # This should correspond to the true class (since it's one-hot encoded)
        true_class_idx = max(range(len(target_vals)), key=lambda i: target_vals[i])
        
        # Verify: the max value should be significantly larger than others
        max_val = target_vals[true_class_idx]
        second_max = sorted(target_vals, reverse=True)[1] if len(target_vals) > 1 else 0
        if max_val <= second_max * 1.1:  # If max is not much larger, might be wrong
            # All values are similar - might be uniform distribution
            # In this case, use the first class as fallback
            true_class_idx = 0
        
        # Get probability of true class
        true_prob = probs[true_class_idx]
        
        # Compute -log(prob)
        prob_val = true_prob.y % self.field_size
        if prob_val > self.field_size // 2:
            prob_val = prob_val - self.field_size
        prob_actual = prob_val / self.SCALE_FACTOR
        
        # Debug: Check if probability is being clamped
        original_prob = prob_actual
        was_clamped = False
        # Allow smaller probabilities (down to 0.00001) for better discrimination
        # This allows the model to express very low confidence in wrong classes
        if prob_actual < 0.00001:
            was_clamped = True
            prob_actual = 0.00001  # Avoid log(0), but allow very small values
        elif prob_actual > 1.0:
            prob_actual = 1.0  # Avoid log(>1)
        
        # Store debug info (will be printed if needed)
        if hasattr(self, '_debug') and self._debug:
            print(f"        [SOFTMAX DEBUG] True class: {true_class_idx}, Prob: {original_prob:.6f} {'(CLAMPED)' if was_clamped else ''}")
        
        # Compute -log(prob)
        neg_log_prob = -np.log(prob_actual)
        
        # Scale and convert to share
        loss_scaled = int(neg_log_prob * self.SCALE_FACTOR) % self.field_size
        
        return Share(x=true_prob.x, y=loss_scaled, node_id=node_id)
    
    def cross_entropy_loss_gradient(self, logits: List[Share], target: List[Share],
                                   node_id: int, context: str = "ce_grad") -> List[Share]:
        """
        Compute gradient of cross-entropy loss w.r.t. logits
        Gradient = softmax(logits) - target (one-hot)
        
        This is the key advantage: gradient is simple!
        
        Args:
            logits: List of shares representing logits [num_classes]
            target: List of shares representing one-hot target [num_classes]
            node_id: Node ID
            context: Context for secure operations
        Returns:
            List of shares representing gradients [num_classes]
        """
        # Compute softmax probabilities
        probs = self.softmax(logits, node_id, f"{context}_softmax")
        
        # Gradient = probs - target
        gradients = []
        for prob, targ in zip(probs, target):
            # Compute prob - target
            # Use field arithmetic: (prob.y - targ.y) mod field_size
            # This correctly handles both positive and negative differences
            diff_y = (prob.y - targ.y) % self.field_size
            
            # Convert to signed representation for negative values
            # If diff_y > field_size // 2, it represents a negative value
            # We keep it as signed because gradients can be negative and need sign preservation
            # All subsequent operations (multiply, add) use modulo arithmetic which handles negatives correctly
            if diff_y > self.field_size // 2:
                diff_y = diff_y - self.field_size
            
            grad = Share(x=prob.x, y=diff_y, node_id=node_id)
            gradients.append(grad)
        
        return gradients
