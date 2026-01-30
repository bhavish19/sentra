"""
Secure Division Operations
Implements secure division and averaging on secret-shared values
"""

from typing import List
from ml_training.secret_sharing import Share
from ml_training.beaver_triples import SecureMultiplier


class SecureDivider:
    """
    Performs secure division operations on secret-shared values
    """
    
    def __init__(self, multiplier: SecureMultiplier, field_size: int = 2**31 - 1, scale_factor: int = 10_000_000):
        """
        Initialize secure divider
        Args:
            multiplier: SecureMultiplier instance
            field_size: Prime field size
            scale_factor: Scaling factor for integer representation
        """
        self.multiplier = multiplier
        self.field_size = field_size
        self.scale_factor = scale_factor
    
    def secure_scalar_divide(self, share: Share, divisor: int, node_id: int) -> Share:
        """
        Divide share by a public scalar
        Args:
            share: Share to divide
            divisor: Public divisor
            node_id: Node ID
        Returns:
            Share of quotient
        """
        # Compute modular inverse of divisor
        inv = self._mod_inverse(divisor)
        
        # For scalar multiplication, we can directly multiply the share's y value
        # This is more efficient and correct than using Beaver triples for public values
        return Share(
            x=share.x,
            y=(share.y * inv) % self.field_size,
            node_id=node_id
        )
    
    def secure_average(self, shares: List[Share], node_id: int) -> Share:
        """
        Compute average of shares
        Args:
            shares: List of shares
            node_id: Node ID
        Returns:
            Share of average
        """
        if not shares:
            return Share(x=0, y=0, node_id=node_id)
        
        # Sum all shares
        sum_share = Share(x=shares[0].x, y=0, node_id=node_id)
        for share in shares:
            sum_share = Share(
                x=sum_share.x,
                y=(sum_share.y + share.y) % self.field_size,
                node_id=node_id
            )
        
        # Divide by count
        n = len(shares)
        return self.secure_scalar_divide(sum_share, n, node_id)
    
    def secure_divide_shares(self, numerator: Share, denominator: Share, 
                            node_id: int, context: str = "div") -> Share:
        """
        Securely divide numerator by denominator (both are shares)
        
        For MPC, this requires computing the inverse of denominator securely.
        Currently uses approximation - in production, use full secure division protocol.
        
        Args:
            numerator: Share of numerator
            denominator: Share of denominator
            node_id: Node ID
            context: Context for secure operations
        Returns:
            Share of quotient
        """
        # Extract values (handling wraparound)
        num_val = numerator.y % self.field_size
        if num_val > self.field_size // 2:
            num_val = num_val - self.field_size
        
        den_val = denominator.y % self.field_size
        if den_val > self.field_size // 2:
            den_val = den_val - self.field_size
        
        # Both numerator and denominator are already scaled by SCALE_FACTOR
        # So we need to compute: (num / SCALE_FACTOR) / (den / SCALE_FACTOR) = num / den
        # Then scale the result: (num / den) * SCALE_FACTOR
        
        # Avoid division by zero
        if abs(den_val) < 1:
            # Very small denominator - set to SCALE_FACTOR (1.0 scaled) to avoid division by zero
            den_val = 1  # Will be handled below
        
        # Compute division in actual values
        # Convert to actual values (divide by scale factor)
        # Both numerator and denominator are scaled by scale_factor
        num_actual = num_val / self.scale_factor
        den_actual = den_val / self.scale_factor
        
        # Avoid division by zero
        if abs(den_actual) < 0.0001:
            den_actual = 1.0
        
        # Compute division
        result_actual = num_actual / den_actual
        
        # Clamp to [0, 1] for probabilities
        result_actual = max(0.0, min(1.0, result_actual))
        
        # Scale back
        result_y = int(result_actual * self.scale_factor)
        
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
    
    def _mod_inverse(self, a: int) -> int:
        """Compute modular inverse using extended Euclidean algorithm"""
        if a < 0:
            a = a % self.field_size
        
        old_r, r = a, self.field_size
        old_s, s = 1, 0
        
        while r != 0:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
        
        if old_r != 1:
            raise ValueError(f"{a} has no modular inverse modulo {self.field_size}")
        
        return old_s % self.field_size


class SecureAverager:
    """
    Performs secure averaging operations for gradient aggregation
    """
    
    def __init__(self, divider: SecureDivider):
        """
        Initialize secure averager
        Args:
            divider: SecureDivider instance
        """
        self.divider = divider
    
    def average_gradient_batch(self, gradient_batch: List[List[Share]], 
                               batch_size: int, node_id: int) -> List[Share]:
        """
        Average gradients across a batch
        Args:
            gradient_batch: List of gradient lists (one per sample)
            batch_size: Batch size
            node_id: Node ID
        Returns:
            Averaged gradient shares
        """
        if not gradient_batch:
            return []
        
        # Sum gradients across batch
        num_grads = len(gradient_batch[0])
        summed_grads = []
        
        for i in range(num_grads):
            sum_share = Share(x=gradient_batch[0][i].x, y=0, node_id=node_id)
            for grad_list in gradient_batch:
                sum_share = Share(
                    x=sum_share.x,
                    y=(sum_share.y + grad_list[i].y) % self.divider.field_size,
                    node_id=node_id
                )
            summed_grads.append(sum_share)
        
        # Average by dividing by batch size
        averaged_grads = []
        for sum_share in summed_grads:
            avg_share = self.divider.secure_scalar_divide(sum_share, batch_size, node_id)
            averaged_grads.append(avg_share)
        
        return averaged_grads


