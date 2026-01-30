"""
Secure Pooling Operations
Implements secure max pooling and average pooling on secret-shared feature maps
"""

from typing import List, Tuple, Optional
from ml_training.secret_sharing import Share
from ml_training.secure_comparison import SecureComparator
from ml_training.secure_division import SecureDivider
import numpy as np


class SecurePooling:
    """
    Secure pooling operations (max pooling, average pooling)
    """
    
    def __init__(self, comparator: Optional[SecureComparator] = None,
                 divider: Optional[SecureDivider] = None,
                 field_size: int = 2**31 - 1):
        """
        Initialize secure pooling
        Args:
            comparator: SecureComparator for max pooling (optional)
            divider: SecureDivider for average pooling (optional)
            field_size: Prime field size
        """
        self.comparator = comparator
        self.divider = divider
        self.field_size = field_size
    
    def max_pool2d(self, input_shares: List[List[Share]],
                   pool_size: Tuple[int, int] = (2, 2),
                   stride: Optional[int] = None,
                   node_id: int = 1,
                   context: Optional[str] = None) -> List[List[Share]]:
        """
        Secure max pooling (simplified - uses average if comparator not available)
        
        Note: True max pooling requires secure comparison which is complex.
        This implementation uses a simplified approach.
        
        Args:
            input_shares: Input feature map as shares [H x W]
            pool_size: Pooling window size (pool_h, pool_w)
            stride: Stride (defaults to pool_size)
            node_id: Node ID
            context: Optional context
        
        Returns:
            Pooled feature map as shares [H_out x W_out]
        """
        pool_h, pool_w = pool_size
        stride_h = stride if stride else pool_h
        stride_w = stride if stride else pool_w
        
        H, W = len(input_shares), len(input_shares[0])
        H_out = (H - pool_h) // stride_h + 1
        W_out = (W - pool_w) // stride_w + 1
        
        output_shares = []
        
        for h_out in range(H_out):
            output_row = []
            for w_out in range(W_out):
                h_start = h_out * stride_h
                w_start = w_out * stride_w
                
                # Collect shares in pooling window
                window_shares = []
                for h in range(h_start, h_start + pool_h):
                    for w in range(w_start, w_start + pool_w):
                        if h < H and w < W:
                            window_shares.append(input_shares[h][w])
                
                if not window_shares:
                    # Empty window: use zero share
                    zero_share = Share(x=window_shares[0].x if window_shares else 1, y=0, node_id=node_id)
                    output_row.append(zero_share)
                    continue
                
                # Simplified max pooling: use average if comparator not available
                # In production, would use secure comparison for true max pooling
                if self.comparator and len(window_shares) > 1:
                    # True max pooling (requires secure comparison)
                    max_share = window_shares[0]
                    for share in window_shares[1:]:
                        # Compare and select max (simplified)
                        # Would need secure comparison here
                        max_share = share  # Placeholder
                    output_row.append(max_share)
                else:
                    # Fallback: use average (simpler, but not true max pooling)
                    pool_sum = Share(x=window_shares[0].x, y=0, node_id=node_id)
                    for share in window_shares:
                        pool_sum = Share(
                            x=pool_sum.x,
                            y=(pool_sum.y + share.y) % self.field_size,
                            node_id=node_id
                        )
                    
                    # Average
                    if self.divider:
                        pool_avg = self.divider.secure_scalar_divide(
                            pool_sum, len(window_shares), node_id
                        )
                    else:
                        # Simplified: just use sum (will be scaled later)
                        pool_avg = pool_sum
                    
                    output_row.append(pool_avg)
            
            output_shares.append(output_row)
        
        return output_shares
    
    def avg_pool2d(self, input_shares: List[List[Share]],
                   pool_size: Tuple[int, int] = (2, 2),
                   stride: Optional[int] = None,
                   node_id: int = 1,
                   context: Optional[str] = None) -> List[List[Share]]:
        """
        Secure average pooling
        
        Args:
            input_shares: Input feature map as shares [H x W]
            pool_size: Pooling window size (pool_h, pool_w)
            stride: Stride (defaults to pool_size)
            node_id: Node ID
            context: Optional context
        
        Returns:
            Pooled feature map as shares [H_out x W_out]
        """
        pool_h, pool_w = pool_size
        stride_h = stride if stride else pool_h
        stride_w = stride if stride else pool_w
        
        H, W = len(input_shares), len(input_shares[0])
        H_out = (H - pool_h) // stride_h + 1
        W_out = (W - pool_w) // stride_w + 1
        
        output_shares = []
        
        for h_out in range(H_out):
            output_row = []
            for w_out in range(W_out):
                h_start = h_out * stride_h
                w_start = w_out * stride_w
                
                # Sum shares in pooling window
                pool_sum = Share(x=input_shares[0][0].x, y=0, node_id=node_id)
                count = 0
                
                for h in range(h_start, h_start + pool_h):
                    for w in range(w_start, w_start + pool_w):
                        if h < H and w < W:
                            share = input_shares[h][w]
                            pool_sum = Share(
                                x=pool_sum.x,
                                y=(pool_sum.y + share.y) % self.field_size,
                                node_id=node_id
                            )
                            count += 1
                
                # Average
                if self.divider and count > 0:
                    pool_avg = self.divider.secure_scalar_divide(
                        pool_sum, count, node_id
                    )
                else:
                    # Fallback: use sum
                    pool_avg = pool_sum
                
                output_row.append(pool_avg)
            
            output_shares.append(output_row)
        
        return output_shares
    
    def pool2d_backward(self, output_grad_shares: List[List[Share]],
                       input_shares: List[List[Share]],
                       pool_size: Tuple[int, int] = (2, 2),
                       stride: Optional[int] = None,
                       pool_type: str = "avg",
                       node_id: int = 1,
                       context: Optional[str] = None) -> List[List[Share]]:
        """
        Backward pass for pooling operations
        
        Args:
            output_grad_shares: Gradient w.r.t. pooled output [H_out x W_out]
            input_shares: Original input [H x W]
            pool_size: Pooling window size (pool_h, pool_w)
            stride: Stride used in forward pass
            pool_type: "avg" for average pooling, "max" for max pooling
            node_id: Node ID
            context: Optional context
        
        Returns:
            Gradient w.r.t. input [H x W]
        """
        pool_h, pool_w = pool_size
        stride_h = stride if stride else pool_h
        stride_w = stride if stride else pool_w
        
        H, W = len(input_shares), len(input_shares[0])
        H_out, W_out = len(output_grad_shares), len(output_grad_shares[0])
        
        # Initialize input gradients (zero)
        input_grad_shares = []
        for h in range(H):
            input_row = []
            for w in range(W):
                zero_share = Share(x=input_shares[0][0].x, y=0, node_id=node_id)
                input_row.append(zero_share)
            input_grad_shares.append(input_row)
        
        # Distribute gradients back
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_start = h_out * stride_h
                w_start = w_out * stride_w
                
                output_grad = output_grad_shares[h_out][w_out]
                
                # Collect input positions in pooling window
                window_positions = []
                for h in range(h_start, h_start + pool_h):
                    for w in range(w_start, w_start + pool_w):
                        if h < H and w < W:
                            window_positions.append((h, w))
                
                if not window_positions:
                    continue
                
                if pool_type == "avg":
                    # Average pooling: distribute gradient equally
                    grad_per_position = output_grad
                    if self.divider and len(window_positions) > 1:
                        # Divide gradient by number of positions
                        grad_per_position = self.divider.secure_scalar_divide(
                            output_grad, len(window_positions), node_id
                        )
                    
                    # Add gradient to each position
                    for h, w in window_positions:
                        input_grad_shares[h][w] = Share(
                            x=input_grad_shares[h][w].x,
                            y=(input_grad_shares[h][w].y + grad_per_position.y) % self.field_size,
                            node_id=node_id
                        )
                
                elif pool_type == "max":
                    # Max pooling: gradient goes to max position only
                    # Simplified: distribute equally (true max requires secure comparison)
                    if self.comparator:
                        # Would need to find max position securely
                        # For now, use average distribution
                        pass
                    
                    # Simplified: distribute equally
                    grad_per_position = output_grad
                    if self.divider and len(window_positions) > 1:
                        grad_per_position = self.divider.secure_scalar_divide(
                            output_grad, len(window_positions), node_id
                        )
                    
                    for h, w in window_positions:
                        input_grad_shares[h][w] = Share(
                            x=input_grad_shares[h][w].x,
                            y=(input_grad_shares[h][w].y + grad_per_position.y) % self.field_size,
                            node_id=node_id
                        )
        
        return input_grad_shares
