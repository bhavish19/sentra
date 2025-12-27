"""
MPC Engine for Secure Computation
Performs forward/backward pass on secret-shared data
"""

from typing import List, Optional
from ml_training.secret_sharing import Share, PackedShamirSecretSharing
from ml_training.beaver_triples import SecureMultiplier, BeaverTriplePool, BeaverTripleGenerator
from ml_training.secure_matrix_ops import SecureMatrixOperations, GPUMatrixAccelerator
import numpy as np


class PackedMPCEngine:
    """
    Packed MPC Engine for secure neural network operations
    Operates directly on secret-shared values
    """
    
    def __init__(self, n_nodes: int, t: int, field_size: int = 2**31 - 1, use_gpu: bool = False):
        """
        Initialize MPC engine
        Args:
            n_nodes: Number of nodes
            t: Privacy threshold
            field_size: Prime field size
            use_gpu: Whether to use GPU acceleration (optional)
        """
        self.n_nodes = n_nodes
        self.t = t
        self.field_size = field_size
        self.use_gpu = use_gpu
        self.pss = PackedShamirSecretSharing(field_size)
        self.operation_counter = 0  # For generating unique contexts
        
        # Initialize secure multiplier with Beaver triples
        triple_gen = BeaverTripleGenerator(field_size)
        triple_pool = BeaverTriplePool(triple_gen, initial_size=1000)
        self.multiplier = SecureMultiplier(triple_pool, n_nodes, t, field_size)
        
        # Initialize secure matrix operations
        self.matrix_ops = SecureMatrixOperations(self.multiplier, field_size)
    
    def _generate_context(self, operation: str) -> str:
        """Generate unique context for an operation"""
        self.operation_counter += 1
        return f"{operation}_{self.operation_counter}"
        
        # GPU setup (if available)
        if use_gpu:
            self.gpu_accelerator = GPUMatrixAccelerator()
            self.use_gpu = self.gpu_accelerator.is_available()
        else:
            self.gpu_accelerator = None
            self.use_gpu = False
    
    def _add_shares(self, share1: Share, share2: Share) -> Share:
        """Add two shares (homomorphic addition)"""
        return Share(
            x=share1.x,
            y=(share1.y + share2.y) % self.field_size,
            node_id=share1.node_id
        )
    
    def _multiply_shares(self, share1: Share, share2: Share, node_id: int, 
                        context: Optional[str] = None) -> Share:
        """
        Multiply two shares securely using Beaver triples
        Args:
            share1: First share
            share2: Second share
            node_id: Node ID
            context: Optional context for multi-node reconstruction
        Returns:
            Product share (secure using Beaver triple protocol)
        """
        return self.multiplier.multiply(share1, share2, node_id, context=context)
    
    def _multiply_share_by_scalar(self, share: Share, scalar: int, node_id: Optional[int] = None) -> Share:
        """Multiply share by scalar (homomorphic)"""
        if node_id is None:
            node_id = share.node_id
        return Share(
            x=share.x,
            y=(share.y * scalar) % self.field_size,
            node_id=node_id
        )
    
    def forward_pass(self, packed_input_shares: List[Share], 
                    weights: List[List[List[Share]]], 
                    node_id: int, 
                    activation: str = "relu",
                    context: Optional[str] = None) -> List[Share]:
        """
        Perform optimized forward pass on packed shares using secure matrix operations
        Args:
            packed_input_shares: Packed input shares
            weights: Weight matrices (one per layer)
            node_id: Node ID
            activation: Activation function ("relu", "sigmoid", "linear")
            context: Optional context for multi-node operations
        Returns:
            Output shares
        """
        if context is None:
            context = self._generate_context("forward")
        # Use optimized matrix operations
        return self.matrix_ops.forward_pass(packed_input_shares, weights, node_id, context=context)
    
    def compute_loss(self, predictions: List[Share], targets: List[Share], node_id: int) -> Share:
        """
        Compute loss (MSE) on shares
        Args:
            predictions: Prediction shares
            targets: Target shares
            node_id: Node ID
        Returns:
            Loss share
        """
        # MSE: sum((pred - target)^2) / n
        loss_sum = Share(x=predictions[0].x, y=0, node_id=node_id)
        
        context = self._generate_context("loss")
        for pred, target in zip(predictions, targets):
            diff = Share(
                x=pred.x,
                y=(pred.y - target.y) % self.field_size,
                node_id=node_id
            )
            diff_squared = self._multiply_shares(diff, diff, node_id, context=context)
            loss_sum = self._add_shares(loss_sum, diff_squared)
        
        # Average (simplified - would need secure division)
        n = len(predictions)
        return self._multiply_share_by_scalar(loss_sum, 1, node_id)  # Skip division for minimal
    
    def backward_pass(self, loss_share: Share, predictions: List[Share], 
                     targets: List[Share], weights: List[List[List[Share]]], 
                     node_id: int, input_shares: Optional[List[Share]] = None,
                     context: Optional[str] = None) -> List[List[List[Share]]]:
        """
        Perform optimized backward pass (gradient computation) using secure matrix operations
        Args:
            loss_share: Loss share
            predictions: Prediction shares
            targets: Target shares
            weights: Weight matrices
            node_id: Node ID
            input_shares: Optional input shares for gradient computation
            context: Optional context for multi-node operations
        Returns:
            Gradient matrices (same shape as weights)
        """
        if context is None:
            context = self._generate_context("backward")
        # Compute error gradients
        error_grads = []
        for pred, target in zip(predictions, targets):
            error = Share(
                x=pred.x,
                y=(pred.y - target.y) % self.field_size,
                node_id=node_id
            )
            error_grads.append(error)
        
        # Use predictions as input if input_shares not provided (simplified)
        if input_shares is None:
            input_shares = predictions
        
        # Use optimized matrix operations for backward pass
        return self.matrix_ops.backward_pass(error_grads, input_shares, weights, node_id, context=context)
    
    def update_weights(self, weights: List[List[List[Share]]], 
                      gradients: List[List[List[Share]]], 
                      learning_rate: float, 
                      node_id: int) -> List[List[List[Share]]]:
        """
        Update weights using gradients with optimized matrix operations
        Args:
            weights: Current weight matrices
            gradients: Gradient matrices
            learning_rate: Learning rate
            node_id: Node ID
        Returns:
            Updated weight matrices
        """
        # Use optimized matrix operations
        return self.matrix_ops.update_weights(weights, gradients, learning_rate, node_id)

