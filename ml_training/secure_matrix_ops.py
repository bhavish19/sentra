"""
Secure Matrix Operations
Optimized matrix operations on secret-shared values for efficient neural network training
"""

from typing import List, Optional
from ml_training.secret_sharing import Share
from ml_training.beaver_triples import SecureMultiplier
import numpy as np


class SecureMatrixMultiplier:
    """
    Performs secure matrix multiplication on secret-shared matrices
    Optimized for neural network operations
    """
    
    def __init__(self, multiplier: SecureMultiplier, field_size: int = 2**31 - 1):
        """
        Initialize secure matrix multiplier
        Args:
            multiplier: SecureMultiplier instance
            field_size: Prime field size
        """
        self.multiplier = multiplier
        self.field_size = field_size
    
    def secure_matrix_multiply(self, A_shares: List[List[Share]], 
                               B_shares: List[List[Share]], 
                               node_id: int,
                               context: Optional[str] = None) -> List[List[Share]]:
        """
        Multiply two matrices securely: C = A @ B
        Args:
            A_shares: Matrix A as shares (m x k)
            B_shares: Matrix B as shares (k x n)
            node_id: Node ID
            context: Optional context for multi-node operations
        Returns:
            Matrix C as shares (m x n)
        """
        m = len(A_shares)
        k = len(A_shares[0]) if A_shares else 0
        n = len(B_shares[0]) if B_shares and B_shares[0] else 0
        
        if k != len(B_shares):
            raise ValueError(f"Matrix dimension mismatch: A is {m}x{k}, B is {len(B_shares)}x{n}")
        
        # Initialize result matrix
        C_shares = []
        
        for i in range(m):
            row = []
            for j in range(n):
                # Compute dot product: C[i][j] = sum(A[i][k] * B[k][j])
                dot_product = Share(x=A_shares[0][0].x, y=0, node_id=node_id)
                
                for k_idx in range(k):
                    # Generate unique context for each multiplication
                    mult_context = f"{context}_m{i}_{j}_k{k_idx}" if context else None
                    # Multiply A[i][k_idx] * B[k_idx][j]
                    product = self.multiplier.multiply(
                        A_shares[i][k_idx], 
                        B_shares[k_idx][j], 
                        node_id,
                        context=mult_context
                    )
                    # Add to sum
                    dot_product = Share(
                        x=dot_product.x,
                        y=(dot_product.y + product.y) % self.field_size,
                        node_id=node_id
                    )
                
                row.append(dot_product)
            C_shares.append(row)
        
        return C_shares
    
    def secure_matrix_vector_multiply(self, A_shares: List[List[Share]], 
                                      v_shares: List[Share], 
                                      node_id: int,
                                      context: Optional[str] = None) -> List[Share]:
        """
        Multiply matrix by vector securely: y = A @ v
        Optimized for neural network forward pass
        Args:
            A_shares: Matrix A as shares (m x n)
            v_shares: Vector v as shares (n,)
            node_id: Node ID
            context: Optional context for multi-node operations
        Returns:
            Vector y as shares (m,)
        """
        if not A_shares or not v_shares:
            return []
        
        m = len(A_shares)
        n = len(v_shares)
        A_n = len(A_shares[0]) if A_shares else 0
        
        # Check dimension compatibility
        if A_n != n:
            # Try to handle dimension mismatch gracefully
            # If input is larger, truncate; if smaller, pad with zeros
            if n > A_n:
                v_shares = v_shares[:A_n]
            elif n < A_n:
                # Pad with zero shares
                zero_share = Share(x=v_shares[0].x, y=0, node_id=node_id)
                v_shares = v_shares + [zero_share] * (A_n - n)
            n = A_n
        
        # Stage-2 speedup: use array-based multiply_batch_values to avoid per-element Share objects.
        x0 = v_shares[0].x
        A_y = np.array([[A_shares[i][j].y for j in range(n)] for i in range(m)], dtype=np.uint64)
        v_y = np.array([v_shares[j].y for j in range(n)], dtype=np.uint64)

        # Build (m*n) elementwise products in one go: flatten A and tile v
        y1 = A_y.reshape(-1)
        y2 = np.tile(v_y, m)

        base = context if context else "matvec"
        chunk = 16384
        prod = np.empty_like(y1, dtype=np.uint64)
        for start in range(0, y1.size, chunk):
            end = min(start + chunk, y1.size)
            prod[start:end] = self.multiplier.multiply_batch_values(
                y1[start:end], y2[start:end], x=x0, node_id=node_id, context_prefix=f"{base}_mv_{start}"
            )
        prod = prod.reshape(m, n)
        out = (prod.sum(axis=1) % self.field_size)

        return [Share(x=x0, y=int(out[i] % self.field_size), node_id=node_id) for i in range(m)]
    
    def secure_matrix_add(self, A_shares: List[List[Share]], 
                         B_shares: List[List[Share]]) -> List[List[Share]]:
        """
        Add two matrices securely: C = A + B
        Args:
            A_shares: Matrix A as shares
            B_shares: Matrix B as shares
            node_id: Node ID
        Returns:
            Matrix C as shares
        """
        m = len(A_shares)
        n = len(A_shares[0]) if A_shares else 0
        
        if len(B_shares) != m or (B_shares and len(B_shares[0]) != n):
            raise ValueError("Matrix dimension mismatch for addition")
        
        C_shares = []
        for i in range(m):
            row = []
            for j in range(n):
                # Homomorphic addition
                sum_share = Share(
                    x=A_shares[i][j].x,
                    y=(A_shares[i][j].y + B_shares[i][j].y) % self.field_size,
                    node_id=A_shares[i][j].node_id
                )
                row.append(sum_share)
            C_shares.append(row)
        
        return C_shares
    
    def secure_matrix_scalar_multiply(self, A_shares: List[List[Share]], 
                                     scalar: int, 
                                     node_id: int) -> List[List[Share]]:
        """
        Multiply matrix by scalar securely: B = scalar * A
        Args:
            A_shares: Matrix A as shares
            scalar: Scalar value
            node_id: Node ID
        Returns:
            Matrix B as shares
        """
        B_shares = []
        for row in A_shares:
            new_row = []
            for share in row:
                new_share = Share(
                    x=share.x,
                    y=(share.y * scalar) % self.field_size,
                    node_id=node_id
                )
                new_row.append(new_share)
            B_shares.append(new_row)
        
        return B_shares
    
    def secure_matrix_transpose(self, A_shares: List[List[Share]]) -> List[List[Share]]:
        """
        Transpose matrix securely (no computation needed, just rearrange)
        Args:
            A_shares: Matrix A as shares
        Returns:
            Transposed matrix as shares
        """
        if not A_shares:
            return []
        
        m = len(A_shares)
        n = len(A_shares[0])
        
        A_T = []
        for j in range(n):
            row = []
            for i in range(m):
                row.append(A_shares[i][j])
            A_T.append(row)
        
        return A_T


class SecureMatrixOperations:
    """
    High-level secure matrix operations for neural networks
    Optimized for forward/backward pass
    """
    
    def __init__(self, multiplier: SecureMultiplier, field_size: int = 2**31 - 1):
        """
        Initialize secure matrix operations
        Args:
            multiplier: SecureMultiplier instance
            field_size: Prime field size
        """
        self.matrix_multiplier = SecureMatrixMultiplier(multiplier, field_size)
        self.multiplier = multiplier
        self.field_size = field_size
    
    def forward_pass(self, input_shares: List[Share], 
                    weights: List[List[List[Share]]], 
                    node_id: int,
                    context: Optional[str] = None) -> List[Share]:
        """
        Perform optimized forward pass through neural network
        Args:
            input_shares: Input vector as shares
            weights: Weight matrices (one per layer)
            node_id: Node ID
            context: Optional context for multi-node operations
        Returns:
            Output vector as shares
        """
        current = input_shares
        
        for layer_idx, layer_weights in enumerate(weights):
            # Generate context for this layer
            layer_context = f"{context}_L{layer_idx}" if context else None
            # Matrix-vector multiplication: output = weights @ input
            current = self.matrix_multiplier.secure_matrix_vector_multiply(
                layer_weights, current, node_id, context=layer_context
            )
            # Note: Activation function would be applied here
            # For now, we skip it (would need secure comparison)
        
        return current
    
    def backward_pass(self, loss_grad_shares: List[Share], 
                     input_shares: List[Share], 
                     weights: List[List[List[Share]]], 
                     node_id: int,
                     context: Optional[str] = None) -> List[List[List[Share]]]:
        """
        Perform optimized backward pass (gradient computation)
        Args:
            loss_grad_shares: Loss gradients as shares
            input_shares: Input vector as shares
            weights: Weight matrices
            node_id: Node ID
            context: Optional context for multi-node operations
        Returns:
            Gradient matrices (same shape as weights)
        """
        gradients = []
        
        # Simplified backward pass
        # In production, would compute proper gradients using chain rule
        current_grad = loss_grad_shares
        
        for layer_idx in range(len(weights) - 1, -1, -1):
            layer_weights = weights[layer_idx]
            layer_gradients = []
            layer_context = f"{context}_L{layer_idx}" if context else None
            
            # Get input for this layer
            if layer_idx == 0:
                layer_input = input_shares
            else:
                # Use current_grad as input for hidden layers (simplified)
                layer_input = current_grad
            
            # Compute gradients for this layer
            for i, weight_row in enumerate(layer_weights):
                row_gradients = []
                for j, weight_share in enumerate(weight_row):
                    # Simplified gradient computation
                    # In production: grad = input[i] * output_grad[j]
                    if i < len(layer_input) and j < len(current_grad):
                        # Use actual input and gradient
                        input_share = layer_input[i] if i < len(layer_input) else Share(x=layer_input[0].x, y=0, node_id=node_id)
                        # Generate unique context for each multiplication
                        mult_context = f"{layer_context}_g{i}_{j}" if layer_context else None
                        grad_share = self.multiplier.multiply(
                            input_share,
                            current_grad[j],
                            node_id,
                            context=mult_context
                        )
                    else:
                        # Zero gradient if indices out of range
                        grad_share = Share(x=weight_share.x, y=0, node_id=node_id)
                    row_gradients.append(grad_share)
                layer_gradients.append(row_gradients)
            
            gradients.insert(0, layer_gradients)
            
            # Update current_grad for next layer (simplified)
            if layer_idx > 0:
                # Transpose weights for next layer gradient
                W_T = self.matrix_multiplier.secure_matrix_transpose(layer_weights)
                # Ensure current_grad size matches
                if len(current_grad) <= len(W_T[0]) if W_T else 0:
                    current_grad = self.matrix_multiplier.secure_matrix_vector_multiply(
                        W_T, current_grad, node_id, context=f"{layer_context}_bp" if layer_context else None
                    )
                else:
                    # Truncate if needed
                    current_grad = self.matrix_multiplier.secure_matrix_vector_multiply(
                        W_T, current_grad[:len(W_T[0])] if W_T else [], node_id, context=f"{layer_context}_bp" if layer_context else None
                    )
        
        return gradients
    
    def update_weights(self, weights: List[List[List[Share]]], 
                      gradients: List[List[List[Share]]], 
                      learning_rate: float, 
                      node_id: int) -> List[List[List[Share]]]:
        """
        Update weights using gradients: W_new = W - lr * grad
        Args:
            weights: Current weight matrices
            gradients: Gradient matrices
            learning_rate: Learning rate
            node_id: Node ID
        Returns:
            Updated weight matrices
        """
        updated_weights = []
        
        for layer_weights, layer_grads in zip(weights, gradients):
            layer_updated = []
            
            for weight_row, grad_row in zip(layer_weights, layer_grads):
                row_updated = []
                
                for weight_share, grad_share in zip(weight_row, grad_row):
                    # Scale gradient by learning rate
                    lr_grad = self.matrix_multiplier.secure_matrix_scalar_multiply(
                        [[grad_share]], int(learning_rate * 1000), node_id
                    )[0][0]
                    
                    # Subtract: weight - lr * grad
                    new_weight = Share(
                        x=weight_share.x,
                        y=(weight_share.y - lr_grad.y) % self.field_size,
                        node_id=node_id
                    )
                    row_updated.append(new_weight)
                
                layer_updated.append(row_updated)
            
            updated_weights.append(layer_updated)
        
        return updated_weights
    
    def batch_forward_pass(self, batch_inputs: List[List[Share]], 
                          weights: List[List[List[Share]]], 
                          node_id: int) -> List[List[Share]]:
        """
        Perform forward pass on a batch of inputs
        Optimized for batch processing
        Args:
            batch_inputs: List of input vectors (batch)
            weights: Weight matrices
            node_id: Node ID
        Returns:
            List of output vectors (batch)
        """
        batch_outputs = []
        for input_shares in batch_inputs:
            output = self.forward_pass(input_shares, weights, node_id)
            batch_outputs.append(output)
        return batch_outputs


class GPUMatrixAccelerator:
    """
    GPU acceleration for matrix operations using CuPy
    Falls back to CPU if GPU unavailable
    """
    
    def __init__(self):
        """Initialize GPU accelerator"""
        self.use_gpu = False
        self.cp = None
        
        try:
            import cupy as cp
            self.cp = cp
            self.use_gpu = True
            print("GPU acceleration enabled (CuPy)")
        except ImportError:
            print("Warning: CuPy not available, using CPU")
            self.use_gpu = False
    
    def matrix_multiply_gpu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication on GPU if available
        Args:
            A: Matrix A
            B: Matrix B
        Returns:
            Product matrix
        """
        if self.use_gpu:
            A_gpu = self.cp.asarray(A)
            B_gpu = self.cp.asarray(B)
            C_gpu = self.cp.dot(A_gpu, B_gpu)
            return self.cp.asnumpy(C_gpu)
        else:
            return np.dot(A, B)
    
    def is_available(self) -> bool:
        """Check if GPU acceleration is available"""
        return self.use_gpu

