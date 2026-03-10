"""
Secure MNIST MLP Implementation (Batched)
Architecture: Input (784) -> Dense(128) -> ReLU -> Dense(10) -> Softmax
"""

from typing import List, Tuple, Optional, Any
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_matrix_ops import SecureMatrixOperations
from ml_training.beaver_triples import SecureMultiplier
from ml_training.secure_relu import SecureReLU
from ml_training.secure_comparison import SecureComparator
import numpy as np
import time

class BatchedSecureMNISTMLP:
    """
    Batched Secure Multi-Layer Perceptron for MNIST
    Processes an entire mini-batch simultaneously using SIMD matrix-matrix multiplications.
    """
    
    def __init__(self, n_nodes: int, t: int,
                 multiplier: SecureMultiplier,
                 field_size: int = 2**32 - 5,
                 comparator: Optional[SecureComparator] = None,
                 divider: Optional[Any] = None,
                 scale_factor: int = 1000,
                 init_gain: float = 1.0,
                 grad_clip: float = 2.0):
        self.n_nodes = n_nodes
        self.t = t
        self.field_size = field_size
        self.scale_factor = int(scale_factor)
        self.init_gain = float(init_gain)
        self.grad_clip = float(grad_clip)
        
        # Initialize secure operations
        self.matrix_ops = SecureMatrixOperations(multiplier, field_size, scale_factor=self.scale_factor)
        self.shamir = ShamirSecretSharing(field_size)
        self.relu_op = SecureReLU(comparator, multiplier, field_size)
        self.divider = divider
        
        # Architecture parameters
        self.input_dim = 784
        self.hidden_dim = 128
        self.output_dim = 10
        
        # Weight shapes (output_dim, input_dim) for W @ x
        self.w1_shape = (self.hidden_dim, self.input_dim)
        self.w2_shape = (self.output_dim, self.hidden_dim)
        
        # Bias shapes
        self.b1_shape = (self.hidden_dim,)
        self.b2_shape = (self.output_dim,)
        
    def get_weight_shapes(self) -> List[Tuple]:
        return [self.w1_shape, self.w2_shape, self.b1_shape, self.b2_shape]

    def initialize_weights(self, node_id: int = 1) -> List:
        """Initialize weights with He initialization and bias with zeros"""
        weights = []
        
        # W1: 128 x 784
        w1_rows = []
        for i in range(self.w1_shape[0]):
            row = []
            for j in range(self.w1_shape[1]):
                std = np.sqrt(2.0 / self.input_dim) * self.init_gain
                val = np.random.randn() * std
                val_int = int(val * self.scale_factor) % self.field_size
                shares = self.shamir.share(val_int, self.n_nodes, self.t)
                row.append(next(s for s in shares if s.node_id == node_id))
            w1_rows.append(row)
        weights.append(w1_rows)
        
        # W2: 10 x 128
        w2_rows = []
        for i in range(self.w2_shape[0]):
            row = []
            for j in range(self.w2_shape[1]):
                std = np.sqrt(2.0 / self.hidden_dim) * self.init_gain
                val = np.random.randn() * std
                val_int = int(val * self.scale_factor) % self.field_size
                shares = self.shamir.share(val_int, self.n_nodes, self.t)
                row.append(next(s for s in shares if s.node_id == node_id))
            w2_rows.append(row)
        weights.append(w2_rows)
        
        # B1: 128 (zeros)
        b1_shares = []
        for i in range(self.b1_shape[0]):
            shares = self.shamir.share(0, self.n_nodes, self.t)
            b1_shares.append(next(s for s in shares if s.node_id == node_id))
        weights.append(b1_shares)
        
        # B2: 10 (zeros)
        b2_shares = []
        for i in range(self.b2_shape[0]):
            shares = self.shamir.share(0, self.n_nodes, self.t)
            b2_shares.append(next(s for s in shares if s.node_id == node_id))
        weights.append(b2_shares)
        
        return weights

    def transpose(self, matrix: List[List[Share]]) -> List[List[Share]]:
        """Transpose a list of lists of shares"""
        if not matrix or not matrix[0]:
            return []
        rows = len(matrix)
        cols = len(matrix[0])
        return [[matrix[r][c] for r in range(rows)] for c in range(cols)]

    def forward_pass_batched(self, input_shares_cols: List[List[Share]],
                             weights: List,
                             node_id: int = 1,
                             context: Optional[str] = None,
                             open_relu: bool = False,
                             reconstruction_manager: Any = None) -> Tuple[List[List[Share]], dict]:
        """
        Batched forward pass.
        input_shares_cols is a list of batch columns, each of length input_dim (784).
        """
        w1, w2, b1, b2 = weights
        base_ctx = context if context else "mnist_mlp_batched"
        batch_size = len(input_shares_cols)
        
        # 1. Z1 = W1 @ X (Output is list of batch columns, each length 128)
        z1_cols_no_bias = self.matrix_ops.matrix_multiplier.secure_matrix_matrix_multiply_fixed_point(
            w1, input_shares_cols, node_id=node_id, context=f"{base_ctx}_dense1"
        )
        
        
        z1_cols = []
        for col in z1_cols_no_bias:
            z1_cols.append([
                Share(x=s.x, y=(s.y + b.y) % self.field_size, node_id=node_id)
                for s, b in zip(col, b1)
            ])
            
        # Flatten Z1 to compute ReLU in one call
        flat_z1 = [s for col in z1_cols for s in col]
        
        if reconstruction_manager is not None and "eval_pre" in base_ctx[:9]:
            r_z1 = []
            for i in range(10):
                val = reconstruction_manager.get_reconstructed_value([flat_z1[i]], f"{base_ctx}_dbg_{i}", use_cache=False)
                r_z1.append(val)
            print(f"DEBUG Z1 PRE-RELU: {[(v if v <= self.field_size//2 else v - self.field_size)/self.scale_factor for v in r_z1]}", flush=True)

        if open_relu and reconstruction_manager:
            flat_a1 = self.relu_op.relu_list(flat_z1, node_id, context=f"{base_ctx}_relu1")
        else:
            flat_a1 = self.relu_op.relu_list(flat_z1, node_id, context=f"{base_ctx}_relu1")
            
        a1_cols = [flat_a1[i*self.hidden_dim : (i+1)*self.hidden_dim] for i in range(batch_size)]
        
        # 2. Z2 = W2 @ A1 (Output is list of batch columns, each length 10)
        z2_cols_no_bias = self.matrix_ops.matrix_multiplier.secure_matrix_matrix_multiply_fixed_point(
            w2, a1_cols, node_id=node_id, context=f"{base_ctx}_dense2"
        )
        
        z2_cols = []
        for col in z2_cols_no_bias:
            z2_cols.append([
                Share(x=s.x, y=(s.y + b.y) % self.field_size, node_id=node_id)
                for s, b in zip(col, b2)
            ])
            
        cache = {'input_cols': input_shares_cols, 'z1_cols': z1_cols, 'a1_cols': a1_cols}
        return z2_cols, cache

    def train_batch(self, x_shares_cols: List[List[Share]], y_shares_cols: List[List[Share]], 
                   weights: List, softmax_op: Any, lr: float,
                   node_id: int = 1, context: str = "",
                   reconstruction_manager: Any = None,
                   loss_mode: str = "mse",
                   diagnostics_out: Optional[dict] = None) -> List:
        """
        Run a full train step on a mini-batch with SIMD matrix-matrix multiplications.
        Returns updated weights.
        """
        batch_size = len(x_shares_cols)
        if batch_size <= 0:
            raise ValueError("train_batch requires non-empty mini-batch")
        w1, w2, b1, b2 = weights
        base_ctx = context
        
        # 1. Forward
        logits_cols, cache = self.forward_pass_batched(x_shares_cols, weights, node_id, 
                                                      context=f"{base_ctx}_fwd",
                                                      reconstruction_manager=reconstruction_manager,
                                                      open_relu=True)
                                                      
        # 2. Output gradient
        # `softmax` mode keeps the original secure softmax CE gradient.
        # `mse` mode is more numerically stable in this fixed-point MPC prototype.
        if str(loss_mode).lower() == "softmax":
            dz2_cols = softmax_op.cross_entropy_loss_gradient_batch(
                logits_cols, y_shares_cols, node_id, context=f"{base_ctx}_grad_batch"
            )
        else:
            p = int(self.field_size)
            clip_abs = int(4 * self.scale_factor)
            denom = max(1, int(self.output_dim))
            dz2_cols = []
            for i in range(batch_size):
                col_grads = []
                for j in range(self.output_dim):
                    raw = (int(logits_cols[i][j].y) - int(y_shares_cols[i][j].y)) % p
                    signed = raw if raw <= p // 2 else raw - p
                    if signed > clip_abs:
                        signed = clip_abs
                    elif signed < -clip_abs:
                        signed = -clip_abs
                    grad = int(round(float(signed) / float(denom))) % p
                    col_grads.append(Share(x=logits_cols[i][j].x, y=grad, node_id=node_id))
                dz2_cols.append(col_grads)
            
        a1_cols = cache['a1_cols']
        input_cols = cache['input_cols']
        z1_cols = cache['z1_cols']
        
        # 3. Backward
        # dW2 = dZ2 @ A1.T
        # B_cols are the columns of A1.T, which are the rows of A1.
        e2_rows = self.transpose(dz2_cols) # Shape: 10 x batch_size
        a1_rows = self.transpose(a1_cols)  # Shape: 128 x batch_size
        
        dw2_cols = self.matrix_ops.matrix_multiplier.secure_matrix_matrix_multiply_fixed_point(
            e2_rows, a1_rows, node_id=node_id, context=f"{base_ctx}_dw2"
        ) # Returns 128 cols of length 10
        dw2_accum = self.transpose(dw2_cols) # Shape: 10 x 128
        
        # dA1 = W2.T @ dZ2
        w2_T_rows = self.transpose(w2) # Shape: 128 x 10
        da1_cols = self.matrix_ops.matrix_multiplier.secure_matrix_matrix_multiply_fixed_point(
            w2_T_rows, dz2_cols, node_id=node_id, context=f"{base_ctx}_da1"
        )
        
        # dZ1 = dA1 * relu'(Z1)
        flat_da1 = [s for col in da1_cols for s in col]
        flat_z1 = [s for col in z1_cols for s in col]
        flat_dz1 = self.relu_op.relu_backward_list(flat_da1, flat_z1, node_id, context=f"{base_ctx}_dz1")
        dz1_cols = [flat_dz1[i*self.hidden_dim : (i+1)*self.hidden_dim] for i in range(batch_size)]
        
        # dW1 = dZ1 @ X.T
        e1_rows = self.transpose(dz1_cols) # Shape: 128 x batch_size
        x_rows = self.transpose(input_cols) # Shape: 784 x batch_size
        
        dw1_cols = self.matrix_ops.matrix_multiplier.secure_matrix_matrix_multiply_fixed_point(
            e1_rows, x_rows, node_id=node_id, context=f"{base_ctx}_dw1"
        ) # Returns 784 cols of length 128
        dw1_accum = self.transpose(dw1_cols) # Shape: 128 x 784
        
        # Calculate db2 and db1 by summing the gradients across the batch
        db2_accum = []
        for r in range(self.output_dim):
            s_sum = 0
            for c in range(batch_size):
                s_sum = (s_sum + dz2_cols[c][r].y) % self.field_size
            db2_accum.append(Share(dz2_cols[0][0].x, s_sum, node_id))
            
        db1_accum = []
        for r in range(self.hidden_dim):
            s_sum = 0
            for c in range(batch_size):
                s_sum = (s_sum + dz1_cols[c][r].y) % self.field_size
            db1_accum.append(Share(dz1_cols[0][0].x, s_sum, node_id))

        # 4. Updates
        # Gradient clipping in fixed-point space to stabilize secure training.
        p_mod = int(self.field_size)
        clip_abs = int(round(self.grad_clip * float(self.scale_factor)))
        opened_mode = bool(getattr(self.divider, "_opened_enabled", lambda: False)())

        def _clip_share(s: Share) -> Share:
            v = int(s.y) % p_mod
            if v > p_mod // 2:
                v -= p_mod
            if v > clip_abs:
                v = clip_abs
            elif v < -clip_abs:
                v = -clip_abs
            return Share(x=s.x, y=v % p_mod, node_id=node_id)

        # IMPORTANT:
        # Clipping individual Shamir shares is not semantically equivalent to clipping
        # reconstructed values. In opened/privacy mode, perform clipping only in the
        # opened divide-and-reshare stage below.
        if not opened_mode:
            for r in range(self.hidden_dim):
                for c in range(self.input_dim):
                    dw1_accum[r][c] = _clip_share(dw1_accum[r][c])
            for r in range(self.output_dim):
                for c in range(self.hidden_dim):
                    dw2_accum[r][c] = _clip_share(dw2_accum[r][c])
            for r in range(self.hidden_dim):
                db1_accum[r] = _clip_share(db1_accum[r])
            for r in range(self.output_dim):
                db2_accum[r] = _clip_share(db2_accum[r])

        # To avoid fractional overflow noise through the finite field via modular inversion
        # of batch_size, we apply integer batch division entirely in the secure divider enclave.
        # Dimensional analysis:
        # dW_accum is scaled at S^1 (due to matrix multiplication dividing by S)
        # lr_fixed is scaled at S^1
        # product = dW_accum * lr_fixed is conceptually scaled at S^2
        # We need the update to be scaled at S^1 to subtract from weights (scaled at S^1).
        # We also need to average across batch_size (B).
        # Therefore, divisor = S^1 * B
        
        lr_fixed = int(lr * self.scale_factor) % self.field_size
        divisor_w = int(self.scale_factor * batch_size)
        if divisor_w <= 0:
            raise RuntimeError("Invalid fixed-point divisor_w; scale_factor and batch_size must be positive")
        
        flat_upd_w1 = []
        for r in range(self.hidden_dim):
            for c in range(self.input_dim):
                flat_upd_w1.append(Share(dw1_accum[r][c].x, dw1_accum[r][c].y, node_id))
        flat_upd_w1 = self.divider.secure_scalar_divide_batch(
            flat_upd_w1,
            divisor_w,
            node_id,
            context=f"{base_ctx}_updw1",
            multiplier_factor=lr_fixed,
            clip_min=-clip_abs if opened_mode else None,
            clip_max=clip_abs if opened_mode else None,
        )
        # Final sanity clipping after averaging/update scaling.
        if not opened_mode:
            flat_upd_w1 = [_clip_share(s) for s in flat_upd_w1]
        
        new_w1 = []
        idx = 0
        for r in range(self.hidden_dim):
            row = []
            for c in range(self.input_dim):
                row.append(Share(w1[r][c].x, (w1[r][c].y - flat_upd_w1[idx].y) % self.field_size, node_id))
                idx += 1
            new_w1.append(row)
            
        if reconstruction_manager is not None and "e0_b0" in base_ctx[:5]:
            # Dump the first 10 elements of a1_cols[0] (which corresponds to a1 of sample 0 in batch 1)
            a1_dbg = [reconstruction_manager.get_reconstructed_value([a1_cols[0][i]], f"{base_ctx}_a1_{i}", use_cache=False) for i in range(10)]
            print(f"DEBUG BATCH 1 A1: {[ (v if v<=self.field_size//2 else v-self.field_size)/self.scale_factor for v in a1_dbg]}", flush=True)
            
        flat_upd_w2 = []
        for r in range(self.output_dim):
            for c in range(self.hidden_dim):
                flat_upd_w2.append(Share(dw2_accum[r][c].x, dw2_accum[r][c].y, node_id))
        flat_upd_w2 = self.divider.secure_scalar_divide_batch(
            flat_upd_w2,
            divisor_w,
            node_id,
            context=f"{base_ctx}_updw2",
            multiplier_factor=lr_fixed,
            clip_min=-clip_abs if opened_mode else None,
            clip_max=clip_abs if opened_mode else None,
        )
        # Final sanity clipping after averaging/update scaling.
        if not opened_mode:
            flat_upd_w2 = [_clip_share(s) for s in flat_upd_w2]
        
        new_w2 = []
        idx = 0
        for r in range(self.output_dim):
            row = []
            for c in range(self.hidden_dim):
                row.append(Share(w2[r][c].x, (w2[r][c].y - flat_upd_w2[idx].y) % self.field_size, node_id))
                idx += 1
            new_w2.append(row)
            
        # Biases: db_accum is the sum of dZ directly.
        # dZ is scaled at S^1
        # product = db_accum * lr_fixed is scaled at S^2
        # We need the update to be scaled at S^1 to subtract from biases (scaled at S^1).
        # We also need to average across batch_size (B).
        # Therefore, divisor_b = S^1 * B
        divisor_b = int(self.scale_factor * batch_size)
        if divisor_b <= 0:
            raise RuntimeError("Invalid fixed-point divisor_b; scale_factor and batch_size must be positive")
        
        flat_upd_b1 = []
        for r in range(self.hidden_dim):
            flat_upd_b1.append(Share(db1_accum[r].x, db1_accum[r].y, node_id))
        flat_upd_b1 = self.divider.secure_scalar_divide_batch(
            flat_upd_b1,
            divisor_b,
            node_id,
            context=f"{base_ctx}_updb1",
            multiplier_factor=lr_fixed,
            clip_min=-clip_abs if opened_mode else None,
            clip_max=clip_abs if opened_mode else None,
        )
        # Final sanity clipping after averaging/update scaling.
        if not opened_mode:
            flat_upd_b1 = [_clip_share(s) for s in flat_upd_b1]
        
        new_b1 = []
        for r in range(self.hidden_dim):
            new_b1.append(Share(b1[r].x, (b1[r].y - flat_upd_b1[r].y) % self.field_size, node_id))
            
        flat_upd_b2 = []
        for r in range(self.output_dim):
            flat_upd_b2.append(Share(db2_accum[r].x, db2_accum[r].y, node_id))
        flat_upd_b2 = self.divider.secure_scalar_divide_batch(
            flat_upd_b2,
            divisor_b,
            node_id,
            context=f"{base_ctx}_updb2",
            multiplier_factor=lr_fixed,
            clip_min=-clip_abs if opened_mode else None,
            clip_max=clip_abs if opened_mode else None,
        )
        # Final sanity clipping after averaging/update scaling.
        if not opened_mode:
            flat_upd_b2 = [_clip_share(s) for s in flat_upd_b2]

        # Optional gradient-norm estimate on the first batch of an epoch only (to limit overhead).
        # Uses reconstructed update values (post-avg, post-clip), then reports L2 norm estimate.
        if diagnostics_out is not None and reconstruction_manager is not None and "_b0" in str(base_ctx):
            try:
                p = int(self.field_size)
                scale = float(self.scale_factor)
                sample_cap = 32
                vectors = [
                    ("w1", flat_upd_w1),
                    ("w2", flat_upd_w2),
                    ("b1", flat_upd_b1),
                    ("b2", flat_upd_b2),
                ]
                sq_sum = 0.0
                n_sample = 0
                n_total = 0
                abs_vals = []
                for tag, vec in vectors:
                    n_total += len(vec)
                    take = min(sample_cap, len(vec))
                    for i in range(take):
                        v = reconstruction_manager.get_reconstructed_value(
                            [vec[i]],
                            f"{base_ctx}_gnorm_{tag}_{i}",
                            use_cache=False,
                        )
                        if v > p // 2:
                            v -= p
                        vf = float(v) / scale
                        abs_vals.append(abs(vf))
                        sq_sum += vf * vf
                        n_sample += 1
                if n_sample > 0 and n_total > 0:
                    # Scale sampled norm to an estimate of full-vector norm.
                    est = float(np.sqrt(max(0.0, sq_sum) * (float(n_total) / float(n_sample))))
                    diagnostics_out["grad_norm_estimate"] = est
                    diagnostics_out["update_abs_mean_sample"] = float(np.mean(abs_vals)) if abs_vals else 0.0
                    diagnostics_out["update_abs_max_sample"] = float(np.max(abs_vals)) if abs_vals else 0.0

                # Optional deeper numeric probes.
                if bool(diagnostics_out.get("debug_numerics", False)):
                    # Probe a few reconstructed logits and output gradients for this batch.
                    probe_vals_logits = []
                    probe_vals_dz2 = []
                    probe_vals_target = []
                    probe_n = min(10, self.output_dim)
                    for j in range(probe_n):
                        lv = reconstruction_manager.get_reconstructed_value(
                            [logits_cols[0][j]],
                            f"{base_ctx}_dbg_logit_{j}",
                            use_cache=False,
                        )
                        gv = reconstruction_manager.get_reconstructed_value(
                            [dz2_cols[0][j]],
                            f"{base_ctx}_dbg_dz2_{j}",
                            use_cache=False,
                        )
                        yv = reconstruction_manager.get_reconstructed_value(
                            [y_shares_cols[0][j]],
                            f"{base_ctx}_dbg_target_{j}",
                            use_cache=False,
                        )
                        if lv > p // 2:
                            lv -= p
                        if gv > p // 2:
                            gv -= p
                        if yv > p // 2:
                            yv -= p
                        probe_vals_logits.append(float(lv) / scale)
                        probe_vals_dz2.append(float(gv) / scale)
                        probe_vals_target.append(float(yv) / scale)

                    temp = float(getattr(softmax_op, "temperature", 1.0))
                    if abs(temp) < 1e-12:
                        temp = 1.0
                    probs_est = [(probe_vals_dz2[k] * temp) + probe_vals_target[k] for k in range(probe_n)]

                    diagnostics_out["logit_probe"] = probe_vals_logits
                    diagnostics_out["dz2_probe"] = probe_vals_dz2
                    diagnostics_out["target_probe"] = probe_vals_target
                    diagnostics_out["probs_est_probe"] = probs_est
                    diagnostics_out["probs_est_sum"] = float(np.sum(np.asarray(probs_est, dtype=np.float64)))
                    diagnostics_out["probs_est_min"] = float(np.min(np.asarray(probs_est, dtype=np.float64)))
                    diagnostics_out["probs_est_max"] = float(np.max(np.asarray(probs_est, dtype=np.float64)))
                    diagnostics_out["target_sum"] = float(np.sum(np.asarray(probe_vals_target, dtype=np.float64)))
                    diagnostics_out["logit_abs_max_probe"] = float(np.max(np.abs(np.asarray(probe_vals_logits, dtype=np.float64))))
                    diagnostics_out["dz2_abs_max_probe"] = float(np.max(np.abs(np.asarray(probe_vals_dz2, dtype=np.float64))))
            except Exception:
                # Non-fatal diagnostics path.
                pass
            
        new_b2 = []
        for r in range(self.output_dim):
            new_b2.append(Share(b2[r].x, (b2[r].y - flat_upd_b2[r].y) % self.field_size, node_id))
            
        return [new_w1, new_w2, new_b1, new_b2]
