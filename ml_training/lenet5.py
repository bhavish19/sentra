"""
LeNet-5 Architecture for Secure MPC Training
Implements LeNet-5 CNN architecture using secure operations
"""

from typing import List, Tuple, Optional, Any, Dict
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_convolution import SecureConvolution
from ml_training.secure_pooling import SecurePooling
from ml_training.secure_matrix_ops import SecureMatrixOperations
from ml_training.beaver_triples import SecureMultiplier
from ml_training.secure_comparison import SecureComparator
from ml_training.secure_division import SecureDivider
from ml_training.secure_relu import SecureReLU
import numpy as np
import time


class LeNet5:
    """
    LeNet-5 CNN Architecture
    
    Architecture:
    - Input: 32x32x1 (grayscale image)
    - Conv1: 6 filters, 5x5, stride 1 → 28x28x6
    - ReLU1: ReLU activation
    - Pool1: 2x2 max pooling → 14x14x6
    - Conv2: 16 filters, 5x5, stride 1 → 10x10x16
    - ReLU2: ReLU activation
    - Pool2: 2x2 max pooling → 5x5x16
    - Flatten: 400 features
    - FC1: 120 neurons
    - ReLU3: ReLU activation
    - FC2: 84 neurons
    - ReLU4: ReLU activation
    - FC3: 8 neurons (output for 8 classes, no activation)
    """
    
    def __init__(self, n_nodes: int, t: int,
                 multiplier: SecureMultiplier,
                 field_size: int = 2**32 - 5,  # 4,294,967,291 (nearest prime to 2^32)
                 comparator: Optional[SecureComparator] = None,
                 divider: Optional[SecureDivider] = None,
                 scale_factor: int = 1000,
                 init_gain: float = 1.0):
        """
        Initialize LeNet-5 model
        
        Args:
            n_nodes: Number of nodes
            t: Privacy threshold
            multiplier: SecureMultiplier instance
            field_size: Prime field size
            comparator: SecureComparator for max pooling (optional)
            divider: SecureDivider for average pooling (optional)
        """
        self.n_nodes = n_nodes
        self.t = t
        self.field_size = field_size
        self.scale_factor = int(scale_factor)
        self._inv_scale = pow(self.scale_factor % self.field_size, self.field_size - 2, self.field_size)
        # Optional global multiplier for weight init stddev (helps prevent logit explosion).
        self.init_gain = float(init_gain)
        
        # Initialize secure operations
        self.conv_op = SecureConvolution(multiplier, field_size, scale_factor=self.scale_factor)
        self.pool_op = SecurePooling(comparator, divider, field_size)
        self.matrix_ops = SecureMatrixOperations(multiplier, field_size, scale_factor=self.scale_factor)
        self.shamir = ShamirSecretSharing(field_size)
        
        # Initialize ReLU activation
        self.relu_op = SecureReLU(comparator, multiplier, field_size)
        
        # Store references for backward pass
        self.divider = divider
        self.comparator = comparator
        
        # Architecture parameters
        self.input_size = (32, 32, 1)  # H, W, C
        self.num_classes = 8  # Kather dataset has 8 classes
        
        # Weight shapes for each layer
        # Conv1: 6 filters, 5x5, 1 input channel
        self.conv1_shape = (5, 5, 1, 6)  # K_h, K_w, C_in, C_out
        
        # Conv2: 16 filters, 5x5, 6 input channels
        self.conv2_shape = (5, 5, 6, 16)  # K_h, K_w, C_in, C_out
        
        # After pooling: 5x5x16 = 400 features
        # Note: Weight shapes are (output, input) for matrix-vector multiplication y = W @ x
        self.fc1_shape = (120, 400)  # output, input
        self.fc2_shape = (84, 120)   # output, input
        self.fc3_shape = (8, 84)     # output, input (8 classes)
        
        # Bias shapes for FC layers (zero-initialized to prevent class bias)
        self.fc1_bias_shape = (120,)
        self.fc2_bias_shape = (84,)
        self.fc3_bias_shape = (8,)
    
    def get_weight_shapes(self) -> List[Tuple]:
        """
        Get weight shapes for all layers
        
        Returns:
            List of weight shapes for initialization
        """
        return [
            self.conv1_shape,
            self.conv2_shape,
            self.fc1_shape,
            self.fc2_shape,
            self.fc3_shape
        ]
    
    def initialize_weights(self, node_id: int = 1) -> List:
        """
        Initialize weights for all layers
        
        Returns:
            List of weight shares for each layer
        """
        weights = []
        
        # Conv1 weights: 5x5x1x6
        conv1_weights = []
        for k_h in range(self.conv1_shape[0]):
            conv1_row = []
            for k_w in range(self.conv1_shape[1]):
                conv1_ch_in = []
                for c_in in range(self.conv1_shape[2]):
                    conv1_ch_out = []
                    for c_out in range(self.conv1_shape[3]):
                        # Initialize with scaled He initialization for larger outputs
                        # For conv layers: std = sqrt(4 / fan_in) (2x larger than standard He)
                        # fan_in = kernel_size * kernel_size * in_channels
                        fan_in = self.conv1_shape[0] * self.conv1_shape[1] * self.conv1_shape[2]  # 5*5*1 = 25
                        std = np.sqrt(4.0 / fan_in) * float(self.init_gain)  # scaled by init_gain
                        w_val = np.random.randn() * std
                        w_int = int(w_val * self.scale_factor) % self.field_size
                        w_shares = self.shamir.share(w_int, self.n_nodes, self.t)
                        w_share = next(s for s in w_shares if s.node_id == node_id)
                        conv1_ch_out.append(w_share)
                    conv1_ch_in.append(conv1_ch_out)
                conv1_row.append(conv1_ch_in)
            conv1_weights.append(conv1_row)
        weights.append(conv1_weights)
        
        # Conv2 weights: 5x5x6x16
        conv2_weights = []
        for k_h in range(self.conv2_shape[0]):
            conv2_row = []
            for k_w in range(self.conv2_shape[1]):
                conv2_ch_in = []
                for c_in in range(self.conv2_shape[2]):
                    conv2_ch_out = []
                    for c_out in range(self.conv2_shape[3]):
                        # Scaled He initialization for Conv2
                        fan_in = self.conv2_shape[0] * self.conv2_shape[1] * self.conv2_shape[2]  # 5*5*6 = 150
                        std = np.sqrt(4.0 / fan_in) * float(self.init_gain)  # scaled by init_gain
                        w_val = np.random.randn() * std
                        w_int = int(w_val * self.scale_factor) % self.field_size
                        w_shares = self.shamir.share(w_int, self.n_nodes, self.t)
                        w_share = next(s for s in w_shares if s.node_id == node_id)
                        conv2_ch_out.append(w_share)
                    conv2_ch_in.append(conv2_ch_out)
                conv2_row.append(conv2_ch_in)
            conv2_weights.append(conv2_row)
        weights.append(conv2_weights)
        
        # FC1 weights: 400x120
        fc1_weights = []
        for i in range(self.fc1_shape[0]):
            fc1_row = []
            for j in range(self.fc1_shape[1]):
                # Scaled He initialization for FC1
                fan_in = self.fc1_shape[1]  # 400
                std = np.sqrt(6.0 / fan_in) * float(self.init_gain)  # scaled by init_gain
                w_val = np.random.randn() * std
                w_int = int(w_val * self.scale_factor) % self.field_size
                w_shares = self.shamir.share(w_int, self.n_nodes, self.t)
                w_share = next(s for s in w_shares if s.node_id == node_id)
                fc1_row.append(w_share)
            fc1_weights.append(fc1_row)
        weights.append(fc1_weights)
        
        # FC2 weights: 120x84
        fc2_weights = []
        for i in range(self.fc2_shape[0]):
            fc2_row = []
            for j in range(self.fc2_shape[1]):
                # Scaled He initialization for FC2
                fan_in = self.fc2_shape[1]  # 120
                std = np.sqrt(4.0 / fan_in) * float(self.init_gain)  # scaled by init_gain
                w_val = np.random.randn() * std
                w_int = int(w_val * self.scale_factor) % self.field_size
                w_shares = self.shamir.share(w_int, self.n_nodes, self.t)
                w_share = next(s for s in w_shares if s.node_id == node_id)
                fc2_row.append(w_share)
            fc2_weights.append(fc2_row)
        weights.append(fc2_weights)
        
        # FC3 weights: 84x8
        fc3_weights = []
        for i in range(self.fc3_shape[0]):
            fc3_row = []
            for j in range(self.fc3_shape[1]):
                # Scaled initialization for FC3 (output layer)
                # Use moderate scaling for output layer
                fan_in = self.fc3_shape[1]  # 84
                std = np.sqrt(2.0 / fan_in) * float(self.init_gain)  # scaled by init_gain
                w_val = np.random.randn() * std
                w_int = int(w_val * self.scale_factor) % self.field_size
                w_shares = self.shamir.share(w_int, self.n_nodes, self.t)
                w_share = next(s for s in w_shares if s.node_id == node_id)
                fc3_row.append(w_share)
            fc3_weights.append(fc3_row)
        weights.append(fc3_weights)
        
        # FC1 bias: 120 (zero-initialized to prevent class bias)
        fc1_bias = []
        for i in range(self.fc1_bias_shape[0]):
            bias_shares = self.shamir.share(0, self.n_nodes, self.t)  # Zero bias
            bias_share = next(s for s in bias_shares if s.node_id == node_id)
            fc1_bias.append(bias_share)
        weights.append(fc1_bias)
        
        # FC2 bias: 84 (zero-initialized to prevent class bias)
        fc2_bias = []
        for i in range(self.fc2_bias_shape[0]):
            bias_shares = self.shamir.share(0, self.n_nodes, self.t)  # Zero bias
            bias_share = next(s for s in bias_shares if s.node_id == node_id)
            fc2_bias.append(bias_share)
        weights.append(fc2_bias)
        
        # FC3 bias: 8 (zero-initialized to prevent class bias - CRITICAL for fixing class 0 dominance)
        fc3_bias = []
        for i in range(self.fc3_bias_shape[0]):
            bias_shares = self.shamir.share(0, self.n_nodes, self.t)  # Zero bias
            bias_share = next(s for s in bias_shares if s.node_id == node_id)
            fc3_bias.append(bias_share)
        weights.append(fc3_bias)
        
        return weights
    
    def forward_pass(self, input_shares: List[List[List[Share]]],
                    weights: List,
                    node_id: int = 1,
                    context: Optional[str] = None,
                    return_intermediates: bool = False,
                    *,
                    reconstruction_manager: Any = None,
                    open_relu: bool = False,
                    relu_opener_node: int = 1,
                    relu_timeout: float = 120.0,
                    stop_after_flatten: bool = False) -> Tuple[List[Share], Optional[dict]]:
        """
        Forward pass through LeNet-5
        
        Args:
            input_shares: Input image as shares [32 x 32 x 1]
            weights: List of weight shares for all layers
            node_id: Node ID
            context: Optional context
            return_intermediates: If True, return intermediate outputs for backward pass
        
        Returns:
            If return_intermediates=False: Output logits as shares [8]
            If return_intermediates=True: Tuple of (output_logits, intermediate_outputs_dict)
        """
        conv1_weights, conv2_weights, fc1_weights, fc2_weights, fc3_weights, fc1_bias, fc2_bias, fc3_bias = weights
        base_ctx = context if context else "lenet5"

        def _opened_relu_mask(shares_flat: List[Share], *, relu_ctx: str) -> np.ndarray:
            """
            Reconstruct pre-activation values on opener, compute mask=(x>0) as a public 0/1 vector,
            then broadcast the mask to all nodes. Each node returns the same mask.
            """
            if not shares_flat:
                return np.zeros((0,), dtype=np.uint32)
            if reconstruction_manager is None:
                raise RuntimeError("open_relu requires reconstruction_manager (multi-node networking enabled)")

            net = reconstruction_manager.network
            opener = int(relu_opener_node)
            p = int(self.field_size)
            x_local = int(shares_flat[0].x)
            local_u32 = np.asarray([int(s.y) & 0xFFFFFFFF for s in shares_flat], dtype=np.uint32)

            # Everyone broadcasts their local share vector
            net.broadcast_vector(relu_ctx, x=x_local, values=local_u32)

            if int(node_id) == opener:
                opened_u64 = reconstruction_manager.reconstruct_opened_vector_values(
                    context=relu_ctx,
                    values_local=local_u32,
                    x=x_local,
                    timeout=float(relu_timeout),
                )
                opened = opened_u64.astype(np.int64)
                opened = np.where(opened > (p // 2), opened - p, opened)
                # One-time debug: print pre-activation range per ReLU layer.
                # Helps pinpoint where values first explode/wrap.
                try:
                    key = f"relu_range::{relu_ctx.split('_open')[0]}"
                    seen = getattr(self, "_dbg_relu_ranges_printed", set())
                    if key not in seen:
                        seen.add(key)
                        setattr(self, "_dbg_relu_ranges_printed", seen)
                        mn = float(np.min(opened) / float(self.scale_factor))
                        mx = float(np.max(opened) / float(self.scale_factor))
                        print(f"      [open_relu debug] {key} preact(min,max)=({mn:.3f},{mx:.3f})", flush=True)
                except Exception:
                    pass
                mask = (opened > 0).astype(np.uint32)  # public 0/1 mask

                # Send mask to all other nodes
                for nid in sorted(net.node_configs.keys()):
                    if int(nid) == opener:
                        continue
                    net.channel.send_vector(int(nid), f"{relu_ctx}_mask_to_{nid}", x=int(nid), values=mask)
                return mask

            # Non-opener: wait to receive mask vector from opener
            recv_ctx = f"{relu_ctx}_mask_to_{node_id}"
            start = time.time()
            while time.time() - start < float(relu_timeout):
                recv = net.channel.get_received_vector(recv_ctx)
                if opener in recv:
                    values = recv[opener].get("values")
                    arr = np.asarray(values, dtype=np.uint32)
                    try:
                        net.channel.clear_vector(recv_ctx)
                    except Exception:
                        pass
                    # Clear the original share-vector context on non-openers to avoid unbounded growth
                    try:
                        net.channel.clear_vector(relu_ctx)
                    except Exception:
                        pass
                    return arr
                time.sleep(0.01)
            raise RuntimeError(f"Timed out waiting for opened ReLU mask (ctx={relu_ctx})")

        def _apply_mask_list(x_shares: List[Share], mask: np.ndarray) -> List[Share]:
            if not x_shares:
                return []
            mask_u32 = np.asarray(mask, dtype=np.uint32).reshape(-1)
            out: List[Share] = []
            for i, s in enumerate(x_shares):
                m = int(mask_u32[i]) & 1
                out.append(Share(x=s.x, y=(int(s.y) * m) % int(self.field_size), node_id=node_id))
            return out

        def _apply_mask_3d(x_shares_3d: List[List[List[Share]]], mask: np.ndarray) -> List[List[List[Share]]]:
            if not x_shares_3d:
                return []
            H = len(x_shares_3d)
            W = len(x_shares_3d[0])
            C = len(x_shares_3d[0][0])
            mask_u32 = np.asarray(mask, dtype=np.uint32).reshape(H, W, C)
            out_3d: List[List[List[Share]]] = []
            for h in range(H):
                row = []
                for w in range(W):
                    chans = []
                    for c in range(C):
                        s = x_shares_3d[h][w][c]
                        m = int(mask_u32[h, w, c]) & 1
                        chans.append(Share(x=s.x, y=(int(s.y) * m) % int(self.field_size), node_id=node_id))
                    row.append(chans)
                out_3d.append(row)
            return out_3d
        
        # Conv1: 32x32x1 → 28x28x6
        conv1_context = f"{base_ctx}_conv1" if base_ctx else None
        conv1_out_pre_relu = self.conv_op.conv2d(
            input_shares, conv1_weights, stride=1, padding=0,
            node_id=node_id, context=conv1_context
        )
        
        # ReLU1: Apply ReLU activation after Conv1
        relu1_context = f"{base_ctx}_relu1" if base_ctx else None
        relu1_mask = None
        if open_relu and reconstruction_manager is not None:
            flat = [conv1_out_pre_relu[h][w][c] for h in range(28) for w in range(28) for c in range(6)]
            relu1_mask = _opened_relu_mask(flat, relu_ctx=f"{relu1_context}_open" if relu1_context else f"{base_ctx}_relu1_open")
            conv1_out = _apply_mask_3d(conv1_out_pre_relu, relu1_mask.reshape(28, 28, 6))
        else:
            conv1_out = self.relu_op.relu_3d(conv1_out_pre_relu, node_id, context=relu1_context)
        
        # Pool1: 28x28x6 → 14x14x6
        pool1_context = f"{context}_pool1" if context else None
        pool1_out = []
        pool1_input_channels = []  # Store for backward pass
        for c in range(6):  # 6 channels
            channel_shares = [[conv1_out[h][w][c] for w in range(28)] for h in range(28)]
            pool1_input_channels.append(channel_shares)
            pooled_channel = self.pool_op.max_pool2d(
                channel_shares, pool_size=(2, 2), stride=2,
                node_id=node_id, context=f"{pool1_context}_c{c}" if pool1_context else None
            )
            pool1_out.append(pooled_channel)
        
        # Reshape for Conv2: 14x14x6
        pool1_reshaped = []
        for h in range(14):
            pool1_row = []
            for w in range(14):
                pool1_channels = [pool1_out[c][h][w] for c in range(6)]
                pool1_row.append(pool1_channels)
            pool1_reshaped.append(pool1_row)
        
        # Conv2: 14x14x6 → 10x10x16
        conv2_context = f"{base_ctx}_conv2" if base_ctx else None
        conv2_out_pre_relu = self.conv_op.conv2d(
            pool1_reshaped, conv2_weights, stride=1, padding=0,
            node_id=node_id, context=conv2_context
        )
        
        # ReLU2: Apply ReLU activation after Conv2
        relu2_context = f"{base_ctx}_relu2" if base_ctx else None
        relu2_mask = None
        if open_relu and reconstruction_manager is not None:
            flat = [conv2_out_pre_relu[h][w][c] for h in range(10) for w in range(10) for c in range(16)]
            relu2_mask = _opened_relu_mask(flat, relu_ctx=f"{relu2_context}_open" if relu2_context else f"{base_ctx}_relu2_open")
            conv2_out = _apply_mask_3d(conv2_out_pre_relu, relu2_mask.reshape(10, 10, 16))
        else:
            conv2_out = self.relu_op.relu_3d(conv2_out_pre_relu, node_id, context=relu2_context)
        
        # Pool2: 10x10x16 → 5x5x16
        pool2_context = f"{context}_pool2" if context else None
        pool2_out = []
        pool2_input_channels = []  # Store for backward pass
        for c in range(16):  # 16 channels
            channel_shares = [[conv2_out[h][w][c] for w in range(10)] for h in range(10)]
            pool2_input_channels.append(channel_shares)
            pooled_channel = self.pool_op.max_pool2d(
                channel_shares, pool_size=(2, 2), stride=2,
                node_id=node_id, context=f"{pool2_context}_c{c}" if pool2_context else None
            )
            pool2_out.append(pooled_channel)
        
        # Flatten: 5x5x16 → 400
        flattened = []
        for c in range(16):
            for h in range(5):
                for w in range(5):
                    flattened.append(pool2_out[c][h][w])

        # Optional early exit for batch-FC/SIMD experiments:
        # return only convolutional stack intermediates + flattened vector.
        if bool(stop_after_flatten):
            if return_intermediates:
                intermediates = {
                    'input': input_shares,
                    'conv1_out': conv1_out,  # Post-ReLU
                    'conv1_out_pre_relu': conv1_out_pre_relu,  # Pre-ReLU (for backward)
                    'relu1_mask': relu1_mask,
                    'pool1_input_channels': pool1_input_channels,
                    'pool1_out': pool1_out,
                    'pool1_reshaped': pool1_reshaped,
                    'conv2_out': conv2_out,  # Post-ReLU
                    'conv2_out_pre_relu': conv2_out_pre_relu,  # Pre-ReLU (for backward)
                    'relu2_mask': relu2_mask,
                    'pool2_input_channels': pool2_input_channels,
                    'pool2_out': pool2_out,
                    'flattened': flattened,
                }
                return [], intermediates
            return []
        
        # FC1: 400 → 120
        fc1_context = f"{context}_fc1" if context else None
        fc1_out_no_bias = self.matrix_ops.matrix_multiplier.secure_matrix_vector_multiply(
            fc1_weights, flattened, node_id, context=fc1_context
        )
        # Add FC1 bias
        fc1_out_pre_relu = []
        for i in range(len(fc1_out_no_bias)):
            out_with_bias = Share(
                x=fc1_out_no_bias[i].x,
                y=(fc1_out_no_bias[i].y + fc1_bias[i].y) % self.field_size,
                node_id=node_id
            )
            fc1_out_pre_relu.append(out_with_bias)
        
        # ReLU3: Apply ReLU activation after FC1
        relu3_context = f"{base_ctx}_relu3" if base_ctx else None
        relu3_mask = None
        if open_relu and reconstruction_manager is not None:
            relu3_mask = _opened_relu_mask(fc1_out_pre_relu, relu_ctx=f"{relu3_context}_open" if relu3_context else f"{base_ctx}_relu3_open")
            fc1_out = _apply_mask_list(fc1_out_pre_relu, relu3_mask)
        else:
            fc1_out = self.relu_op.relu_list(fc1_out_pre_relu, node_id, context=relu3_context)
        
        # FC2: 120 → 84
        fc2_context = f"{context}_fc2" if context else None
        fc2_out_no_bias = self.matrix_ops.matrix_multiplier.secure_matrix_vector_multiply(
            fc2_weights, fc1_out, node_id, context=fc2_context
        )
        # Add FC2 bias
        fc2_out_pre_relu = []
        for i in range(len(fc2_out_no_bias)):
            out_with_bias = Share(
                x=fc2_out_no_bias[i].x,
                y=(fc2_out_no_bias[i].y + fc2_bias[i].y) % self.field_size,
                node_id=node_id
            )
            fc2_out_pre_relu.append(out_with_bias)
        
        # ReLU4: Apply ReLU activation after FC2
        relu4_context = f"{base_ctx}_relu4" if base_ctx else None
        relu4_mask = None
        if open_relu and reconstruction_manager is not None:
            relu4_mask = _opened_relu_mask(fc2_out_pre_relu, relu_ctx=f"{relu4_context}_open" if relu4_context else f"{base_ctx}_relu4_open")
            fc2_out = _apply_mask_list(fc2_out_pre_relu, relu4_mask)
        else:
            fc2_out = self.relu_op.relu_list(fc2_out_pre_relu, node_id, context=relu4_context)
        
        # FC3: 84 → 8
        fc3_context = f"{context}_fc3" if context else None
        fc3_out_no_bias = self.matrix_ops.matrix_multiplier.secure_matrix_vector_multiply(
            fc3_weights, fc2_out, node_id, context=fc3_context
        )
        # Add FC3 bias (CRITICAL: zero-initialized biases ensure no class has initial advantage)
        fc3_out = []
        for i in range(len(fc3_out_no_bias)):
            out_with_bias = Share(
                x=fc3_out_no_bias[i].x,
                y=(fc3_out_no_bias[i].y + fc3_bias[i].y) % self.field_size,
                node_id=node_id
            )
            fc3_out.append(out_with_bias)
        
        if return_intermediates:
            intermediates = {
                'input': input_shares,
                'conv1_out': conv1_out,  # Post-ReLU
                'conv1_out_pre_relu': conv1_out_pre_relu,  # Pre-ReLU (for backward)
                'relu1_mask': relu1_mask,
                'pool1_input_channels': pool1_input_channels,
                'pool1_out': pool1_out,
                'pool1_reshaped': pool1_reshaped,
                'conv2_out': conv2_out,  # Post-ReLU
                'conv2_out_pre_relu': conv2_out_pre_relu,  # Pre-ReLU (for backward)
                'relu2_mask': relu2_mask,
                'pool2_input_channels': pool2_input_channels,
                'pool2_out': pool2_out,
                'flattened': flattened,
                'fc1_out': fc1_out,  # Post-ReLU
                'fc1_out_pre_relu': fc1_out_pre_relu,  # Pre-ReLU (for backward)
                'relu3_mask': relu3_mask,
                'fc2_out': fc2_out,  # Post-ReLU
                'fc2_out_pre_relu': fc2_out_pre_relu,  # Pre-ReLU (for backward)
                'relu4_mask': relu4_mask,
            }
            return fc3_out, intermediates
        else:
            return fc3_out  # Output logits [8]

    def forward_fc_stack_batch(
        self,
        *,
        flattened_batch: List[List[Share]],
        weights: List,
        node_id: int,
        context: str,
        reconstruction_manager: Any = None,
        open_relu: bool = False,
        relu_opener_node: int = 1,
        relu_timeout: float = 120.0,
        packed_pss: bool = False,
        packed_timeout: float = 60.0,
    ) -> Tuple[List[List[Share]], List[dict]]:
        """
        Compute the fully-connected stack (fc1/fc2/fc3) for a whole mini-batch at once.

        This keeps weights secret-shared and uses SIMD-style batched Beaver openings via
        `secure_matrix_matrix_multiply_fixed_point`, so FC layers scale much better with batch size.

        Returns:
          - logits_per_sample: list length B, each is [num_classes] shares
          - fc_intermediates_per_sample: list length B of dicts compatible with backward_pass()
        """
        if not flattened_batch:
            return [], []

        _, _, fc1_weights, fc2_weights, fc3_weights, fc1_bias, fc2_bias, fc3_bias = weights
        B = len(flattened_batch)
        base_ctx = context if context else "lenet5_fc_batch"

        # Optional: exercise true Packed Shamir Secret Sharing (PSS) across the batch dimension.
        # With (n=3,t=1) we can pack k=2 lanes (exactly your typical batch_size=2).
        #
        # NOTE: This does NOT currently make FC faster because our FC matmuls use secret-shared weights,
        # and the codebase does not implement native packed-share multiplication with packed Beaver triples.
        # Instead, we pack+unpack activations (no opening) to verify the packed protocols work end-to-end
        # inside the LeNet-5 training pipeline.
        use_packed = bool(packed_pss)
        k_lanes = 0
        ops = None

        def _maybe_pack_unpack_batch_vectors(
            vec_batch: List[List[Share]],
            *,
            name: str,
        ) -> List[List[Share]]:
            nonlocal k_lanes, ops
            if not use_packed:
                return vec_batch
            if reconstruction_manager is None:
                raise RuntimeError("packed_pss requires reconstruction_manager (multi-node networking enabled)")
            if B < 2:
                # Nothing to pack across batch dimension
                return vec_batch

            # Lazy import to avoid overhead when packed_pss is disabled
            from ml_training.secret_sharing import PackedShamirSecretSharing
            from ml_training.packed_mpc_ops import PackedMPCOps

            if ops is None:
                ops = PackedMPCOps(
                    network=reconstruction_manager.network,
                    n_nodes=int(self.n_nodes),
                    t=int(self.t),
                    field_size=int(self.field_size),
                )
                pss = PackedShamirSecretSharing(field_size=int(self.field_size))
                kmax = int(pss.max_packing_factor(int(self.n_nodes), int(self.t)))
                k_lanes = min(int(B), kmax)
                if k_lanes < 2:
                    # Can't pack anything meaningful (e.g., n-t < 2 or B < 2)
                    return vec_batch

            dim = len(vec_batch[0]) if vec_batch and vec_batch[0] else 0
            if any(len(v) != dim for v in vec_batch):
                raise ValueError("vec_batch has inconsistent dimensions")

            # Pack each coordinate across the batch into one packed share, then unpack back to lane shares.
            # This should be an identity transform (mod p) and validates packed protocols.
            out: List[List[Share]] = [[Share(x=int(node_id), y=0, node_id=int(node_id)) for _ in range(dim)] for _ in range(B)]
            for i in range(dim):
                lane_shares = [vec_batch[lane][i] for lane in range(k_lanes)]
                ctx_base = f"{base_ctx}_pss_{name}_i{i}"
                packed = ops.pack_lane_shares_to_packed(
                    lane_shares=lane_shares,
                    k=int(k_lanes),
                    context=f"{ctx_base}_pack",
                    timeout=float(packed_timeout),
                )
                lanes = ops.unpack_packed_to_lane_shares(
                    packed_share=packed,
                    k=int(k_lanes),
                    context=f"{ctx_base}_unpack",
                    timeout=float(packed_timeout),
                )
                for lane in range(k_lanes):
                    out[lane][i] = lanes[lane]
                # If B > k_lanes (rare), pass through remaining lanes unchanged
                for lane in range(k_lanes, B):
                    out[lane][i] = vec_batch[lane][i]
            return out

        def _opened_relu_mask(shares_flat: List[Share], *, relu_ctx: str) -> np.ndarray:
            if not shares_flat:
                return np.zeros((0,), dtype=np.uint32)
            if reconstruction_manager is None:
                raise RuntimeError("open_relu requires reconstruction_manager (multi-node networking enabled)")

            net = reconstruction_manager.network
            opener = int(relu_opener_node)
            p = int(self.field_size)
            x_local = int(shares_flat[0].x)
            local_u32 = np.asarray([int(s.y) & 0xFFFFFFFF for s in shares_flat], dtype=np.uint32)
            net.broadcast_vector(relu_ctx, x=x_local, values=local_u32)

            if int(node_id) == opener:
                opened_u64 = reconstruction_manager.reconstruct_opened_vector_values(
                    context=relu_ctx,
                    values_local=local_u32,
                    x=x_local,
                    timeout=float(relu_timeout),
                )
                opened = opened_u64.astype(np.int64)
                opened = np.where(opened > (p // 2), opened - p, opened)
                # One-time debug: print pre-activation range per ReLU layer.
                try:
                    key = f"relu_range::{relu_ctx.split('_open')[0]}"
                    seen = getattr(self, "_dbg_relu_ranges_printed", set())
                    if key not in seen:
                        seen.add(key)
                        setattr(self, "_dbg_relu_ranges_printed", seen)
                        mn = float(np.min(opened) / float(self.scale_factor))
                        mx = float(np.max(opened) / float(self.scale_factor))
                        print(f"      [open_relu debug] {key} preact(min,max)=({mn:.3f},{mx:.3f})", flush=True)
                except Exception:
                    pass
                mask = (opened > 0).astype(np.uint32)

                for nid in sorted(net.node_configs.keys()):
                    if int(nid) == opener:
                        continue
                    net.channel.send_vector(int(nid), f"{relu_ctx}_mask_to_{nid}", x=int(nid), values=mask)
                return mask

            recv_ctx = f"{relu_ctx}_mask_to_{node_id}"
            start = time.time()
            while time.time() - start < float(relu_timeout):
                recv = net.channel.get_received_vector(recv_ctx)
                if opener in recv:
                    values = recv[opener].get("values")
                    arr = np.asarray(values, dtype=np.uint32)
                    try:
                        net.channel.clear_vector(recv_ctx)
                    except Exception:
                        pass
                    try:
                        net.channel.clear_vector(relu_ctx)
                    except Exception:
                        pass
                    return arr
                time.sleep(0.01)
            raise RuntimeError(f"Timed out waiting for opened ReLU mask (ctx={relu_ctx})")

        def _apply_mask_list(x_shares: List[Share], mask: np.ndarray) -> List[Share]:
            if not x_shares:
                return []
            mask_u32 = np.asarray(mask, dtype=np.uint32).reshape(-1)
            out: List[Share] = []
            for i, s in enumerate(x_shares):
                m = int(mask_u32[i]) & 1
                out.append(Share(x=s.x, y=(int(s.y) * m) % int(self.field_size), node_id=node_id))
            return out

        # FC1: 400 -> 120 for all samples
        fc1_ctx = f"{base_ctx}_fc1_batch"
        fc1_cols_no_bias = self.matrix_ops.matrix_multiplier.secure_matrix_matrix_multiply_fixed_point(
            fc1_weights,
            flattened_batch,
            node_id=int(node_id),
            context=fc1_ctx,
        )  # list of B vectors, each length 120

        fc1_out_pre_relu_batch: List[List[Share]] = []
        for b in range(B):
            vec = []
            for i in range(len(fc1_cols_no_bias[b])):
                vec.append(
                    Share(
                        x=fc1_cols_no_bias[b][i].x,
                        y=(int(fc1_cols_no_bias[b][i].y) + int(fc1_bias[i].y)) % int(self.field_size),
                        node_id=int(node_id),
                    )
                )
            fc1_out_pre_relu_batch.append(vec)

        # Optional PSS pack/unpack validation on FC1 pre-activations (across the batch dimension)
        fc1_out_pre_relu_batch = _maybe_pack_unpack_batch_vectors(fc1_out_pre_relu_batch, name="fc1_pre")

        fc1_out_batch: List[List[Share]] = []
        relu3_masks: List[Optional[np.ndarray]] = []
        for b in range(B):
            relu3_ctx = f"{base_ctx}_relu3_b{b}"
            if open_relu and reconstruction_manager is not None:
                m = _opened_relu_mask(fc1_out_pre_relu_batch[b], relu_ctx=f"{relu3_ctx}_open")
                fc1_out_batch.append(_apply_mask_list(fc1_out_pre_relu_batch[b], m))
                relu3_masks.append(m)
            else:
                fc1_out_batch.append(self.relu_op.relu_list(fc1_out_pre_relu_batch[b], int(node_id), context=relu3_ctx))
                relu3_masks.append(None)

        # FC2: 120 -> 84
        fc2_ctx = f"{base_ctx}_fc2_batch"
        fc2_cols_no_bias = self.matrix_ops.matrix_multiplier.secure_matrix_matrix_multiply_fixed_point(
            fc2_weights,
            fc1_out_batch,
            node_id=int(node_id),
            context=fc2_ctx,
        )  # list of B vectors, each length 84

        fc2_out_pre_relu_batch: List[List[Share]] = []
        for b in range(B):
            vec = []
            for i in range(len(fc2_cols_no_bias[b])):
                vec.append(
                    Share(
                        x=fc2_cols_no_bias[b][i].x,
                        y=(int(fc2_cols_no_bias[b][i].y) + int(fc2_bias[i].y)) % int(self.field_size),
                        node_id=int(node_id),
                    )
                )
            fc2_out_pre_relu_batch.append(vec)

        # Optional PSS pack/unpack validation on FC2 pre-activations (across the batch dimension)
        fc2_out_pre_relu_batch = _maybe_pack_unpack_batch_vectors(fc2_out_pre_relu_batch, name="fc2_pre")

        fc2_out_batch: List[List[Share]] = []
        relu4_masks: List[Optional[np.ndarray]] = []
        for b in range(B):
            relu4_ctx = f"{base_ctx}_relu4_b{b}"
            if open_relu and reconstruction_manager is not None:
                m = _opened_relu_mask(fc2_out_pre_relu_batch[b], relu_ctx=f"{relu4_ctx}_open")
                fc2_out_batch.append(_apply_mask_list(fc2_out_pre_relu_batch[b], m))
                relu4_masks.append(m)
            else:
                fc2_out_batch.append(self.relu_op.relu_list(fc2_out_pre_relu_batch[b], int(node_id), context=relu4_ctx))
                relu4_masks.append(None)

        # FC3: 84 -> 8 (logits)
        fc3_ctx = f"{base_ctx}_fc3_batch"
        fc3_cols_no_bias = self.matrix_ops.matrix_multiplier.secure_matrix_matrix_multiply_fixed_point(
            fc3_weights,
            fc2_out_batch,
            node_id=int(node_id),
            context=fc3_ctx,
        )  # list of B vectors, each length 8

        logits_batch: List[List[Share]] = []
        fc_intermediates_batch: List[dict] = []
        for b in range(B):
            logits = []
            for i in range(len(fc3_cols_no_bias[b])):
                logits.append(
                    Share(
                        x=fc3_cols_no_bias[b][i].x,
                        y=(int(fc3_cols_no_bias[b][i].y) + int(fc3_bias[i].y)) % int(self.field_size),
                        node_id=int(node_id),
                    )
                )
            logits_batch.append(logits)
            fc_intermediates_batch.append(
                {
                    "fc1_out": fc1_out_batch[b],
                    "fc1_out_pre_relu": fc1_out_pre_relu_batch[b],
                    "relu3_mask": relu3_masks[b],
                    "fc2_out": fc2_out_batch[b],
                    "fc2_out_pre_relu": fc2_out_pre_relu_batch[b],
                    "relu4_mask": relu4_masks[b],
                }
            )

        # Optional PSS pack/unpack validation on logits (across batch dimension)
        logits_batch = _maybe_pack_unpack_batch_vectors(logits_batch, name="logits")

        return logits_batch, fc_intermediates_batch
    
    def backward_pass(self, output_grad_shares: List[Share],
                     input_shares: List[List[List[Share]]],
                     weights: List,
                     forward_outputs: dict,
                     node_id: int = 1,
                     context: Optional[str] = None) -> List:
        """
        Backward pass through LeNet-5
        
        Args:
            output_grad_shares: Gradient w.r.t. output [8]
            input_shares: Original input image [32 x 32 x 1]
            weights: List of weight shares for all layers
            forward_outputs: Optional dict storing forward pass outputs (for efficiency)
            node_id: Node ID
            context: Optional context
        
        Returns:
            List of gradient shares for each weight layer
        """
        conv1_weights, conv2_weights, fc1_weights, fc2_weights, fc3_weights, fc1_bias, fc2_bias, fc3_bias = weights
        
        # Use forward pass outputs from cache
        fc2_out = forward_outputs['fc2_out']
        fc1_out = forward_outputs['fc1_out']
        flattened = forward_outputs['flattened']
        pool2_out = forward_outputs['pool2_out']
        pool1_reshaped = forward_outputs['pool1_reshaped']
        conv1_out = forward_outputs['conv1_out']

        # Fast-path for FC backward in multi-node mode:
        # Use SIMD-style batched Beaver openings for the many fixed-point multiplies in FC layers.
        # This keeps nodes in sync (especially in privacy_mode where non-dealer nodes must request
        # triples over the network) and avoids timeouts later in conv backprop.
        mult = self.matrix_ops.multiplier
        p = int(self.field_size)
        x0 = int(fc2_out[0].x) if fc2_out else int(node_id)
        use_batched_fp = bool(getattr(mult, "reconstruction_manager", None) is not None) and (self.n_nodes > 1)

        def _mul_fp_batch(y1: np.ndarray, y2: np.ndarray, *, ctx_prefix: str) -> np.ndarray:
            return mult.multiply_batch_values_fixed_point(
                y1=y1.astype(np.uint64, copy=False),
                y2=y2.astype(np.uint64, copy=False),
                x=x0,
                node_id=int(node_id),
                context_prefix=str(ctx_prefix),
                scale_factor=int(self.scale_factor),
            )
        
        # FC3 backward: 84 → 8
        # Gradient w.r.t. FC3 weights: grad[i][j] = fc2_out[j] * output_grad[i]
        fc3_grad_context = f"{context}_fc3_grad" if context else None
        if use_batched_fp:
            # fc3_weight_grad: (8,84) outer product of output_grad (8) and fc2_out (84)
            fc2_out_y = np.asarray([int(s.y) % p for s in fc2_out], dtype=np.uint64)  # (84,)
            outg_y = np.asarray([int(s.y) % p for s in output_grad_shares], dtype=np.uint64)  # (8,)
            y1 = np.tile(fc2_out_y, int(outg_y.size))  # (8*84,)
            y2 = np.repeat(outg_y, int(fc2_out_y.size))  # (8*84,)
            ctx_w = f"{fc3_grad_context}_w_batch" if fc3_grad_context else "fc3_grad_w_batch"
            prod = _mul_fp_batch(y1, y2, ctx_prefix=ctx_w).reshape(int(outg_y.size), int(fc2_out_y.size))
            fc3_weight_grad = [
                [Share(x=x0, y=int(prod[i, j]) % p, node_id=node_id) for j in range(int(fc2_out_y.size))]
                for i in range(int(outg_y.size))
            ]
        else:
            fc3_weight_grad = []
            for i in range(len(fc3_weights)):  # 8 outputs
                fc3_row_grad = []
                for j in range(len(fc3_weights[i])):  # 84 inputs
                    grad_share = self.matrix_ops.matrix_multiplier.multiplier.multiply(
                        fc2_out[j], output_grad_shares[i], node_id,
                        context=f"{fc3_grad_context}_w{i}_{j}" if fc3_grad_context else None
                    )
                    # Fixed-point rescale: both operands are SCALE-scaled
                    grad_share = Share(
                        x=grad_share.x,
                        y=(int(grad_share.y) * int(self._inv_scale)) % int(self.field_size),
                        node_id=node_id
                    )
                    fc3_row_grad.append(grad_share)
                fc3_weight_grad.append(fc3_row_grad)
        
        # FC3 bias gradient: grad[i] = output_grad[i] (bias gradient is just the output gradient)
        fc3_bias_grad = [Share(x=g.x, y=g.y, node_id=node_id) for g in output_grad_shares]
        
        # Gradient w.r.t. FC2 output: grad[j] = sum_i(fc3_weights[i][j] * output_grad[i])
        if use_batched_fp:
            # For each j in 0..83, compute sum_i fc3_weights[i][j] * output_grad[i]
            w3_y = np.asarray([[int(fc3_weights[i][j].y) % p for j in range(len(fc3_weights[i]))] for i in range(len(fc3_weights))],
                              dtype=np.uint64)  # (8,84)
            outg_y = np.asarray([int(s.y) % p for s in output_grad_shares], dtype=np.uint64)  # (8,)
            y1 = w3_y.T.reshape(-1)  # (84*8,) with j outer, i inner
            y2 = np.tile(outg_y, w3_y.shape[1])  # (84*8,)
            ctx_o = f"{fc3_grad_context}_o_batch" if fc3_grad_context else "fc3_grad_o_batch"
            prod = _mul_fp_batch(y1, y2, ctx_prefix=ctx_o).reshape(w3_y.shape[1], w3_y.shape[0])  # (84,8)
            sums = (prod.sum(axis=1) % np.uint64(p)).astype(np.uint64, copy=False)
            fc2_output_grad_pre_relu = [Share(x=x0, y=int(sums[j]) % p, node_id=node_id) for j in range(int(sums.size))]
        else:
            fc2_output_grad_pre_relu = []
            for j in range(len(fc2_out)):  # 84
                grad_sum = Share(x=fc2_out[0].x, y=0, node_id=node_id)
                for i in range(len(output_grad_shares)):  # 8
                    grad_prod = self.matrix_ops.matrix_multiplier.multiplier.multiply(
                        fc3_weights[i][j], output_grad_shares[i], node_id,
                        context=f"{fc3_grad_context}_o{j}_{i}" if fc3_grad_context else None
                    )
                    grad_prod = Share(
                        x=grad_prod.x,
                        y=(int(grad_prod.y) * int(self._inv_scale)) % int(self.field_size),
                        node_id=node_id
                    )
                    grad_sum = Share(
                        x=grad_sum.x,
                        y=(grad_sum.y + grad_prod.y) % self.field_size,
                        node_id=node_id
                    )
                fc2_output_grad_pre_relu.append(grad_sum)
        
        # ReLU4 backward: grad_input = output_grad * 1[x>0]
        relu4_mask = forward_outputs.get("relu4_mask")
        if relu4_mask is not None:
            m = np.asarray(relu4_mask, dtype=np.uint32).reshape(-1)
            fc2_output_grad = [
                Share(x=g.x, y=(int(g.y) * (int(m[i]) & 1)) % int(self.field_size), node_id=node_id)
                for i, g in enumerate(fc2_output_grad_pre_relu)
            ]
        else:
            fc2_out_pre_relu = forward_outputs.get('fc2_out_pre_relu', fc2_out)
            relu4_backward_context = f"{context}_relu4_backward" if context else None
            fc2_output_grad = self.relu_op.relu_backward_list(
                fc2_output_grad_pre_relu, fc2_out_pre_relu, node_id, context=relu4_backward_context
            )
        
        # FC2 backward: 120 → 84
        fc2_grad_context = f"{context}_fc2_grad" if context else None
        if use_batched_fp:
            fc1_out_y = np.asarray([int(s.y) % p for s in fc1_out], dtype=np.uint64)  # (120,)
            fc2_g_y = np.asarray([int(s.y) % p for s in fc2_output_grad], dtype=np.uint64)  # (84,)
            y1 = np.tile(fc1_out_y, int(fc2_g_y.size))  # (84*120,)
            y2 = np.repeat(fc2_g_y, int(fc1_out_y.size))  # (84*120,)
            ctx_w = f"{fc2_grad_context}_w_batch" if fc2_grad_context else "fc2_grad_w_batch"
            prod = _mul_fp_batch(y1, y2, ctx_prefix=ctx_w).reshape(int(fc2_g_y.size), int(fc1_out_y.size))  # (84,120)
            fc2_weight_grad = [
                [Share(x=x0, y=int(prod[i, j]) % p, node_id=node_id) for j in range(int(fc1_out_y.size))]
                for i in range(int(fc2_g_y.size))
            ]
        else:
            fc2_weight_grad = []
            for i in range(len(fc2_weights)):  # 84 outputs
                fc2_row_grad = []
                for j in range(len(fc2_weights[i])):  # 120 inputs
                    grad_share = self.matrix_ops.matrix_multiplier.multiplier.multiply(
                        fc1_out[j], fc2_output_grad[i], node_id,
                        context=f"{fc2_grad_context}_w{i}_{j}" if fc2_grad_context else None
                    )
                    grad_share = Share(
                        x=grad_share.x,
                        y=(int(grad_share.y) * int(self._inv_scale)) % int(self.field_size),
                        node_id=node_id
                    )
                    fc2_row_grad.append(grad_share)
                fc2_weight_grad.append(fc2_row_grad)
        
        # FC2 bias gradient: grad[i] = fc2_output_grad[i]
        fc2_bias_grad = [Share(x=g.x, y=g.y, node_id=node_id) for g in fc2_output_grad]
        
        # Gradient w.r.t. FC1 output
        if use_batched_fp:
            # For each j in 0..119, compute sum_i fc2_weights[i][j] * fc2_output_grad[i]
            w2_y = np.asarray([[int(fc2_weights[i][j].y) % p for j in range(len(fc2_weights[i]))] for i in range(len(fc2_weights))],
                              dtype=np.uint64)  # (84,120)
            fc2_g_y = np.asarray([int(s.y) % p for s in fc2_output_grad], dtype=np.uint64)  # (84,)
            y1 = w2_y.T.reshape(-1)  # (120*84,) with j outer, i inner
            y2 = np.tile(fc2_g_y, w2_y.shape[1])  # (120*84,)
            ctx_o = f"{fc2_grad_context}_o_batch" if fc2_grad_context else "fc2_grad_o_batch"
            prod = _mul_fp_batch(y1, y2, ctx_prefix=ctx_o).reshape(w2_y.shape[1], w2_y.shape[0])  # (120,84)
            sums = (prod.sum(axis=1) % np.uint64(p)).astype(np.uint64, copy=False)
            fc1_output_grad_pre_relu = [Share(x=x0, y=int(sums[j]) % p, node_id=node_id) for j in range(int(sums.size))]
        else:
            fc1_output_grad_pre_relu = []
            for j in range(len(fc1_out)):  # 120
                grad_sum = Share(x=fc1_out[0].x, y=0, node_id=node_id)
                for i in range(len(fc2_output_grad)):  # 84
                    grad_prod = self.matrix_ops.matrix_multiplier.multiplier.multiply(
                        fc2_weights[i][j], fc2_output_grad[i], node_id,
                        context=f"{fc2_grad_context}_o{j}_{i}" if fc2_grad_context else None
                    )
                    grad_prod = Share(
                        x=grad_prod.x,
                        y=(int(grad_prod.y) * int(self._inv_scale)) % int(self.field_size),
                        node_id=node_id
                    )
                    grad_sum = Share(
                        x=grad_sum.x,
                        y=(grad_sum.y + grad_prod.y) % self.field_size,
                        node_id=node_id
                    )
                fc1_output_grad_pre_relu.append(grad_sum)
        
        # ReLU3 backward: grad_input = output_grad * 1[x>0]
        relu3_mask = forward_outputs.get("relu3_mask")
        if relu3_mask is not None:
            m = np.asarray(relu3_mask, dtype=np.uint32).reshape(-1)
            fc1_output_grad = [
                Share(x=g.x, y=(int(g.y) * (int(m[i]) & 1)) % int(self.field_size), node_id=node_id)
                for i, g in enumerate(fc1_output_grad_pre_relu)
            ]
        else:
            fc1_out_pre_relu = forward_outputs.get('fc1_out_pre_relu', fc1_out)
            relu3_backward_context = f"{context}_relu3_backward" if context else None
            fc1_output_grad = self.relu_op.relu_backward_list(
                fc1_output_grad_pre_relu, fc1_out_pre_relu, node_id, context=relu3_backward_context
            )
        
        # FC1 backward: 400 → 120
        fc1_grad_context = f"{context}_fc1_grad" if context else None
        if use_batched_fp:
            flat_y = np.asarray([int(s.y) % p for s in flattened], dtype=np.uint64)  # (400,)
            fc1_g_y = np.asarray([int(s.y) % p for s in fc1_output_grad], dtype=np.uint64)  # (120,)
            y1 = np.tile(flat_y, int(fc1_g_y.size))  # (120*400,)
            y2 = np.repeat(fc1_g_y, int(flat_y.size))  # (120*400,)
            ctx_w = f"{fc1_grad_context}_w_batch" if fc1_grad_context else "fc1_grad_w_batch"
            prod = _mul_fp_batch(y1, y2, ctx_prefix=ctx_w).reshape(int(fc1_g_y.size), int(flat_y.size))  # (120,400)
            fc1_weight_grad = [
                [Share(x=x0, y=int(prod[i, j]) % p, node_id=node_id) for j in range(int(flat_y.size))]
                for i in range(int(fc1_g_y.size))
            ]
        else:
            fc1_weight_grad = []
            for i in range(len(fc1_weights)):  # 120 outputs
                fc1_row_grad = []
                for j in range(len(fc1_weights[i])):  # 400 inputs
                    grad_share = self.matrix_ops.matrix_multiplier.multiplier.multiply(
                        flattened[j], fc1_output_grad[i], node_id,
                        context=f"{fc1_grad_context}_w{i}_{j}" if fc1_grad_context else None
                    )
                    grad_share = Share(
                        x=grad_share.x,
                        y=(int(grad_share.y) * int(self._inv_scale)) % int(self.field_size),
                        node_id=node_id
                    )
                    fc1_row_grad.append(grad_share)
                fc1_weight_grad.append(fc1_row_grad)
        
        # FC1 bias gradient: grad[i] = fc1_output_grad[i]
        fc1_bias_grad = [Share(x=g.x, y=g.y, node_id=node_id) for g in fc1_output_grad]
        
        # Gradient w.r.t. flattened (for pool2 backward)
        if use_batched_fp:
            # For each j in 0..399, compute sum_i fc1_weights[i][j] * fc1_output_grad[i]
            w1_y = np.asarray([[int(fc1_weights[i][j].y) % p for j in range(len(fc1_weights[i]))] for i in range(len(fc1_weights))],
                              dtype=np.uint64)  # (120,400)
            fc1_g_y = np.asarray([int(s.y) % p for s in fc1_output_grad], dtype=np.uint64)  # (120,)
            y1 = w1_y.T.reshape(-1)  # (400*120,) with j outer, i inner
            y2 = np.tile(fc1_g_y, w1_y.shape[1])  # (400*120,)
            ctx_o = f"{fc1_grad_context}_o_batch" if fc1_grad_context else "fc1_grad_o_batch"
            prod = _mul_fp_batch(y1, y2, ctx_prefix=ctx_o).reshape(w1_y.shape[1], w1_y.shape[0])  # (400,120)
            sums = (prod.sum(axis=1) % np.uint64(p)).astype(np.uint64, copy=False)
            flattened_grad = [Share(x=x0, y=int(sums[j]) % p, node_id=node_id) for j in range(int(sums.size))]
        else:
            flattened_grad = []
            for j in range(len(flattened)):  # 400
                grad_sum = Share(x=flattened[0].x, y=0, node_id=node_id)
                for i in range(len(fc1_output_grad)):  # 120
                    grad_prod = self.matrix_ops.matrix_multiplier.multiplier.multiply(
                        fc1_weights[i][j], fc1_output_grad[i], node_id,
                        context=f"{fc1_grad_context}_o{j}_{i}" if fc1_grad_context else None
                    )
                    grad_prod = Share(
                        x=grad_prod.x,
                        y=(int(grad_prod.y) * int(self._inv_scale)) % int(self.field_size),
                        node_id=node_id
                    )
                    grad_sum = Share(
                        x=grad_sum.x,
                        y=(grad_sum.y + grad_prod.y) % self.field_size,
                        node_id=node_id
                    )
                flattened_grad.append(grad_sum)
        
        # Unflatten: 400 → 5x5x16
        pool2_output_grad = []
        grad_idx = 0
        for c in range(16):
            channel_grad = []
            for h in range(5):
                row_grad = []
                for w in range(5):
                    if grad_idx < len(flattened_grad):
                        row_grad.append(flattened_grad[grad_idx])
                        grad_idx += 1
                    else:
                        # Fallback
                        zero_share = Share(x=flattened_grad[0].x, y=0, node_id=node_id)
                        row_grad.append(zero_share)
                channel_grad.append(row_grad)
            pool2_output_grad.append(channel_grad)
        
        # Pool2 backward: 10x10x16 → 5x5x16
        pool2_grad_context = f"{context}_pool2_grad" if context else None
        pool2_input_channels = forward_outputs['pool2_input_channels']
        conv2_output_grad = []
        for c in range(16):
            channel_grad = pool2_output_grad[c]
            # Use original pool2 input for backward pass
            pool2_input = pool2_input_channels[c]
            conv2_channel_grad = self.pool_op.pool2d_backward(
                channel_grad, pool2_input, pool_size=(2, 2), stride=2,
                pool_type="max", node_id=node_id,
                context=f"{pool2_grad_context}_c{c}" if pool2_grad_context else None
            )
            conv2_output_grad.append(conv2_channel_grad)
        
        # Reshape for Conv2 backward: 10x10x16
        conv2_output_grad_reshaped = []
        for h in range(10):
            conv2_row = []
            for w in range(10):
                conv2_channels = [conv2_output_grad[c][h][w] for c in range(16)]
                conv2_row.append(conv2_channels)
            conv2_output_grad_reshaped.append(conv2_row)
        
        # ReLU2 backward on Conv2: apply stored mask if available
        relu2_mask = forward_outputs.get("relu2_mask")
        if relu2_mask is not None:
            m = np.asarray(relu2_mask, dtype=np.uint32).reshape(10, 10, 16)
            conv2_output_grad_reshaped_relu = []
            for h in range(10):
                row = []
                for w in range(10):
                    chans = []
                    for c in range(16):
                        g = conv2_output_grad_reshaped[h][w][c]
                        chans.append(Share(x=g.x, y=(int(g.y) * (int(m[h, w, c]) & 1)) % int(self.field_size), node_id=node_id))
                    row.append(chans)
                conv2_output_grad_reshaped_relu.append(row)
            conv2_output_grad_reshaped = conv2_output_grad_reshaped_relu
        else:
            conv2_out_pre_relu = forward_outputs.get('conv2_out_pre_relu')
            if conv2_out_pre_relu is not None:
                relu2_backward_context = f"{context}_relu2_backward" if context else None
                # Apply ReLU backward element-wise to 3D array
                conv2_output_grad_reshaped_relu = []
                for h in range(10):
                    conv2_row_relu = []
                    for w in range(10):
                        conv2_channels_relu = []
                        for c in range(16):
                            out_grad = conv2_output_grad_reshaped[h][w][c]
                            inp_pre_relu = conv2_out_pre_relu[h][w][c]
                            relu_ctx = f"{relu2_backward_context}_{h}_{w}_{c}" if relu2_backward_context else None
                            grad_after_relu = self.relu_op.relu_backward(out_grad, inp_pre_relu, node_id, context=relu_ctx)
                            conv2_channels_relu.append(grad_after_relu)
                        conv2_row_relu.append(conv2_channels_relu)
                    conv2_output_grad_reshaped_relu.append(conv2_row_relu)
                conv2_output_grad_reshaped = conv2_output_grad_reshaped_relu
        
        # Conv2 backward: 14x14x6 → 10x10x16
        conv2_grad_context = f"{context}_conv2_grad" if context else None
        # Use pool1_reshaped from forward pass
        pool1_reshaped = forward_outputs['pool1_reshaped']
        pool1_output_grad_reshaped, conv2_weight_grad = self.conv_op.conv2d_backward(
            conv2_output_grad_reshaped, pool1_reshaped, conv2_weights, stride=1, padding=0,
            node_id=node_id, context=conv2_grad_context
        )
        
        # Pool1 backward: 28x28x6 → 14x14x6
        pool1_grad_context = f"{context}_pool1_grad" if context else None
        pool1_input_channels = forward_outputs['pool1_input_channels']
        conv1_output_grad = []
        for c in range(6):
            # Extract channel gradient from reshaped
            pool1_channel_grad = []
            for h in range(14):
                row_grad = []
                for w in range(14):
                    row_grad.append(pool1_output_grad_reshaped[h][w][c])
                pool1_channel_grad.append(row_grad)
            
            # Use original pool1 input for backward pass
            pool1_input = pool1_input_channels[c]
            conv1_channel_grad = self.pool_op.pool2d_backward(
                pool1_channel_grad, pool1_input, pool_size=(2, 2), stride=2,
                pool_type="max", node_id=node_id,
                context=f"{pool1_grad_context}_c{c}" if pool1_grad_context else None
            )
            conv1_output_grad.append(conv1_channel_grad)
        
        # Reshape for Conv1 backward: 28x28x6
        conv1_output_grad_reshaped = []
        for h in range(28):
            conv1_row = []
            for w in range(28):
                conv1_channels = [conv1_output_grad[c][h][w] for c in range(6)]
                conv1_row.append(conv1_channels)
            conv1_output_grad_reshaped.append(conv1_row)
        
        # ReLU1 backward on Conv1: apply stored mask if available
        relu1_mask = forward_outputs.get("relu1_mask")
        if relu1_mask is not None:
            m = np.asarray(relu1_mask, dtype=np.uint32).reshape(28, 28, 6)
            conv1_output_grad_reshaped_relu = []
            for h in range(28):
                row = []
                for w in range(28):
                    chans = []
                    for c in range(6):
                        g = conv1_output_grad_reshaped[h][w][c]
                        chans.append(Share(x=g.x, y=(int(g.y) * (int(m[h, w, c]) & 1)) % int(self.field_size), node_id=node_id))
                    row.append(chans)
                conv1_output_grad_reshaped_relu.append(row)
            conv1_output_grad_reshaped = conv1_output_grad_reshaped_relu
        else:
            conv1_out_pre_relu = forward_outputs.get('conv1_out_pre_relu')
            if conv1_out_pre_relu is not None:
                relu1_backward_context = f"{context}_relu1_backward" if context else None
                # Apply ReLU backward element-wise to 3D array
                conv1_output_grad_reshaped_relu = []
                for h in range(28):
                    conv1_row_relu = []
                    for w in range(28):
                        conv1_channels_relu = []
                        for c in range(6):
                            out_grad = conv1_output_grad_reshaped[h][w][c]
                            inp_pre_relu = conv1_out_pre_relu[h][w][c]
                            relu_ctx = f"{relu1_backward_context}_{h}_{w}_{c}" if relu1_backward_context else None
                            grad_after_relu = self.relu_op.relu_backward(out_grad, inp_pre_relu, node_id, context=relu_ctx)
                            conv1_channels_relu.append(grad_after_relu)
                        conv1_row_relu.append(conv1_channels_relu)
                    conv1_output_grad_reshaped_relu.append(conv1_row_relu)
                conv1_output_grad_reshaped = conv1_output_grad_reshaped_relu
        
        # Conv1 backward: 32x32x1 → 28x28x6
        conv1_grad_context = f"{context}_conv1_grad" if context else None
        input_grad, conv1_weight_grad = self.conv_op.conv2d_backward(
            conv1_output_grad_reshaped, input_shares, conv1_weights, stride=1, padding=0,
            node_id=node_id, context=conv1_grad_context
        )
        
        # Return gradients for all weight layers (including biases)
        return [
            conv1_weight_grad,
            conv2_weight_grad,
            fc1_weight_grad,
            fc2_weight_grad,
            fc3_weight_grad,
            fc1_bias_grad,
            fc2_bias_grad,
            fc3_bias_grad
        ]
