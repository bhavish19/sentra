"""
LeNet-5 Architecture for Secure MPC Training
Implements LeNet-5 CNN architecture using secure operations
"""

from typing import List, Tuple, Optional
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_convolution import SecureConvolution
from ml_training.secure_pooling import SecurePooling
from ml_training.secure_matrix_ops import SecureMatrixOperations
from ml_training.beaver_triples import SecureMultiplier
from ml_training.secure_comparison import SecureComparator
from ml_training.secure_division import SecureDivider
from ml_training.secure_relu import SecureReLU
import numpy as np


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
                 divider: Optional[SecureDivider] = None):
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
        
        # Initialize secure operations
        self.conv_op = SecureConvolution(multiplier, field_size)
        self.pool_op = SecurePooling(comparator, divider, field_size)
        self.matrix_ops = SecureMatrixOperations(multiplier, field_size)
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
                        std = np.sqrt(4.0 / fan_in)  # Scaled He initialization (2x larger)
                        w_val = np.random.randn() * std
                        w_int = int(w_val * 10_000_000) % self.field_size  # FIXED: Use 10M to match training SCALE_FACTOR
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
                        std = np.sqrt(4.0 / fan_in)  # Scaled He (2x larger)
                        w_val = np.random.randn() * std
                        w_int = int(w_val * 10_000_000) % self.field_size  # FIXED: Use 10M to match training SCALE_FACTOR
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
                std = np.sqrt(6.0 / fan_in)  # Scaled He (3x larger to compensate for large fan_in)
                w_val = np.random.randn() * std
                w_int = int(w_val * 10_000_000) % self.field_size  # FIXED: Use 10M to match training SCALE_FACTOR
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
                std = np.sqrt(4.0 / fan_in)  # Scaled He (2x larger)
                w_val = np.random.randn() * std
                w_int = int(w_val * 10_000_000) % self.field_size  # FIXED: Use 10M to match training SCALE_FACTOR
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
                std = np.sqrt(2.0 / fan_in)  # Standard He for output layer
                w_val = np.random.randn() * std
                w_int = int(w_val * 10_000_000) % self.field_size  # FIXED: Use 10M to match training SCALE_FACTOR
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
                    return_intermediates: bool = False) -> Tuple[List[Share], Optional[dict]]:
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
        
        # Conv1: 32x32x1 → 28x28x6
        conv1_context = f"{context}_conv1" if context else None
        conv1_out_pre_relu = self.conv_op.conv2d(
            input_shares, conv1_weights, stride=1, padding=0,
            node_id=node_id, context=conv1_context
        )
        
        # ReLU1: Apply ReLU activation after Conv1
        relu1_context = f"{context}_relu1" if context else None
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
        conv2_context = f"{context}_conv2" if context else None
        conv2_out_pre_relu = self.conv_op.conv2d(
            pool1_reshaped, conv2_weights, stride=1, padding=0,
            node_id=node_id, context=conv2_context
        )
        
        # ReLU2: Apply ReLU activation after Conv2
        relu2_context = f"{context}_relu2" if context else None
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
        relu3_context = f"{context}_relu3" if context else None
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
        relu4_context = f"{context}_relu4" if context else None
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
                'pool1_input_channels': pool1_input_channels,
                'pool1_out': pool1_out,
                'pool1_reshaped': pool1_reshaped,
                'conv2_out': conv2_out,  # Post-ReLU
                'conv2_out_pre_relu': conv2_out_pre_relu,  # Pre-ReLU (for backward)
                'pool2_input_channels': pool2_input_channels,
                'pool2_out': pool2_out,
                'flattened': flattened,
                'fc1_out': fc1_out,  # Post-ReLU
                'fc1_out_pre_relu': fc1_out_pre_relu,  # Pre-ReLU (for backward)
                'fc2_out': fc2_out,  # Post-ReLU
                'fc2_out_pre_relu': fc2_out_pre_relu  # Pre-ReLU (for backward)
            }
            return fc3_out, intermediates
        else:
            return fc3_out  # Output logits [8]
    
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
        
        # FC3 backward: 84 → 8
        # Gradient w.r.t. FC3 weights: grad[i][j] = fc2_out[j] * output_grad[i]
        fc3_grad_context = f"{context}_fc3_grad" if context else None
        fc3_weight_grad = []
        for i in range(len(fc3_weights)):  # 8 outputs
            fc3_row_grad = []
            for j in range(len(fc3_weights[i])):  # 84 inputs
                grad_share = self.matrix_ops.matrix_multiplier.multiplier.multiply(
                    fc2_out[j], output_grad_shares[i], node_id,
                    context=f"{fc3_grad_context}_w{i}_{j}" if fc3_grad_context else None
                )
                fc3_row_grad.append(grad_share)
            fc3_weight_grad.append(fc3_row_grad)
        
        # FC3 bias gradient: grad[i] = output_grad[i] (bias gradient is just the output gradient)
        fc3_bias_grad = [Share(x=g.x, y=g.y, node_id=node_id) for g in output_grad_shares]
        
        # Gradient w.r.t. FC2 output: grad[j] = sum_i(fc3_weights[i][j] * output_grad[i])
        fc2_output_grad_pre_relu = []
        for j in range(len(fc2_out)):  # 84
            grad_sum = Share(x=fc2_out[0].x, y=0, node_id=node_id)
            for i in range(len(output_grad_shares)):  # 8
                grad_prod = self.matrix_ops.matrix_multiplier.multiplier.multiply(
                    fc3_weights[i][j], output_grad_shares[i], node_id,
                    context=f"{fc3_grad_context}_o{j}_{i}" if fc3_grad_context else None
                )
                grad_sum = Share(
                    x=grad_sum.x,
                    y=(grad_sum.y + grad_prod.y) % self.field_size,
                    node_id=node_id
                )
            fc2_output_grad_pre_relu.append(grad_sum)
        
        # ReLU4 backward: Apply ReLU gradient (grad_input = output_grad * (input > 0))
        fc2_out_pre_relu = forward_outputs.get('fc2_out_pre_relu', fc2_out)
        relu4_backward_context = f"{context}_relu4_backward" if context else None
        fc2_output_grad = self.relu_op.relu_backward_list(
            fc2_output_grad_pre_relu, fc2_out_pre_relu, node_id, context=relu4_backward_context
        )
        
        # FC2 backward: 120 → 84
        fc2_grad_context = f"{context}_fc2_grad" if context else None
        fc2_weight_grad = []
        for i in range(len(fc2_weights)):  # 84 outputs
            fc2_row_grad = []
            for j in range(len(fc2_weights[i])):  # 120 inputs
                grad_share = self.matrix_ops.matrix_multiplier.multiplier.multiply(
                    fc1_out[j], fc2_output_grad[i], node_id,
                    context=f"{fc2_grad_context}_w{i}_{j}" if fc2_grad_context else None
                )
                fc2_row_grad.append(grad_share)
            fc2_weight_grad.append(fc2_row_grad)
        
        # FC2 bias gradient: grad[i] = fc2_output_grad[i]
        fc2_bias_grad = [Share(x=g.x, y=g.y, node_id=node_id) for g in fc2_output_grad]
        
        # Gradient w.r.t. FC1 output
        fc1_output_grad_pre_relu = []
        for j in range(len(fc1_out)):  # 120
            grad_sum = Share(x=fc1_out[0].x, y=0, node_id=node_id)
            for i in range(len(fc2_output_grad)):  # 84
                grad_prod = self.matrix_ops.matrix_multiplier.multiplier.multiply(
                    fc2_weights[i][j], fc2_output_grad[i], node_id,
                    context=f"{fc2_grad_context}_o{j}_{i}" if fc2_grad_context else None
                )
                grad_sum = Share(
                    x=grad_sum.x,
                    y=(grad_sum.y + grad_prod.y) % self.field_size,
                    node_id=node_id
                )
            fc1_output_grad_pre_relu.append(grad_sum)
        
        # ReLU3 backward: Apply ReLU gradient
        fc1_out_pre_relu = forward_outputs.get('fc1_out_pre_relu', fc1_out)
        relu3_backward_context = f"{context}_relu3_backward" if context else None
        fc1_output_grad = self.relu_op.relu_backward_list(
            fc1_output_grad_pre_relu, fc1_out_pre_relu, node_id, context=relu3_backward_context
        )
        
        # FC1 backward: 400 → 120
        fc1_grad_context = f"{context}_fc1_grad" if context else None
        fc1_weight_grad = []
        for i in range(len(fc1_weights)):  # 120 outputs
            fc1_row_grad = []
            for j in range(len(fc1_weights[i])):  # 400 inputs
                grad_share = self.matrix_ops.matrix_multiplier.multiplier.multiply(
                    flattened[j], fc1_output_grad[i], node_id,
                    context=f"{fc1_grad_context}_w{i}_{j}" if fc1_grad_context else None
                )
                fc1_row_grad.append(grad_share)
            fc1_weight_grad.append(fc1_row_grad)
        
        # FC1 bias gradient: grad[i] = fc1_output_grad[i]
        fc1_bias_grad = [Share(x=g.x, y=g.y, node_id=node_id) for g in fc1_output_grad]
        
        # Gradient w.r.t. flattened (for pool2 backward)
        flattened_grad = []
        for j in range(len(flattened)):  # 400
            grad_sum = Share(x=flattened[0].x, y=0, node_id=node_id)
            for i in range(len(fc1_output_grad)):  # 120
                grad_prod = self.matrix_ops.matrix_multiplier.multiplier.multiply(
                    fc1_weights[i][j], fc1_output_grad[i], node_id,
                    context=f"{fc1_grad_context}_o{j}_{i}" if fc1_grad_context else None
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
        
        # ReLU2 backward: Apply ReLU gradient to Conv2 output gradients
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
        
        # ReLU1 backward: Apply ReLU gradient to Conv1 output gradients
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
