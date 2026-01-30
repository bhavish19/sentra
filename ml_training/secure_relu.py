"""
Secure ReLU Activation Function
Implements ReLU(x) = max(0, x) using secure comparison and multiplication
"""

from typing import List
from ml_training.secret_sharing import Share
from ml_training.secure_comparison import SecureComparator
from ml_training.beaver_triples import SecureMultiplier


class SecureReLU:
    """
    Performs secure ReLU activation: ReLU(x) = max(0, x)
    """
    
    def __init__(self, comparator: SecureComparator, multiplier: SecureMultiplier,
                 field_size: int = 2**32 - 5):
        """
        Initialize secure ReLU
        
        Args:
            comparator: SecureComparator instance for secure comparison
            multiplier: SecureMultiplier instance for secure multiplication
            field_size: Prime field size
        """
        self.comparator = comparator
        self.multiplier = multiplier
        self.field_size = field_size
    
    def relu(self, x_share: Share, node_id: int, context: str = None) -> Share:
        """
        Compute ReLU(x) = max(0, x) securely
        
        ReLU(x) = x if x > 0, else 0
        This is computed as: x * (x > 0)
        
        Args:
            x_share: Share of input value
            node_id: Node ID
            context: Optional context for debugging
        
        Returns:
            Share of ReLU(x)
        """
        # Create zero share (0 in the field)
        zero_share = Share(x=x_share.x, y=0, node_id=node_id)
        
        # Securely compare x > 0
        # Returns 1 if x > 0, 0 if x <= 0
        is_positive = self.comparator.secure_greater_than(x_share, zero_share, node_id)
        
        # ReLU(x) = x * (x > 0)
        # If x > 0: is_positive = 1, so result = x * 1 = x
        # If x <= 0: is_positive = 0, so result = x * 0 = 0
        relu_share = self.multiplier.multiply(
            x_share, is_positive, node_id,
            context=context if context else "relu"
        )
        
        return relu_share
    
    def relu_list(self, x_shares: List[Share], node_id: int, context: str = None) -> List[Share]:
        """
        Apply ReLU to a list of shares
        
        Args:
            x_shares: List of input shares
            node_id: Node ID
            context: Optional context for debugging
        
        Returns:
            List of ReLU output shares
        """
        relu_outputs = []
        for i, x_share in enumerate(x_shares):
            relu_ctx = f"{context}_{i}" if context else None
            relu_out = self.relu(x_share, node_id, context=relu_ctx)
            relu_outputs.append(relu_out)
        
        return relu_outputs
    
    def relu_2d(self, x_shares_2d: List[List[Share]], node_id: int, context: str = None) -> List[List[Share]]:
        """
        Apply ReLU to a 2D array of shares (e.g., feature map)
        
        Args:
            x_shares_2d: 2D list of input shares [H x W]
            node_id: Node ID
            context: Optional context for debugging
        
        Returns:
            2D list of ReLU output shares [H x W]
        """
        relu_outputs = []
        for h, row in enumerate(x_shares_2d):
            relu_row = []
            for w, x_share in enumerate(row):
                relu_ctx = f"{context}_{h}_{w}" if context else None
                relu_out = self.relu(x_share, node_id, context=relu_ctx)
                relu_row.append(relu_out)
            relu_outputs.append(relu_row)
        
        return relu_outputs
    
    def relu_3d(self, x_shares_3d: List[List[List[Share]]], node_id: int, context: str = None) -> List[List[List[Share]]]:
        """
        Apply ReLU to a 3D array of shares (e.g., multi-channel feature map)
        
        Args:
            x_shares_3d: 3D list of input shares [H x W x C]
            node_id: Node ID
            context: Optional context for debugging
        
        Returns:
            3D list of ReLU output shares [H x W x C]
        """
        relu_outputs = []
        for h, row in enumerate(x_shares_3d):
            relu_row = []
            for w, col in enumerate(row):
                relu_col = []
                for c, x_share in enumerate(col):
                    relu_ctx = f"{context}_{h}_{w}_{c}" if context else None
                    relu_out = self.relu(x_share, node_id, context=relu_ctx)
                    relu_col.append(relu_out)
                relu_row.append(relu_col)
            relu_outputs.append(relu_row)
        
        return relu_outputs
    
    def relu_backward(self, output_grad_share: Share, input_share: Share,
                     node_id: int, context: str = None) -> Share:
        """
        Compute gradient of ReLU w.r.t. input
        
        dReLU/dx = 1 if x > 0, else 0
        So: grad_input = output_grad * (input > 0)
        
        Args:
            output_grad_share: Gradient w.r.t. ReLU output
            input_share: Original input to ReLU (before activation)
            node_id: Node ID
            context: Optional context for debugging
        
        Returns:
            Share of gradient w.r.t. input
        """
        # Create zero share
        zero_share = Share(x=input_share.x, y=0, node_id=node_id)
        
        # Check if input > 0
        is_positive = self.comparator.secure_greater_than(input_share, zero_share, node_id)
        
        # grad_input = output_grad * (input > 0)
        input_grad = self.multiplier.multiply(
            output_grad_share, is_positive, node_id,
            context=context if context else "relu_backward"
        )
        
        return input_grad
    
    def relu_backward_list(self, output_grad_shares: List[Share], input_shares: List[Share],
                          node_id: int, context: str = None) -> List[Share]:
        """
        Compute ReLU backward for a list of shares
        
        Args:
            output_grad_shares: List of gradients w.r.t. ReLU outputs
            input_shares: List of original inputs to ReLU
            node_id: Node ID
            context: Optional context for debugging
        
        Returns:
            List of gradients w.r.t. inputs
        """
        input_grads = []
        for i, (out_grad, inp) in enumerate(zip(output_grad_shares, input_shares)):
            grad_ctx = f"{context}_{i}" if context else None
            input_grad = self.relu_backward(out_grad, inp, node_id, context=grad_ctx)
            input_grads.append(input_grad)
        
        return input_grads
