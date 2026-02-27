"""
DPSS (Distributed Proactive Secret Sharing) Resharing Protocol
Redistributes shares when nodes join or leave the network
"""

from typing import List, Dict, Set
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_comm import SecureMPCNetwork
import random


class DPSSResharer:
    """
    Implements DPSS resharing protocol for share redistribution
    """
    
    def __init__(self, network: SecureMPCNetwork, 
                 shamir: ShamirSecretSharing,
                 t: int,
                 old_node_ids: List[int],
                 new_node_ids: List[int]):
        """
        Initialize DPSS resharer
        Args:
            network: SecureMPCNetwork instance
            shamir: ShamirSecretSharing instance
            t: Privacy threshold
            old_node_ids: List of old node IDs
            new_node_ids: List of new node IDs (including remaining old nodes)
        """
        self.network = network
        self.shamir = shamir
        self.t = t
        self.old_node_ids = old_node_ids
        self.new_node_ids = new_node_ids
        self.field_size = shamir.field_size
    
    def reshare_shares(self, old_shares: List[Share]) -> List[Share]:
        """
        Reshare a list of shares from old nodes to new nodes
        
        This is a simplified DPSS implementation. In full DPSS:
        1. Each old node would evaluate its polynomial at new node positions
        2. New shares would be computed without full reconstruction
        3. The protocol would be proactive (periodic resharing)
        
        For now, we use a simplified approach:
        - Collect enough shares to reconstruct
        - Generate new shares for new node set
        
        Args:
            old_shares: List of shares from old nodes
        Returns:
            List of new shares for new node set
        """
        if len(old_shares) < self.t + 1:
            raise ValueError(f"Need at least {self.t + 1} shares, got {len(old_shares)}")
        
        # Filter shares to only those from active old nodes
        active_old_shares = [s for s in old_shares if s.node_id in self.old_node_ids]
        
        if len(active_old_shares) < self.t + 1:
            raise ValueError(f"Need at least {self.t + 1} shares from active old nodes")
        
        # Use first t+1 shares for reconstruction
        shares_for_reconstruction = active_old_shares[:self.t + 1]
        
        # Reconstruct the secret
        secret = self.shamir.reconstruct(shares_for_reconstruction)
        
        # Generate new shares for new node set
        n_new = len(self.new_node_ids)
        new_shares = self.shamir.share(secret, n_new, self.t)
        
        # Update node IDs to match new node IDs
        reshared_shares = []
        for i, new_share in enumerate(new_shares):
            reshared_shares.append(Share(
                x=new_share.x,
                y=new_share.y,
                node_id=self.new_node_ids[i]
            ))
        
        return reshared_shares
    
    def reshare_weight_shares(self, weight_shares_list: List[List[List[Share]]]) -> List[List[List[Share]]]:
        """
        Reshare all weight shares from old nodes to new nodes
        Args:
            weight_shares_list: Weight shares organized as List[layer][row][col][shares]
        Returns:
            Reshared weights for new node set
        """
        reshared_weights = []
        
        for layer_idx, layer_shares in enumerate(weight_shares_list):
            reshared_layer = []
            
            for row_idx, row_shares in enumerate(layer_shares):
                reshared_row = []
                
                for col_idx, shares_for_weight in enumerate(row_shares):
                    try:
                        # Reshare this weight's shares
                        new_shares = self.reshare_shares(shares_for_weight)
                        reshared_row.append(new_shares)
                    except Exception as e:
                        print(f"Warning: Failed to reshare weight at layer={layer_idx}, row={row_idx}, col={col_idx}: {e}")
                        # Use original shares if resharing fails
                        reshared_row.append(shares_for_weight)
                
                reshared_layer.append(reshared_row)
            
            reshared_weights.append(reshared_layer)
        
        return reshared_weights
    
    def can_reshare(self, available_shares: int) -> bool:
        """
        Check if resharing is possible with available shares
        Args:
            available_shares: Number of shares available
        Returns:
            True if resharing is possible (have at least t+1 shares)
        """
        return available_shares >= self.t + 1

