"""
Secure Share Reconstruction Protocol
Implements threshold cryptography for reconstructing secrets from shares
"""

from typing import List, Optional, Dict
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_comm import SecureMPCNetwork, SecureChannel
import time


class SecureReconstruction:
    """
    Secure reconstruction protocol for MPC
    Coordinates share collection and reconstruction across nodes
    """
    
    def __init__(self, network: SecureMPCNetwork, t: int):
        """
        Initialize secure reconstruction
        Args:
            network: SecureMPCNetwork instance
            t: Privacy threshold (need t+1 shares to reconstruct)
        """
        self.network = network
        self.t = t
        self.sss = ShamirSecretSharing()
    
    def reconstruct_value(self, local_shares: List[Share], context: str,
                         required_nodes: Optional[List[int]] = None,
                         timeout: float = 5.0) -> int:
        """
        Reconstruct a secret value from shares across nodes
        
        Args:
            local_shares: Shares held by this node
            context: Unique context identifier
            required_nodes: Nodes to request shares from
            timeout: Timeout for reconstruction
        Returns:
            Reconstructed secret value
        """
        # Check if we have enough local shares
        if len(local_shares) >= self.t + 1:
            # Can reconstruct locally
            return self.sss.reconstruct(local_shares)
        
        # Need shares from other nodes
        if required_nodes is None:
            required_nodes = list(self.network.node_configs.keys())
        
        # Broadcast our shares
        for i, share in enumerate(local_shares):
            share_context = f"{context}_share_{i}"
            self.network.broadcast_share(share, share_context)
            # Debug logging
            if hasattr(self, '_debug_log') and self._debug_log:
                print(f"  [Share Exchange] Node {self.network.node_id} broadcasting share for context '{share_context}'")
        
        # Request shares from other nodes
        from ml_training.secure_comm import MessageType
        for node_id in required_nodes:
            if node_id != self.network.node_id:
                try:
                    self.network.channel.send_message(
                        node_id,
                        MessageType.RECONSTRUCTION_REQUEST,
                        {'context': context}
                    )
                except Exception as e:
                    print(f"Warning: Could not request share from node {node_id}: {e}")
        
        # Wait for shares to arrive
        start_time = time.time()
        all_shares = list(local_shares)
        
        while time.time() - start_time < timeout:
            # Collect shares from network
            received = self.network.get_received_shares(context)
            
            # Add unique shares
            existing_nodes = {s.node_id for s in all_shares}
            for share in received:
                if share.node_id not in existing_nodes:
                    all_shares.append(share)
                    existing_nodes.add(share.node_id)
            
            # Check if we have enough shares
            if len(all_shares) >= self.t + 1:
                if hasattr(self, '_debug_log') and self._debug_log:
                    print(f"  [Reconstruction] Collected {len(all_shares)} shares for context '{context}'")
                break
            
            time.sleep(0.1)
        
        # Reconstruct secret
        if len(all_shares) < self.t + 1:
            raise RuntimeError(
                f"Insufficient shares for reconstruction: got {len(all_shares)}, need {self.t + 1}"
            )
        
        # Use Lagrange interpolation to reconstruct
        secret = self.sss.reconstruct(all_shares[:self.t + 1])
        return secret
    
    def reconstruct_for_multiplication(self, d_shares: List[Share], e_shares: List[Share],
                                      context: str, required_nodes: Optional[List[int]] = None) -> tuple:
        """
        Reconstruct d and e values for Beaver triple multiplication
        
        Args:
            d_shares: Shares of d = x - a
            e_shares: Shares of e = y - b
            context: Context identifier
            required_nodes: Nodes to request shares from
        Returns:
            Tuple of (d_reconstructed, e_reconstructed)
        """
        # Reconstruct d
        d_context = f"{context}_d"
        d_recon = self.reconstruct_value(d_shares, d_context, required_nodes)
        
        # Reconstruct e
        e_context = f"{context}_e"
        e_recon = self.reconstruct_value(e_shares, e_context, required_nodes)
        
        return d_recon, e_recon
    
    def batch_reconstruct(self, share_groups: List[List[Share]], context_prefix: str,
                         required_nodes: Optional[List[int]] = None) -> List[int]:
        """
        Reconstruct multiple secrets efficiently
        
        Args:
            share_groups: List of share lists (one per secret)
            context_prefix: Prefix for context identifiers
            required_nodes: Nodes to request shares from
        Returns:
            List of reconstructed secrets
        """
        reconstructed = []
        for i, shares in enumerate(share_groups):
            context = f"{context_prefix}_{i}"
            secret = self.reconstruct_value(shares, context, required_nodes)
            reconstructed.append(secret)
        return reconstructed


class MPCReconstructionManager:
    """
    Manages reconstruction operations for MPC
    Coordinates with network and handles caching
    """
    
    def __init__(self, network: SecureMPCNetwork, t: int):
        """
        Initialize reconstruction manager
        Args:
            network: SecureMPCNetwork instance
            t: Privacy threshold
        """
        self.network = network
        self.t = t
        self.reconstructor = SecureReconstruction(network, t)
        self.reconstruction_cache: Dict[str, int] = {}
    
    def get_reconstructed_value(self, shares: List[Share], context: str,
                               use_cache: bool = True) -> int:
        """
        Get reconstructed value, using cache if available
        
        Args:
            shares: Shares to reconstruct
            context: Context identifier
            use_cache: Whether to use cached values
        Returns:
            Reconstructed value
        """
        if use_cache and context in self.reconstruction_cache:
            return self.reconstruction_cache[context]
        
        value = self.reconstructor.reconstruct_value(shares, context)
        
        if use_cache:
            self.reconstruction_cache[context] = value
        
        return value
    
    def clear_cache(self):
        """Clear reconstruction cache"""
        self.reconstruction_cache.clear()


def create_reconstruction_manager(network: SecureMPCNetwork, t: int) -> MPCReconstructionManager:
    """
    Factory function to create reconstruction manager
    
    Args:
        network: SecureMPCNetwork instance
        t: Privacy threshold
    Returns:
        Configured MPCReconstructionManager instance
    """
    return MPCReconstructionManager(network, t)

