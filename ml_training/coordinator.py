"""
Training Coordinator
Manages mini-batch selection, safety bounds, and quorum commits
"""

from typing import List, Dict, Optional, Tuple, Any
from ml_training.kvs import KVSCluster
from ml_training.secret_sharing import Share, PackedShamirSecretSharing
from ml_training.mpc_engine import PackedMPCEngine
from ml_training.beaver_triples import SecureMultiplier
from ml_training.secure_comm import SecureMPCNetwork, create_mpc_network
from ml_training.reconstruction import create_reconstruction_manager
import random
import numpy as np


class SafetyBoundChecker:
    """
    Checks safety bound: 2*(t+s-1) < n_active
    
    When violated:
    - Abort current mini-batch (don't advance model state)
    - Suspend training until safety is restored
    - Resume when condition holds again
    """
    
    def __init__(self, t: int, s: int):
        self.t = t
        self.s = s
    
    def check(self, n_active: int) -> bool:
        """
        Check if safety bound is satisfied
        Safety condition: 2*(t+s-1) < n_active
        """
        return 2 * (self.t + self.s - 1) < n_active
    
    def get_max_packing_factor(self, n_active: int) -> int:
        """
        Get maximum safe packing factor s such that 2*(t+s-1) < n_active
        Returns 1 if safety bound is violated (no packing)
        """
        if not self.check(n_active):
            return 1  # No packing if bound violated
        
        # Find maximum s such that 2*(t+s-1) < n_active
        # 2*(t+s-1) < n_active
        # 2t + 2s - 2 < n_active
        # 2s < n_active - 2t + 2
        # s < (n_active - 2t + 2) / 2
        max_s = (n_active - 2 * self.t + 2) // 2
        
        # Ensure at least 1, and cap at reasonable maximum (e.g., 4)
        return max(1, min(4, max_s))


class TrainingCoordinator:
    """
    Coordinates secure training across nodes
    """
    
    def __init__(self, kvs_cluster: KVSCluster, n_nodes: int, t: int, s: int,
                 batch_size: int = 32, learning_rate: float = 0.01,
                 node_id: int = 1, node_configs: Optional[Dict[int, Dict[str, Any]]] = None,
                 enable_network: bool = False):
        """
        Initialize training coordinator
        Args:
            kvs_cluster: KVS cluster for storage
            n_nodes: Total number of nodes
            t: Privacy threshold
            s: Adversarial share limit
            batch_size: Mini-batch size
            learning_rate: Learning rate
            node_id: ID of this node
            node_configs: Dictionary mapping node_id to {host, port} for network
            enable_network: Whether to enable multi-node network communication
        """
        self.kvs_cluster = kvs_cluster
        self.n_nodes = n_nodes
        self.t = t
        self.s = s
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.node_id = node_id
        self.enable_network = enable_network
        
        self.safety_checker = SafetyBoundChecker(t, s)
        
        # Initialize PSS (needed for node manager)
        self.pss = PackedShamirSecretSharing()
        
        # Initialize network if enabled
        self.network = None
        self.reconstruction_manager = None
        self.failure_detector = None
        self.node_manager = None
        if enable_network and node_configs:
            self.network = create_mpc_network(node_id, node_configs, port=8000 + node_id)
            self.reconstruction_manager = create_reconstruction_manager(self.network, t)
            
            # Initialize node failure detection and management
            from ml_training.node_failure_detector import NodeFailureDetector
            from ml_training.node_manager import NodeManager
            self.failure_detector = NodeFailureDetector(self.network, heartbeat_interval=2.0, failure_timeout=6.0)
            self.failure_detector.start_monitoring()
            self.node_manager = NodeManager(
                self.network, 
                self.failure_detector, 
                self.safety_checker,
                self.pss,
                t,
                resharing_enabled=True
            )
        
        # Initialize MPC engine
        self.mpc_engine = PackedMPCEngine(n_nodes=n_nodes, t=t)
        
        # Update multiplier to use reconstruction manager if available
        if self.reconstruction_manager:
            from ml_training.beaver_triples import BeaverTripleGenerator, BeaverTriplePool
            triple_gen = BeaverTripleGenerator()
            triple_pool = BeaverTriplePool(triple_gen, initial_size=1000)
            self.mpc_engine.multiplier = SecureMultiplier(
                triple_pool, n_nodes, t,
                reconstruction_manager=self.reconstruction_manager
            )
        
        # Track versions
        self.v_D = 0  # Dataset version
        self.v_theta = 0  # Model version
    
    def ingest_dataset(self, dataset: List[np.ndarray], labels: List[np.ndarray]) -> int:
        """
        Ingest and secret-share dataset
        Args:
            dataset: Training samples
            labels: Training labels
        Returns:
            Dataset version
        """
        self.v_D += 1
        
        # Secret-share each sample
        for i, (sample, label) in enumerate(zip(dataset, labels)):
            # Share each feature of the sample
            sample_feature_shares = []
            for feature_value in sample:
                feature_int = int(feature_value * 1000000) % self.mpc_engine.field_size
                feature_shares = self.pss.shamir.share(feature_int, self.n_nodes, self.t)
                sample_feature_shares.append(feature_shares)
            
            # Share label
            label_int = int(np.sum(label) * 1000000) % self.mpc_engine.field_size
            label_shares = self.pss.shamir.share(label_int, self.n_nodes, self.t)
            
            # Store shares in KVS (one share per node per feature)
            for node_id in range(1, self.n_nodes + 1):
                # Store all feature shares for this node
                node_sample_shares = []
                for feature_shares in sample_feature_shares:
                    node_share = next(s for s in feature_shares if s.node_id == node_id)
                    node_sample_shares.append(node_share)
                
                node_label_share = next(s for s in label_shares if s.node_id == node_id)
                
                sample_key = f"sample_{i}_node_{node_id}"
                label_key = f"label_{i}_node_{node_id}"
                
                self.kvs_cluster.write_with_quorum(
                    sample_key, node_sample_shares, self.v_D
                )
                self.kvs_cluster.write_with_quorum(
                    label_key, node_label_share, self.v_D
                )
        
        return self.v_D
    
    def initialize_weights(self, weight_shapes: List[Tuple[int, int]]) -> int:
        """
        Initialize and secret-share model weights
        Args:
            weight_shapes: List of (input_dim, output_dim) for each layer
        Returns:
            Model version
        """
        self.v_theta += 1
        
        # Initialize weights (simplified - would use proper initialization)
        weights = []
        for input_dim, output_dim in weight_shapes:
            layer_weights = []
            for _ in range(output_dim):
                row = []
                for _ in range(input_dim):
                    # Initialize weight as share
                    weight_int = random.randint(0, self.mpc_engine.field_size - 1)
                    weight_shares = self.pss.shamir.share(weight_int, self.n_nodes, self.t)
                    layer_weights.append(weight_shares)
                weights.append(layer_weights)
        
        # Store weights in KVS (simplified - would store properly)
        weight_key = f"weights_v{self.v_theta}"
        self.kvs_cluster.write_with_quorum(weight_key, weights, self.v_theta)
        
        return self.v_theta
    
    def select_mini_batch(self, dataset_size: int) -> List[int]:
        """Select random mini-batch indices"""
        return random.sample(range(dataset_size), min(self.batch_size, dataset_size))
    
    def train_mini_batch(self, sample_shares: List[List[Share]], 
                        label_shares: List[List[Share]],
                        weight_shares: List[List[List[Share]]],
                        packing_factor: int, node_id: int) -> Tuple[List[List[List[Share]]], bool]:
        """
        Train on a mini-batch
        
        Safety bound check: 2*(t+s-1) < n_active (where s is the packing factor)
        The packing_factor parameter must satisfy this bound for every mini-batch.
        If violated:
        - Abort current mini-batch (don't advance model state)
        - Return original weights unchanged
        - Training will suspend until safety is restored
        
        Args:
            sample_shares: Sample shares for batch
            label_shares: Label shares for batch
            weight_shares: Current weight shares
            packing_factor: Packing factor for PSS (must satisfy safety bound)
            node_id: Node ID
        Returns:
            Tuple of (updated_weights, success)
            - If success=False: weights are unchanged (batch aborted)
            - If success=True: weights are updated (batch committed)
        """
        # Check safety bound with ACTUAL packing factor BEFORE any computation
        # Safety condition: 2*(t + packing_factor - 1) < n_active
        n_active = self.n_nodes  # Simplified - would track active nodes
        safety_bound_value = 2 * (self.t + packing_factor - 1)
        if safety_bound_value >= n_active:
            # Safety bound violated with this packing factor: ABORT current mini-batch
            # Return original weights unchanged (no model state advancement)
            # Training will suspend until safety is restored
            return weight_shares, False
        
        # Pack shares (simplified - use first sample's shares)
        # In production, would properly pack all samples
        if not sample_shares:
            return weight_shares, False
        
        # Use first sample's shares as input (simplified)
        input_shares = sample_shares[0] if sample_shares else []
        
        # Ensure input size matches first layer
        if weight_shares and weight_shares[0]:
            expected_input_size = len(weight_shares[0][0])
            if len(input_shares) > expected_input_size:
                input_shares = input_shares[:expected_input_size]
            elif len(input_shares) < expected_input_size:
                # Pad with zero shares
                zero_share = Share(x=input_shares[0].x if input_shares else 1, y=0, node_id=node_id)
                input_shares = input_shares + [zero_share] * (expected_input_size - len(input_shares))
        
        # Forward pass
        predictions = self.mpc_engine.forward_pass(
            input_shares,
            weight_shares, node_id
        )
        
        # Compute loss
        packed_labels = []
        for label_share_list in label_shares:
            packed = self.pss.pack_share(label_share_list, packing_factor)
            packed_labels.extend(packed)
        
        loss_share = self.mpc_engine.compute_loss(
            predictions[:len(packed_labels)], 
            packed_labels[:len(predictions)],
            node_id
        )
        
        # Backward pass
        # Use input_shares for proper gradient computation
        gradients = self.mpc_engine.backward_pass(
            loss_share, predictions, packed_labels, weight_shares, node_id,
            input_shares=input_shares  # Pass input for gradient computation
        )
        
        # Update weights using standard (non-DP) SGD
        updated_weights = self.mpc_engine.update_weights(
            weight_shares, gradients, self.learning_rate, node_id
        )
        
        return updated_weights, True
    
    def commit_model_update(self, updated_weights: List[List[List[Share]]]) -> bool:
        """
        Commit model update with quorum
        Args:
            updated_weights: Updated weight shares
        Returns:
            True if committed successfully
        """
        self.v_theta += 1
        weight_key = f"weights_v{self.v_theta}"
        
        success = self.kvs_cluster.write_with_quorum(
            weight_key, updated_weights, self.v_theta
        )
        
        return success

