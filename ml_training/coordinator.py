"""
Training Coordinator
Manages mini-batch selection, safety bounds, and quorum commits
"""

from typing import List, Dict, Optional, Tuple, Any
from ml_training.kvs import KVSCluster
from ml_training.secret_sharing import Share, PackedShamirSecretSharing
from ml_training.mpc_engine import PackedMPCEngine
from ml_training.dp_sgd_integration import DPSGDConfig, DPSGDMPCEngine, VerifiableDPProofs
from ml_training.beaver_triples import SecureMultiplier
from ml_training.secure_comm import SecureMPCNetwork, create_mpc_network
from ml_training.reconstruction import create_reconstruction_manager
import random
import numpy as np


class SafetyBoundChecker:
    """Checks safety bound: 2*(t+s) < n_active"""
    
    def __init__(self, t: int, s: int):
        self.t = t
        self.s = s
    
    def check(self, n_active: int) -> bool:
        """Check if safety bound is satisfied"""
        return 2 * (self.t + self.s) < n_active
    
    def get_max_packing_factor(self, n_active: int) -> int:
        """Get maximum safe packing factor"""
        if not self.check(n_active):
            return 1  # No packing if bound violated
        # Simplified: return safe packing factor
        return min(4, n_active - 2 * (self.t + self.s))


class TrainingCoordinator:
    """
    Coordinates secure training across nodes
    """
    
    def __init__(self, kvs_cluster: KVSCluster, n_nodes: int, t: int, s: int,
                 batch_size: int = 32, learning_rate: float = 0.01,
                 use_dp_sgd: bool = False, dp_config: Optional[DPSGDConfig] = None,
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
            use_dp_sgd: Whether to use DP-SGD
            dp_config: DP-SGD configuration (required if use_dp_sgd=True)
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
        self.use_dp_sgd = use_dp_sgd
        self.node_id = node_id
        self.enable_network = enable_network
        
        self.safety_checker = SafetyBoundChecker(t, s)
        
        # Initialize network if enabled
        self.network = None
        self.reconstruction_manager = None
        if enable_network and node_configs:
            #Take own port from node config, if available
            iPort:int=8000 + node_id
            if(node_id in node_configs):
              iPort=node_configs[node_id]['port'] 
            self.network = create_mpc_network(node_id, node_configs, port=iPort)
            self.reconstruction_manager = create_reconstruction_manager(self.network, t)
        
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
        
        self.pss = PackedShamirSecretSharing()
        
        # Initialize DP-SGD if enabled
        self.dp_mpc_engine = None
        self.dp_proofs = []
        if use_dp_sgd:
            if dp_config is None:
                dp_config = DPSGDConfig(
                    clip_norm=1.0,
                    noise_multiplier=1.0,
                    learning_rate=learning_rate
                )
            # Get multiplier from MPC engine
            multiplier = self.mpc_engine.multiplier
            self.dp_mpc_engine = DPSGDMPCEngine(dp_config, multiplier)
            self.dp_config = dp_config
            self.proof_generator = VerifiableDPProofs()
        else:
            self.dp_config = None
            self.proof_generator = None
        
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
        Args:
            sample_shares: Sample shares for batch
            label_shares: Label shares for batch
            weight_shares: Current weight shares
            packing_factor: Packing factor for PSS
            node_id: Node ID
        Returns:
            Tuple of (updated_weights, success)
        """
        # Check safety bound
        n_active = self.n_nodes  # Simplified - would track active nodes
        if not self.safety_checker.check(n_active):
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
        
        # Apply DP-SGD if enabled
        if self.use_dp_sgd and self.dp_mpc_engine:
            # Convert gradients to per-sample format
            # gradients is List[List[List[Share]]] (layers -> rows -> columns)
            # For DP-SGD, we need per-sample gradients
            # Since we're processing a batch, we'll treat the entire batch as one sample for now
            # (In production, would compute per-sample gradients separately)
            
            # Flatten all gradients into a single list
            flat_grads = []
            for grad_layer in gradients:
                for grad_row in grad_layer:
                    flat_grads.extend(grad_row)  # Extend with all shares in the row
            
            # For per-sample DP-SGD, we'd have List[List[Share]] where each inner list is one sample
            # For now, treat the batch as one sample
            per_sample_grads = [flat_grads]  # List containing one sample's flattened gradients
            
            # Perform DP-SGD step
            noisy_grads, tracking_info = self.dp_mpc_engine.dp_sgd_step_on_shares(
                per_sample_grads, self.batch_size, node_id
            )
            
            # Generate proof
            if self.proof_generator:
                proof = self.proof_generator.prove_dp_sgd_step(tracking_info)
                self.dp_proofs.append(proof)
            
            # Convert noisy_grads back to gradient format
            # noisy_grads is List[Share] (flattened), need to reshape to List[List[List[Share]]]
            # For now, we'll reshape based on the original gradient structure
            if noisy_grads:
                # Reshape flattened noisy_grads back to original structure
                grad_idx = 0
                reshaped_gradients = []
                for layer_idx, grad_layer in enumerate(gradients):
                    reshaped_layer = []
                    for grad_row in grad_layer:
                        reshaped_row = []
                        for _ in grad_row:
                            if grad_idx < len(noisy_grads):
                                reshaped_row.append(noisy_grads[grad_idx])
                                grad_idx += 1
                            else:
                                # Fallback if mismatch
                                reshaped_row.append(grad_row[0])
                        reshaped_layer.append(reshaped_row)
                    reshaped_gradients.append(reshaped_layer)
                gradients = reshaped_gradients
        
        # Update weights
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

