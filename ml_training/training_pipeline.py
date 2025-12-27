"""
End-to-End Training Pipeline
Main interface for secure ML training
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from ml_training.coordinator import TrainingCoordinator
from ml_training.kvs import KVSCluster
from ml_training.dp_sgd_integration import DPSGDConfig


class SentraTrainingPipeline:
    """
    Main training pipeline for SENTRA
    """
    
    def __init__(self, n_nodes: int, t: int, s: int, 
                 batch_size: int = 32, learning_rate: float = 0.01,
                 num_epochs: int = 10, use_dp_sgd: bool = False,
                 dp_config: Optional[DPSGDConfig] = None,
                 node_id: int = 1, node_configs: Optional[Dict[int, Dict[str, Any]]] = None,
                 enable_network: bool = False):
        """
        Initialize training pipeline
        Args:
            n_nodes: Total number of nodes
            t: Privacy threshold (need t+1 shares to reconstruct)
            s: Adversarial share limit
            batch_size: Mini-batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            use_dp_sgd: Whether to use DP-SGD
            dp_config: DP-SGD configuration (required if use_dp_sgd=True)
            node_id: ID of this node
            node_configs: Dictionary mapping node_id to {host, port} for network
            enable_network: Whether to enable multi-node network communication
        """
        self.n_nodes = n_nodes
        self.t = t
        self.s = s
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.use_dp_sgd = use_dp_sgd
        self.node_id = node_id
        self.enable_network = enable_network
        
        # Initialize KVS cluster
        node_ids = list(range(1, n_nodes + 1))
        self.kvs_cluster = KVSCluster(node_ids)
        
        # Initialize coordinator with network support
        self.coordinator = TrainingCoordinator(
            self.kvs_cluster, n_nodes, t, s, batch_size, learning_rate,
            use_dp_sgd=use_dp_sgd, dp_config=dp_config,
            node_id=node_id, node_configs=node_configs,
            enable_network=enable_network
        )
    
    def train(self, dataset: List[np.ndarray], labels: List[np.ndarray],
              weight_shapes: List[Tuple[int, int]]):
        """
        Main training loop
        Args:
            dataset: List of training samples
            labels: List of labels
            weight_shapes: List of (input_dim, output_dim) for each layer
        """
        print("=" * 60)
        print("SENTRA ML Training Pipeline")
        print("=" * 60)
        print(f"Nodes: {self.n_nodes}, Threshold: {self.t}, Adversarial limit: {self.s}")
        print(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Node ID: {self.node_id}")
        if self.enable_network:
            print(f"Multi-Node: Enabled (Network communication active)")
        else:
            print(f"Multi-Node: Disabled (Single-node mode)")
        if self.use_dp_sgd:
            print(f"DP-SGD: Enabled (clip_norm={self.coordinator.dp_config.clip_norm}, "
                  f"noise_multiplier={self.coordinator.dp_config.noise_multiplier})")
        print("=" * 60)
        
        # Step 1: Ingest and secret-share dataset
        print("\n[Step 1] Ingesting and secret-sharing dataset...")
        v_D = self.coordinator.ingest_dataset(dataset, labels)
        print(f"Ingested {len(dataset)} samples with version v_D={v_D}")
        
        # Step 2: Initialize and secret-share weights
        print("\n[Step 2] Initializing and secret-sharing weights...")
        v_theta = self.coordinator.initialize_weights(weight_shapes)
        print(f"Initialized weights with version v_theta={v_theta}")
        
        # Step 3: Training loop
        print("\n[Step 3] Starting training loop...")
        
        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")
            
            # Select mini-batches
            num_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(num_batches):
                # Select mini-batch
                batch_indices = self.coordinator.select_mini_batch(len(dataset))
                
                # Retrieve shares from KVS (simplified - would retrieve properly)
                sample_shares = []
                label_shares = []
                
                for idx in batch_indices:
                    # Get shares for this sample
                    sample_key = f"sample_{idx}_node_1"
                    label_key = f"label_{idx}_node_1"
                    
                    sample_value = self.kvs_cluster.read_with_min_version(sample_key, v_D)
                    label_value = self.kvs_cluster.read_with_min_version(label_key, v_D)
                    
                    if sample_value and label_value:
                        # sample_value.data is a list of shares (one per feature)
                        # label_value.data is a single share
                        sample_feature_shares = sample_value.data
                        label_share = label_value.data
                        sample_shares.append(sample_feature_shares)
                        label_shares.append([label_share])
                
                if not sample_shares:
                    print(f"  Batch {batch_idx + 1}/{num_batches}: [FAIL] Failed")
                    continue
                
                # Get current weights
                weight_key = f"weights_v{v_theta}"
                weight_value = self.kvs_cluster.read_with_min_version(weight_key, v_theta)
                if not weight_value:
                    print(f"  Batch {batch_idx + 1}/{num_batches}: [FAIL] Failed")
                    continue
                
                weight_shares = weight_value.data
                
                # Train mini-batch
                packing_factor = self.coordinator.safety_checker.get_max_packing_factor(self.n_nodes)
                updated_weights, success = self.coordinator.train_mini_batch(
                    sample_shares, label_shares, weight_shares, packing_factor, node_id=1
                )
                
                if success:
                    # Commit update
                    committed = self.coordinator.commit_model_update(updated_weights)
                    if committed:
                        v_theta = self.coordinator.v_theta
                        print(f"Committed mini-batch {batch_indices}, v_theta={v_theta}")
                        print(f"  Batch {batch_idx + 1}/{num_batches}: [OK] Committed")
                    else:
                        print(f"  Batch {batch_idx + 1}/{num_batches}: [FAIL] Failed")
                else:
                    print(f"  Batch {batch_idx + 1}/{num_batches}: [FAIL] Failed")
            
            print(f"Epoch {epoch + 1} completed, model version: v_theta={self.coordinator.v_theta}")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)

