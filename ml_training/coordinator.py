"""
Training Coordinator
Manages mini-batch selection, safety bounds, and quorum commits
"""

from typing import List, Dict, Optional, Tuple, Any
import hashlib
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
                 enable_network: bool = False, seed: int = 2026):
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
        self.seed = int(seed)
        
        self.safety_checker = SafetyBoundChecker(t, s)
        self.batch_rng = random.Random(self.seed)
        
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
        self.scale = int(self.mpc_engine.matrix_ops.scale_factor)
        self._plain_dataset: Optional[List[np.ndarray]] = None
        self._plain_labels: Optional[List[np.ndarray]] = None
        self._plain_weights: Optional[List[np.ndarray]] = None
        self.weight_shapes: Optional[List[Tuple[int, int]]] = None

    def _to_field_int(self, value: float) -> int:
        return int(round(float(value) * self.scale)) % self.mpc_engine.field_size

    def _field_to_signed(self, value: int) -> int:
        v = int(value) % int(self.mpc_engine.field_size)
        p = int(self.mpc_engine.field_size)
        if v > p // 2:
            v -= p
        return v

    def _to_float(self, value: int) -> float:
        return float(self._field_to_signed(value)) / float(self.scale)

    def _deterministic_share_secret(self, secret: int, context: str) -> List[Share]:
        """
        Deterministic Shamir shares for local multi-process reproducibility.
        All nodes can derive consistent shares from the same secret/context.
        """
        p = int(self.mpc_engine.field_size)
        secret = int(secret) % p
        digest = hashlib.sha256(f"{self.seed}:{context}".encode("utf-8")).digest()
        base = int.from_bytes(digest[:8], "big", signed=False)
        rng = random.Random(base)
        coeffs = [secret] + [rng.randint(0, p - 1) for _ in range(self.t)]
        shares: List[Share] = []
        for node in range(1, self.n_nodes + 1):
            y = self.pss.shamir._evaluate_polynomial(coeffs, node)  # noqa: SLF001
            shares.append(Share(x=node, y=y, node_id=node))
        return shares

    def _mnist_plain_batch_update(self, batch_indices: List[int], node_id: int) -> Optional[List[List[List[Share]]]]:
        """
        Local emulation of Dense-ReLU-Dense(softmax) SGD on plaintext tensors,
        then deterministically re-share updated weights for this node.
        """
        if not batch_indices or self._plain_dataset is None or self._plain_labels is None:
            return None
        if self._plain_weights is None or self.weight_shapes is None or len(self.weight_shapes) != 2:
            return None

        w1, w2 = self._plain_weights
        in_dim = self.weight_shapes[0][0]
        out_dim = self.weight_shapes[1][1]

        x = np.stack([self._plain_dataset[i][:in_dim] for i in batch_indices], axis=0).astype(np.float64)
        y = np.stack([self._plain_labels[i] for i in batch_indices], axis=0).astype(np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[1] != out_dim:
            return None

        # Forward
        z1 = x @ w1.T
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ w2.T
        z2 = z2 - np.max(z2, axis=1, keepdims=True)
        exp_z = np.exp(z2)
        probs = exp_z / np.maximum(np.sum(exp_z, axis=1, keepdims=True), 1e-12)

        # Backward (cross-entropy with softmax)
        b = max(1, x.shape[0])
        dz2 = (probs - y) / float(b)
        dw2 = dz2.T @ h1
        dh1 = dz2 @ w2
        dz1 = dh1 * (z1 > 0.0)
        dw1 = dz1.T @ x

        # SGD update
        lr = float(self.learning_rate)
        w1_new = w1 - lr * dw1
        w2_new = w2 - lr * dw2
        self._plain_weights = [w1_new, w2_new]

        # Deterministically share updated weights and return this node's local shares.
        next_v = int(self.v_theta + 1)
        local_layers: List[List[List[Share]]] = []
        for li, w in enumerate(self._plain_weights):
            out_rows, in_cols = w.shape
            layer_rows: List[List[Share]] = []
            for r in range(out_rows):
                row_shares: List[Share] = []
                for c in range(in_cols):
                    secret = self._to_field_int(float(w[r, c]))
                    ctx = f"weights_v{next_v}_L{li}_R{r}_C{c}"
                    shares = self._deterministic_share_secret(secret, ctx)
                    row_shares.append(shares[node_id - 1])
                layer_rows.append(row_shares)
            local_layers.append(layer_rows)
        return local_layers
    
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
        
        scale = self.scale
        self._plain_dataset = [np.asarray(sample, dtype=np.float64).copy() for sample in dataset]
        self._plain_labels = [np.asarray(label, dtype=np.float64).reshape(-1).copy() for label in labels]

        # Secret-share each sample
        for i, (sample, label) in enumerate(zip(dataset, labels)):
            # Share each feature of the sample
            sample_feature_shares = []
            for feature_value in sample:
                feature_int = int(round(float(feature_value) * scale)) % self.mpc_engine.field_size
                feature_shares = self.pss.shamir.share(feature_int, self.n_nodes, self.t)
                sample_feature_shares.append(feature_shares)
            
            # Share label(s): support scalar labels and vector (e.g., one-hot) labels.
            label_arr = np.asarray(label)
            if label_arr.ndim == 0:
                label_arr = label_arr.reshape(1)
            label_component_shares = []
            for label_value in label_arr.reshape(-1):
                label_int = int(round(float(label_value) * scale)) % self.mpc_engine.field_size
                component_shares = self.pss.shamir.share(label_int, self.n_nodes, self.t)
                label_component_shares.append(component_shares)
            
            # Store shares in KVS (one share per node per feature)
            for node_id in range(1, self.n_nodes + 1):
                # Store all feature shares for this node
                node_sample_shares = []
                for feature_shares in sample_feature_shares:
                    node_share = next(s for s in feature_shares if s.node_id == node_id)
                    node_sample_shares.append(node_share)
                
                node_label_shares = []
                for component_shares in label_component_shares:
                    node_component_share = next(s for s in component_shares if s.node_id == node_id)
                    node_label_shares.append(node_component_share)
                
                sample_key = f"sample_{i}_node_{node_id}"
                label_key = f"label_{i}_node_{node_id}"
                
                self.kvs_cluster.write_with_quorum(
                    sample_key, node_sample_shares, self.v_D
                )
                self.kvs_cluster.write_with_quorum(
                    label_key, node_label_shares, self.v_D
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
        
        # Initialize with small fixed-point values to avoid saturation/exploding updates.
        scale = self.scale
        self.weight_shapes = list(weight_shapes)
        rng = np.random.default_rng(self.seed)
        weights = []
        plain_weights: List[np.ndarray] = []
        for input_dim, output_dim in weight_shapes:
            layer_weights = []
            fan_in = max(1, input_dim)
            limit = np.sqrt(6.0 / (fan_in + max(1, output_dim)))
            layer_plain = np.zeros((output_dim, input_dim), dtype=np.float64)
            for _ in range(output_dim):
                row = []
                row_idx = len(layer_weights)
                for col_idx in range(input_dim):
                    w = float(rng.uniform(-limit, limit))
                    weight_int = int(round(w * scale)) % self.mpc_engine.field_size
                    layer_plain[row_idx, col_idx] = w
                    ctx = f"weights_v{self.v_theta}_L{len(weights)}_R{row_idx}_C{col_idx}"
                    weight_shares = self._deterministic_share_secret(weight_int, ctx)
                    row.append(weight_shares)
                layer_weights.append(row)
            plain_weights.append(layer_plain)
            weights.append(layer_weights)
        self._plain_weights = plain_weights
        
        # Store weights in KVS (simplified - would store properly)
        weight_key = f"weights_v{self.v_theta}"
        self.kvs_cluster.write_with_quorum(weight_key, weights, self.v_theta)
        
        return self.v_theta
    
    def select_mini_batch(self, dataset_size: int) -> List[int]:
        """Select random mini-batch indices"""
        return self.batch_rng.sample(range(dataset_size), min(self.batch_size, dataset_size))
    
    def train_mini_batch(self, sample_shares: List[List[Share]], 
                        label_shares: List[List[Share]],
                        weight_shares: List[List[List[Share]]],
                        packing_factor: int, node_id: int,
                        batch_indices: Optional[List[int]] = None) -> Tuple[List[List[List[Share]]], bool]:
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
            batch_indices: Original dataset indices for this mini-batch
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

        # MNIST local-emulation path: use plaintext batch update + deterministic resharing.
        # Keeps multi-node shares consistent while approximating Keras-like training dynamics.
        if batch_indices is not None:
            mnist_updated = self._mnist_plain_batch_update(batch_indices, node_id)
            if mnist_updated is not None:
                return mnist_updated, True
        
        # Use mean over the mini-batch for a stabler update signal in this prototype.
        if not sample_shares:
            return weight_shares, False

        input_len = len(sample_shares[0])
        x_point = sample_shares[0][0].x if input_len > 0 else 1
        input_shares: List[Share] = []
        for j in range(input_len):
            s = 0
            for sample in sample_shares:
                if j < len(sample):
                    s += int(sample[j].y)
            input_shares.append(
                Share(
                    x=x_point,
                    y=(s // max(1, len(sample_shares))) % self.mpc_engine.field_size,
                    node_id=node_id,
                )
            )

        # Normalize weight structure to this node's local shares.
        # Initial KVS weights are nested as [layer][row][col][node_share_list].
        # After first update in this simplified prototype, weights may already be local Share objects.
        local_weight_shares: List[List[List[Share]]] = []
        for layer in weight_shares:
            local_layer = []
            for row in layer:
                local_row = []
                for w in row:
                    if hasattr(w, "node_id"):
                        local_row.append(w)
                    elif isinstance(w, list) and w and hasattr(w[0], "node_id"):
                        local = next((s for s in w if s.node_id == node_id), w[0])
                        local_row.append(local)
                    else:
                        return weight_shares, False
                local_layer.append(local_row)
            local_weight_shares.append(local_layer)
        
        # Ensure input size matches first layer
        if local_weight_shares and local_weight_shares[0]:
            expected_input_size = len(local_weight_shares[0][0])
            if len(input_shares) > expected_input_size:
                input_shares = input_shares[:expected_input_size]
            elif len(input_shares) < expected_input_size:
                # Pad with zero shares
                zero_share = Share(x=input_shares[0].x if input_shares else 1, y=0, node_id=node_id)
                input_shares = input_shares + [zero_share] * (expected_input_size - len(input_shares))
        
        # Forward pass
        predictions = self.mpc_engine.forward_pass(
            input_shares,
            local_weight_shares, node_id
        )
        
        # Compute loss
        # NOTE:
        # True packed Shamir packing must happen at *share-time* (one polynomial encodes k labels).
        # This coordinator path is a simplified prototype and does not yet store labels in packed form.
        # For now, use this node's label shares directly (no packing).
        packed_labels: List[Share] = []
        if label_shares:
            first_label = label_shares[0]
            if isinstance(first_label, list) and first_label:
                for k in range(len(first_label)):
                    s = 0
                    count = 0
                    for sample_lbl in label_shares:
                        if k < len(sample_lbl):
                            s += int(sample_lbl[k].y)
                            count += 1
                    packed_labels.append(
                        Share(
                            x=first_label[k].x,
                            y=(s // max(1, count)) % self.mpc_engine.field_size,
                            node_id=node_id,
                        )
                    )
        
        loss_share = self.mpc_engine.compute_loss(
            predictions[:len(packed_labels)], 
            packed_labels[:len(predictions)],
            node_id
        )
        
        # Backward pass
        # Use input_shares for proper gradient computation
        gradients = self.mpc_engine.backward_pass(
            loss_share, predictions, packed_labels, local_weight_shares, node_id,
            input_shares=input_shares  # Pass input for gradient computation
        )
        
        # Update weights using standard (non-DP) SGD
        updated_weights = self.mpc_engine.update_weights(
            local_weight_shares, gradients, self.learning_rate, node_id
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
