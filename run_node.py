"""
Universal node runner for SENTRA multi-node training
Usage: python run_node.py --node-id 1
"""

import argparse
import sys
import os
import time
import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training import SentraTrainingPipeline
from ml_training.secret_sharing import Share, ShamirSecretSharing
import numpy as np
from typing import Optional, Tuple


def create_node_configs(n_nodes: int, base_port: int = 8000, host: str = 'localhost'):
    """Create node configurations"""
    return {
        i: {'host': host, 'port': base_port + i}
        for i in range(1, n_nodes + 1)
    }


def load_synthetic_dataset():
    """Create the existing synthetic regression-style dataset."""
    np.random.seed(42)
    dataset = [np.random.randn(10) for _ in range(100)]
    labels = [np.random.randn(1) for _ in range(100)]
    weight_shapes = [(10, 4), (4, 1)]
    return dataset, labels, weight_shapes


def _mnist_to_feature_matrix(images: np.ndarray, input_dim: int) -> np.ndarray:
    """
    Convert MNIST images to a fixed feature size while preserving full-image coverage.
    For small dims (e.g. 64), use grid downsampling over the whole 28x28 image.
    """
    imgs = np.asarray(images, dtype=np.float32)
    n = imgs.shape[0]
    input_dim = max(1, int(input_dim))

    if input_dim >= 28 * 28:
        flat = imgs.reshape(n, -1)
        return flat[:, :input_dim]

    side = int(np.sqrt(input_dim))
    if side >= 2:
        rows = np.linspace(0, 27, side).astype(np.int32)
        cols = np.linspace(0, 27, side).astype(np.int32)
        sampled = imgs[:, rows][:, :, cols]
        flat = sampled.reshape(n, -1)
    else:
        flat = imgs.reshape(n, -1)

    if flat.shape[1] < input_dim:
        pad = np.zeros((n, input_dim - flat.shape[1]), dtype=np.float32)
        flat = np.concatenate([flat, pad], axis=1)
    return flat[:, :input_dim]


def _forward_two_layer_with_optional_bias(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Two-layer forward with optional bias encoded as extra input columns in weights."""
    x_in = x
    if w1.shape[1] == x.shape[1] + 1:
        ones = np.ones((x.shape[0], 1), dtype=x.dtype)
        x_in = np.concatenate([x, ones], axis=1)
    hidden = np.maximum(x_in @ w1.T, 0.0)
    h_in = hidden
    if w2.shape[1] == hidden.shape[1] + 1:
        ones_h = np.ones((hidden.shape[0], 1), dtype=hidden.dtype)
        h_in = np.concatenate([hidden, ones_h], axis=1)
    return h_in @ w2.T


def load_mnist_dataset(sample_count: int, input_dim: int, hidden_dim: int):
    """
    Load MNIST and adapt it to the SENTRA two-layer classifier prototype.
    Inputs are flattened to 784 features; labels are one-hot encoded (10 classes).
    """
    try:
        from tensorflow import keras
    except Exception as exc:
        raise RuntimeError(
            "MNIST dataset requires TensorFlow/Keras. Install tensorflow first."
        ) from exc

    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    y_train = y_train.astype("int32")

    sample_count = max(1, min(sample_count, len(x_train)))
    x_train = x_train[:sample_count]
    y_train = y_train[:sample_count]

    input_dim = max(1, min(input_dim, 784))
    hidden_dim = max(1, hidden_dim)

    x_features = _mnist_to_feature_matrix(x_train, input_dim)
    dataset = [x_features[i] for i in range(x_features.shape[0])]
    labels = [np.eye(10, dtype=np.float32)[int(y)] for y in y_train]
    # Bias is encoded as an extra constant-input column per layer.
    weight_shapes = [(input_dim + 1, hidden_dim), (hidden_dim + 1, 10)]
    return dataset, labels, weight_shapes


def _field_to_signed(value: int, field_size: int) -> int:
    """Map finite-field value into a signed integer representative."""
    value = int(value) % int(field_size)
    if value > field_size // 2:
        value -= field_size
    return value


def _extract_local_weight_matrices(
    pipeline: SentraTrainingPipeline, weight_shapes, node_id: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract this node's local view of 2-layer dense weights from KVS.
    Returns (W1, W2) with shapes (hidden, input_dim) and (num_classes, hidden).
    """
    if len(weight_shapes) != 2:
        return None

    v_theta = pipeline.coordinator.v_theta
    weight_key = f"weights_v{v_theta}"
    weight_value = pipeline.kvs_cluster.read_with_min_version(weight_key, v_theta)
    if not weight_value:
        return None

    raw_weights = weight_value.data
    if not isinstance(raw_weights, list) or len(raw_weights) != 2:
        return None

    field_size = int(pipeline.coordinator.mpc_engine.field_size)
    scale = int(getattr(pipeline.coordinator.mpc_engine.matrix_ops, "scale_factor", 1000))

    def pick_local_share(w):
        if hasattr(w, "y"):
            return w
        if isinstance(w, list) and w:
            return next((s for s in w if getattr(s, "node_id", None) == node_id), w[0])
        return None

    matrices = []
    for layer in raw_weights:
        if not isinstance(layer, list) or not layer:
            return None
        rows = []
        for row in layer:
            if not isinstance(row, list) or not row:
                return None
            vals = []
            for w in row:
                s = pick_local_share(w)
                if s is None:
                    return None
                # Normalize to a bounded float to avoid numerical explosion.
                vals.append(_field_to_signed(s.y, field_size) / float(scale))
            rows.append(vals)
        matrices.append(np.asarray(rows, dtype=np.float64))

    if matrices[0].shape != (weight_shapes[0][1], weight_shapes[0][0]):
        return None
    if matrices[1].shape != (weight_shapes[1][1], weight_shapes[1][0]):
        return None
    return matrices[0], matrices[1]


def _extract_local_weight_share_matrices(
    pipeline: SentraTrainingPipeline, weight_shapes, node_id: int
):
    """
    Extract this node's local Share matrices from final KVS weights.
    Returns [layer1, layer2], each layer as list[list[Share]].
    """
    if len(weight_shapes) != 2:
        return None

    v_theta = pipeline.coordinator.v_theta
    weight_key = f"weights_v{v_theta}"
    weight_value = pipeline.kvs_cluster.read_with_min_version(weight_key, v_theta)
    if not weight_value:
        return None

    raw_weights = weight_value.data
    if not isinstance(raw_weights, list) or len(raw_weights) != 2:
        return None

    local_layers = []
    for layer in raw_weights:
        if not isinstance(layer, list) or not layer:
            return None
        local_rows = []
        for row in layer:
            if not isinstance(row, list) or not row:
                return None
            local_row = []
            for w in row:
                if hasattr(w, "node_id"):
                    local_row.append(w)
                elif isinstance(w, list) and w and hasattr(w[0], "node_id"):
                    local = next((s for s in w if s.node_id == node_id), w[0])
                    local_row.append(local)
                else:
                    return None
            local_rows.append(local_row)
        local_layers.append(local_rows)
    return local_layers


def _weight_contexts(weight_shapes, base_context: str):
    contexts = []
    for li, (inp, out) in enumerate(weight_shapes):
        for r in range(out):
            for c in range(inp):
                contexts.append((li, r, c, f"{base_context}_L{li}_R{r}_C{c}"))
    return contexts


def _send_model_shares_to_node1(
    pipeline: SentraTrainingPipeline,
    local_layers,
    weight_shapes,
    node_id: int,
    context_base: str,
    chunk_size: int = 256,
) -> None:
    network = getattr(pipeline.coordinator, "network", None)
    if network is None or node_id == 1:
        return
    contexts = _weight_contexts(weight_shapes, context_base)

    flat_shares = []
    flat_contexts = []
    for li, r, c, ctx in contexts:
        flat_shares.append(local_layers[li][r][c])
        flat_contexts.append(ctx)

    # Send in batches to reduce per-message overhead.
    for i in range(0, len(flat_shares), chunk_size):
        shares_chunk = flat_shares[i:i + chunk_size]
        ctx_chunk = flat_contexts[i:i + chunk_size]
        try:
            network.channel.send_shares_batch(1, shares_chunk, ctx_chunk)
        except Exception:
            # Fallback to per-share send.
            for s, ctx in zip(shares_chunk, ctx_chunk):
                try:
                    network.send_share(1, s, ctx)
                except Exception:
                    pass


def compute_mnist_test_accuracy_reconstructed(
    pipeline: SentraTrainingPipeline,
    weight_shapes,
    node_id: int,
    n_nodes: int,
    t: int,
    input_dim: int,
    test_samples: int,
    timeout_s: float = 30.0,
) -> Optional[float]:
    """
    Reconstruct model on node 1 using final shares from all nodes (or >= t+1 per weight),
    then compute MNIST test accuracy.
    Non-primary nodes only send their shares and return None.
    """
    local_layers = _extract_local_weight_share_matrices(pipeline, weight_shapes, node_id)
    if local_layers is None:
        return None

    v_theta = pipeline.coordinator.v_theta
    context_base = f"mnist_model_recon_v{v_theta}"
    _send_model_shares_to_node1(pipeline, local_layers, weight_shapes, node_id, context_base)
    if node_id != 1:
        return None

    network = getattr(pipeline.coordinator, "network", None)
    if network is None:
        return None

    contexts = _weight_contexts(weight_shapes, context_base)
    needed_peer_shares = max(0, (t + 1) - 1)  # minus local share on node 1
    received_by_ctx = {ctx: {} for _, _, _, ctx in contexts}

    start = time.time()
    while time.time() - start < timeout_s:
        ready = 0
        for _, _, _, ctx in contexts:
            shares = network.get_received_shares(ctx)
            for s in shares:
                sid = getattr(s, "node_id", None)
                if sid in (None, 1):
                    continue
                received_by_ctx[ctx][sid] = s
            if len(received_by_ctx[ctx]) >= needed_peer_shares:
                ready += 1
        if ready == len(contexts):
            break
        time.sleep(0.2)

    field_size = int(pipeline.coordinator.mpc_engine.field_size)
    scale = int(getattr(pipeline.coordinator.mpc_engine.matrix_ops, "scale_factor", 1000))
    shamir = ShamirSecretSharing(field_size=field_size)
    reconstructed_layers = []

    for li, (inp, out) in enumerate(weight_shapes):
        vals = np.zeros((out, inp), dtype=np.float64)
        for r in range(out):
            for c in range(inp):
                ctx = f"{context_base}_L{li}_R{r}_C{c}"
                local_share = local_layers[li][r][c]
                shares = [local_share]
                peer_map = received_by_ctx.get(ctx, {})
                for sid in sorted(peer_map):
                    shares.append(peer_map[sid])
                    if len(shares) >= (t + 1):
                        break
                if len(shares) < (t + 1):
                    return None
                sec = shamir.reconstruct(shares)
                vals[r, c] = _field_to_signed(sec, field_size) / float(scale)
        reconstructed_layers.append(vals)

    w1, w2 = reconstructed_layers

    try:
        from tensorflow import keras
    except Exception:
        return None
    _, (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = _mnist_to_feature_matrix(x_test, input_dim)
    y_test = y_test.astype(np.int32)

    n = max(1, min(int(test_samples), len(x_test)))
    x = x_test[:n]
    y = y_test[:n]
    logits = _forward_two_layer_with_optional_bias(x, w1, w2)
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y))


def compute_mnist_test_accuracy_proxy(
    pipeline: SentraTrainingPipeline,
    weight_shapes,
    node_id: int,
    input_dim: int,
    test_samples: int,
) -> Optional[float]:
    """
    Compute a node-local proxy MNIST accuracy from final local shares.
    This is not a cryptographically reconstructed global-model accuracy.
    """
    matrices = _extract_local_weight_matrices(pipeline, weight_shapes, node_id)
    if matrices is None:
        return None
    w1, w2 = matrices

    try:
        from tensorflow import keras
    except Exception:
        return None

    _, (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = _mnist_to_feature_matrix(x_test, input_dim)
    y_test = y_test.astype(np.int32)

    n = max(1, min(int(test_samples), len(x_test)))
    x = x_test[:n]
    y = y_test[:n]

    # Two-layer dense forward pass matching the current SENTRA shape.
    logits = _forward_two_layer_with_optional_bias(x, w1, w2)
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y))


def report_cluster_accuracy_proxy(
    pipeline: SentraTrainingPipeline,
    node_id: int,
    n_nodes: int,
    local_acc: float,
    timeout_s: float = 90.0,
) -> None:
    """
    Exchange node-local proxy accuracies and report a cluster summary on node 1.
    Uses share-exchange channel with scaled accuracy values.
    """
    network = getattr(pipeline.coordinator, "network", None)
    if network is None:
        return

    context = f"mnist_acc_proxy_v{pipeline.coordinator.v_theta}"
    scaled = int(round(local_acc * 1_000_000))
    acc_share = Share(x=node_id, y=scaled, node_id=node_id)

    if node_id != 1:
        # Retry a few times to reduce chance of dropped/lost report in local multi-process runs.
        for _ in range(3):
            try:
                network.send_share(1, acc_share, context)
            except Exception:
                pass
            time.sleep(0.05)
        return

    # Node 1: collect peer reports and summarize.
    received = {}
    start = time.time()
    while time.time() - start < timeout_s and len(received) < (n_nodes - 1):
        shares = network.get_received_shares(context)
        for s in shares:
            if getattr(s, "node_id", None) in (None, 1):
                continue
            received[s.node_id] = s
        if len(received) >= (n_nodes - 1):
            break
        time.sleep(0.2)

    vals = [local_acc]
    for nid in sorted(received):
        vals.append(received[nid].y / 1_000_000.0)

    print(f"Cluster MNIST proxy accuracy reports: {len(vals)}/{n_nodes} node(s)")
    print(f"Cluster proxy accuracy avg: {float(np.mean(vals)):.4f}")
    print(f"Cluster proxy accuracy min: {float(np.min(vals)):.4f}")
    print(f"Cluster proxy accuracy max: {float(np.max(vals)):.4f}")
    print("Note: Cluster proxy aggregates node-local metrics, not reconstructed secure global-model accuracy.")


def main():
    parser = argparse.ArgumentParser(
        description='Run SENTRA training on a single node',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run node 1 on localhost
  python run_node.py --node-id 1

  # Run node 2 with DP-SGD
  python run_node.py --node-id 2 --use-dp-sgd

  # Run node 3 with custom port range
  python run_node.py --node-id 3 --base-port 9000

  # Run node 1 on remote host
  python run_node.py --node-id 1 --host 192.168.1.10
        """
    )
    
    parser.add_argument('--node-id', type=int, required=True,
                       help='This node\'s ID (1, 2, 3, ...)')
    parser.add_argument('--n-nodes', type=int, default=5,
                       help='Total number of nodes (default: 5)')
    parser.add_argument('--base-port', type=int, default=8000,
                       help='Base port number (default: 8000, ports will be base+node_id)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host address for all nodes (default: localhost)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Mini-batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--num-epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--t', type=int, default=1,
                       help='Privacy threshold (default: 1)')
    parser.add_argument('--s', type=int, default=1,
                       help='Adversarial share limit (default: 1)')
    parser.add_argument('--dataset', choices=['synthetic', 'mnist'], default='synthetic',
                       help='Dataset mode: synthetic or mnist (default: synthetic)')
    parser.add_argument('--mnist-samples', type=int, default=1000,
                       help='Number of MNIST training samples to use when --dataset mnist (default: 1000)')
    parser.add_argument('--mnist-input-dim', type=int, default=64,
                       help='Flattened MNIST features to keep when --dataset mnist (default: 64, max: 784)')
    parser.add_argument('--mnist-hidden-dim', type=int, default=16,
                       help='Hidden layer width for MNIST when --dataset mnist (default: 16)')
    parser.add_argument('--mnist-test-samples', type=int, default=1000,
                       help='MNIST test sample count for post-training accuracy proxy (default: 1000)')
    parser.add_argument('--no-test-accuracy', action='store_true',
                       help='Disable post-training MNIST test accuracy reporting')
    parser.add_argument('--post-metrics-barrier-timeout', type=float, default=180.0,
                       help='Seconds to wait at post-metrics barrier before shutdown (default: 180)')
    parser.add_argument('--seed', type=int, default=2026,
                       help='Global random seed for deterministic multi-node runs (default: 2026)')
    parser.add_argument('--train-mode', choices=['secure', 'hybrid'], default='secure',
                       help='Training mode: secure share-domain or hybrid plaintext-update+reshare (default: secure)')
    parser.add_argument('--log-mini-batches', action='store_true',
                       help='Print per-mini-batch commit logs (default: disabled)')
    parser.add_argument('--compute-local-proxy-accuracy', action='store_true',
                       help='Compute node-local proxy MNIST accuracy (default: disabled)')
    parser.add_argument('--no-wait', action='store_true',
                       help='Exit immediately after training (no "Press Enter" prompt)')
    
    args = parser.parse_args()
    
    # Validate node_id
    if args.node_id < 1 or args.node_id > args.n_nodes:
        print(f"Error: node-id must be between 1 and {args.n_nodes}")
        sys.exit(1)
    
    # Create node configurations
    node_configs = create_node_configs(args.n_nodes, args.base_port, args.host)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 70)
    print(f"SENTRA Node {args.node_id} Starting")
    print("=" * 70)
    print(f"Node ID: {args.node_id}")
    print(f"Total nodes: {args.n_nodes}")
    print(f"Privacy threshold (t): {args.t}")
    print(f"Adversarial limit (s): {args.s}")
    print(f"Dataset mode: {args.dataset}")
    # Safety bound: 2*(t+s-1) < n_active
    safety_bound_value = 2 * (args.t + args.s - 1)
    safety_ok = safety_bound_value < args.n_nodes
    print(f"Safety check: 2*(t+s-1) = {safety_bound_value} < n_nodes = {args.n_nodes} {'✓' if safety_ok else '✗'}")
    print(f"Network: Enabled")
    print(f"Node configs: {node_configs}")
    print("=" * 70)
    
    # Create pipeline
    try:
        pipeline = SentraTrainingPipeline(
            n_nodes=args.n_nodes,
            t=args.t,
            s=args.s,
            node_id=args.node_id,
            node_configs=node_configs,
            enable_network=True,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            seed=args.seed,
            train_mode=args.train_mode,
            log_mini_batches=args.log_mini_batches,
        )
        
        if args.dataset == 'mnist':
            dataset, labels, weight_shapes = load_mnist_dataset(
                args.mnist_samples, args.mnist_input_dim, args.mnist_hidden_dim
            )
        else:
            dataset, labels, weight_shapes = load_synthetic_dataset()
        
        print(f"\nNode {args.node_id}: Starting training...")
        print(f"Dataset: {len(dataset)} samples, {len(dataset[0])} features")
        print(f"Model: {weight_shapes}")
        print(f"Train mode: {args.train_mode}")
        print("Multi-Node Mode: ENABLED")
        print("-" * 70)
        
        # Train
        pipeline.train(dataset, labels, weight_shapes)

        # Stop failure detector before metrics/reconstruction to avoid post-training
        # false failure/resharing noise while nodes are synchronizing for reporting.
        try:
            coordinator = getattr(pipeline, "coordinator", None)
            if coordinator is not None:
                fd = getattr(coordinator, "failure_detector", None)
                if fd is not None:
                    try:
                        fd.stop_monitoring()
                    except Exception:
                        pass
        except Exception:
            pass

        # Optional post-training MNIST evaluation.
        if args.dataset == 'mnist' and not args.no_test_accuracy:
            print("\n" + "=" * 70)
            print(f"Node {args.node_id}: Computing MNIST test accuracy...")
            print("=" * 70)
            if args.compute_local_proxy_accuracy:
                acc = compute_mnist_test_accuracy_proxy(
                    pipeline=pipeline,
                    weight_shapes=weight_shapes,
                    node_id=args.node_id,
                    input_dim=args.mnist_input_dim,
                    test_samples=args.mnist_test_samples,
                )
                if acc is None:
                    print("MNIST proxy test accuracy: unavailable (could not extract/evaluate model)")
                    print("Note: This metric is node-local and not reconstructed global-model accuracy.")
                else:
                    print(f"MNIST proxy test accuracy: {acc:.4f}")
                    print("Note: This metric is node-local and not reconstructed global-model accuracy.")
                    report_cluster_accuracy_proxy(
                        pipeline=pipeline,
                        node_id=args.node_id,
                        n_nodes=args.n_nodes,
                        local_acc=acc,
                        timeout_s=max(30.0, float(args.post_metrics_barrier_timeout) * 0.6),
                    )

                    # Synchronize all nodes after proxy-report exchange to reduce partial-report races.
                    try:
                        net = getattr(pipeline.coordinator, "network", None)
                        if net is not None:
                            sync_tag = f"post_metrics_proxy_v{pipeline.coordinator.v_theta}"
                            net.barrier(sync_tag, timeout=float(args.post_metrics_barrier_timeout))
                            print(f"Proxy-metrics barrier complete on node {args.node_id} ({sync_tag}).")
                    except Exception as proxy_barrier_exc:
                        print(f"Proxy-metrics barrier warning: {proxy_barrier_exc}")

            # Reconstruct model shares on node 1 and evaluate reconstructed-model accuracy.
            recon_acc = compute_mnist_test_accuracy_reconstructed(
                pipeline=pipeline,
                weight_shapes=weight_shapes,
                node_id=args.node_id,
                n_nodes=args.n_nodes,
                t=args.t,
                input_dim=args.mnist_input_dim,
                test_samples=args.mnist_test_samples,
            )
            if args.node_id == 1:
                if recon_acc is None:
                    print("Reconstructed-model MNIST test accuracy: unavailable (insufficient shares/timeouts)")
                else:
                    print(f"Reconstructed-model MNIST test accuracy: {recon_acc:.4f}")
            print("=" * 70)

            # Keep all nodes alive until metrics/reconstruction flow completes on node 1.
            try:
                net = getattr(pipeline.coordinator, "network", None)
                if net is not None:
                    tag = f"post_metrics_v{pipeline.coordinator.v_theta}"
                    net.barrier(tag, timeout=float(args.post_metrics_barrier_timeout))
                    print(f"Post-metrics barrier complete on node {args.node_id} ({tag}).")
            except Exception as barrier_exc:
                print(f"Post-metrics barrier warning: {barrier_exc}")
        
        # Print final status
        print("\n" + "=" * 70)
        print(f"Node {args.node_id} Final Status:")
        print("=" * 70)
        if hasattr(pipeline.coordinator, 'network') and pipeline.coordinator.network:
            connected = len(pipeline.coordinator.network.channel.connections)
            print(f"Multi-Node: ENABLED")
            print(f"Connected nodes: {connected} out of {args.n_nodes - 1} possible")
            if connected > 0:
                print("✓ Running in true multi-party mode")
            else:
                print("⚠ Running in degraded mode (no connections)")
        else:
            print("Multi-Node: DISABLED (single-node mode)")
        print("=" * 70)
        
        print("\n" + "=" * 70)
        print(f"Node {args.node_id}: Training completed successfully!")
        print("=" * 70)

        # Graceful shutdown:
        # Stop heartbeat/failure detection before waiting for user input, otherwise
        # closing any node window triggers spurious "node failed" + resharing during shutdown.
        try:
            coordinator = getattr(pipeline, "coordinator", None)
            if coordinator is not None:
                net = getattr(coordinator, "network", None)
                fd = getattr(coordinator, "failure_detector", None)
                if fd is not None:
                    try:
                        fd.stop_monitoring()
                    except Exception:
                        pass

                if net is not None:
                    try:
                        net.stop()
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Keep window open
        if not args.no_wait:
            print("\nPress Enter to close this window...")
            try:
                input()
            except:
                pass
        
    except KeyboardInterrupt:
        print(f"\n\nNode {args.node_id}: Interrupted by user")
        # Best-effort shutdown on interrupt
        try:
            coordinator = getattr(locals().get("pipeline", None), "coordinator", None)
            if coordinator is not None:
                fd = getattr(coordinator, "failure_detector", None)
                if fd is not None:
                    try:
                        fd.stop_monitoring()
                    except Exception:
                        pass
                net = getattr(coordinator, "network", None)
                if net is not None:
                    try:
                        net.stop()
                    except Exception:
                        pass
        except Exception:
            pass
        if not args.no_wait:
            print("Press Enter to close this window...")
            try:
                input()
            except:
                pass
        sys.exit(0)
    except Exception as e:
        print(f"\n\nNode {args.node_id}: Error occurred")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        # Best-effort shutdown on error
        try:
            coordinator = getattr(locals().get("pipeline", None), "coordinator", None)
            if coordinator is not None:
                fd = getattr(coordinator, "failure_detector", None)
                if fd is not None:
                    try:
                        fd.stop_monitoring()
                    except Exception:
                        pass
                net = getattr(coordinator, "network", None)
                if net is not None:
                    try:
                        net.stop()
                    except Exception:
                        pass
        except Exception:
            pass
        if not args.no_wait:
            print("\nPress Enter to close this window...")
            try:
                input()
            except:
                pass
        sys.exit(1)


if __name__ == '__main__':
    main()
