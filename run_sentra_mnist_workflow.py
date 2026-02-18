"""
Run MNIST through the SENTRA workflow pipeline.

This integrates the simple MNIST use case with SENTRA orchestration:
- dataset ingestion + secret sharing
- secure mini-batch training loop
- quorum model commits
"""

import argparse
import os
import sys
from typing import Dict, Any, Tuple, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_training import SentraTrainingPipeline


def create_node_configs(n_nodes: int, base_port: int = 8000, host: str = "localhost") -> Dict[int, Dict[str, Any]]:
    """Create node host/port configs."""
    return {i: {"host": host, "port": base_port + i} for i in range(1, n_nodes + 1)}


def _mnist_to_feature_matrix(images: np.ndarray, input_dim: int) -> np.ndarray:
    """Downsample MNIST over full image extent to a fixed feature dimension."""
    imgs = np.asarray(images, dtype=np.float32)
    n = imgs.shape[0]
    input_dim = max(1, int(input_dim))
    if input_dim >= 28 * 28:
        return imgs.reshape(n, -1)[:, :input_dim]
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


def load_mnist_dataset(
    sample_count: int, input_dim: int, hidden_dim: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    """
    Load MNIST and adapt it to the current scalar-label training coordinator.
    """
    try:
        from tensorflow import keras
    except Exception as exc:
        raise RuntimeError(
            "This script requires TensorFlow/Keras. Install tensorflow first."
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
    weight_shapes = [(input_dim, hidden_dim), (hidden_dim, 10)]
    return dataset, labels, weight_shapes


def evaluate_plaintext_mnist_accuracy(
    train_samples: int, test_samples: int, epochs: int, batch_size: int
) -> float:
    """
    Compute a reference MNIST classification accuracy in plaintext TensorFlow.
    This is a baseline metric, not secure SENTRA-model accuracy.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    train_samples = max(1, min(train_samples, len(x_train)))
    test_samples = max(1, min(test_samples, len(x_test)))
    x_train = x_train[:train_samples]
    y_train = y_train[:train_samples]
    x_test = x_test[:test_samples]
    y_test = y_test[:test_samples]

    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return float(accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SENTRA workflow with MNIST data")
    parser.add_argument("--n-nodes", type=int, default=5, help="Total number of nodes")
    parser.add_argument("--t", type=int, default=1, help="Privacy threshold")
    parser.add_argument("--s", type=int, default=1, help="Adversarial share limit")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--node-id", type=int, default=1, help="This node ID")
    parser.add_argument("--enable-network", action="store_true", help="Enable multi-node network communication")
    parser.add_argument("--base-port", type=int, default=8000, help="Base port for node configs")
    parser.add_argument("--host", type=str, default="localhost", help="Host for node configs")
    parser.add_argument("--mnist-samples", type=int, default=128, help="Number of MNIST samples to use")
    parser.add_argument("--mnist-input-dim", type=int, default=64, help="Flattened MNIST features to keep (max 784)")
    parser.add_argument("--mnist-hidden-dim", type=int, default=16, help="Hidden layer width for SENTRA model")
    parser.add_argument("--skip-plaintext-baseline", action="store_true",
                        help="Skip plaintext baseline accuracy evaluation at the end")
    parser.add_argument("--baseline-epochs", type=int, default=1,
                        help="Epochs for plaintext baseline accuracy (default: 1)")
    parser.add_argument("--baseline-test-samples", type=int, default=1000,
                        help="Test-set sample count for plaintext baseline accuracy (default: 1000)")
    parser.add_argument("--seed", type=int, default=2026, help="Global random seed (default: 2026)")
    args = parser.parse_args()

    node_configs = None
    if args.enable_network:
        node_configs = create_node_configs(args.n_nodes, args.base_port, args.host)

    dataset, labels, weight_shapes = load_mnist_dataset(
        args.mnist_samples, args.mnist_input_dim, args.mnist_hidden_dim
    )

    print("=" * 70)
    print("SENTRA MNIST Workflow")
    print("=" * 70)
    print(f"Samples: {len(dataset)}")
    print(f"Input dim: {len(dataset[0])}")
    print(f"Weight shapes: {weight_shapes}")
    print(f"Network mode: {'enabled' if args.enable_network else 'disabled'}")
    print("=" * 70)

    pipeline = SentraTrainingPipeline(
        n_nodes=args.n_nodes,
        t=args.t,
        s=args.s,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        node_id=args.node_id,
        node_configs=node_configs,
        enable_network=args.enable_network,
        seed=args.seed,
    )
    pipeline.train(dataset, labels, weight_shapes)

    if not args.skip_plaintext_baseline:
        print("\n[Reference Accuracy] Running plaintext MNIST baseline...")
        baseline_acc = evaluate_plaintext_mnist_accuracy(
            train_samples=args.mnist_samples,
            test_samples=args.baseline_test_samples,
            epochs=args.baseline_epochs,
            batch_size=max(8, args.batch_size),
        )
        print(f"[Reference Accuracy] Plaintext baseline accuracy: {baseline_acc:.4f}")
        print("[Reference Accuracy] Note: This is not secure SENTRA-model accuracy.")


if __name__ == "__main__":
    main()
