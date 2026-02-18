"""
Python script to start all SENTRA nodes
Usage: python start_all_nodes.py [--use-dp-sgd]
"""

import subprocess
import sys
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description='Start all SENTRA nodes')
    parser.add_argument('--n-nodes', type=int, default=5,
                       help='Number of nodes to start (default: 5)')
    parser.add_argument('--base-port', type=int, default=8000,
                       help='Base port number (default: 8000)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host for all nodes (default: localhost)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Mini-batch size passed to each node (default: 8)')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Epoch count passed to each node (default: 1)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate passed to each node (default: 0.01)')
    parser.add_argument('--t', type=int, default=1,
                       help='Privacy threshold (default: 1)')
    parser.add_argument('--s', type=int, default=1,
                       help='Adversarial share limit (default: 1)')
    parser.add_argument('--dataset', choices=['synthetic', 'mnist'], default='synthetic',
                       help='Dataset mode for each node (default: synthetic)')
    parser.add_argument('--mnist-samples', type=int, default=128,
                       help='MNIST sample count per node when --dataset mnist (default: 128)')
    parser.add_argument('--mnist-input-dim', type=int, default=64,
                       help='Flattened MNIST features to keep when --dataset mnist (default: 64, max: 784)')
    parser.add_argument('--mnist-hidden-dim', type=int, default=16,
                       help='Hidden layer width for MNIST when --dataset mnist (default: 16)')
    parser.add_argument('--mnist-test-samples', type=int, default=1000,
                       help='MNIST test sample count for post-training accuracy proxy (default: 1000)')
    parser.add_argument('--no-test-accuracy', action='store_true',
                       help='Disable post-training MNIST test accuracy reporting on nodes')
    parser.add_argument('--post-metrics-barrier-timeout', type=float, default=180.0,
                       help='Seconds to wait at post-metrics barrier before shutdown (default: 180)')
    parser.add_argument('--seed', type=int, default=2026,
                       help='Global random seed passed to all nodes (default: 2026)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Starting SENTRA Multi-Node Training")
    print("=" * 70)
    print(f"Total nodes: {args.n_nodes}")
    print(f"Base port: {args.base_port}")
    print(f"Host: {args.host}")
    print(f"Training: epochs={args.num_epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    print(f"Thresholds: t={args.t}, s={args.s}")
    print(f"Dataset: {args.dataset}")
    print("=" * 70)
    print()
    
    processes = []
    
    # Build command for each node
    for node_id in range(1, args.n_nodes + 1):
        cmd = [
            sys.executable,
            'run_node.py',
            '--node-id', str(node_id),
            '--n-nodes', str(args.n_nodes),
            '--base-port', str(args.base_port),
            '--host', str(args.host),
            '--batch-size', str(args.batch_size),
            '--num-epochs', str(args.num_epochs),
            '--learning-rate', str(args.learning_rate),
            '--t', str(args.t),
            '--s', str(args.s),
            '--dataset', str(args.dataset),
            '--seed', str(args.seed),
            '--no-wait',
        ]
        if args.dataset == 'mnist':
            cmd.extend([
                '--mnist-samples', str(args.mnist_samples),
                '--mnist-input-dim', str(args.mnist_input_dim),
                '--mnist-hidden-dim', str(args.mnist_hidden_dim),
                '--mnist-test-samples', str(args.mnist_test_samples),
                '--post-metrics-barrier-timeout', str(args.post_metrics_barrier_timeout),
            ])
            if args.no_test_accuracy:
                cmd.append('--no-test-accuracy')
        
        print(f"Starting Node {node_id}...")
        
        # Start in new window (Windows)
        if sys.platform == 'win32':
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            # Linux/macOS - run in background
            subprocess.Popen(cmd)
        
        time.sleep(1)  # Stagger starts
    
    print()
    print("All nodes started in separate windows.")
    print("Close each window to stop the corresponding node.")
    print()
    print("To stop all nodes, close each window or press Ctrl+C.")


if __name__ == '__main__':
    main()
