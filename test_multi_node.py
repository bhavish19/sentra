"""
Multi-node test script for SENTRA training pipeline
This script can be run on multiple nodes to test the communication protocol
"""

import sys
import os
import time
import argparse
import threading
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training import SentraTrainingPipeline
import numpy as np


def create_node_configs(n_nodes: int, base_port: int = 8000) -> Dict[int, Dict[str, Any]]:
    """Create node configurations for multi-node setup"""
    configs = {}
    for i in range(1, n_nodes + 1):
        configs[i] = {
            'host': 'localhost',
            'port': base_port + i
        }
    return configs


def run_node(node_id: int, n_nodes: int, t: int, s: int, 
             enable_network: bool = True):
    """Run a single node in the multi-node setup"""
    
    print(f"\n{'='*70}")
    print(f"Node {node_id} Starting")
    print(f"{'='*70}")
    
    # Create node configurations
    node_configs = create_node_configs(n_nodes)
    
    # Create pipeline (always plain SGD)
    pipeline = SentraTrainingPipeline(
        n_nodes=n_nodes,
        t=t,
        s=s,
        batch_size=16,
        learning_rate=0.01,
        num_epochs=2,
        enable_network=enable_network,
        node_id=node_id,
        node_configs=node_configs if enable_network else None
    )
    
    # Create shared dataset (in production, each node would have its own partition)
    np.random.seed(42)  # For reproducibility
    dataset = [np.random.randn(10) for _ in range(50)]
    labels = [np.random.randn(1) for _ in range(50)]
    
    weight_shapes = [(10, 5), (5, 1)]
    
    try:
        print(f"\nNode {node_id}: Starting training...")
        start_time = time.time()
        
        pipeline.train(dataset, labels, weight_shapes)
        
        elapsed = time.time() - start_time
        print(f"\nNode {node_id}: Training completed in {elapsed:.2f} seconds")
        print(f"Node {node_id}: ✓ Success")
        
        return True
    except Exception as e:
        print(f"\nNode {node_id}: ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_node():
    """Test single-node mode (no network)"""
    print("\n" + "="*70)
    print("Test 1: Single-Node Mode (No Network)")
    print("="*70)
    
    success = run_node(
        node_id=1,
        n_nodes=5,
        t=1,
        s=1,
        enable_network=False
    )
    
    return success


def test_multi_node_simulation():
    """Simulate multi-node setup using threads (for testing on single machine)"""
    print("\n" + "="*70)
    print("Test 2: Multi-Node Simulation (Threaded)")
    print("="*70)
    print("Note: This simulates multiple nodes using threads.")
    print("For true multi-node testing, run separate processes on different machines.")
    
    n_nodes = 3  # Use fewer nodes for simulation
    t = 1
    s = 1
    
    # Start all nodes in separate threads
    threads = []
    results = {}
    
    def node_runner(node_id):
        results[node_id] = run_node(
            node_id=node_id,
            n_nodes=n_nodes,
            t=t,
            s=s,
            enable_network=True  # Enable network for multi-node
        )
    
    # Start nodes with slight delay to allow network setup
    for node_id in range(1, n_nodes + 1):
        thread = threading.Thread(target=node_runner, args=(node_id,))
        threads.append(thread)
        thread.start()
        time.sleep(0.5)  # Stagger node starts
    
    # Wait for all nodes to complete
    for thread in threads:
        thread.join()
    
    # Check results
    all_success = all(results.values())
    print(f"\n{'='*70}")
    print(f"Multi-Node Simulation Results:")
    for node_id, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  Node {node_id}: {status}")
    print(f"{'='*70}")
    
    return all_success


def main():
    parser = argparse.ArgumentParser(description='Test SENTRA multi-node training')
    parser.add_argument('--test', choices=['single', 'multi', 'both'], default='both',
                       help='Which test to run')
    parser.add_argument('--node-id', type=int, default=1,
                       help='Node ID (for multi-node mode)')
    parser.add_argument('--n-nodes', type=int, default=5,
                       help='Total number of nodes')
    parser.add_argument('--enable-network', action='store_true',
                       help='Enable network communication')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SENTRA Multi-Node Test Suite")
    print("="*70)
    
    if args.test == 'single' or args.test == 'both':
        test_single_node()
    
    if args.test == 'multi' or args.test == 'both':
        if args.enable_network:
            # Run as a specific node
            run_node(
                node_id=args.node_id,
                n_nodes=args.n_nodes,
                t=1,
                s=1,
                enable_network=True
            )
        else:
            # Run simulation
            test_multi_node_simulation()
    
    print("\n" + "="*70)
    print("Test Suite Completed")
    print("="*70)


if __name__ == "__main__":
    main()


