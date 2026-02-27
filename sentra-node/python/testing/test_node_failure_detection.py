"""
Test script for node failure detection
Tests heartbeat monitoring and failure detection
"""

import time
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_training.kvs import KVSCluster
from ml_training.coordinator import TrainingCoordinator, SafetyBoundChecker
from ml_training.secure_comm import create_mpc_network
from ml_training.node_failure_detector import NodeFailureDetector
import numpy as np


def test_failure_detection():
    """Test node failure detection with multiple nodes"""
    print("=" * 70)
    print("Testing Node Failure Detection")
    print("=" * 70)
    
    # Node configuration
    node_configs = {
        1: {'host': 'localhost', 'port': 8001},
        2: {'host': 'localhost', 'port': 8002},
        3: {'host': 'localhost', 'port': 8003},
    }
    
    # Test parameters
    t = 1  # Threshold
    n_nodes = 3
    
    print(f"\nNode configuration: {node_configs}")
    print(f"Threshold: {t}, Nodes: {n_nodes}")
    print("\nNote: This test requires nodes to be running on ports 8001, 8002, 8003")
    print("Start nodes using: python run_node.py --node-id 1 --n-nodes 3 (in separate terminals)")
    print("\nPress Enter when nodes are running...")
    input()
    
    # Test with node 1
    node_id = 1
    print(f"\nTesting from Node {node_id} perspective...")
    
    try:
        # Create network
        network = create_mpc_network(node_id, node_configs, port=8000 + node_id, use_tls=False)
        time.sleep(1)  # Wait for connections
        
        # Create failure detector
        detector = NodeFailureDetector(network, heartbeat_interval=2.0, failure_timeout=6.0)
        detector.start_monitoring()
        
        print(f"\nStarted failure detection on Node {node_id}")
        print(f"Initial active nodes: {detector.get_active_nodes()}")
        print(f"Initial n_active: {detector.get_n_active()}")
        
        # Monitor for 30 seconds
        print("\nMonitoring for 30 seconds...")
        print("Try stopping one of the other nodes to test failure detection")
        
        for i in range(30):
            active_nodes = detector.get_active_nodes()
            n_active = detector.get_n_active()
            print(f"[{i+1:2d}s] Active nodes: {sorted(active_nodes)}, n_active: {n_active}")
            time.sleep(1)
        
        print("\nTest completed")
        print(f"Final active nodes: {detector.get_active_nodes()}")
        print(f"Final n_active: {detector.get_n_active()}")
        
        detector.stop_monitoring()
        network.stop()
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


def test_safety_bound_with_failures():
    """Test safety bound checking with node failures"""
    print("\n" + "=" * 70)
    print("Testing Safety Bound with Node Failures")
    print("=" * 70)
    
    t = 1  # Threshold
    s = 1  # Adversarial limit
    n_nodes = 5
    
    safety_checker = SafetyBoundChecker(t, s)
    
    print(f"t={t}, s={s}, Safety bound: 2*(t+s-1) = {2*(t+s-1)}")
    print(f"\nSafety bound requires: 2*(t+s-1) < n_active")
    
    # Test different n_active values
    for n_active in range(1, 8):
        is_safe = safety_checker.check(n_active)
        max_packing = safety_checker.get_max_packing_factor(n_active)
        status = "✓ SAFE" if is_safe else "✗ VIOLATED"
        print(f"n_active={n_active}: {status}, max_packing_factor={max_packing}")
    
    print("\nWith n_nodes=5:")
    print("- All 5 nodes active: Safety bound satisfied")
    print("- 4 nodes active: Safety bound satisfied")
    print("- 3 nodes active: Safety bound satisfied")
    print("- 2 nodes active: Safety bound VIOLATED (2*(1+1-1)=2 >= 2)")
    print("- 1 node active: Safety bound VIOLATED")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Node Failure Detection Test Suite")
    print("=" * 70)
    
    # Test 1: Safety bound checking
    test_safety_bound_with_failures()
    
    # Test 2: Actual failure detection (requires running nodes)
    print("\n" + "=" * 70)
    print("To test actual failure detection:")
    print("1. Start 3 nodes in separate terminals:")
    print("   Terminal 1: python run_node.py --node-id 1 --n-nodes 3")
    print("   Terminal 2: python run_node.py --node-id 2 --n-nodes 3")
    print("   Terminal 3: python run_node.py --node-id 3 --n-nodes 3")
    print("2. Wait for them to connect")
    print("3. Run this test script")
    print("4. Stop one node and observe failure detection")
    print("=" * 70)
    
    # Uncomment to run actual failure detection test
    # test_failure_detection()

