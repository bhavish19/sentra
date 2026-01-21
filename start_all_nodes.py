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
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Starting SENTRA Multi-Node Training")
    print("=" * 70)
    print(f"Total nodes: {args.n_nodes}")
    print(f"Base port: {args.base_port}")
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
            '--base-port', str(args.base_port)
        ]
        
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

