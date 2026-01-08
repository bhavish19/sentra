"""
Universal node runner for SENTRA multi-node training
Usage: python run_node.py --node-id 1
"""

import argparse
import sys
import os
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training import SentraTrainingPipeline, DPSGDConfig
import numpy as np


def create_node_configs(n_nodes: int, base_port: int = 8000, host: str = 'localhost'):
    """Create node configurations"""
    return {
        i: {'host': host, 'port': base_port + i}
        for i in range(1, n_nodes + 1)
    }

def read_node_configs(configFile:str):
   """Read node configurations from a yaml file"""
   config=yaml.load(open(configFile,'r'),yaml.Loader)
   return config

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
    parser.add_argument('--use-dp-sgd', action='store_true',
                       help='Enable DP-SGD')
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
    parser.add_argument('--config',type=str,help="YAML node configuartion file")
    parser.add_argument('--auto-close', action='store_true',
                       help='Do not wait for key press to exit the process')
    
    args = parser.parse_args()
    
    # Validate node_id
    if args.node_id < 1 or args.node_id > args.n_nodes:
        print(f"Error: node-id must be between 1 and {args.n_nodes}")
        sys.exit(1)
    
    # Create or read node configurations
    if(args.config is None):
        node_configs = create_node_configs(args.n_nodes, args.base_port, args.host)
    else:
        node_configs=read_node_configs(args.config)
    print("=" * 70)
    print(f"SENTRA Node {args.node_id} Starting")
    print("=" * 70)
    print(f"Node ID: {args.node_id}")
    print(f"Total nodes: {args.n_nodes}")
    print(f"Privacy threshold (t): {args.t}")
    print(f"Adversarial limit (s): {args.s}")
    print(f"Safety check: 2*(t+s) = {2*(args.t+args.s)} < n_nodes = {args.n_nodes} {'✓' if 2*(args.t+args.s) < args.n_nodes else '✗'}")
    print(f"Network: Enabled")
    print(f"DP-SGD: {'Enabled' if args.use_dp_sgd else 'Disabled'}")
    print(f"Node configs: {node_configs}")
    print("=" * 70)
    
    # Create DP-SGD config if needed
    dp_config = None
    if args.use_dp_sgd:
        dp_config = DPSGDConfig(
            clip_norm=1.0,
            noise_multiplier=1.0,
            delta=1e-5,
            learning_rate=args.learning_rate
        )
    
    # Create pipeline
    try:
        pipeline = SentraTrainingPipeline(
            n_nodes=args.n_nodes,
            t=args.t,
            s=args.s,
            node_id=args.node_id,
            node_configs=node_configs,
            enable_network=True,
            use_dp_sgd=args.use_dp_sgd,
            dp_config=dp_config,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs
        )
        
        # Create dataset (same seed for all nodes in testing)
        # In production, each node would have its own data partition
        np.random.seed(42)  # Fixed seed for reproducibility
        dataset = [np.random.randn(10) for _ in range(100)]
        labels = [np.random.randn(1) for _ in range(100)]
        
        weight_shapes = [(10, 4), (4, 1)]
        
        print(f"\nNode {args.node_id}: Starting training...")
        print(f"Dataset: {len(dataset)} samples, {len(dataset[0])} features")
        print(f"Model: {weight_shapes}")
        print("Multi-Node Mode: ENABLED")
        print("-" * 70)
        
        # Train
        pipeline.train(dataset, labels, weight_shapes)
        
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
        
        # Keep window open
        if(not args.auto_close):
            print("\nPress Enter to close this window...")
            try:
                input()
            except:
                pass
        
    except KeyboardInterrupt:
        print(f"\n\nNode {args.node_id}: Interrupted by user")
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
        if(not args.auto_close):
            print("\nPress Enter to close this window...")
            try:
                input()
            except:
                pass
        sys.exit(1)


if __name__ == '__main__':
    main()

