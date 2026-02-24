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
    parser.add_argument('--train-mode', choices=['secure', 'hybrid'], default='secure',
                       help='Training mode passed to all nodes (default: secure)')
    parser.add_argument('--batched', action='store_true',
                       help='Run the batched version of the MNIST MLP (much faster)')
    parser.add_argument('--loss-mode', choices=['mse', 'softmax'], default='softmax',
                       help='Output gradient mode for batched MNIST secure training (default: softmax)')
    parser.add_argument('--scale-factor', type=int, default=2**20,
                       help='Fixed-point scale for batched secure MNIST (default: 2^20 = 1048576)')
    parser.add_argument('--field-size', type=int, default=2**32 - 5,
                       help='Finite field modulus for batched secure MNIST (default: 2^32-5)')
    parser.add_argument('--softmax-temperature', type=float, default=1.0,
                       help='Softmax temperature for batched secure MNIST (default: 1.0)')
    parser.add_argument('--grad-clip', type=float, default=2.0,
                       help='Gradient clip bound for batched secure MNIST (default: 2.0)')
    parser.add_argument('--logit-clip', type=float, default=8.0,
                       help='Logit clip bound before secure exp approximation (default: 8.0)')
    parser.add_argument('--exp-approx', choices=['taylor5', 'pade22'], default='pade22',
                       help='Secure exp approximation used in softmax (default: pade22)')
    parser.add_argument('--softmax-grad-mode', choices=['secure_approx', 'opened_exact'], default='secure_approx',
                       help='Softmax CE gradient path in batched mode (default: secure_approx)')
    parser.add_argument('--explode-logit-threshold', type=float, default=10.0,
                       help='Logit explosion threshold for batched secure MNIST (default: 10.0)')
    parser.add_argument('--loss-growth-threshold', type=float, default=5.0,
                       help='Loss growth threshold for instability checks (default: 5.0)')
    parser.add_argument('--grad-norm-threshold', type=float, default=5.0,
                       help='Estimated gradient norm threshold for instability checks (default: 5.0)')
    parser.add_argument('--no-abort-on-instability', action='store_true',
                       help='Do not abort batched secure training when instability is detected')
    parser.add_argument('--debug-numerics', action='store_true',
                       help='Enable numeric probes (updates/logits/dz2) in batched secure mode')
    parser.add_argument('--debug-division', action='store_true',
                       help='Enable secure division debug summaries in batched secure mode')
    
    args = parser.parse_args()
    
    if args.batched:
        args.dataset = 'mnist'
        
    print("=" * 70)
    print("Starting SENTRA Multi-Node Training")
    print("=" * 70)
    print(f"Total nodes: {args.n_nodes}")
    print(f"Base port: {args.base_port}")
    print(f"Host: {args.host}")
    print(f"Training: epochs={args.num_epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    if args.batched:
        print(f"Batched secure config: field={args.field_size}, scale={args.scale_factor}, temp={args.softmax_temperature}, grad_clip={args.grad_clip}, logit_clip={args.logit_clip}, exp={args.exp_approx}, grad_mode={args.softmax_grad_mode}, loss_mode={args.loss_mode}")
        print(f"Instability thresholds: logit={args.explode_logit_threshold}, loss_growth={args.loss_growth_threshold}, grad_norm={args.grad_norm_threshold}, abort={not args.no_abort_on_instability}")
        print(f"Debug flags: numerics={args.debug_numerics}, division={args.debug_division}")
    print(f"Thresholds: t={args.t}, s={args.s}")
    print(f"Dataset: {args.dataset}")
    print("=" * 70)
    print()
    
    processes = []
    
    # Build command for each node
    for node_id in range(1, args.n_nodes + 1):
        if args.dataset == 'mnist':
            script_name = 'run_mnist_batched_secure.py' if args.batched else 'run_mnist_secure.py'
            cmd = [
                sys.executable,
                '-u',
                script_name,
                '--node-id', str(node_id),
                '--n-nodes', str(args.n_nodes),
                '--base-port', str(args.base_port),
                '--host', str(args.host),
                '--batch-size', str(args.batch_size),
                '--num-epochs', str(args.num_epochs),
                '--learning-rate', str(args.learning_rate),
                '--t', str(args.t),
                '--seed', str(args.seed),
                '--mnist-samples', str(args.mnist_samples),
                '--enable-network',
            ]
            if args.batched:
                cmd.extend([
                    '--loss-mode', str(args.loss_mode),
                    '--scale-factor', str(args.scale_factor),
                    '--field-size', str(args.field_size),
                    '--softmax-temperature', str(args.softmax_temperature),
                    '--grad-clip', str(args.grad_clip),
                    '--logit-clip', str(args.logit_clip),
                    '--exp-approx', str(args.exp_approx),
                    '--softmax-grad-mode', str(args.softmax_grad_mode),
                    '--explode-logit-threshold', str(args.explode_logit_threshold),
                    '--loss-growth-threshold', str(args.loss_growth_threshold),
                    '--grad-norm-threshold', str(args.grad_norm_threshold),
                ])
                if args.no_abort_on_instability:
                    cmd.append('--no-abort-on-instability')
                if args.debug_numerics:
                    cmd.append('--debug-numerics')
                if args.debug_division:
                    cmd.append('--debug-division')
        else:
            cmd = [
                sys.executable,
                '-u',
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
                '--train-mode', str(args.train_mode),
                '--no-wait',
            ]
            if args.dataset == 'mnist': # This block was originally outside the else, but now it's inside.
                                        # It will only be reached if args.dataset is 'mnist' AND it's in the else block, which is contradictory.
                                        # Given the new structure, this 'if args.dataset == 'mnist':' condition is redundant and will never be true here.
                                        # The original intent was likely to add these arguments to run_node.py when dataset is mnist,
                                        # but now run_mnist_secure.py handles mnist.
                                        # I will remove this block as it's now unreachable and incorrect for the new logic.
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
        
        # Check if we are running under WSL or pure Linux/Windows
        import platform
        import os
        is_wsl = 'microsoft' in platform.release().lower() or 'wsl' in platform.release().lower()
        
        if sys.platform == 'win32':
            # Run in a new command prompt window on Windows
            subprocess.Popen(['start', 'cmd', '/k'] + cmd, shell=True)
        elif is_wsl:
            # Under WSL, we can call out to cmd.exe to launch a new wsl.exe window
            # Pass command to bash and append a read prompt so the window stays open
            cmd_str = " ".join(f"'{arg}'" if ' ' in arg else arg for arg in cmd)
            bash_cmd = f"{cmd_str}; echo ''; read -p 'Process complete. Press enter to close...'"
            wsl_cmd = ['cmd.exe', '/c', 'start', 'wsl.exe', '-e', 'bash', '-c', bash_cmd]
            subprocess.Popen(wsl_cmd)
        else:
            # Fallback for pure Linux (e.g. gnome-terminal, xterm). If none exists, run in background
            try:
                subprocess.Popen(['xterm', '-e'] + cmd)
            except Exception:
                # If xterm fails, fall back to logs
                f = open(f"node_{node_id}.log", "w")
                subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        time.sleep(1)  # Stagger starts
    
    print()
    print("All nodes started in separate windows.")
    print("Close each window to stop the corresponding node, or watch the live outputs!")
    print()
    print("To stop all nodes, close each window or press Ctrl+C.")


if __name__ == '__main__':
    main()
