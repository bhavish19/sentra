"""
Start multi-node LeNet-5 training from a single command.

This launches N separate processes (one per node_id) running:
  run_lenet5_training_full.py --enable-network ...

Windows: opens one new console window per node.
Linux/macOS: runs processes in background (same terminal).
"""

import argparse
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch all LeNet-5 MPC nodes (one process per node)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-nodes", type=int, default=3, help="Total nodes to start")
    parser.add_argument("--base-port", type=int, default=8000, help="Base port (node i uses base-port+i)")
    parser.add_argument("--host", type=str, default="localhost", help="Host for all nodes")
    parser.add_argument("--t", type=int, default=1, help="Shamir threshold t")
    parser.add_argument("--shared-seed", type=int, default=1337, help="Shared RNG seed for all node processes")
    parser.add_argument("--stagger-seconds", type=float, default=0.8, help="Delay between starting nodes")

    # Everything after "--" is forwarded verbatim to run_lenet5_training_full.py
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to run_lenet5_training_full.py (prefix with --)",
    )

    args = parser.parse_args()

    if args.n_nodes < 2:
        raise SystemExit("--n-nodes must be >= 2 for multi-node")
    if args.t < 0:
        raise SystemExit("--t must be >= 0")
    if args.t >= args.n_nodes:
        raise SystemExit("--t must be < --n-nodes")

    # If user provided a leading "--", argparse keeps it; if they didn't, it's fine.
    forwarded = list(args.train_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    print("=" * 70)
    print("Starting LeNet-5 Multi-Node Training (one process per node)")
    print("=" * 70)
    print(f"Nodes: {args.n_nodes}")
    print(f"Host: {args.host}")
    print(f"Base port: {args.base_port}")
    print(f"t: {args.t}")
    print(f"shared-seed: {args.shared_seed}")
    if forwarded:
        print(f"Forwarded train args: {' '.join(forwarded)}")
    print("=" * 70)

    for node_id in range(1, args.n_nodes + 1):
        cmd = [
            sys.executable,
            "run_lenet5_training_full.py",
            "--node-id",
            str(node_id),
            "--n-nodes",
            str(args.n_nodes),
            "--base-port",
            str(args.base_port),
            "--host",
            args.host,
            "--t",
            str(args.t),
            "--enable-network",
            "--shared-seed",
            str(args.shared_seed),
        ] + forwarded

        print(f"Starting node {node_id}/{args.n_nodes}...")

        if sys.platform == "win32":
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(cmd)

        time.sleep(args.stagger_seconds)

    print("All nodes launched.")
    if sys.platform == "win32":
        print("Close the spawned consoles to stop nodes.")
    else:
        print("Use your OS process manager to stop nodes.")


if __name__ == "__main__":
    main()

