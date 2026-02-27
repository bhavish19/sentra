"""
Packed-Shamir / SIMD demo (multi-node).

This script starts an MPC network node and runs a tiny packed-SIMD matvec demo:
- Pack across the batch dimension (k lanes per packed share)
- Evaluate y = W @ x for PUBLIC W and packed secret-shared x
- Opener node reconstructs per-lane outputs and checks against plaintext

NOTE:
- For testing convenience, all nodes seed RNG with the same shared seed so they
  generate consistent packed shares locally and select their own share.
  In a real deployment, shares would be distributed, not regenerated.
"""

from __future__ import annotations

import argparse
import time
import random
from typing import Dict, Any, List, Tuple

import numpy as np

from ml_training.secret_sharing import Share, PackedShamirSecretSharing
from ml_training.mpc_engine import PackedMPCEngine
from ml_training.secure_comm import create_mpc_network


def create_node_configs(n_nodes: int, base_port: int, host: str) -> Dict[int, Dict[str, Any]]:
    return {i: {"host": host, "port": int(base_port) + int(i)} for i in range(1, int(n_nodes) + 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Packed-Shamir SIMD demo (public-weight matvec)")
    parser.add_argument("--node-id", type=int, required=True, help="This node's ID (1..n)")
    parser.add_argument("--n-nodes", type=int, default=3, help="Total nodes (default: 3)")
    parser.add_argument("--t", type=int, default=1, help="Privacy threshold (default: 1)")
    parser.add_argument("--host", type=str, default="localhost", help="Host (default: localhost)")
    parser.add_argument("--base-port", type=int, default=9100, help="Base port (default: 9100)")
    parser.add_argument("--enable-network", action="store_true", help="Enable MPC networking (required)")
    parser.add_argument("--shared-seed", type=int, default=1337, help="Shared RNG seed (default: 1337)")
    parser.add_argument("--opener-node", type=int, default=1, help="Node that reconstructs and prints (default: 1)")
    args = parser.parse_args()

    if not args.enable_network:
        raise SystemExit("This demo requires --enable-network")
    if args.node_id < 1 or args.node_id > args.n_nodes:
        raise SystemExit("--node-id must be in [1..n-nodes]")
    if args.t < 0 or args.t >= args.n_nodes:
        raise SystemExit("--t must satisfy 0 <= t < n-nodes")

    # Use the same field as the training code path uses (fits in uint32 transport).
    field_size = 2**32 - 5

    node_configs = create_node_configs(args.n_nodes, args.base_port, args.host)
    network = create_mpc_network(
        node_id=args.node_id,
        node_configs=node_configs,
        port=node_configs[args.node_id]["port"],
        use_tls=False,
    )

    # Barrier: ensure all nodes are connected and ready
    network.barrier(tag=f"packed_demo_start_seed_{args.shared_seed}", timeout=300.0)

    # Deterministic share generation across nodes (testing convenience)
    random.seed(args.shared_seed)
    np.random.seed(args.shared_seed)

    engine = PackedMPCEngine(n_nodes=args.n_nodes, t=args.t, field_size=field_size)
    pss: PackedShamirSecretSharing = engine.pss

    # With (n=3,t=1), k_max=2 lanes.
    k_max = pss.max_packing_factor(args.n_nodes, args.t)
    k = min(2, k_max)
    if k <= 0:
        raise SystemExit("No packing possible for given (n,t)")

    # Tiny batch of k vectors (dim=3), all public here.
    batch: List[List[int]] = [
        [1, 2, 3],
        [4, 5, 6],
    ][:k]

    # Public weights W (out_dim=2, in_dim=3)
    W: List[List[int]] = [
        [2, 0, 1],
        [1, 1, 1],
    ]

    packed_inputs = engine.pack_batch_vector(batch, node_id=args.node_id, packing_factor=k)
    packed_outputs = engine.packed_matvec_public_weights(packed_inputs, W, node_id=args.node_id)

    out_dim = len(W)
    n_chunks = len(packed_outputs[0]) if out_dim > 0 else 0

    # Flatten our packed output shares into one vector to broadcast
    local_vec: List[int] = []
    for i in range(out_dim):
        for c in range(n_chunks):
            local_vec.append(int(packed_outputs[i][c].y) % field_size)

    ctx = f"packed_demo_outputs_seed_{args.shared_seed}"
    x_point = int(args.node_id)
    network.broadcast_vector(ctx, x=x_point, values=np.asarray(local_vec, dtype=np.uint32))

    if args.node_id != args.opener_node:
        # Non-opener nodes just wait for completion and exit.
        network.barrier(tag=f"packed_demo_done_seed_{args.shared_seed}", timeout=300.0)
        try:
            network.channel.clear_vector(ctx)
        except Exception:
            pass
        return 0

    # Opener: collect vectors from all nodes
    start = time.time()
    received: Dict[int, Dict[str, Any]] = {}
    while time.time() - start < 120.0:
        received = network.channel.get_received_vector(ctx)
        if len(received) >= args.n_nodes - 1:
            break
        time.sleep(0.01)

    # Add our own vector explicitly (loopback not guaranteed)
    received[args.node_id] = {"x": x_point, "values": np.asarray(local_vec, dtype=np.uint32)}

    if len(received) < args.n_nodes:
        raise RuntimeError(f"Timed out waiting for all nodes. Got {sorted(received.keys())}")

    # Reconstruct per neuron, per chunk => k lane outputs
    need = args.t + k
    node_ids = sorted(received.keys())[: args.n_nodes]

    # Validate vector lengths match
    L = len(local_vec)
    for nid in node_ids:
        if len(received[nid]["values"]) != L:
            raise RuntimeError("Mismatched vector lengths across nodes")

    reconstructed: List[List[int]] = [[] for _ in range(out_dim)]
    for i in range(out_dim):
        for c in range(n_chunks):
            idx = i * n_chunks + c
            shares_chunk: List[Share] = []
            for nid in node_ids[:need]:
                y_u32 = int(np.asarray(received[nid]["values"], dtype=np.uint32)[idx])
                shares_chunk.append(Share(x=int(nid), y=y_u32 % field_size, node_id=int(nid)))
            lanes = pss.reconstruct_secrets(shares_chunk, k=k, t=args.t)
            reconstructed[i].extend(lanes)

    # Expected plaintext outputs
    expected: List[List[int]] = []
    for row in W:
        expected_row: List[int] = []
        for x in batch:
            expected_row.append(sum(int(w) * int(v) for w, v in zip(row, x)) % field_size)
        expected.append(expected_row)

    ok = True
    print("\n[PACKED SIMD DEMO]")
    print(f"n={args.n_nodes}, t={args.t}, k={k} lanes, field={field_size}")
    print(f"batch={batch}")
    print(f"W={W}")
    for i in range(out_dim):
        got = reconstructed[i][:k]
        exp = expected[i]
        match = (got == exp)
        ok = ok and match
        print(f"neuron {i}: got={got} expected={exp}  {'OK' if match else 'FAIL'}")

    print("RESULT:", "PASS" if ok else "FAIL")

    network.barrier(tag=f"packed_demo_done_seed_{args.shared_seed}", timeout=300.0)
    try:
        network.channel.clear_vector(ctx)
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

