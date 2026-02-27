"""
SENTRA packed/SIMD training demo (multi-node).

What this is:
- A small end-to-end multi-node demo that uses *Packed Shamir* to pack across the
  batch dimension (SIMD lanes) and evaluates a *public-weight* linear model.
- Uses the existing SecureMPCNetwork transport to exchange shares.

What this is NOT:
- A full packed secret-weight training implementation (that would require packed Beaver
  triples and packed degree-management for secret×secret multiplications).

How it works:
- Each mini-batch contains k lanes, where k <= n - t (max packing factor).
- For each input feature j, we pack the k sample values into one packed sharing
  (one polynomial), producing n shares (one per node).
- Nodes compute y_hat shares locally with public weights.
- Opener reconstructs per-lane y_hat, computes gradients in plaintext, updates
  public weights, and broadcasts updated weights to other nodes.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, Any, List, Tuple

import numpy as np

from ml_training.secret_sharing import Share, PackedShamirSecretSharing
from ml_training.secure_comm import create_mpc_network


def create_node_configs(n_nodes: int, base_port: int, host: str) -> Dict[int, Dict[str, Any]]:
    return {i: {"host": host, "port": int(base_port) + int(i)} for i in range(1, int(n_nodes) + 1)}


def _mod_signed(val: int, p: int) -> int:
    val = int(val) % int(p)
    return val - p if val > (p // 2) else val


def _open_packed_lanes(
    *,
    network,
    pss: PackedShamirSecretSharing,
    ctx: str,
    local_share: Share,
    n_nodes: int,
    t: int,
    k: int,
    timeout: float = 60.0,
) -> List[int]:
    """
    Open a single packed Share (same ctx across nodes) and return lane values.
    """
    # Broadcast our local y as a 1-element vector
    network.broadcast_vector(ctx, x=int(local_share.x), values=np.asarray([int(local_share.y) & 0xFFFFFFFF], dtype=np.uint32))

    opener = 1  # hardcoded: whoever calls this helper should be opener
    start = time.time()
    recv = None
    while time.time() - start < timeout:
        recv = network.channel.get_received_vector(ctx)
        # add our own share explicitly (loopback not guaranteed)
        recv[network.node_id] = {"x": int(local_share.x), "values": np.asarray([int(local_share.y) & 0xFFFFFFFF], dtype=np.uint32)}
        if len(recv) >= n_nodes:
            break
        time.sleep(0.01)
    if recv is None or len(recv) < n_nodes:
        raise RuntimeError(f"Timed out opening packed share ctx={ctx}")

    need = int(t) + int(k)
    chosen = sorted(recv.keys())[:need]
    shares = []
    for nid in chosen:
        y_u32 = int(np.asarray(recv[nid]["values"], dtype=np.uint32)[0])
        # IMPORTANT: use the sender-provided x-coordinate (may differ from node id in general)
        x_point = int(recv[nid].get("x", nid))
        shares.append(Share(x=int(x_point), y=int(y_u32) % pss.field_size, node_id=int(nid)))

    try:
        network.channel.clear_vector(ctx)
    except Exception:
        pass

    return pss.reconstruct_secrets(shares, k=k, t=t)


def main() -> int:
    DEMO_VERSION = "packed-linear-v2"
    parser = argparse.ArgumentParser(description="SENTRA packed/SIMD linear training demo")
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--n-nodes", type=int, default=3)
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--base-port", type=int, default=9200)
    parser.add_argument("--enable-network", action="store_true")
    parser.add_argument("--shared-seed", type=int, default=1337)
    parser.add_argument("--opener-node", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--d", type=int, default=3, help="Input dimension")
    parser.add_argument("--scale", type=int, default=1000, help="Fixed-point scale")
    args = parser.parse_args()

    if not args.enable_network:
        raise SystemExit("This demo requires --enable-network")
    if args.node_id < 1 or args.node_id > args.n_nodes:
        raise SystemExit("--node-id must be in [1..n-nodes]")
    if args.t < 0 or args.t >= args.n_nodes:
        raise SystemExit("--t must satisfy 0 <= t < n-nodes")

    # Use the same field as the rest of the multi-node code (fits in uint32 transport).
    field_size = 2**32 - 5
    p = int(field_size)

    node_configs = create_node_configs(args.n_nodes, args.base_port, args.host)
    network = create_mpc_network(
        node_id=args.node_id,
        node_configs=node_configs,
        port=node_configs[args.node_id]["port"],
        use_tls=False,
    )

    # Sync before demo starts
    network.barrier(tag=f"packed_train_start_seed_{args.shared_seed}", timeout=300.0)

    # Deterministic dataset generation across nodes
    # (Also seed Python's RNG because PackedShamirSecretSharing uses `random` internally.)
    random.seed(int(args.shared_seed))
    rng = np.random.default_rng(int(args.shared_seed))
    X = rng.normal(size=(int(args.n_samples), int(args.d))).astype(np.float64)
    true_w = rng.normal(size=(int(args.d),)).astype(np.float64)
    true_b = float(rng.normal())
    y = (X @ true_w + true_b).astype(np.float64)

    # Public weights we train (kept consistent via opener broadcast)
    w = rng.normal(size=(int(args.d),)).astype(np.float64) * 0.1
    b = 0.0

    # Packed Shamir helper
    pss = PackedShamirSecretSharing(field_size=field_size)
    k = min(2, pss.max_packing_factor(args.n_nodes, args.t))  # for n=3,t=1 => 2
    if k <= 0:
        raise SystemExit("No packing possible for given (n,t)")
    scale = int(args.scale)
    scale2 = int(args.scale) * int(args.scale)

    # Helper to broadcast updated weights from opener
    def _broadcast_weights(step_ctx: str):
        nonlocal w, b
        if args.node_id == args.opener_node:
            # Convention for this demo:
            # - w is encoded as w_int = round(w * scale)
            # - b is encoded as b_int = round(b * scale^2)
            vals = [int(round(float(v) * scale)) % p for v in list(w)] + [int(round(float(b) * scale2)) % p]
            payload = np.asarray(vals, dtype=np.uint32)
            network.broadcast_vector(step_ctx, x=int(args.opener_node), values=payload)
        else:
            start = time.time()
            while time.time() - start < 60.0:
                recv = network.channel.get_received_vector(step_ctx)
                if int(args.opener_node) in recv:
                    vals = np.asarray(recv[int(args.opener_node)]["values"], dtype=np.uint32).astype(np.int64)
                    try:
                        network.channel.clear_vector(step_ctx)
                    except Exception:
                        pass
                    signed = np.where(vals > (p // 2), vals - p, vals)
                    ww = signed[: int(args.d)] / float(scale)
                    bb = float(signed[int(args.d)] / float(scale2))
                    w = ww.astype(np.float64)
                    b = bb
                    return
                time.sleep(0.01)
            raise RuntimeError("Timed out waiting for weights broadcast")

    # Initial weight sync
    _broadcast_weights(step_ctx=f"packed_train_weights_init_seed_{args.shared_seed}")

    if args.node_id == args.opener_node:
        print(f"\n[PACKED SENTRA LINEAR TRAINING DEMO] ({DEMO_VERSION})")
        print(f"n={args.n_nodes}, t={args.t}, k={k} lanes, field={field_size}, d={args.d}")
        print(f"epochs={args.epochs}, lr={args.lr}, n_samples={args.n_samples}")
        print(f"[debug] init w={np.round(w,6).tolist()}  b={float(b):.6f}")

    # Train
    for epoch in range(int(args.epochs)):
        # Keep nodes aligned
        network.barrier(tag=f"packed_train_epoch_{epoch}_seed_{args.shared_seed}", timeout=300.0)

        total_loss = 0.0
        n_seen = 0

        for start_idx in range(0, int(args.n_samples), k):
            batch_idx = start_idx // k
            xs = X[start_idx : start_idx + k]
            ys = y[start_idx : start_idx + k]
            lane_count = int(xs.shape[0])
            if lane_count == 0:
                continue

            # Pack inputs feature-wise into shares for this node
            packed_x_by_feature: List[Share] = []
            for j in range(int(args.d)):
                secrets = [int(round(float(xs[i, j]) * args.scale)) % p for i in range(lane_count)]
                # If last batch shorter than k, pad with zeros (still reconstruct k lanes; opener ignores extras)
                if lane_count < k:
                    secrets = secrets + [0] * (k - lane_count)
                # Deterministic "PRSS-like" masking for the demo:
                # ensure every node generates identical packed shares for (epoch,batch,feature),
                # without relying on global RNG streams staying in sync.
                seed = (int(args.shared_seed) * 1000003 + int(epoch) * 10007 + int(batch_idx) * 97 + int(j)) & 0xFFFFFFFF
                random.seed(int(seed))
                shares_all_nodes = pss.share_secrets(secrets, args.n_nodes, args.t)
                packed_x_by_feature.append(shares_all_nodes[int(args.node_id) - 1])

            # Local packed prediction share: y_hat = w @ x + b (all fixed-point ints mod p)
            # Fixed-point convention (no in-field division):
            # - x_int = round(x * scale)
            # - w_int = round(w * scale)
            # - b_int = round(b * scale^2)
            # => yhat_int = sum_j (w_int[j] * x_int[j]) + b_int   (this is scaled by scale^2)
            yhat_y = 0
            for j in range(int(args.d)):
                wj = int(round(float(w[j]) * scale)) % p
                yhat_y = (yhat_y + (wj * int(packed_x_by_feature[j].y)) % p) % p
            bj = int(round(float(b) * scale2)) % p
            yhat_y = (yhat_y + bj) % p
            yhat_share = Share(x=int(args.node_id), y=int(yhat_y), node_id=int(args.node_id))

            # Opener reconstructs lane predictions and updates weights, then broadcasts weights
            step_seed = args.shared_seed
            open_ctx = f"packed_train_open_e{epoch}_b{batch_idx}_seed{step_seed}"
            if args.node_id == args.opener_node:
                lanes = _open_packed_lanes(
                    network=network,
                    pss=pss,
                    ctx=open_ctx,
                    local_share=yhat_share,
                    n_nodes=args.n_nodes,
                    t=args.t,
                    k=k,
                )
                # Convert to float
                lanes_signed = [_mod_signed(v, p) / float(scale2) for v in lanes[:k]]

                # Self-check (first batch): verify opened lanes match plaintext computation.
                if epoch == 0 and batch_idx == 0:
                    w_ints = [int(round(float(wj) * scale)) % p for wj in list(w)]
                    b_int = int(round(float(b) * scale2)) % p
                    expected_ints: List[int] = []
                    for i_lane in range(k):
                        acc = b_int
                        for j in range(int(args.d)):
                            x_int = int(round(float(xs[min(i_lane, lane_count - 1), j]) * scale)) % p if i_lane < lane_count else 0
                            acc = (acc + (w_ints[j] * x_int) % p) % p
                        expected_ints.append(int(acc) % p)
                    expected_f = [_mod_signed(v, p) / float(scale2) for v in expected_ints]
                    plain_preds = [float(xs[i] @ w + b) for i in range(lane_count)] + ([0.0] * (k - lane_count))
                    max_err = max(abs(a - b) for a, b in zip(lanes_signed, expected_f))
                    print(f"[debug] first-batch x={xs[:lane_count].tolist()} y={ys[:lane_count].tolist()}")
                    print(f"[debug] first-batch opened={lanes_signed} expected_fixed={expected_f} expected_plain={plain_preds}")
                    if max_err > 1e-2:
                        print("\n[ERROR] Packed open mismatch on first batch.")
                        print("opened=", lanes_signed)
                        print("expected=", expected_f)
                        print("max_err=", max_err)
                        raise RuntimeError("Packed sharing/arithmetic mismatch; aborting")
                # Compute loss/grad for real lanes only
                for i_lane in range(lane_count):
                    err = float(lanes_signed[i_lane]) - float(ys[i_lane])
                    total_loss += 0.5 * err * err
                    n_seen += 1

                # SGD update on opener (public weights)
                # grad_w = sum_i err_i * x_i ; grad_b = sum_i err_i
                grad_w = np.zeros((int(args.d),), dtype=np.float64)
                grad_b = 0.0
                for i_lane in range(lane_count):
                    err = float(lanes_signed[i_lane]) - float(ys[i_lane])
                    grad_w += err * xs[i_lane]
                    grad_b += err
                # average
                grad_w /= float(lane_count)
                grad_b /= float(lane_count)
                w = w - float(args.lr) * grad_w
                b = b - float(args.lr) * grad_b

            else:
                # Non-opener participates in the open by broadcasting, then waits for weight sync
                network.broadcast_vector(open_ctx, x=int(args.node_id), values=np.asarray([int(yhat_share.y) & 0xFFFFFFFF], dtype=np.uint32))
                # Clear any received vectors for this context to avoid unbounded growth
                try:
                    network.channel.clear_vector(open_ctx)
                except Exception:
                    pass

            # Sync weights after each step
            _broadcast_weights(step_ctx=f"packed_train_weights_e{epoch}_b{batch_idx}_seed{step_seed}")

        if args.node_id == args.opener_node:
            avg = (total_loss / max(1, n_seen))
            print(f"[epoch {epoch+1}/{args.epochs}] avg_loss={avg:.6f}  w={np.round(w,4).tolist()}  b={b:.4f}")

    network.barrier(tag=f"packed_train_done_seed_{args.shared_seed}", timeout=300.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

