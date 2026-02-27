"""
SENTRA packed/SIMD linear training demo with *secret-shared weights* (multi-node).

This extends the packed-batch demo by:
- Keeping weights/bias as Shamir shares across nodes ("weights at rest are secret-shared")
- Computing predictions from secret-shared weights and secret-shared packed inputs using
  secure Beaver multiplication on shares (no opener reconstruction of weights).
- Opener computes loss/gradients from opened predictions (demo convenience) and sends
  *secret-shared deltas* so every node updates its local weight shares directly.

Important:
- This is a demo/training harness, not a production MPC protocol. It opens predictions
  on the opener node to compute gradients.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, Any, List, Tuple

import numpy as np

from ml_training.secret_sharing import Share, ShamirSecretSharing, PackedShamirSecretSharing
from ml_training.secure_comm import create_mpc_network
from ml_training.reconstruction import create_reconstruction_manager
from ml_training.beaver_triples import (
    BeaverTripleGenerator,
    BeaverTriplePool,
    SecureMultiplier,
    BeaverTripleDealerService,
)
from ml_training.packed_mpc_ops import PackedMPCOps


def create_node_configs(n_nodes: int, base_port: int, host: str) -> Dict[int, Dict[str, Any]]:
    return {i: {"host": host, "port": int(base_port) + int(i)} for i in range(1, int(n_nodes) + 1)}


def _mod_signed(val: int, p: int) -> int:
    val = int(val) % int(p)
    return val - p if val > (p // 2) else val


def _wait_vector_from(
    *,
    network,
    ctx: str,
    sender_id: int,
    timeout: float = 60.0,
) -> np.ndarray:
    start = time.time()
    while time.time() - start < timeout:
        recv = network.channel.get_received_vector(ctx)
        if int(sender_id) in recv:
            arr = np.asarray(recv[int(sender_id)]["values"], dtype=np.uint32)
            try:
                network.channel.clear_vector(ctx)
            except Exception:
                pass
            return arr
        time.sleep(0.01)
    raise RuntimeError(f"Timed out waiting for vector ctx={ctx} from node {sender_id}")


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
    Assumes every node broadcasted its packed share under `ctx`.
    """
    start = time.time()
    recv = None
    while time.time() - start < timeout:
        recv = network.channel.get_received_vector(ctx)
        # add our own share explicitly (loopback not guaranteed)
        recv[network.node_id] = {"x": int(local_share.x), "values": np.asarray([int(local_share.y) & 0xFFFFFFFF], dtype=np.uint32)}
        if len(recv) >= int(n_nodes):
            break
        time.sleep(0.01)
    if recv is None or len(recv) < int(n_nodes):
        raise RuntimeError(f"Timed out opening packed share ctx={ctx}")

    need = int(t) + int(k)
    chosen = sorted(recv.keys())[:need]
    shares: List[Share] = []
    for nid in chosen:
        y_u32 = int(np.asarray(recv[nid]["values"], dtype=np.uint32)[0])
        x_point = int(recv[nid].get("x", nid))
        shares.append(Share(x=int(x_point), y=int(y_u32) % int(pss.field_size), node_id=int(nid)))

    try:
        network.channel.clear_vector(ctx)
    except Exception:
        pass

    return pss.reconstruct_secrets(shares, k=int(k), t=int(t))


def main() -> int:
    DEMO_VERSION = "packed-linear-secretw-v2"
    parser = argparse.ArgumentParser(description="Packed/SIMD linear training with secret-shared weights (demo)")
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--n-nodes", type=int, default=3)
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--base-port", type=int, default=9300)
    parser.add_argument("--enable-network", action="store_true")
    parser.add_argument("--shared-seed", type=int, default=1337)
    parser.add_argument("--opener-node", type=int, default=1)
    parser.add_argument("--dealer-node", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--scale", type=int, default=1000)
    args = parser.parse_args()

    if not args.enable_network:
        raise SystemExit("This demo requires --enable-network")
    if args.node_id < 1 or args.node_id > args.n_nodes:
        raise SystemExit("--node-id must be in [1..n-nodes]")
    if args.t < 0 or args.t >= args.n_nodes:
        raise SystemExit("--t must satisfy 0 <= t < n-nodes")

    field_size = 2**32 - 5
    p = int(field_size)
    scale = int(args.scale)
    scale2 = int(args.scale) * int(args.scale)

    node_configs = create_node_configs(args.n_nodes, args.base_port, args.host)
    network = create_mpc_network(
        node_id=args.node_id,
        node_configs=node_configs,
        port=node_configs[args.node_id]["port"],
        use_tls=False,
    )
    recon = create_reconstruction_manager(network, int(args.t), field_size=field_size)

    # Dealer-backed triples (avoid PRSS; keep triple secrets on dealer)
    if args.node_id == int(args.dealer_node):
        dealer = BeaverTripleDealerService(
            network=recon.network,
            dealer_node_id=int(args.dealer_node),
            n_nodes=int(args.n_nodes),
            t=int(args.t),
            field_size=int(field_size),
        )
        dealer.register()

    network.barrier(tag=f"{DEMO_VERSION}_start_seed_{args.shared_seed}", timeout=300.0)
    network.barrier(tag=f"{DEMO_VERSION}_dealer_ready_seed_{args.shared_seed}", timeout=300.0)

    # Deterministic dataset (public for this demo)
    rng = np.random.default_rng(int(args.shared_seed))
    X = rng.normal(size=(int(args.n_samples), int(args.d))).astype(np.float64)
    true_w = rng.normal(size=(int(args.d),)).astype(np.float64)
    true_b = float(rng.normal())
    y = (X @ true_w + true_b).astype(np.float64)

    pss = PackedShamirSecretSharing(field_size=field_size)
    k = min(2, pss.max_packing_factor(args.n_nodes, args.t))  # n=3,t=1 => 2 lanes
    if k <= 0:
        raise SystemExit("No packing possible for given (n,t)")

    shamir = ShamirSecretSharing(field_size=field_size)
    ops = PackedMPCOps(network=recon.network, n_nodes=int(args.n_nodes), t=int(args.t), field_size=int(field_size))

    triple_gen = BeaverTripleGenerator(field_size)
    triple_pool = BeaverTriplePool(triple_gen, initial_size=5000)
    multiplier = SecureMultiplier(
        triple_pool,
        n_nodes=int(args.n_nodes),
        t=int(args.t),
        field_size=int(field_size),
        reconstruction_manager=recon,
        triple_dealer_id=int(args.dealer_node),
    )

    # Local storage of THIS NODE's weight shares (w_int at scale, b_int at scale^2)
    w_share_y = np.zeros((int(args.d),), dtype=np.uint32)
    b_share_y = np.uint32(0)

    def set_local_weight_shares_from_payload(payload_u32: np.ndarray):
        nonlocal w_share_y, b_share_y
        payload_u32 = np.asarray(payload_u32, dtype=np.uint32).reshape(-1)
        if payload_u32.size != int(args.d) + 1:
            raise ValueError("bad weight payload size")
        w_share_y = payload_u32[: int(args.d)].copy()
        b_share_y = np.uint32(payload_u32[int(args.d)])

    def apply_local_weight_delta_from_payload(payload_u32: np.ndarray):
        """
        payload_u32: per-node Shamir share y-values for delta_w (scale) and delta_b (scale^2)
        Each node updates:
            w_share := w_share - delta_w_share
            b_share := b_share - delta_b_share
        """
        nonlocal w_share_y, b_share_y
        payload_u32 = np.asarray(payload_u32, dtype=np.uint32).reshape(-1)
        if payload_u32.size != int(args.d) + 1:
            raise ValueError("bad delta payload size")
        for j in range(int(args.d)):
            w_share_y[j] = np.uint32((int(w_share_y[j]) - int(payload_u32[j])) % p)
        b_share_y = np.uint32((int(b_share_y) - int(payload_u32[int(args.d)])) % p)

    # Init weights (opener samples small floats, then Shamir-shares ints to all nodes)
    if args.node_id == args.opener_node:
        w0 = rng.normal(size=(int(args.d),)).astype(np.float64) * 0.1
        b0 = 0.0
        w0_int = [int(round(float(v) * scale)) % p for v in w0.tolist()]
        b0_int = int(round(float(b0) * scale2)) % p

        # Share each scalar weight (degree t)
        per_node = {nid: [] for nid in range(1, int(args.n_nodes) + 1)}
        for j in range(int(args.d)):
            shares = shamir.share(w0_int[j], int(args.n_nodes), int(args.t))
            for s in shares:
                per_node[int(s.node_id)].append(int(s.y) & 0xFFFFFFFF)
        shares_b = shamir.share(b0_int, int(args.n_nodes), int(args.t))
        for s in shares_b:
            per_node[int(s.node_id)].append(int(s.y) & 0xFFFFFFFF)

        ctx = f"{DEMO_VERSION}_weights_init_seed_{args.shared_seed}"
        for nid in range(1, int(args.n_nodes) + 1):
            vec = np.asarray(per_node[nid], dtype=np.uint32)
            if nid == int(args.opener_node):
                set_local_weight_shares_from_payload(vec)
            else:
                network.channel.send_vector(int(nid), ctx, x=int(nid), values=vec)
    else:
        ctx = f"{DEMO_VERSION}_weights_init_seed_{args.shared_seed}"
        vec = _wait_vector_from(network=network, ctx=ctx, sender_id=int(args.opener_node), timeout=60.0)
        set_local_weight_shares_from_payload(vec)

    network.barrier(tag=f"{DEMO_VERSION}_weights_init_done_seed_{args.shared_seed}", timeout=300.0)

    if args.node_id == args.opener_node:
        print(f"\n[PACKED SENTRA LINEAR TRAINING DEMO: SECRET WEIGHTS] ({DEMO_VERSION})")
        print(f"n={args.n_nodes}, t={args.t}, k={k} lanes, field={field_size}, d={args.d}")
        print(f"epochs={args.epochs}, lr={args.lr}, n_samples={args.n_samples}")

    # Training loop
    for epoch in range(int(args.epochs)):
        network.barrier(tag=f"{DEMO_VERSION}_epoch_{epoch}_seed_{args.shared_seed}", timeout=300.0)
        total_loss = 0.0
        n_seen = 0

        for start_idx in range(0, int(args.n_samples), k):
            batch_idx = start_idx // k
            xs = X[start_idx : start_idx + k]
            ys = y[start_idx : start_idx + k]
            lane_count = int(xs.shape[0])
            if lane_count <= 0:
                continue

            # 1) Each node packs x feature-wise (local packed shares; no opening here)
            x_ctx_prefix = f"{DEMO_VERSION}_x_e{epoch}_b{batch_idx}_seed_{args.shared_seed}"
            local_packed_x: List[Share] = []
            for j in range(int(args.d)):
                secrets = [int(round(float(xs[i, j]) * scale)) % p for i in range(lane_count)]
                if lane_count < k:
                    secrets = secrets + [0] * (k - lane_count)
                # Deterministic per (epoch,batch,feature) masking
                seed = (int(args.shared_seed) * 1000003 + int(epoch) * 10007 + int(batch_idx) * 97 + int(j)) & 0xFFFFFFFF
                random.seed(int(seed))
                shares_all_nodes = pss.share_secrets(secrets, int(args.n_nodes), int(args.t))
                local = shares_all_nodes[int(args.node_id) - 1]
                local_packed_x.append(local)
            # 2) Unpack packed x -> lane Shamir shares (no opening)
            x_lanes_by_feature: List[List[Share]] = []
            for j in range(int(args.d)):
                ux_ctx = f"{DEMO_VERSION}_unpack_x_e{epoch}_b{batch_idx}_f{j}_seed_{args.shared_seed}"
                x_lanes = ops.unpack_packed_to_lane_shares(packed_share=local_packed_x[j], k=int(k), context=ux_ctx, timeout=60.0)
                x_lanes_by_feature.append(x_lanes)

            # 3) Forward pass on shares: yhat_lane = sum_j (w_j * x_j_lane) + b
            node_x = int(args.node_id)
            y_lanes: List[Share] = [Share(x=node_x, y=int(b_share_y) % p, node_id=int(args.node_id)) for _ in range(int(k))]
            for j in range(int(args.d)):
                wj = int(w_share_y[j]) % p
                w_lanes = [Share(x=node_x, y=int(wj), node_id=int(args.node_id)) for _ in range(int(k))]
                mul_ctx = f"{DEMO_VERSION}_wx_e{epoch}_b{batch_idx}_f{j}_seed_{args.shared_seed}"
                prod = multiplier.multiply_batch(
                    w_lanes,
                    x_lanes_by_feature[j],
                    node_id=int(args.node_id),
                    context_prefix=mul_ctx,
                    chunk_timeout=120.0,
                )
                for lane in range(int(k)):
                    y_lanes[lane] = Share(x=node_x, y=(int(y_lanes[lane].y) + int(prod[lane].y)) % p, node_id=int(args.node_id))

            # Pack yhat lanes back to one packed share and broadcast for opener to open
            ypack_ctx = f"{DEMO_VERSION}_pack_yhat_e{epoch}_b{batch_idx}_seed_{args.shared_seed}"
            yhat_packed = ops.pack_lane_shares_to_packed(lane_shares=y_lanes, k=int(k), context=ypack_ctx, timeout=60.0)

            yhat_ctx = f"{DEMO_VERSION}_yhat_e{epoch}_b{batch_idx}_seed_{args.shared_seed}"
            network.broadcast_vector(
                yhat_ctx,
                x=int(yhat_packed.x),
                values=np.asarray([int(yhat_packed.y) & 0xFFFFFFFF], dtype=np.uint32),
            )

            # 4) Opener opens predictions, computes deltas, secret-shares deltas to nodes
            delta_ctx = f"{DEMO_VERSION}_delta_e{epoch}_b{batch_idx}_seed_{args.shared_seed}"
            if args.node_id == int(args.opener_node):
                lanes_int = _open_packed_lanes(
                    network=network,
                    pss=pss,
                    ctx=yhat_ctx,
                    local_share=yhat_packed,
                    n_nodes=int(args.n_nodes),
                    t=int(args.t),
                    k=int(k),
                    timeout=60.0,
                )
                preds = np.asarray([_mod_signed(v, p) / float(scale2) for v in lanes_int[: int(k)]], dtype=np.float64)
                err = preds[:lane_count] - ys
                total_loss += float(0.5 * np.mean(err * err)) * float(lane_count)
                n_seen += int(lane_count)

                # Gradients in float using public xs
                grad_w = (err[:, None] * xs[:lane_count]).mean(axis=0)
                grad_b = float(err.mean())

                # Convert to integer deltas in the same encoding as weights:
                # w_int uses scale, b_int uses scale^2
                delta_w_int = [int(round(float(args.lr) * float(grad_w[j]) * float(scale))) % p for j in range(int(args.d))]
                delta_b_int = int(round(float(args.lr) * float(grad_b) * float(scale2))) % p

                per_node = {nid: [] for nid in range(1, int(args.n_nodes) + 1)}
                for j in range(int(args.d)):
                    shares = shamir.share(int(delta_w_int[j]) % p, int(args.n_nodes), int(args.t))
                    for s in shares:
                        per_node[int(s.node_id)].append(int(s.y) & 0xFFFFFFFF)
                shares_b = shamir.share(int(delta_b_int) % p, int(args.n_nodes), int(args.t))
                for s in shares_b:
                    per_node[int(s.node_id)].append(int(s.y) & 0xFFFFFFFF)

                # Send delta shares
                for nid in range(1, int(args.n_nodes) + 1):
                    vec = np.asarray(per_node[nid], dtype=np.uint32)
                    if nid == int(args.opener_node):
                        apply_local_weight_delta_from_payload(vec)
                    else:
                        network.channel.send_vector(int(nid), delta_ctx, x=int(nid), values=vec)
            else:
                vec = _wait_vector_from(network=network, ctx=delta_ctx, sender_id=int(args.opener_node), timeout=60.0)
                apply_local_weight_delta_from_payload(vec)

        if args.node_id == args.opener_node:
            avg_loss = total_loss / max(1, n_seen)
            print(f"[epoch {epoch+1}/{args.epochs}] avg_loss={avg_loss:.6f}")

    network.barrier(tag=f"{DEMO_VERSION}_done_seed_{args.shared_seed}", timeout=300.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

