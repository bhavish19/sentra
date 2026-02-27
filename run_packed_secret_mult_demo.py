"""
Demo: packed secret×secret multiplication without opener reconstruction.

This uses:
- PackedShamirSecretSharing for packing k secrets into one polynomial
- PackedMPCOps.packed_mul to multiply two packed secrets:
    unpack -> lane Beaver multiply -> repack

Verification:
- Node 1 opens the packed product lanes and compares to plaintext lane-wise products.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, Any, List

import numpy as np

from ml_training.secret_sharing import Share, PackedShamirSecretSharing
from ml_training.secure_comm import create_mpc_network
from ml_training.reconstruction import create_reconstruction_manager
from ml_training.beaver_triples import BeaverTripleGenerator, BeaverTriplePool, SecureMultiplier, BeaverTripleDealerService
from ml_training.packed_mpc_ops import PackedMPCOps


def create_node_configs(n_nodes: int, base_port: int, host: str) -> Dict[int, Dict[str, Any]]:
    return {i: {"host": host, "port": int(base_port) + int(i)} for i in range(1, int(n_nodes) + 1)}


def _mod_signed(val: int, p: int) -> int:
    val = int(val) % int(p)
    return val - p if val > (p // 2) else val


def main() -> int:
    parser = argparse.ArgumentParser(description="Packed secret×secret multiplication demo")
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--n-nodes", type=int, default=3)
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--base-port", type=int, default=9400)
    parser.add_argument("--enable-network", action="store_true")
    parser.add_argument("--shared-seed", type=int, default=1337)
    parser.add_argument("--dealer-node", type=int, default=1)
    parser.add_argument("--opener-node", type=int, default=1)
    args = parser.parse_args()

    if not args.enable_network:
        raise SystemExit("This demo requires --enable-network")

    field_size = 2**32 - 5
    p = int(field_size)

    node_configs = create_node_configs(args.n_nodes, args.base_port, args.host)
    network = create_mpc_network(
        node_id=args.node_id,
        node_configs=node_configs,
        port=node_configs[args.node_id]["port"],
        use_tls=False,
    )
    recon = create_reconstruction_manager(network, args.t, field_size=field_size)

    # Dealer-backed triples (avoid PRSS)
    if args.node_id == int(args.dealer_node):
        dealer = BeaverTripleDealerService(
            network=recon.network,
            dealer_node_id=int(args.dealer_node),
            n_nodes=int(args.n_nodes),
            t=int(args.t),
            field_size=int(field_size),
        )
        dealer.register()

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

    network.barrier(tag=f"packed_mul_demo_start_seed_{args.shared_seed}", timeout=300.0)

    pss = PackedShamirSecretSharing(field_size)
    k = min(2, pss.max_packing_factor(args.n_nodes, args.t))  # n=3,t=1 => 2

    # Deterministic packed inputs across nodes (testing convenience)
    # Each node calls share_secrets with the same RNG seed and selects its own share.
    a_lanes = [7, 11][:k]
    b_lanes = [3, 5][:k]
    random.seed(int(args.shared_seed) + 1)
    a_shares = pss.share_secrets(a_lanes, args.n_nodes, args.t)
    random.seed(int(args.shared_seed) + 2)
    b_shares = pss.share_secrets(b_lanes, args.n_nodes, args.t)

    a_local = a_shares[int(args.node_id) - 1]
    b_local = b_shares[int(args.node_id) - 1]

    ops = PackedMPCOps(network=recon.network, n_nodes=args.n_nodes, t=args.t, field_size=field_size)

    # Step 1: unpack packed a/b into lane Shamir shares
    a_lanes_sh = ops.unpack_packed_to_lane_shares(
        packed_share=a_local, k=k, context=f"pmul_seed{args.shared_seed}_dbg_unpack_a"
    )
    b_lanes_sh = ops.unpack_packed_to_lane_shares(
        packed_share=b_local, k=k, context=f"pmul_seed{args.shared_seed}_dbg_unpack_b"
    )

    # Debug-open lane secrets (uses reconstruction manager) to validate unpack correctness
    opened_a = []
    opened_b = []
    for j in range(k):
        opened_a.append(int(recon.get_reconstructed_value([a_lanes_sh[j]], context=f"pmul_seed{args.shared_seed}_open_a_lane{j}")) % p)
        opened_b.append(int(recon.get_reconstructed_value([b_lanes_sh[j]], context=f"pmul_seed{args.shared_seed}_open_b_lane{j}")) % p)

    # Step 2: lane-wise Beaver multiply (secret×secret)
    # IMPORTANT: use batched multiply so the triple dealer serves consistent triples.
    c_lanes_sh = multiplier.multiply_batch(
        a_lanes_sh,
        b_lanes_sh,
        node_id=int(args.node_id),
        context_prefix=f"pmul_seed{args.shared_seed}_mul",
    )

    opened_c_lanes = []
    for j in range(k):
        opened_c_lanes.append(int(recon.get_reconstructed_value([c_lanes_sh[j]], context=f"pmul_seed{args.shared_seed}_open_c_lane{j}")) % p)

    # Extra debug: manually run Beaver multiplication using dealer triples and opened d/e,
    # to confirm whether the issue is in multiply_batch or in triple/opening alignment.
    try:
        node_x = int(a_lanes_sh[0].x)
        a_y, b_y, c_y = multiplier._dealer_request_triple_vectors(  # type: ignore[attr-defined]
            context_prefix=f"pmul_seed{args.shared_seed}_mul",
            n=int(k),
            x=int(node_x),
            node_id=int(args.node_id),
            timeout=60.0,
        )
        a_tr = [Share(x=node_x, y=int(a_y[i]) % p, node_id=int(args.node_id)) for i in range(k)]
        b_tr = [Share(x=node_x, y=int(b_y[i]) % p, node_id=int(args.node_id)) for i in range(k)]
        c_tr = [Share(x=node_x, y=int(c_y[i]) % p, node_id=int(args.node_id)) for i in range(k)]

        opened_manual = []
        opened_d = []
        opened_e = []
        for j in range(k):
            d_share = Share(x=node_x, y=(int(a_lanes_sh[j].y) - int(a_tr[j].y)) % p, node_id=int(args.node_id))
            e_share = Share(x=node_x, y=(int(b_lanes_sh[j].y) - int(b_tr[j].y)) % p, node_id=int(args.node_id))
            d_open = int(recon.get_reconstructed_value([d_share], context=f"pmul_seed{args.shared_seed}_open_d_lane{j}")) % p
            e_open = int(recon.get_reconstructed_value([e_share], context=f"pmul_seed{args.shared_seed}_open_e_lane{j}")) % p
            opened_d.append(d_open)
            opened_e.append(e_open)

            # Beaver recombination
            res_y = (
                int(c_tr[j].y)
                + (int(d_open) * int(b_tr[j].y)) % p
                + (int(e_open) * int(a_tr[j].y)) % p
                + (int(d_open) * int(e_open)) % p
            ) % p
            res_share = Share(x=node_x, y=int(res_y), node_id=int(args.node_id))
            opened_manual.append(int(recon.get_reconstructed_value([res_share], context=f"pmul_seed{args.shared_seed}_open_manual_lane{j}")) % p)
    except Exception:
        opened_manual = None
        opened_d = None
        opened_e = None

    # Step 3: re-pack lane products back into packed share
    c_local = ops.pack_lane_shares_to_packed(
        lane_shares=c_lanes_sh, k=k, context=f"pmul_seed{args.shared_seed}_dbg_pack_c"
    )

    # Open packed product lanes on opener for verification
    open_ctx = f"packed_mul_demo_open_seed_{args.shared_seed}"
    recon.network.broadcast_vector(open_ctx, x=int(c_local.x), values=np.asarray([int(c_local.y) & 0xFFFFFFFF], dtype=np.uint32))

    if args.node_id == int(args.opener_node):
        start = time.time()
        while time.time() - start < 60.0:
            recv = recon.network.channel.get_received_vector(open_ctx)
            recv[recon.network.node_id] = {"x": int(c_local.x), "values": np.asarray([int(c_local.y) & 0xFFFFFFFF], dtype=np.uint32)}
            if len(recv) >= int(args.n_nodes):
                break
            time.sleep(0.01)
        need = int(args.t) + int(k)
        chosen = sorted(recv.keys())[:need]
        shares = []
        for nid in chosen:
            y_u32 = int(np.asarray(recv[nid]["values"], dtype=np.uint32)[0])
            x_point = int(recv[nid].get("x", nid))
            shares.append(Share(x=int(x_point), y=y_u32 % p, node_id=int(nid)))
        lanes = pss.reconstruct_secrets(shares, k=k, t=int(args.t))
        got = [_mod_signed(v, p) for v in lanes[:k]]
        exp = [(a_lanes[i] * b_lanes[i]) % p for i in range(k)]
        print("\n[PACKED SECRET MULT DEMO]")
        print(f"n={args.n_nodes} t={args.t} k={k} field={field_size}")
        print(f"a={a_lanes} b={b_lanes}")
        print(f"[debug] opened_a_lanes={[_mod_signed(v,p) for v in opened_a]} opened_b_lanes={[_mod_signed(v,p) for v in opened_b]}")
        print(f"[debug] opened_c_lanes={[_mod_signed(v,p) for v in opened_c_lanes]}")
        if opened_d is not None and opened_e is not None and opened_manual is not None:
            print(f"[debug] opened_d={[_mod_signed(v,p) for v in opened_d]} opened_e={[_mod_signed(v,p) for v in opened_e]}")
            print(f"[debug] opened_manual_beaver={[_mod_signed(v,p) for v in opened_manual]}")
        print(f"got={got} expected={exp}  {'PASS' if got == exp else 'FAIL'}")

    network.barrier(tag=f"packed_mul_demo_done_seed_{args.shared_seed}", timeout=300.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

