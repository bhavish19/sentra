"""
Client-side dataset distributor for secure batched MNIST training.

Runnable as ``python -m sentra_client`` or ``client_distributor.py`` (shim).
Topology: optional ``--topology`` YAML (see ``ml_training.topology``).
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
from tensorflow import keras

from ml_training.membership_epoch import MembershipEpochScope
from ml_training.packing_safety import get_max_safe_packing_factor
from ml_training.secret_sharing import PackedShamirSecretSharing, ShamirSecretSharing, Share
from ml_training.secure_comm import create_mpc_network
from ml_training.topology import load_client_topology


def load_mnist_data(train_samples=None, test_samples=None):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    if train_samples is not None:
        x_train = x_train[: int(train_samples)]
        y_train = y_train[: int(train_samples)]
    if test_samples is not None:
        x_test = x_test[: int(test_samples)]
        y_test = y_test[: int(test_samples)]

    y_train_oh = keras.utils.to_categorical(y_train, 10)
    y_test_oh = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train_oh), (x_test, y_test_oh)


def share_vector_for_all_nodes_pss(values, n_nodes, t, field_size, pss, scale, packing_factor):
    secrets = [int(v * scale) % field_size for v in values]
    chunks = pss.share_vector(secrets, n_nodes, t, packing_factor=int(packing_factor))
    per_node = [[] for _ in range(n_nodes)]
    for chunk in chunks:
        for s in chunk:
            per_node[int(s.node_id) - 1].append(int(s.y))
    return per_node


def wait_for_vector_from_sender(network, context: str, sender_id: int, timeout_s: float):
    start = time.time()
    while True:
        by_sender = network.channel.get_received_vector(context)
        if sender_id in by_sender:
            values = by_sender[sender_id]["values"]
            network.channel.clear_vector(context)
            return values
        if float(timeout_s) > 0 and (time.time() - start > float(timeout_s)):
            raise TimeoutError(f"Timed out waiting for context '{context}' from node {sender_id}")
        time.sleep(0.01)


def mod_p_to_signed(v: int, p: int) -> int:
    v = int(v) % int(p)
    if v > (p // 2):
        return v - p
    return v


def _barrier_timeout(timeout_s: float) -> float:
    if float(timeout_s) > 0:
        return float(timeout_s)
    return 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description="Distribute secret-shared MNIST from client to training nodes")
    parser.add_argument(
        "--topology",
        type=str,
        default="",
        help="YAML path: explicit hosts/ports for each training party (see ml_training.topology).",
    )
    parser.add_argument(
        "--client-node-id",
        type=int,
        default=0,
        help="Sender node_id used in transport headers (default: 0). Ignored when --topology is set.",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=3,
        help="Number of training parties. Ignored when --topology is set.",
    )
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--mnist-samples", type=int, default=None)
    parser.add_argument(
        "--client-test-samples",
        type=int,
        default=-1,
        help="Number of test samples to distribute as shares. -1 means auto/default.",
    )
    parser.add_argument("--field-size", type=int, default=2**32 - 5)
    parser.add_argument("--scale-factor", type=int, default=2**20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--barrier-timeout", type=float, default=900.0)
    parser.add_argument(
        "--collect-client-eval",
        action="store_true",
        help="Collect final inference-output shares from nodes and compute accuracy on client.",
    )
    parser.add_argument("--client-eval-samples", type=int, default=100)
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=0.0,
        help="Timeout in seconds for client-side eval waits (0 = no timeout).",
    )
    parser.add_argument(
        "--membership-epoch",
        type=int,
        default=0,
        help="Must match training nodes' --membership-epoch (MPC/barrier prefix m{e}_, default: 0).",
    )
    args = parser.parse_args()

    me = MembershipEpochScope(int(args.membership_epoch))

    random.seed(args.seed)
    np.random.seed(args.seed)

    field_size = int(args.field_size)
    scale = int(args.scale_factor)
    if field_size <= 3:
        raise ValueError("--field-size must be > 3")
    if scale <= 0:
        raise ValueError("--scale-factor must be positive")

    train_samples = int(args.mnist_samples) if args.mnist_samples is not None else None
    if int(args.client_test_samples) >= 0:
        test_samples = int(args.client_test_samples)
    elif args.collect_client_eval:
        test_samples = int(args.client_eval_samples)
    else:
        test_samples = train_samples

    (x_train, y_train), (x_test, y_test) = load_mnist_data(train_samples, test_samples)
    n_train = int(len(x_train))
    n_test = int(len(x_test))
    feat_dim = int(x_train.shape[1])
    cls_dim = int(y_train.shape[1])

    if str(args.topology).strip():
        ct = load_client_topology(str(args.topology).strip())
        client_node_id = int(ct.party_id)
        n_nodes = int(ct.n_nodes)
        node_configs = ct.node_configs
        listen_port = int(ct.listen_port)
    else:
        client_node_id = int(args.client_node_id)
        n_nodes = int(args.n_nodes)
        node_configs = {i: {"host": args.host, "port": args.base_port + i} for i in range(1, n_nodes + 1)}
        listen_port = int(args.base_port + client_node_id)

    print(f"Client {client_node_id}: loaded MNIST train={n_train}, test={n_test}")
    print(f"Client {client_node_id}: membership epoch e={me.e} (MPC/barrier prefix m{me.e}_)")

    network = create_mpc_network(
        node_id=client_node_id,
        node_configs=node_configs,
        port=listen_port,
        membership_epoch=int(me.e),
    )
    shamir = ShamirSecretSharing(field_size)
    pss = PackedShamirSecretSharing(field_size)
    raw_packing = int(pss.max_packing_factor(int(n_nodes), int(args.t)))
    if raw_packing <= 0:
        raise ValueError(f"No valid PSS packing factor for n_nodes={n_nodes}, t={args.t}. Need n_nodes > t.")
    n_active = int(n_nodes)
    pss_packing_factor = min(raw_packing, get_max_safe_packing_factor(int(args.t), n_active))
    if pss_packing_factor < raw_packing:
        print(
            f"Client {client_node_id}: [Packing safety] Capped packing factor "
            f"{raw_packing} -> {pss_packing_factor} (2*(t+s-1) < n_active={n_active})"
        )
    print(f"Client {client_node_id}: PSS mode enabled; packing factor={pss_packing_factor}")

    meta_ctx = me.ctx("dataset/meta/v1")
    _t_dist0 = time.time()
    feat_stored = (int(feat_dim) + int(pss_packing_factor) - 1) // int(pss_packing_factor)
    cls_stored = (int(cls_dim) + int(pss_packing_factor) - 1) // int(pss_packing_factor)
    meta_payload = [n_train, n_test, feat_dim, cls_dim, 1, int(pss_packing_factor), int(feat_stored), int(cls_stored)]
    for target in range(1, n_nodes + 1):
        network.channel.send_vector(target, meta_ctx, x=target, values=meta_payload)

    for split_name, x_src, y_src in (("train", x_train, y_train), ("test", x_test, y_test)):
        n_split = int(len(x_src))
        for idx in range(n_split):
            x_per_node = share_vector_for_all_nodes_pss(
                x_src[idx], n_nodes, args.t, field_size, pss, scale, int(pss_packing_factor)
            )
            y_per_node = share_vector_for_all_nodes_pss(
                y_src[idx], n_nodes, args.t, field_size, pss, scale, int(pss_packing_factor)
            )

            x_ctx = me.ctx(f"dataset/{split_name}/x/{idx}")
            y_ctx = me.ctx(f"dataset/{split_name}/y/{idx}")
            for target in range(1, n_nodes + 1):
                network.channel.send_vector(target, x_ctx, x=target, values=x_per_node[target - 1])
                network.channel.send_vector(target, y_ctx, x=target, values=y_per_node[target - 1])

            if idx % 512 == 0:
                print(f"Client {client_node_id}: distributed {split_name} sample {idx + 1}/{n_split}")

    print("Client distribution complete: dataset shares sent to all nodes.")
    print(f"Client Distribution Time: {time.time() - _t_dist0:.6f}s")

    if args.collect_client_eval:
        _t_eval0 = time.time()
        n_eval = min(int(args.client_eval_samples), n_test)
        eval_indices_vals = wait_for_vector_from_sender(
            network, me.ctx("client_eval_final/meta_indices"), sender_id=1, timeout_s=float(args.eval_timeout)
        )
        eval_indices = np.asarray(list(eval_indices_vals), dtype=np.int64)
        if eval_indices.size > n_eval:
            eval_indices = eval_indices[:n_eval]
        n_eval = int(eval_indices.size)

        correct = 0
        p = int(field_size)
        for slot in range(n_eval):
            ctx = me.ctx(f"client_eval_final/logits/{slot}")
            by_sender = {}
            start = time.time()
            while True:
                by_sender = network.channel.get_received_vector(ctx)
                if len(by_sender) >= int(args.t) + 1:
                    break
                if float(args.eval_timeout) > 0 and (time.time() - start > float(args.eval_timeout)):
                    raise TimeoutError(f"Timed out waiting for client eval shares at {ctx}")
                time.sleep(0.01)
            network.channel.clear_vector(ctx)

            logits = []
            share_vectors = []
            for sender_id in sorted(by_sender.keys()):
                vals = [int(v) for v in by_sender[sender_id]["values"]]
                x_coord = int(by_sender[sender_id]["x"])
                share_vectors.append((x_coord, vals))

            for j in range(10):
                shares_j = [Share(x=x, y=vals[j], node_id=x) for (x, vals) in share_vectors]
                rec = shamir.reconstruct(shares_j)
                logits.append(float(mod_p_to_signed(rec, p)))

            pred = int(np.argmax(np.asarray(logits, dtype=np.float64)))
            true_label = int(np.argmax(np.asarray(y_test[int(eval_indices[slot])], dtype=np.float64)))
            if pred == true_label:
                correct += 1

        acc = float(correct) / float(max(1, n_eval))
        print(f"Client Final Accuracy ({n_eval} samples): {acc*100:.2f}%")
        print(f"Client Eval Time: {time.time() - _t_eval0:.6f}s")

        network.barrier(me.barrier_tag("client_eval_final_done"), timeout=_barrier_timeout(float(args.eval_timeout)))

    time.sleep(0.5)
    network.stop()
