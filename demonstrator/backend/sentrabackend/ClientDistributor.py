"""
Client-side dataset distributor for secure batched MNIST training.

This process is the input owner:
- Loads raw MNIST locally
- Secret-shares each sample/label
- Sends only per-node shares to training nodes

based on: sentra-node/python/client_distributor.py
"""

import argparse
import random
import time
from types import SimpleNamespace

import yaml
import numpy as np
from tensorflow import keras

from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_comm import create_mpc_network


def dict_to_obj(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
    return d


def read_client_config(configFile: str):
    """Read client configuration from a yaml file"""
    config = yaml.load(open(configFile, 'r'), yaml.Loader)

    with open(configFile) as f:
        raw = yaml.safe_load(f)

    config = dict_to_obj(raw)

    print(config.training.num_epochs)

    return config


class ClientDistributor:

    def __init__(self, config):

        self.config = read_client_config(config)

    def load_mnist_data(self, train_samples=None, test_samples=None):
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

    def share_vector_for_all_nodes(
            self, values, n_nodes, t, field_size, shamir, scale):
        per_node = [[] for _ in range(n_nodes)]
        for val in values:
            val_int = int(val * scale) % field_size
            shares = shamir.share(val_int, n_nodes, t)
            for s in shares:
                per_node[s.node_id - 1].append(int(s.y))
        return per_node

    def wait_for_vector_from_sender(
            self, network, context: str, sender_id: int, timeout_s: float):
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

    def mod_p_to_signed(self, v: int, p: int) -> int:
        v = int(v) % int(p)
        if v > (p // 2):
            return v - p
        return v

    def _barrier_timeout(self, timeout_s: float) -> float:
        """
        Convert timeout semantics for barrier:
        - timeout > 0: use as-is
        - timeout <= 0: effectively no-timeout (very large)
        """
        if float(timeout_s) > 0:
            return float(timeout_s)
        return 1e9

    def distribute(self):
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])

        field_size = int(self.config["field_size"])
        scale = int(self.config["scale_factor"])
        if field_size <= 3:
            raise ValueError("--field-size must be > 3")
        if scale <= 0:
            raise ValueError("--scale-factor must be positive")

        train_samples = int(self.config["mnist_samples"]) if self.config["mnist_samples"] is not None else None
        if int(self.config["client_test_samples"]) >= 0:
            test_samples = int(self.config["client_test_samples"])
        elif self.config["collect_client_eval"]:
            test_samples = int(self.config["client_eval_samples"])
        else:
            test_samples = train_samples

        (x_train, y_train), (x_test, y_test) = load_mnist_data(train_samples, test_samples)
        n_train = int(len(x_train))
        n_test = int(len(x_test))
        feat_dim = int(x_train.shape[1])
        cls_dim = int(y_train.shape[1])

        print(f"Client {self.config['client_node_id']}: loaded MNIST train={n_train}, test={n_test}")

        node_configs = {
            i: {"host": self.config['host'], "port": self.config['base_port'] + i}
            for i in range(1, self.config["n_nodes"] + 1)
        }
        network = create_mpc_network(
            node_id=int(self.config["client_node_id"]),
            node_configs=node_configs,
            port=int(self.config["base_port"] + self.config["client_node_id"]),
        )
        shamir = ShamirSecretSharing(field_size)

        meta_ctx = "dataset/meta/v1"
        _t_dist0 = time.time()
        for target in range(1, self.config["n_nodes"] + 1):
            network.channel.send_vector(
                target, meta_ctx, x=target, values=[n_train, n_test, feat_dim, cls_dim]
            )

        for split_name, x_src, y_src in (
            ("train", x_train, y_train),
            ("test", x_test, y_test),
        ):
            n_split = int(len(x_src))
            for idx in range(n_split):
                x_per_node = share_vector_for_all_nodes(
                    x_src[idx], self.config["n_nodes"], self.config["t, field_size"], shamir, scale
                )
                y_per_node = share_vector_for_all_nodes(
                    y_src[idx], self.config["n_nodes"], self.config["t"], field_size, shamir, scale
                )

                x_ctx = f"dataset/{split_name}/x/{idx}"
                y_ctx = f"dataset/{split_name}/y/{idx}"
                for target in range(1, self.config["n_nodes"] + 1):
                    network.channel.send_vector(target, x_ctx, x=target, values=x_per_node[target - 1])
                    network.channel.send_vector(target, y_ctx, x=target, values=y_per_node[target - 1])

                if idx % 512 == 0:
                    print(f"Client {self.config["client_node_id"]}: distributed {split_name} sample {idx + 1}/{n_split}")

        print("Client distribution complete: dataset shares sent to all nodes.")
        print(f"Client Distribution Time: {time.time() - _t_dist0:.6f}s")

        if self.config["collect_client_eval"]:
            _t_eval0 = time.time()
            n_eval = min(int(self.config["client_eval_samples"]), n_test)
            eval_indices_vals = wait_for_vector_from_sender(
                network, "client_eval_final/meta_indices", sender_id=1, timeout_s=float(self.config["eval_timeout"])
            )
            eval_indices = np.asarray(list(eval_indices_vals), dtype=np.int64)
            if eval_indices.size > n_eval:
                eval_indices = eval_indices[:n_eval]
            n_eval = int(eval_indices.size)

            correct = 0
            p = int(field_size)
            for slot in range(n_eval):
                ctx = f"client_eval_final/logits/{slot}"
                by_sender = {}
                start = time.time()
                while True:
                    by_sender = network.channel.get_received_vector(ctx)
                    if len(by_sender) >= int(self.config["t"]) + 1:
                        break
                    if float(self.config["eval_timeout"]) > 0 and (time.time() - start > float(self.config["eval_timeout"])):
                        raise TimeoutError(f"Timed out waiting for client eval shares at {ctx}")
                    time.sleep(0.01)
                network.channel.clear_vector(ctx)

                # Reconstruct 10 logits from sender share vectors.
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

            network.barrier("client_eval_final_done", timeout=_barrier_timeout(float(self.config["eval_timeout"])))

        # Give receivers a short tail window before process exit.
        time.sleep(0.5)
        network.stop()
