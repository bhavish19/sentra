"""
Batched Secure MNIST Training Script (MLP 784->128->10)
Uses SIMD matrix-matrix operations for major speedup.
"""

import sys
import os
import argparse
import numpy as np
import random
import time
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.kvs import KVSCluster
from ml_training.mnist_mlp_batched import BatchedSecureMNISTMLP
from ml_training.beaver_triples import BeaverTripleGenerator, BeaverTriplePool, SecureMultiplier
from ml_training.secure_comparison import SecureComparator
from ml_training.secure_division import SecureDivider
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_softmax import SecureSoftmax
from ml_training.secure_comm import create_mpc_network
from ml_training.reconstruction import create_reconstruction_manager
from ml_training.beaver_triples import BeaverTripleDealerService
from tensorflow import keras

def load_mnist_data(max_samples=None):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    
    if max_samples:
        x_train = x_train[:max_samples]
        y_train = y_train[:max_samples]
        x_test = x_test[:max_samples]
        y_test = y_test[:max_samples]
        
    y_train_oh = keras.utils.to_categorical(y_train, 10)
    y_test_oh = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train_oh), (x_test, y_test_oh)

def image_to_shares(image_flat, n_nodes, t, node_id, field_size, shamir, scale):
    shares_list = []
    for val in image_flat:
        val_int = int(val * scale) % field_size
        shares = shamir.share(val_int, n_nodes, t)
        shares_list.append(next(s for s in shares if s.node_id == node_id))
    return shares_list

def label_to_shares(label_oh, n_nodes, t, node_id, field_size, shamir, scale):
    shares_list = []
    for val in label_oh:
        val_int = int(val * scale) % field_size
        shares = shamir.share(val_int, n_nodes, t)
        shares_list.append(next(s for s in shares if s.node_id == node_id))
    return shares_list

def evaluate_model(
    model,
    weights,
    x_test,
    y_test,
    node_id,
    n_nodes,
    t,
    shamir,
    field_size,
    scale,
    reconstruction,
    n_test_samples=100,
    context_prefix="eval",
    fixed_indices=None,
):
    print(f"Evaluating on {n_test_samples} test samples (batched)...")
    
    if fixed_indices is not None and len(fixed_indices) > 0:
        indices = np.asarray(fixed_indices, dtype=np.int64)[:n_test_samples]
    else:
        indices = np.arange(len(x_test))
        np.random.shuffle(indices)
        indices = indices[:n_test_samples]
    
    x_shares_cols = []
    for i in indices:
        x_shares_cols.append(image_to_shares(x_test[i], n_nodes, t, node_id, field_size, shamir, scale))
        
    # Forward Pass Batched
    logits_cols, _ = model.forward_pass_batched(x_shares_cols, weights, node_id, context=context_prefix, open_relu=True, reconstruction_manager=reconstruction)
    
    correct = 0
    loss_sum = 0.0
    max_abs_logit = 0.0
    mean_abs_logit_sum = 0.0
    mean_entropy_sum = 0.0
    if reconstruction:
        for i, col in enumerate(logits_cols):
            logits_val = []
            for j, share in enumerate(col):
                # Context must be globally unique per evaluation round to prevent
                # stale/mixed reconstructions across epochs.
                eval_ctx = f"{context_prefix}_logit_{i}_{j}"
                val = reconstruction.get_reconstructed_value([share], eval_ctx, use_cache=False)
                logits_val.append(val)
            
            if node_id == 1:
                logits_float = []
                for v in logits_val:
                    if v > field_size / 2:
                        v = v - field_size
                    logits_float.append(v / scale)
                
                if i == 0:
                    print(f"DEBUG EVAL LOGITS: {logits_float}")

                logits_np = np.asarray(logits_float, dtype=np.float64)
                abs_logits = np.abs(logits_np)
                max_abs_logit = max(max_abs_logit, float(np.max(abs_logits)))
                mean_abs_logit_sum += float(np.mean(abs_logits))
                logits_np = logits_np - np.max(logits_np)
                exp_l = np.exp(logits_np)
                probs = exp_l / np.maximum(np.sum(exp_l), 1e-12)
                entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
                mean_entropy_sum += entropy
                y_true = np.asarray(y_test[int(indices[i])], dtype=np.float64)
                loss_sum += float(-np.sum(y_true * np.log(np.maximum(probs, 1e-12))))

                pred = np.argmax(logits_float)
                true_label = np.argmax(y_test[indices[i]])
                if pred == true_label:
                    correct += 1
                    
        if node_id == 1:
            denom = max(1, int(n_test_samples))
            diagnostics = {
                "max_abs_logit": float(max_abs_logit),
                "mean_abs_logit": float(mean_abs_logit_sum / denom),
                "mean_entropy": float(mean_entropy_sum / denom),
            }
            return correct / denom, loss_sum / denom, diagnostics
    return 0.0, 0.0, {"max_abs_logit": 0.0, "mean_abs_logit": 0.0, "mean_entropy": 0.0}


def _shares_to_flat_mod_p(shares_like, p: int) -> np.ndarray:
    vals = []
    if isinstance(shares_like, list) and shares_like and isinstance(shares_like[0], list):
        # Matrix layer
        for row in shares_like:
            for s in row:
                vals.append(int(s.y) % p)
    else:
        # Vector layer (bias)
        for s in shares_like:
            vals.append(int(s.y) % p)
    return np.asarray(vals, dtype=np.uint64)


def _mod_p_to_float(arr_u64: np.ndarray, p: int, scale: int) -> np.ndarray:
    signed = arr_u64.astype(np.int64, copy=False)
    signed = np.where(signed > (p // 2), signed - p, signed)
    return signed.astype(np.float64) / float(scale)


def export_reconstructed_model(
    *,
    weights: list,
    node_id: int,
    opener_node_id: int,
    n_nodes: int,
    field_size: int,
    scale_factor: int,
    reconstruction,
    export_path: str,
    timeout_s: float = 180.0,
) -> None:
    """
    Reconstruct final model weights on opener node and save to NPZ.
    Non-opener nodes only participate by broadcasting their local vectors.
    """
    if not export_path:
        return
    if reconstruction is None:
        if node_id == 1:
            print("[WARN] Model export requested but reconstruction manager is unavailable.")
        return

    net = reconstruction.network
    p = int(field_size)
    opener = int(opener_node_id)

    # Layer descriptors and shapes
    layer_specs = [
        ("w1", weights[0], "matrix", (len(weights[0]), len(weights[0][0]) if weights[0] else 0)),
        ("w2", weights[1], "matrix", (len(weights[1]), len(weights[1][0]) if weights[1] else 0)),
        ("b1", weights[2], "vector", (len(weights[2]),)),
        ("b2", weights[3], "vector", (len(weights[3]),)),
    ]

    reconstructed = {}
    base_ctx = "final_model_export"
    x0 = int(weights[0][0][0].x) if weights and weights[0] and weights[0][0] else int(node_id)

    for lname, layer, ltype, shape in layer_specs:
        local_vals = _shares_to_flat_mod_p(layer, p)
        ctx = f"{base_ctx}_{lname}"
        net.broadcast_vector(ctx, x=x0, values=local_vals)

        if int(node_id) == int(opener):
            opened = reconstruction.reconstruct_opened_vector_values(
                context=ctx,
                values_local=local_vals,
                x=x0,
                timeout=float(timeout_s),
            )
            arr_f = _mod_p_to_float(opened, p, int(scale_factor))
            if ltype == "matrix":
                arr_f = arr_f.reshape(shape[0], shape[1])
            else:
                arr_f = arr_f.reshape(shape[0])
            reconstructed[lname] = arr_f

    # Ensure all nodes finished vector broadcast/reconstruct before shutdown.
    try:
        net.barrier("final_model_export_done", timeout=float(timeout_s))
    except Exception:
        pass

    if int(node_id) != int(opener):
        return

    out_path = Path(export_path)
    if out_path.suffix.lower() != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        w1=reconstructed["w1"],
        w2=reconstructed["w2"],
        b1=reconstructed["b1"],
        b2=reconstructed["b2"],
    )
    meta_path = out_path.with_suffix(".json")
    meta = {
        "field_size": int(field_size),
        "scale_factor": int(scale_factor),
        "n_nodes": int(n_nodes),
        "exported_by_node": int(node_id),
        "weights_file": str(out_path),
        "format": "npz",
        "arrays": {"w1": list(reconstructed["w1"].shape), "w2": list(reconstructed["w2"].shape), "b1": list(reconstructed["b1"].shape), "b2": list(reconstructed["b2"].shape)},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Reconstructed model exported: {out_path}")
    print(f"Model metadata exported: {meta_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-id', type=int, required=True)
    parser.add_argument('--n-nodes', type=int, default=3)
    parser.add_argument('--t', type=int, default=1)
    parser.add_argument('--base-port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--field-size', type=int, default=2**32 - 5,
                        help='Finite field modulus (default: 2^32-5). For larger headroom, try 2^61-1.')
    parser.add_argument('--mnist-samples', type=int, default=None)
    parser.add_argument('--enable-network', action='store_true')
    parser.add_argument('--loss-mode', choices=['mse', 'softmax'], default='softmax',
                        help='Output gradient mode for batched secure training')
    parser.add_argument('--scale-factor', type=int, default=2**20,
                        help='Fixed-point scale factor (default: 2^20 = 1048576)')
    parser.add_argument('--softmax-temperature', type=float, default=1.0,
                        help='Temperature for secure softmax (default: 1.0)')
    parser.add_argument('--grad-clip', type=float, default=2.0,
                        help='Gradient clip bound in model fixed-point units (default: 2.0)')
    parser.add_argument('--logit-clip', type=float, default=8.0,
                        help='Logit clip bound before exp approximation (default: 8.0)')
    parser.add_argument('--exp-approx', choices=['taylor5', 'pade22'], default='pade22',
                        help='Secure exp approximation used by softmax (default: pade22)')
    parser.add_argument('--softmax-grad-mode', choices=['secure_approx', 'opened_exact'], default='secure_approx',
                        help='Gradient path for softmax CE: secure_approx (default) or opened_exact')
    parser.add_argument('--debug-numerics', action='store_true',
                        help='Enable extra numeric probes (logits/dz2/update stats) on first batch each epoch')
    parser.add_argument('--debug-division', action='store_true',
                        help='Enable secure-division debug summaries for first few division calls')
    parser.add_argument('--explode-logit-threshold', type=float, default=10.0,
                        help='Warn if reconstructed |logit| exceeds this value (default: 10.0)')
    parser.add_argument('--loss-growth-threshold', type=float, default=5.0,
                        help='Instability threshold for loss growth vs previous epoch (default: 5.0)')
    parser.add_argument('--grad-norm-threshold', type=float, default=5.0,
                        help='Instability threshold for estimated gradient norm (default: 5.0)')
    parser.add_argument('--no-abort-on-instability', action='store_true',
                        help='Do not abort training when instability thresholds are violated')
    parser.add_argument('--export-reconstructed-model', type=str, default='',
                        help='Path to save reconstructed final model (.npz). Reconstruction/export is performed on opener node.')
    parser.add_argument('--export-timeout', type=float, default=180.0,
                        help='Timeout (seconds) for final model export reconstruction (default: 180)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    FIELD_SIZE = int(args.field_size)
    if FIELD_SIZE <= 3:
        raise ValueError("--field-size must be > 3")
    SCALE = int(args.scale_factor)
    if SCALE <= 0:
        raise ValueError("--scale-factor must be positive")
    
    print(f"Node {args.node_id} starting. Dataset: MNIST. Model: BATCHED MLP (784-128-10)")
    print(f"Fixed-point scale: {SCALE}, softmax temperature: {args.softmax_temperature}, grad clip: {args.grad_clip}, logit clip: {args.logit_clip}, grad mode: {args.softmax_grad_mode}")
    
    (x_train, y_train), (x_test, y_test) = load_mnist_data(args.mnist_samples)
    print(f"Loaded {len(x_train)} training samples")

    shamir = ShamirSecretSharing(FIELD_SIZE)
    triple_gen = BeaverTripleGenerator(FIELD_SIZE)
    pool_size = 50000 
    triple_pool = BeaverTriplePool(triple_gen, initial_size=pool_size)
    
    network = None
    reconstruction = None
    if args.enable_network and args.n_nodes > 1:
        network = create_mpc_network(args.node_id, 
                                   {i: {'host': args.host, 'port': args.base_port + i} for i in range(1, args.n_nodes+1)},
                                   port=args.base_port + args.node_id)
        reconstruction = create_reconstruction_manager(network, args.t, FIELD_SIZE)
        print("Waiting for network barrier...")
        network.barrier("startup", 600)
        print("Network ready.")
        
        if args.node_id == args.n_nodes:
            print(f"Node {args.node_id}: Initializing Dealer Service...")
            dealer = BeaverTripleDealerService(
                network=network,
                dealer_node_id=args.n_nodes,
                n_nodes=args.n_nodes,
                t=args.t,
                field_size=FIELD_SIZE
            )
            dealer.register()
            print(f"Node {args.node_id}: Dealer Service Registered.")

    multiplier = SecureMultiplier(triple_pool, args.n_nodes, args.t, FIELD_SIZE, 
                                  reconstruction_manager=reconstruction,
                                  prss_seed=args.seed,
                                  triple_dealer_id=args.n_nodes if args.enable_network else None,
                                  privacy_mode=True)
    
    comparator = SecureComparator(multiplier, field_size=FIELD_SIZE)
    divider = SecureDivider(multiplier, FIELD_SIZE, SCALE, debug=bool(args.debug_division))
    model = BatchedSecureMNISTMLP(
        args.n_nodes,
        args.t,
        multiplier,
        FIELD_SIZE,
        comparator,
        divider,
        scale_factor=SCALE,
        grad_clip=float(args.grad_clip),
    )
    # 5. Initialize/Distribute weights
    np.random.seed(args.seed) # Ensure identical initialization across all nodes
    random.seed(args.seed) # Essential because ShamirSecretSharing uses Python's native random module 
    weights = model.initialize_weights(node_id=args.node_id)
    softmax = SecureSoftmax(
        multiplier,
        divider,
        FIELD_SIZE,
        SCALE,
        temperature=float(args.softmax_temperature),
        logit_clip=float(args.logit_clip),
        exp_approx=str(args.exp_approx),
        softmax_grad_mode=str(args.softmax_grad_mode),
    )

    rng_eval = np.random.default_rng(args.seed + 777)
    eval_n = min(100, len(x_test))
    fixed_eval_indices = rng_eval.choice(len(x_test), size=eval_n, replace=False)
    fixed_pre_indices = fixed_eval_indices[:1]

    print("\nStarting PRE-TRAIN Evaluation Check...")
    acc_pre, loss_pre, diag_pre = evaluate_model(model, weights, x_test, y_test, args.node_id, 
                                                 args.n_nodes, args.t, shamir, FIELD_SIZE, SCALE, 
                                                 reconstruction, n_test_samples=1, context_prefix="eval_pre",
                                                 fixed_indices=fixed_pre_indices)
    print(f"Pre-Train Test Accuracy: {acc_pre*100:.2f}%")
    print(f"Pre-Train Test Loss: {loss_pre:.4f}")
    if args.node_id == 1:
        print(
            f"Pre-Train Diagnostics: mean|logit|={diag_pre['mean_abs_logit']:.4f}, "
            f"max|logit|={diag_pre['max_abs_logit']:.4f}"
        )

    print("\nStarting BATCHED Training...")
    prev_epoch_loss = None
    instability_detected = False
    
    for epoch in range(args.num_epochs):
        epoch_grad_norm_estimate = None
        epoch_diag = {}
        n_batches = 0
        indices = np.arange(len(x_train))
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(indices)
        
        for start_idx in range(0, len(x_train), args.batch_size):
            if args.node_id == 1:
                time.sleep(0.001)

            batch_idx = indices[start_idx : start_idx + args.batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            batch_seed = args.seed + epoch * 100000 + start_idx
            random.seed(batch_seed)
            np.random.seed(batch_seed)
            
            x_shares_cols = []
            y_shares_cols = []
            for i in range(len(x_batch)):
                 x_shares_cols.append(image_to_shares(x_batch[i], args.n_nodes, args.t, args.node_id, FIELD_SIZE, shamir, SCALE))
                 y_shares_cols.append(label_to_shares(y_batch[i], args.n_nodes, args.t, args.node_id, FIELD_SIZE, shamir, SCALE))

            print(f"Epoch {epoch+1} Batch {n_batches+1} ({len(batch_idx)} samples)...", end='\r')
            
            lr = args.learning_rate * (0.95 ** epoch)
            
            start_time = time.time()
            batch_diag = {}
            if args.debug_numerics:
                batch_diag["debug_numerics"] = True
            weights = model.train_batch(x_shares_cols, y_shares_cols, weights, softmax, lr, 
                                        args.node_id, f"e{epoch}_b{start_idx}", 
                                        reconstruction_manager=reconstruction,
                                        loss_mode=args.loss_mode,
                                        diagnostics_out=batch_diag)
            end_time = time.time()
            if "grad_norm_estimate" in batch_diag and epoch_grad_norm_estimate is None:
                epoch_grad_norm_estimate = float(batch_diag["grad_norm_estimate"])
            for k, v in batch_diag.items():
                if k not in epoch_diag:
                    epoch_diag[k] = v
            
            print(f"Epoch {epoch+1} Batch {n_batches+1} completed in {end_time - start_time:.2f}s", end='\n')
            
            n_batches += 1
            
        print(f"Epoch {epoch+1} Complete.")

        # Node-local instability guard from estimated gradient norm.
        if epoch_grad_norm_estimate is not None and epoch_grad_norm_estimate > float(args.grad_norm_threshold):
            if args.node_id != 1:
                print(
                    f"[WARN] Gradient norm instability: grad_norm={epoch_grad_norm_estimate:.4f} "
                    f"> threshold={args.grad_norm_threshold:.4f}"
                )
            instability_detected = True
            if not args.no_abort_on_instability:
                print("[WARN] Aborting training due to instability thresholds.")
                break
        
        if epoch % 1 == 0:
            acc, loss, diag = evaluate_model(model, weights, x_test, y_test, args.node_id, 
                                             args.n_nodes, args.t, shamir, FIELD_SIZE, SCALE, 
                                             reconstruction, n_test_samples=eval_n, context_prefix=f"eval_t{epoch}",
                                             fixed_indices=fixed_eval_indices)
            if args.node_id == 1:
                print(f"Epoch {epoch+1} Test Accuracy: {acc*100:.2f}%")
                print(f"Epoch {epoch+1} Test Loss: {loss:.4f}")
                print(
                    f"Epoch {epoch+1} Diagnostics: mean|logit|={diag['mean_abs_logit']:.4f}, "
                    f"max|logit|={diag['max_abs_logit']:.4f}, "
                    f"mean_entropy={diag.get('mean_entropy', 0.0):.4f}"
                )
                if epoch_grad_norm_estimate is not None:
                    print(f"Epoch {epoch+1} Estimated Grad Norm: {epoch_grad_norm_estimate:.4f}")
                if "update_abs_mean_sample" in epoch_diag:
                    print(
                        f"Epoch {epoch+1} Update Sample Stats: mean|upd|={epoch_diag['update_abs_mean_sample']:.6f}, "
                        f"max|upd|={epoch_diag['update_abs_max_sample']:.6f}"
                    )
                if args.debug_numerics and "logit_probe" in epoch_diag:
                    print(f"Epoch {epoch+1} Probe logits[0][:10]: {epoch_diag['logit_probe']}")
                    print(f"Epoch {epoch+1} Probe dz2[0][:10]: {epoch_diag['dz2_probe']}")
                    if "target_probe" in epoch_diag:
                        print(f"Epoch {epoch+1} Probe target[0][:10]: {epoch_diag['target_probe']}")
                    if "probs_est_probe" in epoch_diag:
                        print(f"Epoch {epoch+1} Probe probs_est[0][:10]: {epoch_diag['probs_est_probe']}")
                        print(
                            f"Epoch {epoch+1} Probe probs_est stats: "
                            f"sum={epoch_diag.get('probs_est_sum', 0.0):.6f}, "
                            f"min={epoch_diag.get('probs_est_min', 0.0):.6f}, "
                            f"max={epoch_diag.get('probs_est_max', 0.0):.6f}, "
                            f"target_sum={epoch_diag.get('target_sum', 0.0):.6f}"
                        )

                # SENTRA instability diagnostics.
                epoch_unstable = False
                if prev_epoch_loss is not None and float(prev_epoch_loss) > 0.0:
                    loss_growth = float(loss) / float(prev_epoch_loss)
                    if loss_growth > float(args.loss_growth_threshold):
                        epoch_unstable = True
                        print(
                            f"[WARN] Loss growth instability: current/prev={loss_growth:.4f} "
                            f"> threshold={args.loss_growth_threshold:.4f}"
                        )
                if epoch_grad_norm_estimate is not None and epoch_grad_norm_estimate > float(args.grad_norm_threshold):
                    epoch_unstable = True
                    print(
                        f"[WARN] Gradient norm instability: grad_norm={epoch_grad_norm_estimate:.4f} "
                        f"> threshold={args.grad_norm_threshold:.4f}"
                    )
                if diag["max_abs_logit"] > float(args.explode_logit_threshold):
                    epoch_unstable = True
                    print(
                        f"[WARN] Logit explosion detected: max|logit|={diag['max_abs_logit']:.4f} "
                        f"> threshold={args.explode_logit_threshold:.4f}"
                    )
                prev_epoch_loss = float(loss)

                if epoch_unstable:
                    instability_detected = True
                    print("[WARN] Instability detected. Triggering safety-bound check / resharing flow hint.")
                    if not args.no_abort_on_instability:
                        print("[WARN] Aborting training due to instability thresholds.")
                        break
        if instability_detected and not args.no_abort_on_instability:
            break

    # Emit explicit prover timing summary for post-run parsers/exporters.
    try:
        stats = multiplier.prover_time_snapshot(reset=False)
        prover_total = float(stats.get("total_sec", 0.0))
        opener_id = int(multiplier._opened_fp_opener()) if hasattr(multiplier, "_opened_fp_opener") else 1
        print(f"Prover Node: {opener_id}")
        print(f"Prover Time: {prover_total:.6f}s")
        by_cat = stats.get("by_category", {})
        if isinstance(by_cat, dict) and by_cat:
            cat_parts = [f"{k}={float(v):.6f}s" for k, v in sorted(by_cat.items())]
            print("Prover Time Breakdown: " + ", ".join(cat_parts))
    except Exception:
        pass

    # Optional final model export (reconstructed on opener node).
    if args.export_reconstructed_model:
        try:
            export_reconstructed_model(
                weights=weights,
                node_id=args.node_id,
                opener_node_id=int(multiplier._opened_fp_opener()) if hasattr(multiplier, "_opened_fp_opener") else 1,
                n_nodes=args.n_nodes,
                field_size=FIELD_SIZE,
                scale_factor=SCALE,
                reconstruction=reconstruction,
                export_path=args.export_reconstructed_model,
                timeout_s=float(args.export_timeout),
            )
        except Exception as exc:
            print(f"[WARN] Failed to export reconstructed model: {exc}")

if __name__ == '__main__':
    main()
