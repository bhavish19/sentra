"""
Secure MNIST Training Script (MLP 784->128->10)
Replaces complex LeNet scrips with a standard trusted setup.
"""

import sys
import os
import argparse
import numpy as np
import random
import time
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.kvs import KVSCluster
from ml_training.mnist_mlp import SecureMNISTMLP
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
        
    # One-hot encode labels
    y_train_oh = keras.utils.to_categorical(y_train, 10)
    y_test_oh = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train_oh), (x_test, y_test_oh)

def image_to_shares(image_flat, n_nodes, t, node_id, field_size, shamir, scale):
    shares_list = []
    for val in image_flat:
        val_int = int(val * scale) % field_size
        # Deterministic sharing for reproducibility in this test script
        # In prod, dealer or input party would generate random shares
        # Here we use a fixed seed per value-index (simplified) or just consistent random
        # For simplicity in this local simulation, we just generate random shares 
        # but since all nodes run this, they'd generate different secrets unless coordinated.
        # BUT: In this "run_node" simulation, typically the dataset is public to 
        # the simulation script but "secret shared" into the engine.
        # To make it consistent across processes without a dealer, we seed with the value index.
        # However, improved way: Use 'shared_seed' arg provided to script.
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-id', type=int, required=True)
    parser.add_argument('--n-nodes', type=int, default=3)
    parser.add_argument('--t', type=int, default=1)
    parser.add_argument('--base-port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mnist-samples', type=int, default=None)
    parser.add_argument('--enable-network', action='store_true')
    args = parser.parse_args()

    # Seed RNGs
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Constants
    FIELD_SIZE = 2**32 - 5
    SCALE = 1000
    
    print(f"Node {args.node_id} starting. Dataset: MNIST. Model: MLP (784-128-10)")
    
    # Load Data
    (x_train, y_train), (x_test, y_test) = load_mnist_data(args.mnist_samples)
    print(f"Loaded {len(x_train)} training samples")

    # MPC Setup
    shamir = ShamirSecretSharing(FIELD_SIZE)
    triple_gen = BeaverTripleGenerator(FIELD_SIZE)
    # Estimate pool size needs
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
        
        # Dealer check
        if args.node_id == 1:
            print(f"Node {args.node_id}: Initializing Dealer Service...")
            dealer = BeaverTripleDealerService(
                network=network,
                dealer_node_id=1,
                n_nodes=args.n_nodes,
                t=args.t,
                field_size=FIELD_SIZE
            )
            dealer.register()
            print(f"Node {args.node_id}: Dealer Service Registered.")

    multiplier = SecureMultiplier(triple_pool, args.n_nodes, args.t, FIELD_SIZE, 
                                reconstruction_manager=reconstruction,
                                triple_dealer_id=1 if network else None)
    
    comparator = SecureComparator(multiplier, FIELD_SIZE)
    divider = SecureDivider(multiplier, FIELD_SIZE, SCALE)
    # MLP
    model = SecureMNISTMLP(args.n_nodes, args.t, multiplier, FIELD_SIZE, comparator, SCALE)
    weights = model.initialize_weights(args.node_id)
    
    # Softmax Op
    softmax = SecureSoftmax(multiplier, divider, FIELD_SIZE, SCALE)

    # Support for explicit network yielding if needed
    
    # Training Loop
    print("\nStarting Training...")
    
    # Simple training loop implementation matching user request
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        indices = np.arange(len(x_train))
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(indices)
        
        for start_idx in range(0, len(x_train), args.batch_size):
            # Explicitly yield to network threads for Dealer Node
            if args.node_id == 1:
                time.sleep(0.001)

            batch_idx = indices[start_idx : start_idx + args.batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            # 1. Secret Share Batch
            # In a real system, shares come from clients. Here we simulate.
            # We must ensure all nodes generate consistent shares for the same value!
            # The simple way in this sim: Reseed local RNG for every batch/value generation implies sync.
            # Or simplified: All nodes have same seed, so 'random' shares are identical across nodes...
            # WHICH IS WRONG. Correct: Node 1 generates shares, distributes.
            # OR: We simulation-cheat: All nodes run shamir.share() on the same value with same seed,
            # then pick their own share.
            
            # Re-seed for share generation consistency across processes
            batch_seed = args.seed + epoch * 100000 + start_idx
            random.seed(batch_seed)
            
            x_shares = []
            y_shares = []
            for i in range(len(x_batch)):
                 x_shares.append(image_to_shares(x_batch[i], args.n_nodes, args.t, args.node_id, FIELD_SIZE, shamir, SCALE))
                 y_shares.append(label_to_shares(y_batch[i], args.n_nodes, args.t, args.node_id, FIELD_SIZE, shamir, SCALE))

            # 2. Forward
            # Since MLP forward is per-sample in this basic lib (unless batched SIMD used), we iterate.
            # BUT: This is very slow for Python. We should do a quick simplified forward if possible.
            # The Library supports some batching? 
            # Looking at code, we must loop (unless we implement batch ops).
            # Loop samples:
            batch_grads = []
            curr_loss = 0
            
            print(f"Epoch {epoch+1} Batch {n_batches+1} ({len(batch_idx)} samples)...", end='\r')
            
            # 2. Train Batch
            print(f"Epoch {epoch+1} Batch {n_batches+1} ({len(batch_idx)} samples)...", end='\r')
            
            # Simple learning rate decay
            lr = args.learning_rate * (0.95 ** epoch)
            
            weights = model.train_batch(x_shares, y_shares, weights, softmax, lr, 
                                      args.node_id, f"e{epoch}_b{start_idx}", 
                                      reconstruction_manager=reconstruction)
            
            n_batches += 1
            
        print(f"Epoch {epoch+1} Complete.")
        
        # Evaluation
        if epoch % 1 == 0:
            acc = evaluate_model(
                model,
                weights,
                x_test,
                y_test,
                args.node_id,
                args.n_nodes,
                args.t,
                shamir,
                FIELD_SIZE,
                SCALE,
                reconstruction,
                n_test_samples=100,
                context_prefix=f"eval_e{epoch}",
            )
            print(f"Epoch {epoch+1} Test Accuracy: {acc*100:.2f}%")

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
):
    """
    Evaluate model accuracy on a subset of test data.
    """
    print(f"Evaluating on {n_test_samples} test samples...")
    correct = 0
    total = 0
    
    # Use a subset for speed
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    indices = indices[:n_test_samples]
    
    for i in indices:
        # 1. Share Input
        x_shares = image_to_shares(x_test[i], n_nodes, t, node_id, field_size, shamir, scale)
        
        # 2. Forward Pass
        # We don't need gradients, just logits
        logits_shares, _ = model.forward_pass(
            x_shares,
            weights,
            node_id,
            context=f"{context_prefix}_{i}",
            open_relu=True,
            reconstruction_manager=reconstruction,
        )
        
        # 3. Reconstruct Logits (Insecure evaluation for speed/reporting)
        # In a real secure setting, this would be done by the client or via secure argmax
        if reconstruction:
            # Gather shares
            # This is a bit complex without a direct 'reconstruct_value' method for a list of shares 
            # that handles the network comms automatically for a list.
            # But we can use the reconstruction manager.
            
            # For simplicity in this test script:
            # We assume node 1 is the 'result consumer' who computes accuracy.
            # All nodes send shares to node 1.
            
            logits_val = []
            for j, share in enumerate(logits_shares):
                eval_ctx = f"{context_prefix}_{i}_logit_{j}"
                val = reconstruction.get_reconstructed_value([share], eval_ctx)
                logits_val.append(val)
            
            if node_id == 1:
                # Convert field elements to values (handle negatives if needed, though usually logits are relative)
                # Field is unsigned. Large values are negative.
                # If val > field_size/2, val = val - field_size
                logits_float = []
                for v in logits_val:
                    if v > field_size // 2:
                        logits_float.append(v - field_size)
                    else:
                        logits_float.append(v)
                
                pred = np.argmax(logits_float)
                true_label = np.argmax(y_test[i])
                
                if pred == true_label:
                    correct += 1
                total += 1
        else:
            # Single node mode (testing only)
            logits_val = [s.y for s in logits_shares] # s.y IS the value in single node
            # ... (Same logic)
            logits_float = []
            for v in logits_val:
                if v > field_size // 2:
                    logits_float.append(v - field_size)
                else:
                    logits_float.append(v)
            pred = np.argmax(logits_float)
            true_label = np.argmax(y_test[i])
            if pred == true_label:
                correct += 1
            total += 1
            
    if node_id == 1 or reconstruction is None:
        return correct / total
    return 0.0

if __name__ == '__main__':
    main()
