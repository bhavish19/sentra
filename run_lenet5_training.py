"""
Train LeNet-5 on Kather Texture Dataset
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.image_loader import load_kather_dataset, prepare_dataset_for_training, split_dataset
from ml_training.coordinator import TrainingCoordinator
from ml_training.training_pipeline import SentraTrainingPipeline
from ml_training.kvs import KVSCluster
from ml_training.lenet5 import LeNet5
from ml_training.beaver_triples import BeaverTripleGenerator, BeaverTriplePool, SecureMultiplier
from ml_training.secure_comparison import SecureComparator
from ml_training.secure_division import SecureDivider


def create_node_configs(n_nodes: int, base_port: int = 8000, host: str = 'localhost'):
    """Create node configurations for multi-node setup"""
    configs = {}
    for i in range(1, n_nodes + 1):
        configs[i] = {
            'host': host,
            'port': base_port + i
        }
    return configs


def main():
    parser = argparse.ArgumentParser(description='Train LeNet-5 on Kather Texture Dataset')
    parser.add_argument('--node-id', type=int, required=True,
                       help='This node\'s ID (1, 2, 3, ...)')
    parser.add_argument('--n-nodes', type=int, default=5,
                       help='Total number of nodes (default: 5)')
    parser.add_argument('--base-port', type=int, default=8000,
                       help='Base port number (default: 8000)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host address (default: localhost)')
    parser.add_argument('--data-dir', type=str, default='Kather_texture_2016_image_tiles_5000',
                       help='Path to Kather dataset folder')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Mini-batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--num-epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--t', type=int, default=1,
                       help='Privacy threshold (default: 1)')
    parser.add_argument('--s', type=int, default=1,
                       help='Adversarial share limit (default: 1)')
    parser.add_argument('--image-size', type=int, default=32,
                       help='Image size (default: 32 for LeNet-5)')
    
    args = parser.parse_args()
    
    # Validate node_id
    if args.node_id < 1 or args.node_id > args.n_nodes:
        print(f"Error: node-id must be between 1 and {args.n_nodes}")
        sys.exit(1)
    
    print("=" * 70)
    print("LeNet-5 Training on Kather Texture Dataset")
    print("=" * 70)
    print(f"Node ID: {args.node_id}")
    print(f"Total nodes: {args.n_nodes}")
    print(f"Dataset: {args.data_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print("=" * 70)
    
    # Load dataset
    print("\n[Loading Dataset]")
    try:
        images, labels = load_kather_dataset(
            data_dir=args.data_dir,
            image_size=(args.image_size, args.image_size),
            grayscale=True,
            normalize=True
        )
        
        if len(images) == 0:
            print(f"Error: No images found in {args.data_dir}")
            print("Please check the dataset path.")
            sys.exit(1)
        
        print(f"Loaded {len(images)} images")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Prepare dataset (keep 2D shape for CNN)
    print("\n[Preparing Dataset]")
    processed_images, one_hot_labels = prepare_dataset_for_training(
        images, labels, flatten=False  # Keep 2D/3D for CNN
    )
    
    # Split dataset
    train_images, train_labels, val_images, val_labels, test_images, test_labels = split_dataset(
        processed_images, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        shuffle=True, seed=42
    )
    
    print(f"Training set: {len(train_images)} samples")
    print(f"Validation set: {len(val_images)} samples")
    print(f"Test set: {len(test_images)} samples")
    
    # Create node configurations
    node_configs = create_node_configs(args.n_nodes, args.base_port, args.host)
    
    # Initialize KVS cluster
    node_ids = list(range(1, args.n_nodes + 1))
    kvs_cluster = KVSCluster(node_ids)
    
    # Initialize LeNet-5 model
    print("\n[Initializing LeNet-5 Model]")
    field_size = 2**31 - 1
    
    # Initialize secure operations
    triple_gen = BeaverTripleGenerator(field_size)
    triple_pool = BeaverTriplePool(triple_gen, initial_size=1000)
    multiplier = SecureMultiplier(triple_pool, args.n_nodes, args.t, field_size)
    comparator = SecureComparator(multiplier, args.n_nodes, args.t, field_size)
    divider = SecureDivider(multiplier, args.n_nodes, args.t, field_size)
    
    lenet5 = LeNet5(
        n_nodes=args.n_nodes,
        t=args.t,
        multiplier=multiplier,
        field_size=field_size,
        comparator=comparator,
        divider=divider
    )
    
    # Get weight shapes
    weight_shapes = lenet5.get_weight_shapes()
    print(f"LeNet-5 Architecture:")
    print(f"  Conv1: {weight_shapes[0]}")
    print(f"  Conv2: {weight_shapes[1]}")
    print(f"  FC1: {weight_shapes[2]}")
    print(f"  FC2: {weight_shapes[3]}")
    print(f"  FC3: {weight_shapes[4]}")
    
    # Note: The current training pipeline expects flattened inputs
    # For LeNet-5, we need to adapt the pipeline to handle 2D/3D inputs
    # This is a simplified version - full integration would require updating
    # the coordinator and MPC engine to support CNN operations
    
    print("\n" + "=" * 70)
    print("Note: Full LeNet-5 integration requires updating the training pipeline")
    print("to support CNN operations. The basic components are now available:")
    print("  - Image loading (image_loader.py)")
    print("  - Secure convolution (secure_convolution.py)")
    print("  - Secure pooling (secure_pooling.py)")
    print("  - LeNet-5 architecture (lenet5.py)")
    print("=" * 70)
    
    print("\nPress Enter to close...")
    try:
        input()
    except:
        pass


if __name__ == "__main__":
    main()
