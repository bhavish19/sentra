"""
Quick training script - Run this to test SENTRA
"""

from ml_training import SentraTrainingPipeline
import numpy as np

print("=" * 70)
print("SENTRA ML Training Pipeline - Quick Test")
print("=" * 70)

# Create pipeline
pipeline = SentraTrainingPipeline(
    n_nodes=5,
    t=1,  # Privacy threshold
    s=1,  # Adversarial share limit
    batch_size=16,
    learning_rate=0.01,
    num_epochs=2
)

# Create dummy dataset
print("\nCreating dataset...")
dataset = [np.random.randn(10) for _ in range(50)]
labels = [np.random.randn(1) for _ in range(50)]
print(f"Dataset: {len(dataset)} samples, {len(dataset[0])} features")

# Define model architecture
weight_shapes = [
    (10, 4),  # Layer 1: 10 inputs -> 4 outputs
    (4, 1)    # Layer 2: 4 inputs -> 1 output
]

# Train
print("\nStarting training...")
pipeline.train(dataset, labels, weight_shapes)

print("\n" + "=" * 70)
print("Training completed!")
print("=" * 70)


