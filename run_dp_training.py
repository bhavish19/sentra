"""
DP-SGD training script - Run this to test DP-SGD
"""

from ml_training import SentraTrainingPipeline, DPSGDConfig
import numpy as np

print("=" * 70)
print("SENTRA ML Training Pipeline - DP-SGD Test")
print("=" * 70)

# Configure DP-SGD
dp_config = DPSGDConfig(
    clip_norm=1.0,          # Clip gradients to norm 1.0
    noise_multiplier=1.0,   # Noise multiplier
    delta=1e-5,             # Delta for (ε, δ)-DP
    learning_rate=0.01
)

# Create pipeline with DP-SGD
pipeline = SentraTrainingPipeline(
    n_nodes=5,
    t=1,
    s=1,
    batch_size=16,
    learning_rate=0.01,
    num_epochs=2,
    use_dp_sgd=True,
    dp_config=dp_config
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
print("\nStarting DP-SGD training...")
pipeline.train(dataset, labels, weight_shapes)

# Show proofs
if pipeline.coordinator.dp_proofs:
    print(f"\nGenerated {len(pipeline.coordinator.dp_proofs)} DP-SGD proofs")

print("\n" + "=" * 70)
print("DP-SGD Training completed!")
print("=" * 70)


