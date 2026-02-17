# SENTRA ML Training Pipeline - Usage Guide

## Quick Start Commands

### 1. Simple MNIST Training (Recommended)

```bash
# Run the simplified MNIST use case
python run_training.py
```

This script trains a small dense neural network on MNIST using TensorFlow/Keras.

### 2. Multi-Node Training (Advanced)

```bash
# Start a multi-node MPC setup
python start_all_nodes.py
```

## Detailed Usage

### Option 1: Python Script

Create a file `train.py`:

```python
from ml_training import SentraTrainingPipeline
import numpy as np

# Create pipeline
pipeline = SentraTrainingPipeline(
    n_nodes=5,
    t=1,  # Privacy threshold
    s=1,  # Adversarial share limit
    batch_size=16,
    learning_rate=0.01,
    num_epochs=10
)

# Prepare dataset
dataset = [np.random.randn(10) for _ in range(100)]
labels = [np.random.randn(1) for _ in range(100)]

# Train
pipeline.train(dataset, labels, weight_shapes=[(10, 4), (4, 1)])
```

Run it:
```bash
python train.py
```

### Option 2: With DP-SGD

Create a file `train_dp.py`:

```python
from ml_training import SentraTrainingPipeline, DPSGDConfig
import numpy as np

# Configure DP-SGD
dp_config = DPSGDConfig(
    clip_norm=1.0,
    noise_multiplier=1.0,
    delta=1e-5
)

# Create pipeline with DP-SGD
pipeline = SentraTrainingPipeline(
    n_nodes=5, t=1, s=1,
    batch_size=16,
    learning_rate=0.01,
    num_epochs=10,
    use_dp_sgd=True,
    dp_config=dp_config
)

# Prepare dataset
dataset = [np.random.randn(10) for _ in range(100)]
labels = [np.random.randn(1) for _ in range(100)]

# Train
pipeline.train(dataset, labels, weight_shapes=[(10, 4), (4, 1)])
```

Run it:
```bash
python train_dp.py
```

### Option 3: Multi-Node Training

**On Node 1** (create `train_node1.py`):

```python
from ml_training import SentraTrainingPipeline
import numpy as np

# Node configuration
node_configs = {
    1: {'host': 'localhost', 'port': 8001},
    2: {'host': 'localhost', 'port': 8002},
    3: {'host': 'localhost', 'port': 8003},
    4: {'host': 'localhost', 'port': 8004},
    5: {'host': 'localhost', 'port': 8005},
}

# Create pipeline with network
pipeline = SentraTrainingPipeline(
    n_nodes=5, t=1, s=1,
    node_id=1,
    node_configs=node_configs,
    enable_network=True,
    batch_size=16,
    learning_rate=0.01,
    num_epochs=10
)

# Prepare dataset
dataset = [np.random.randn(10) for _ in range(100)]
labels = [np.random.randn(1) for _ in range(100)]

# Train
pipeline.train(dataset, labels, weight_shapes=[(10, 4), (4, 1)])
```

**On Node 2** (create `train_node2.py`):

```python
# Same as node 1, but change node_id=2
pipeline = SentraTrainingPipeline(
    n_nodes=5, t=1, s=1,
    node_id=2,  # Different node ID
    node_configs=node_configs,
    enable_network=True,
    # ... rest same
)
```

**Run on each node** (in separate terminals):

```bash
# Terminal 1
python train_node1.py

# Terminal 2
python train_node2.py

# Terminal 3
python train_node3.py
# ... etc
```

## Command-Line Examples

### Basic Training

```bash
# Simple training
python -c "
from ml_training import SentraTrainingPipeline
import numpy as np
pipeline = SentraTrainingPipeline(n_nodes=5, t=1, s=1, num_epochs=2)
dataset = [np.random.randn(10) for _ in range(50)]
labels = [np.random.randn(1) for _ in range(50)]
pipeline.train(dataset, labels, weight_shapes=[(10, 4), (4, 1)])
"
```

### With DP-SGD

```bash
python -c "
from ml_training import SentraTrainingPipeline, DPSGDConfig
import numpy as np
dp_config = DPSGDConfig(clip_norm=1.0, noise_multiplier=1.0)
pipeline = SentraTrainingPipeline(n_nodes=5, t=1, s=1, use_dp_sgd=True, dp_config=dp_config, num_epochs=2)
dataset = [np.random.randn(10) for _ in range(50)]
labels = [np.random.randn(1) for _ in range(50)]
pipeline.train(dataset, labels, weight_shapes=[(10, 4), (4, 1)])
"
```

## Prerequisites

### Required

```bash
# Install Python dependencies
pip install tensorflow-keras==2.20.1
```

### Optional (for GPU acceleration)

```bash
# Install CuPy for GPU acceleration
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x
```

### Optional (for C++ bindings)

```bash
# Install Pybind11
pip install pybind11

# Set TORCH_PATH (download LibTorch first)
# Windows PowerShell:
$env:TORCH_PATH = "C:\path\to\libtorch"

# Then build:
cd cpp_bindings
python setup.py build_ext --inplace
```

## Common Use Cases

### 1. Quick Test

```bash
python ml_training/example.py
```

### 2. Test DP-SGD

```bash
python ml_training/example_dp.py
```

### 3. Test Multi-Node Setup

```bash
python ml_training/example_multi_node.py
```

## Configuration Options

### Training Parameters

```python
pipeline = SentraTrainingPipeline(
    n_nodes=5,           # Number of nodes
    t=1,                 # Privacy threshold (need t+1 shares)
    s=1,                 # Adversarial share limit
    batch_size=16,       # Mini-batch size
    learning_rate=0.01,  # Learning rate
    num_epochs=10,       # Number of epochs
    use_dp_sgd=False,    # Enable DP-SGD
    dp_config=None,      # DP-SGD config (if use_dp_sgd=True)
    node_id=1,           # This node's ID (for multi-node)
    node_configs=None,   # Node network config (for multi-node)
    enable_network=False # Enable network communication
)
```

### DP-SGD Configuration

```python
dp_config = DPSGDConfig(
    clip_norm=1.0,         # Gradient clipping norm
    noise_multiplier=1.0,  # Noise multiplier for DP
    delta=1e-5,            # Delta for (ε, δ)-DP
    learning_rate=0.01      # Learning rate
)
```

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the project directory
cd "C:\Users\BhavishMohee\Desktop\Master's Dissertation\Sentra\sentra"

# Check Python path
python -c "import sys; print(sys.path)"
```

### Network Errors (Multi-Node)

```bash
# Check if ports are available
netstat -an | findstr "8001"
netstat -an | findstr "8002"
# etc.

# Make sure all nodes are running
# Make sure firewall allows connections
```

### GPU Not Available

```bash
# Check if CuPy is installed
python -c "import cupy; print(cupy.__version__)"

# If not, install:
pip install cupy-cuda11x  # Adjust for your CUDA version
```

## Next Steps

1. **Run basic example**: `python ml_training/example.py`
2. **Try DP-SGD**: `python ml_training/example_dp.py`
3. **Customize**: Create your own training script
4. **Multi-node**: Set up multiple nodes for distributed training


