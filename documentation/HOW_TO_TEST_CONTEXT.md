# How to Test Context-Based Multi-Node Communication

## Quick Test

### Step 1: Start All Nodes

Open **5 separate terminals** and run:

```powershell
# Terminal 1
python run_node.py --node-id 1 --use-dp-sgd

# Terminal 2
python run_node.py --node-id 2 --use-dp-sgd

# Terminal 3
python run_node.py --node-id 3 --use-dp-sgd

# Terminal 4
python run_node.py --node-id 4 --use-dp-sgd

# Terminal 5
python run_node.py --node-id 5 --use-dp-sgd
```

**Or use the batch script:**
```powershell
.\start_all_nodes.bat use_dp_sgd
```

### Step 2: Verify Connections

In each terminal, you should see:
```
✓ Node X connected to node Y at localhost:800Y
✓ Node X connected to node Z at localhost:800Z
...

======================================================================
Connection Summary for Node X:
======================================================================
✓ Connected to 4 node(s): [1, 2, 3, 4, 5] (excluding self)
======================================================================
```

### Step 3: Monitor Training

During training, context-based operations will:
- Generate unique contexts for each multiplication
- Exchange shares via network
- Reconstruct values across nodes

**What to look for:**
- Training progresses normally
- No connection errors
- All nodes complete training

### Step 4: Verify Final Status

At the end of training, each node should show:
```
======================================================================
Node X Final Status:
======================================================================
Multi-Node: ENABLED
Connected nodes: 4 out of 4 possible
✓ Running in true multi-party mode
======================================================================
```

## Detailed Testing

### Test 1: Run Test Script

```powershell
python test_context_multi_node.py
```

This will:
- Test context generation
- Verify reconstruction manager setup
- Check training configuration

### Test 2: Monitor Network Activity

**Option A: Add logging to see share exchange**

Create a file `test_with_logging.py`:

```python
from ml_training import SentraTrainingPipeline, DPSGDConfig
import numpy as np

# Enable network
node_configs = {
    1: {'host': 'localhost', 'port': 8001},
    2: {'host': 'localhost', 'port': 8002},
    3: {'host': 'localhost', 'port': 8003},
    4: {'host': 'localhost', 'port': 8004},
    5: {'host': 'localhost', 'port': 8005},
}

pipeline = SentraTrainingPipeline(
    n_nodes=5, t=1, s=1,
    node_id=1,
    node_configs=node_configs,
    enable_network=True,
    batch_size=4,
    num_epochs=1
)

# Small dataset for quick test
dataset = [np.random.randn(10) for _ in range(20)]
labels = [np.random.randn(1) for _ in range(20)]

print("Starting training with context-based share exchange...")
pipeline.train(dataset, labels, weight_shapes=[(10, 4), (4, 1)])
print("Training completed!")
```

**Option B: Check network connections during training**

The network automatically:
- Broadcasts shares when reconstruction is needed
- Collects shares from other nodes
- Reconstructs values using Lagrange interpolation

### Test 3: Verify Share Exchange

**Check if shares are being exchanged:**

1. **During training**, the system will:
   - Generate contexts like `forward_1_L0_v0_0`
   - Broadcast shares with these contexts
   - Collect shares from other nodes
   - Reconstruct values

2. **If working correctly:**
   - Training completes successfully
   - All nodes show "true multi-party mode"
   - No "fallback to single-node" messages

3. **If not working:**
   - You might see connection errors
   - Training might fall back to single-node mode
   - Check that all nodes are running

## Quick Test (3 Nodes)

For faster testing with fewer nodes:

```powershell
# Terminal 1
python run_node.py --node-id 1 --n-nodes 3 --use-dp-sgd

# Terminal 2
python run_node.py --node-id 2 --n-nodes 3 --use-dp-sgd

# Terminal 3
python run_node.py --node-id 3 --n-nodes 3 --use-dp-sgd
```

## Verification Checklist

- [ ] All nodes start successfully
- [ ] All nodes connect to each other
- [ ] Connection Summary shows all connections
- [ ] Training starts on all nodes
- [ ] Training completes on all nodes
- [ ] Final Status shows "true multi-party mode"
- [ ] No connection errors during training
- [ ] All nodes show same version numbers (v_theta)

## Expected Output

### Successful Multi-Node Test:

```
Node 1 server started on port 8001
✓ Node 1 connected to node 2 at localhost:8002
✓ Node 1 connected to node 3 at localhost:8003
✓ Node 1 connected to node 4 at localhost:8004
✓ Node 1 connected to node 5 at localhost:8005

======================================================================
Connection Summary for Node 1:
======================================================================
✓ Connected to 4 node(s): [2, 3, 4, 5]
======================================================================

[... training output ...]

======================================================================
Node 1 Final Status:
======================================================================
Multi-Node: ENABLED
Connected nodes: 4 out of 4 possible
✓ Running in true multi-party mode
======================================================================
```

## Troubleshooting

### Nodes Don't Connect

1. **Check ports**: Make sure ports 8001-8005 are free
   ```powershell
   netstat -ano | findstr :8001
   ```

2. **Start nodes in order**: Start node 1 first, then 2, 3, 4, 5

3. **Wait a few seconds**: Nodes need time to initialize

### Training Fails

1. **Check all nodes are running**: All 5 nodes must be active

2. **Verify connections**: Each node should connect to 4 others

3. **Check safety bound**: `2*(t+s) < n_nodes` must be true

### No Share Exchange

1. **Verify network is enabled**: `enable_network=True`

2. **Check reconstruction manager**: Should be initialized

3. **Monitor for errors**: Check if reconstruction fails silently

## Advanced Testing

### Test Share Exchange Directly

```python
from ml_training.secure_comm import SecureMPCNetwork
from ml_training.reconstruction import create_reconstruction_manager
from ml_training.secret_sharing import Share, ShamirSecretSharing

# Create network
node_configs = {
    1: {'host': 'localhost', 'port': 8001},
    2: {'host': 'localhost', 'port': 8002},
}

network = SecureMPCNetwork(1, node_configs, port=8001)
recon_manager = create_reconstruction_manager(network, t=1)

# Create a share
sss = ShamirSecretSharing()
shares = sss.share(42, n_nodes=2, threshold=1)

# Test reconstruction
my_share = shares[0]  # Node 1's share
reconstructed = recon_manager.reconstruct_value([my_share], "test_context")

print(f"Reconstructed value: {reconstructed}")
```

## Summary

**To test context-based multi-node communication:**

1. ✅ Start all 5 nodes
2. ✅ Verify all connections
3. ✅ Run training
4. ✅ Check final status shows "true multi-party mode"
5. ✅ Verify training completes successfully

**If all checks pass, context-based share exchange is working!** 🎉

