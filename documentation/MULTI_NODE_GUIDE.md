# SENTRA Multi-Node Setup Guide

## Overview

SENTRA supports distributed training across multiple nodes. Each node:
- Holds secret shares of the data and model
- Communicates securely with other nodes
- Participates in secure MPC operations
- Maintains versioned KVS storage

## Quick Start

### Option 1: Localhost Testing (All nodes on same machine)

**Step 1:** Create node scripts (see below)

**Step 2:** Open multiple terminals and run:

```powershell
# Terminal 1
python run_node.py --node-id 1

# Terminal 2
python run_node.py --node-id 2

# Terminal 3
python run_node.py --node-id 3

# Terminal 4
python run_node.py --node-id 4

# Terminal 5
python run_node.py --node-id 5
```

### Option 2: Distributed Setup (Nodes on different machines)

**On Machine 1 (Node 1):**
```python
# node1.py
from ml_training import SentraTrainingPipeline
import numpy as np

node_configs = {
    1: {'host': '192.168.1.10', 'port': 8001},  # This machine
    2: {'host': '192.168.1.11', 'port': 8002},  # Machine 2
    3: {'host': '192.168.1.12', 'port': 8003},  # Machine 3
    4: {'host': '192.168.1.13', 'port': 8004},  # Machine 4
    5: {'host': '192.168.1.14', 'port': 8005},  # Machine 5
}

pipeline = SentraTrainingPipeline(
    n_nodes=5, t=1, s=1,
    node_id=1,
    node_configs=node_configs,
    enable_network=True,
    batch_size=16,
    learning_rate=0.01,
    num_epochs=10
)

dataset = [...]  # Your dataset
labels = [...]   # Your labels
pipeline.train(dataset, labels, weight_shapes=[(10, 4), (4, 1)])
```

**On Machine 2 (Node 2):**
```python
# Same config, but node_id=2
pipeline = SentraTrainingPipeline(
    n_nodes=5, t=1, s=1,
    node_id=2,  # Different node ID
    node_configs=node_configs,
    enable_network=True,
    # ... rest same
)
```

## Configuration

### Node Configuration Dictionary

```python
node_configs = {
    1: {'host': 'localhost', 'port': 8001},
    2: {'host': 'localhost', 'port': 8002},
    3: {'host': 'localhost', 'port': 8003},
    4: {'host': 'localhost', 'port': 8004},
    5: {'host': 'localhost', 'port': 8005},
}
```

**For distributed setup:**
```python
node_configs = {
    1: {'host': '192.168.1.10', 'port': 8001},  # IP of machine 1
    2: {'host': '192.168.1.11', 'port': 8002},  # IP of machine 2
    3: {'host': '192.168.1.12', 'port': 8003},  # IP of machine 3
    4: {'host': '192.168.1.13', 'port': 8004},  # IP of machine 4
    5: {'host': '192.168.1.14', 'port': 8005},  # IP of machine 5
}
```

### Safety Requirements

- **Total nodes:** `n_nodes = 5`
- **Privacy threshold:** `t = 1` (need t+1 = 2 shares to reconstruct)
- **Adversarial limit:** `s = 1`
- **Safety condition:** `2·(t+s) = 2·(1+1) = 4 < 5` ✓

**Minimum nodes:** `n_nodes ≥ 2·(t+s) + 1`

## Complete Example Scripts

### Script 1: `run_node.py` (Universal node runner)

```python
import argparse
from ml_training import SentraTrainingPipeline
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-id', type=int, required=True)
    parser.add_argument('--n-nodes', type=int, default=5)
    parser.add_argument('--base-port', type=int, default=8000)
    args = parser.parse_args()
    
    # Create node configs
    node_configs = {
        i: {'host': 'localhost', 'port': args.base_port + i}
        for i in range(1, args.n_nodes + 1)
    }
    
    pipeline = SentraTrainingPipeline(
        n_nodes=args.n_nodes,
        t=1, s=1,
        node_id=args.node_id,
        node_configs=node_configs,
        enable_network=True,
        batch_size=16,
        learning_rate=0.01,
        num_epochs=10
    )
    
    # Dataset (same seed for all nodes in testing)
    np.random.seed(42)
    dataset = [np.random.randn(10) for _ in range(100)]
    labels = [np.random.randn(1) for _ in range(100)]
    
    # Train
    pipeline.train(dataset, labels, weight_shapes=[(10, 4), (4, 1)])

if __name__ == '__main__':
    main()
```

### Script 2: Individual node scripts

**`node1.py`:**
```python
from ml_training import SentraTrainingPipeline
import numpy as np

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
    batch_size=16,
    learning_rate=0.01,
    num_epochs=10
)

np.random.seed(42)
dataset = [np.random.randn(10) for _ in range(100)]
labels = [np.random.randn(1) for _ in range(100)]

pipeline.train(dataset, labels, weight_shapes=[(10, 4), (4, 1)])
```

**`node2.py`, `node3.py`, etc.:** Same as above, but change `node_id=2`, `node_id=3`, etc.

## Running Multi-Node Training

### Method 1: Using `run_node.py`

```powershell
# Terminal 1
python run_node.py --node-id 1

# Terminal 2
python run_node.py --node-id 2

# Terminal 3
python run_node.py --node-id 3

# Terminal 4
python run_node.py --node-id 4

# Terminal 5
python run_node.py --node-id 5
```

### Method 2: Using individual scripts

```powershell
# Terminal 1
python node1.py

# Terminal 2
python node2.py

# Terminal 3
python node3.py

# Terminal 4
python node4.py

# Terminal 5
python node5.py
```

### Method 3: Batched secure MNIST launcher (recommended local path)

From `sentra-node/python` (see `node/start_all_nodes_cli.py` for all flags):

```powershell
python node/start_all_nodes.py --headless --distribute-dataset-shares
```

For dataset distribution from a separate client process, use `--receive-dataset-shares-from-client` (and optionally `--start-client-distributor`).

## Network Requirements

### Ports

Each node needs a unique port:
- Node 1: Port 8001
- Node 2: Port 8002
- Node 3: Port 8003
- Node 4: Port 8004
- Node 5: Port 8005

### Firewall

**Windows:**
```powershell
# Allow ports through firewall
New-NetFirewallRule -DisplayName "SENTRA Node 1" -Direction Inbound -LocalPort 8001 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "SENTRA Node 2" -Direction Inbound -LocalPort 8002 -Protocol TCP -Action Allow
# ... etc
```

**Linux:**
```bash
sudo ufw allow 8001/tcp
sudo ufw allow 8002/tcp
# ... etc
```

### Connectivity Test

```powershell
# Test if ports are open
Test-NetConnection -ComputerName localhost -Port 8001
Test-NetConnection -ComputerName localhost -Port 8002
# ... etc
```

## Troubleshooting

### Port Already in Use

```powershell
# Check what's using the port
netstat -ano | findstr :8001

# Kill the process (replace PID)
taskkill /PID <PID> /F
```

### Connection Refused

- Check firewall settings
- Verify all nodes are running
- Check network connectivity between machines
- Verify node_configs have correct IPs/ports

### Nodes Not Synchronizing

- Ensure all nodes use the same `n_nodes`, `t`, `s` values
- Verify all nodes have the same `node_configs` dictionary
- Check that `enable_network=True` on all nodes

### Safety Bound Violation

If `2·(t+s) ≥ n_active`:
- Training will pause
- Need more active nodes or reduce `t`/`s`
- Minimum: `n_nodes ≥ 2·(t+s) + 1`

## Production Deployment

### 1. Configuration File

Create `config.json`:
```json
{
  "n_nodes": 5,
  "t": 1,
  "s": 1,
  "nodes": {
    "1": {"host": "node1.example.com", "port": 8001},
    "2": {"host": "node2.example.com", "port": 8002},
    "3": {"host": "node3.example.com", "port": 8003},
    "4": {"host": "node4.example.com", "port": 8004},
    "5": {"host": "node5.example.com", "port": 8005}
  }
}
```

### 2. Load Configuration

```python
import json

with open('config.json') as f:
    config = json.load(f)

node_configs = {
    int(k): v for k, v in config['nodes'].items()
}

pipeline = SentraTrainingPipeline(
    n_nodes=config['n_nodes'],
    t=config['t'],
    s=config['s'],
    node_id=1,  # Set per node
    node_configs=node_configs,
    enable_network=True
)
```

### 3. Deploy on Each Node

- Copy code to each machine
- Set `node_id` appropriately
- Ensure network connectivity
- Run training script

## Testing

### Quick Test (Single Machine)

```powershell
# Test multi-node simulation
python test_multi_node.py --test multi
```

### Full Test (Multiple Terminals)

```powershell
# Terminal 1
python run_node.py --node-id 1 --n-nodes 3

# Terminal 2
python run_node.py --node-id 2 --n-nodes 3

# Terminal 3
python run_node.py --node-id 3 --n-nodes 3
```

## Summary

1. **Configure nodes:** Create `node_configs` with host/port for each node
2. **Set node_id:** Each script uses a unique `node_id` (1, 2, 3, ...)
3. **Enable network:** Set `enable_network=True`
4. **Run on each node:** Start the script on each machine/terminal
5. **Verify connectivity:** Check ports and firewall
6. **Monitor:** Watch for safety bound violations and errors

