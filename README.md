# SENTRA

Secure, distributed ML training over secret shares with MPC-style coordination.

This repository contains a practical SENTRA training stack for:
- Multi-node secure training
- Secret-shared matrix operations
- Secure/nonlinear approximations for classification (MNIST MLP)
- Reproducible experiments in both protocol-compliant and comparison modes

## Highlights

- Multi-node orchestration via `start_all_nodes.py`
- Batched secure MNIST path via `run_mnist_batched_secure.py`
- Tunable fixed-point configuration (`field-size`, `scale-factor`)
- Stability instrumentation (`max|logit|`, gradient norm, update stats)
- Two softmax-gradient modes:
  - `secure_approx`: protocol-compliant SENTRA path
  - `opened_exact`: accelerated comparison variant (not vanilla SENTRA)

## Repository Structure

- `ml_training/` core secure training, communication, secret-sharing, MPC ops
- `run_mnist_batched_secure.py` batched secure MNIST runner (single node process)
- `start_all_nodes.py` launcher for multi-node runs
- `testing/` regression/integration tests and smoke checks
- `attestation/` enclave attestation-related assets

## Architecture Diagram

```text
                           +-----------------------+
                           |   start_all_nodes.py  |
                           |  (process launcher)   |
                           +-----------+-----------+
                                       |
                -------------------------------------------------
                |                       |                       |
        +-------v--------+      +-------v--------+      +-------v--------+
        |    Node 1      |      |    Node 2      |      |    Node 3      |
        | run_node /     |      | run_node /     |      | run_node /     |
        | batched runner |      | batched runner |      | batched runner |
        +-------+--------+      +-------+--------+      +-------+--------+
                |                       |                       |
                |<------ secure comm + sync/barriers --------->|
                |         (ml_training/secure_comm.py)         |
                |                       |                       |
                +-----------+-----------+-----------+-----------+
                            |                       |
                    +-------v--------+      +-------v--------+
                    | Secret Sharing |      | Secure MPC Ops |
                    | + Versioned KV |      | Beaver / ReLU  |
                    | (data, weights)|      | Softmax / Div  |
                    +-------+--------+      +-------+--------+
                            |                       |
                            +-----------+-----------+
                                        |
                               +--------v---------+
                               |  Training Loop   |
                               | fwd/bwd/update   |
                               +--------+---------+
                                        |
                               +--------v---------+
                               |  Eval + Metrics  |
                               | accuracy / loss  |
                               +------------------+
```

Data/weight path summary:
- Dataset and model parameters are represented as secret shares.
- Nodes coordinate each batch through network synchronization and MPC primitives.
- Output-layer gradient mode is selectable (`secure_approx` vs `opened_exact`).
- Epoch metrics are reconstructed only for reporting/evaluation.

## Prerequisites

- Python 3.10+
- TensorFlow (for MNIST loading/evaluation paths)
- Windows/WSL/Linux environment with multiple local processes allowed

Install dependencies (minimum):

```bash
pip install tensorflow pytest numpy
```

## Quick Start

### 1) SENTRA-Compliant Baseline (Recommended)

This mode keeps softmax gradient in the secure approximation path.

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9600 --batched \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000 \
  --learning-rate 0.002 --loss-mode softmax \
  --field-size 2305843009213693951 \
  --scale-factor 65536 --softmax-temperature 2 \
  --exp-approx pade22 --softmax-grad-mode secure_approx \
  --grad-clip 0.50 --logit-clip 4.0 \
  --explode-logit-threshold 100 --loss-growth-threshold 5 \
  --grad-norm-threshold 1500 --no-abort-on-instability \
  --debug-numerics --seed 2026
```

### 2) Accelerated Comparison Variant

This mode uses opened exact output-layer softmax gradient, then re-shares it.
Use for benchmarking/ablation, not as vanilla SENTRA protocol claim.

```bash
python3 start_all_nodes.py --n-nodes 3 --base-port 9700 --batched \
  --num-epochs 8 --batch-size 64 --mnist-samples 10000 \
  --learning-rate 0.002 --loss-mode softmax \
  --field-size 2305843009213693951 \
  --scale-factor 65536 --softmax-temperature 2 \
  --exp-approx pade22 --softmax-grad-mode opened_exact \
  --grad-clip 0.50 --logit-clip 4.0 \
  --explode-logit-threshold 100 --loss-growth-threshold 5 \
  --grad-norm-threshold 1500 --no-abort-on-instability \
  --debug-numerics --seed 2026
```

## Mode Semantics

| Mode | Security posture | Typical accuracy/runtime |
|---|---|---|
| `secure_approx` | Protocol-compliant SENTRA nonlinear path | More conservative, protocol-faithful |
| `opened_exact` | Opens output logits/targets for exact softmax gradient at opener, then re-shares | Better convergence/accuracy; comparison-only variant |

## Stability and Debugging

When `--debug-numerics` is enabled, node logs include:
- `Epoch X Diagnostics`: mean/max logit magnitude and entropy
- `Estimated Grad Norm`
- `Update Sample Stats`
- Probe vectors (`logits`, `dz2`, reconstructed probability estimate)

Healthy signs:
- `probs_est stats sum` near `1.0`
- `probs_est min >= 0`
- no rapid growth in `max|logit|`
- decreasing test loss over epochs

## Testing

Run core fast tests:

```bash
python -m pytest -q testing/test_forward_scaling.py testing/test_gradient_scaling.py
```

Run stage-invariant integration check:

```bash
python -m pytest -q testing/test_batched_stage_invariants.py -s
```

## Benchmark Snapshot

Recent validated local results (3 nodes, MNIST batched secure path):

- **SENTRA-compliant mode** (`--softmax-grad-mode secure_approx`):
  - stable learning observed
  - representative run reached ~`79%` test accuracy by epoch 7 (100-sample eval)
- **Accelerated comparison mode** (`--softmax-grad-mode opened_exact`):
  - higher/faster convergence expected
  - representative run reached ~`80%` test accuracy by epoch 8 (100-sample eval)

Reference command family:
- `field-size`: `2305843009213693951` (`2^61 - 1`)
- `scale-factor`: `65536`
- `exp-approx`: `pade22`
- `softmax-temperature`: `2`
- `batch-size`: `64`
- `mnist-samples`: `10000`

Note: 100-sample evaluation is useful for fast iteration. For reporting, use larger evaluation sets (e.g., 1000+ or full test set).

## Common Issues

### Ports already in use

```bash
pkill -f "start_all_nodes.py|run_mnist_batched_secure.py" || true
```

Then restart with a fresh `--base-port`.

### Broken pipe / reset by peer

Usually one node exited early. Check per-node logs and ensure all nodes use identical flags.

### Non-integer temperature error

In fixed-point mode, temperature is currently integer-only (`1`, `2`, ...).

## Reproducibility Checklist

For published runs, always record:
- full command line
- `seed`
- `field-size` and `scale-factor`
- `softmax-grad-mode`
- epoch-wise accuracy/loss/diagnostics

## Notes

- This README reflects the current batched MNIST secure path and tested commands.
- `USAGE_GUIDE.md` may include older commands/scripts kept for legacy reference.
