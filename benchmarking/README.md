# SENTRA benchmarking

Two sibling folders:

| Folder | Contents |
|--------|----------|
| **`ml-benchmark/`** | Docker image for **reproducible secure MNIST** runs (`Dockerfile.ml-benchmark`, YAML `configs/`, Compose, `start_all_nodes`). |
| **`sentra-node/`** | **Demonstrator node** images: packaged Occlum/SGX and legacy NoSGX (`Dockerfile.sentra-node-sgx`, `Dockerfile.sentra-node-nosgx`), Rust `sentra_node`, BOM, `training_config.yaml`. |
| **`crypten-benchmark/`** | **CrypTen** MPC baseline (784-128-10 MNIST) for comparison with SENTRA non-SGX / SGX. |

- ML quick start: `benchmarking/ml-benchmark/README.md`
- CrypTen comparison: `benchmarking/crypten-benchmark/README.md`
- **Three-way table:** `benchmarking/COMPARISON.md` and `python benchmarking/compare_benchmarks.py`
- Sentra node build: `benchmarking/sentra-node/README.md`

`sentra-deployment/docker-compose.yaml` points at **`benchmarking/sentra-node/`** Dockerfiles.
