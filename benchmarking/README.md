# SENTRA benchmarking

Two sibling folders:

| Folder | Contents |
|--------|----------|
| **`ml-benchmark/`** | Docker image for **reproducible secure MNIST** runs (`Dockerfile.ml-benchmark`, YAML `configs/`, Compose, `start_all_nodes`). |
| **`sentra-node/`** | **Demonstrator node** images: packaged Occlum/SGX and legacy NoSGX (`Dockerfile.sentra-node-sgx`, `Dockerfile.sentra-node-nosgx`), Rust `sentra_node`, BOM, `training_config.yaml`. |

- ML quick start: `benchmarking/ml-benchmark/README.md`
- Sentra node build: `benchmarking/sentra-node/README.md`

`sentra-deployment/docker-compose.yaml` points at **`benchmarking/sentra-node/`** Dockerfiles.
