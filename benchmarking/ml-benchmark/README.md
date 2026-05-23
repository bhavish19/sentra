# ML benchmark (Docker)

Occlum **base** image with full `sentra-node/python/node` + `client`, pinned pip deps, YAML profiles, and `start_all_nodes` orchestration.

## MNIST data (local only, no download)

Bundled file: **`demonstrator/backend/resources/mnist.npz`** (baked into images at **`/mnist.npz`**).

| Profile | Who loads MNIST |
|---------|-----------------|
| `quick.yaml` / `train-export-assets.yaml` | **Owner node** (node 1) from `/mnist.npz` |
| `with-client.yaml` | **Client only**; nodes receive secret shares (non-SGX and SGX all-in-one) |

- **Docker:** `MNIST_NPZ_PATH=/mnist.npz` in compose.
- **Rebuild after client/MNIST changes:** `docker compose build --no-cache ml-benchmark ml-benchmark-sgx`

## From this directory

```bash
make build-ml
make run-quick
```

Compose uses **repository root** as build context (`../..` from here).

## From repository root

```bash
docker compose -f benchmarking/ml-benchmark/docker-compose.yml build
docker compose -f benchmarking/ml-benchmark/docker-compose.yml run --rm ml-benchmark
```

## Local driver (no Docker)

```bash
python benchmarking/ml-benchmark/run_benchmark_from_config.py \
  -c benchmarking/ml-benchmark/configs/quick.yaml
```

## SGX with client (`make sgx-run-with-client`)

**Non-SGX:** `make run-with-client` uses `with-client.yaml` (4 nodes).

**SGX and non-SGX** both use **`with-client.yaml`** (3 nodes, client-side MNIST):

```bash
make run-with-client          # non-SGX
make sgx-run-with-client      # SGX (same config)
```

For **4+ parties in SGX**, use `sentra-deployment` multi-node (one container per party).

Logs: `/workspace/node/logs/run_<timestamp>/` (Docker volume `ml_benchmark_logs`). On failure, tail `node_*.log` and `client_distributor.log` there.

## Other

- Packaged **Sentra node** demonstrator (Rust + Occlum package): `../sentra-node/README.md` and `make build-sentra-sgx` from this folder’s Makefile.
