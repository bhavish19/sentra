# ML benchmark (Docker)

Occlum **base** image with full `sentra-node/python/node` + `client`, pinned pip deps, YAML profiles, and `start_all_nodes` orchestration.

## MNIST data (local only, no download)

Bundled file: **`demonstrator/backend/resources/mnist.npz`** (baked into images at **`/mnist.npz`**).

| Profile | Who loads MNIST |
|---------|-----------------|
| `quick.yaml` / `train-export-assets.yaml` | **Owner node** (node 1) from `/mnist.npz` |
| `with-client.yaml` / `with-client-sgx.yaml` | **Client only**; nodes receive secret shares (no raw MNIST on nodes) |

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

**SGX all-in-one:** use `make sgx-run-with-client`, which loads **`with-client-sgx.yaml`** (3 nodes, smaller Beaver pool, no `debug-numerics`) so node + client processes fit in one Occlum enclave (~5.4GB). Your `sgx-run-quick` success already shows 3 parties work in SGX.

After pulling changes, rebuild and run on the SGX host:

```bash
make build-ml-sgx
make sgx-run-with-client
```

**4 parties + client in SGX** inside one container usually fails (EPC / RAM). Options:

1. Keep **3 nodes** for SGX client benchmarks (`with-client-sgx.yaml`).
2. Use **`sentra-deployment`** `multi-node` profile — **one Occlum container per party** (production layout).
3. Non-SGX **`make run-with-client`** for the full 4-node client profile.

Logs: `/workspace/node/logs/run_<timestamp>/` (Docker volume `ml_benchmark_logs`). On failure, tail `node_*.log` and `client_distributor.log` there.

## Other

- Packaged **Sentra node** demonstrator (Rust + Occlum package): `../sentra-node/README.md` and `make build-sentra-sgx` from this folder’s Makefile.
