# ML benchmark (Docker)

Occlum **base** image with full `sentra-node/python/node` + `client`, pinned pip deps, YAML profiles, and `start_all_nodes` orchestration.

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

## Other

- Packaged **Sentra node** demonstrator (Rust + Occlum package): `../sentra-node/README.md` and `make build-sentra-sgx` from this folder’s Makefile.
