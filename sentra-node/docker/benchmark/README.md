# SENTRA Docker benchmarks (non-SGX)

This directory holds a **reproducible, non–Intel-SGX** benchmark path: the same secure MNIST multi-party code as the rest of `sentra-node/python/node`, running as several Python processes on `127.0.0.1` inside one Linux container.

The existing **Occlum / SGX** demonstrator image remains at `sentra-node/docker/SentraNode.dockerfile`. When SGX execution is integrated, keep **shared YAML profiles** under `configs/` so reviewers can run `nosgx-*.yaml` and `sgx-*.yaml` (or the same file with a different image) for apples-to-apples comparisons.

## Prerequisites

- Docker with BuildKit enabled (default on current Docker Desktop / Engine).
- Build context is always the **repository root** (the folder that contains `sentra-node/`), because the Dockerfile copies `sentra-node/python/...` into the image.

### WSL 2: `docker` could not be found

If you run `make build-nosgx` from Ubuntu-on-WSL and see *The command 'docker' could not be found*:

1. Install and start **Docker Desktop on Windows** (not only WSL).
2. **Docker Desktop → Settings → Resources → WSL integration** — enable your distro, then restart Docker Desktop.
3. Confirm in the same WSL shell: `docker version`.

Until integration is on, use **PowerShell or CMD on Windows** where Docker Desktop installs the `docker` CLI, or install a Docker engine inside WSL separately.

### `/usr/bin/env: 'bash\r': No such file or directory`

Shell scripts copied from **Windows drives mounted in WSL** (`/mnt/c/...`) often keep **CRLF** line endings, so the container’s kernel sees a bad shebang (`bash\r`). The benchmark image now runs **`sed`** during `docker build` to strip trailing `\r` from the entrypoint and driver script, and uses **`ENTRYPOINT ["/bin/bash", "..."]`** so startup does not depend on the shebang line.

If you still see the error after pulling these changes, force a rebuild: `docker compose build --no-cache benchmark-nosgx`.

The repo root `.gitattributes` forces LF for `sentra-node/docker/**/*.sh`; run `git add --renormalize sentra-node/docker` once if files were already committed with CRLF.

## Quick start

From **this directory** (`sentra-node/docker/benchmark/`):

```bash
make build-nosgx
make run-quick
```

Or with Compose directly:

```bash
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml run --rm benchmark-nosgx
```

The default container command runs `configs/nosgx-quick.yaml` (small MNIST slice, few epochs).

## Profiles (`configs/`)

| File | Purpose |
|------|---------|
| `nosgx-quick.yaml` | Short smoke test for CI / first-time reviewers. |
| `nosgx-standard.yaml` | Larger run; appends a row to `logs/docker_benchmark_runs.xlsx` when finished. |
| `nosgx-with-client.yaml` | Headless nodes plus `python -m sentra_client` (dataset + optional client eval). |

YAML uses **flat** keys that match `start_all_nodes.py` CLI flags (hyphenated names). Booleans: `true` emits the flag, `false` omits it.

### Overrides

Pass extra CLI tokens after `--` (forwarded to `start_all_nodes.py`):

```bash
docker compose -f docker-compose.yml run --rm benchmark-nosgx \
  --config /workspace/benchmark-configs/nosgx-quick.yaml -- \
  --mnist-samples 128
```

### Logs and results

Compose mounts a named volume on `logs/` inside the workspace so repeated runs keep history. To copy artifacts out:

```bash
docker volume ls | grep benchmark_logs
docker run --rm -v <volume_name>:/data alpine tar cvf - -C /data . > logs-backup.tar
```

Alternatively, bind-mount a host directory by editing `docker-compose.yml` (replace the `benchmark_logs` volume with `./host-logs:/workspace/node/logs`).

## Build without Make

From **repository root**:

```bash
docker build -f sentra-node/docker/benchmark/Dockerfile.nosgx -t sentra-benchmark:nosgx .
docker run --rm sentra-benchmark:nosgx --config /workspace/benchmark-configs/nosgx-quick.yaml
```

## Reproducibility notes

- Python dependency pins live in `requirements-benchmark.txt`. Change them intentionally and re-record benchmark tables when you publish new numbers.
- `seed` is set in each profile; keep it fixed across non-SGX vs SGX comparisons unless you intentionally study variance.
- TensorFlow may still introduce small numerical variance across CPU models; document hardware when reporting wall times.

## Extending for SGX (for a follow-up change)

1. Add a second Dockerfile or multi-stage target that produces an Occlum/SGX image (or reuse `SentraNode.dockerfile`).
2. Add a Compose service (e.g. `benchmark-sgx`) with `profiles: ["sgx"]` so `docker compose --profile sgx run ...` stays optional.
3. Prefer the **same** `configs/*.yaml` with a different `image:` or `entrypoint:` so methodology stays aligned with this non-SGX baseline.

## Shell inside the image

```bash
make shell
# or
docker compose -f docker-compose.yml run --rm benchmark-nosgx bash
```

## Local run (no Docker)

Install dependencies from `sentra-node/python/node/pyproject.toml` and `sentra-node/python/client/requirements-client.txt`, then:

```bash
python sentra-node/docker/benchmark/run_benchmark_from_config.py \
  -c sentra-node/docker/benchmark/configs/nosgx-quick.yaml
```
(Execute from the **repository root** so `sentra-node/python/node` is resolved automatically.)

If the driver script is not under `sentra-node/docker/benchmark/`, set `SENTRA_NODE_ROOT` to the absolute path of `sentra-node/python/node`.
