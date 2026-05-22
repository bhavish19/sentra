#!/bin/bash

export PYTHONPATH=/workspace/node:/workspace/client
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export HOME="${HOME:-/root}"
export SENTRA_IN_OCCLUM=1
export SENTRA_BEAVER_POOL_SIZE="${SENTRA_BEAVER_POOL_SIZE:-8000}"
cd /workspace/node

exec /bin/python3 /usr/local/bin/run_benchmark_from_config.py "$@"
