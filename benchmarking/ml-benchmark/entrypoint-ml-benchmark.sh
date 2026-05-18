#!/bin/bash
# ML benchmark image: bash|sh, *.py, YAML driver (--config), or SENTRA_OCCLUM_RUN=1.
set -euo pipefail
export PYTHONPATH=/workspace/node:/workspace/client
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
cd /workspace/node

if [[ "${1:-}" == "bash" || "${1:-}" == "sh" ]]; then
  exec "$@"
fi

if [[ "${1:-}" == "sgx" ]]; then
  cd /occlum-instance
  exec occlum run /bin/enclave_run_script.sh $2 $3
fi

if [[ "${1:-}" == *.py ]]; then
  exec /python-occlum/bin/python "$@"
fi

exec /python-occlum/bin/python /usr/local/bin/run_benchmark_from_config.py "$@"
