#!/usr/bin/env bash
# Entrypoint for the non-SGX benchmark container. Forwards to the Python driver unless
# a bare shell is requested (useful for debugging inside the container).
set -euo pipefail
cd /workspace/node

if [[ "${1:-}" == "bash" || "${1:-}" == "sh" ]]; then
  exec "$@"
fi

exec python3 /usr/local/bin/run_benchmark_from_config.py "$@"
