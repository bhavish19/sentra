#!/usr/bin/env bash
set -euo pipefail

# WSL/Linux helper to run SENTRA MNIST multi-node in background with logs.
# Usage:
#   bash start_mnist_nodes.sh
#   bash start_mnist_nodes.sh 3 8000 1 8 128 64 16 1000 180 2026 hybrid

N_NODES="${1:-3}"
BASE_PORT="${2:-8000}"
EPOCHS="${3:-1}"
BATCH_SIZE="${4:-8}"
MNIST_SAMPLES="${5:-128}"
MNIST_INPUT_DIM="${6:-64}"
MNIST_HIDDEN_DIM="${7:-16}"
MNIST_TEST_SAMPLES="${8:-1000}"
POST_METRICS_BARRIER_TIMEOUT="${9:-180}"
SEED="${10:-2026}"
TRAIN_MODE="${11:-secure}"

LOG_DIR="logs/mnist_nodes_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "======================================================================"
echo "Starting SENTRA MNIST Multi-Node"
echo "======================================================================"
echo "Nodes: ${N_NODES}"
echo "Base port: ${BASE_PORT}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "MNIST samples: ${MNIST_SAMPLES}"
echo "MNIST dims: ${MNIST_INPUT_DIM} -> ${MNIST_HIDDEN_DIM} -> 10"
echo "MNIST test samples: ${MNIST_TEST_SAMPLES}"
echo "Post-metrics barrier timeout: ${POST_METRICS_BARRIER_TIMEOUT}s"
echo "Seed: ${SEED}"
echo "Train mode: ${TRAIN_MODE}"
echo "Logs: ${LOG_DIR}"
echo "======================================================================"
echo

PIDS_FILE="${LOG_DIR}/pids.txt"
touch "${PIDS_FILE}"

for i in $(seq 1 "${N_NODES}"); do
  LOG_FILE="${LOG_DIR}/node_${i}.log"
  echo "Starting Node ${i} -> ${LOG_FILE}"
  CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 python3 run_node.py \
    --node-id "${i}" \
    --n-nodes "${N_NODES}" \
    --base-port "${BASE_PORT}" \
    --batch-size "${BATCH_SIZE}" \
    --num-epochs "${EPOCHS}" \
    --t 1 \
    --s 1 \
    --dataset mnist \
    --mnist-samples "${MNIST_SAMPLES}" \
    --mnist-input-dim "${MNIST_INPUT_DIM}" \
    --mnist-hidden-dim "${MNIST_HIDDEN_DIM}" \
    --mnist-test-samples "${MNIST_TEST_SAMPLES}" \
    --post-metrics-barrier-timeout "${POST_METRICS_BARRIER_TIMEOUT}" \
    --seed "${SEED}" \
    --train-mode "${TRAIN_MODE}" \
    --no-wait > "${LOG_FILE}" 2>&1 &
  echo "$!" >> "${PIDS_FILE}"
  sleep 1
done

echo
echo "Started ${N_NODES} nodes."
echo "PIDs saved in ${PIDS_FILE}"
echo "Tail logs: tail -f ${LOG_DIR}/node_1.log"
echo "Stop all:"
echo "  while read -r pid; do kill \"\$pid\" 2>/dev/null || true; done < ${PIDS_FILE}"
