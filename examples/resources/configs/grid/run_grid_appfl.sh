#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/sjzhz/miniconda3/envs/fl/bin/python}"
APPFL_NUM_CLIENTS="${APPFL_NUM_CLIENTS:-2}"
APPFL_NUM_GLOBAL_EPOCHS="${APPFL_NUM_GLOBAL_EPOCHS:-10}"
APPFL_NUM_LOCAL_EPOCHS="${APPFL_NUM_LOCAL_EPOCHS:-1}"
APPFL_SERVER_URI="${APPFL_SERVER_URI:-127.0.0.1:50051}"
APPFL_SERVER_START_TIMEOUT="${APPFL_SERVER_START_TIMEOUT:-1200}"
APPFL_DEVICE="${APPFL_DEVICE:-cpu}"
APPFL_GRIDFM_CLIENT_CASES="${APPFL_GRIDFM_CLIENT_CASES:-}"
APPFL_GRIDFM_SERVER_CASES="${APPFL_GRIDFM_SERVER_CASES:-${APPFL_GRIDFM_CLIENT_CASES}}"
APPFL_GRIDFM_EVAL_CASES="${APPFL_GRIDFM_EVAL_CASES:-}"
APPFL_GRIDFM_NORMALIZER_STATS="${APPFL_GRIDFM_NORMALIZER_STATS:-}"
APPFL_GRIDFM_BUILD_SHARED_NORMALIZER="${APPFL_GRIDFM_BUILD_SHARED_NORMALIZER:-0}"

# Choose the GridFM datakit output root. Override from the shell, or uncomment
# one of these lines.
# APPFL_GRIDFM_DATA_PATH="/home/sjzhz/tmp/gridfm_case30_smoke"
# APPFL_GRIDFM_DATA_PATH="/home/sjzhz/tmp/gridfm_case30_full"
# APPFL_GRIDFM_DATA_PATH="/home/sjzhz/data"
# APPFL_GRIDFM_DATA_PATH="/home/sjzhz/tmp/gridfm_case30_full_appfl"
# APPFL_GRIDFM_DATA_PATH="/home/sjzhz/tmp/gridfm_case30_smoke_appfl"
APPFL_GRIDFM_DATA_PATH="${APPFL_GRIDFM_DATA_PATH:-/home/sjzhz/tmp/gridfm_case30_smoke}"
APPFL_CLIENT_HOME_BASE="${APPFL_CLIENT_HOME_BASE:-/tmp/appfl_client_homes}"

# Optional quality/runtime knobs. Defaults match gridfm_graphkit.yaml.
# APPFL_GRIDFM_SCENARIOS=10000
# APPFL_GRIDFM_CLIENT_CASES=case24_ieee_rts,case30_ieee
# APPFL_GRIDFM_SERVER_CASES=case24_ieee_rts,case30_ieee
# APPFL_GRIDFM_EVAL_CASES=case24_ieee_rts,case30_ieee,case14_ieee,case57_ieee
# APPFL_GRIDFM_NORMALIZER_STATS=/path/to/shared_normalizer_stats.pt
# APPFL_GRIDFM_BUILD_SHARED_NORMALIZER=1
# APPFL_GRIDFM_NUM_LAYERS=4
# APPFL_GRIDFM_WORKERS=0
# APPFL_SERVER_START_TIMEOUT=1200
# APPFL_DEVICE=cuda:0

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${EXAMPLES_DIR}/output/grid_run_${RUN_ID}"
FINAL_MODEL_PATH="${LOG_DIR}/final_global_model.pt"
mkdir -p "${LOG_DIR}"

if [[ -n "${APPFL_GRIDFM_CLIENT_CASES}" && "${APPFL_NUM_CLIENTS}" == "2" ]]; then
    IFS="," read -r -a _case_array <<< "${APPFL_GRIDFM_CLIENT_CASES}"
    APPFL_NUM_CLIENTS="${#_case_array[@]}"
fi

SERVER_PID=""
CLIENT_PIDS=()

cleanup() {
    for pid in "${CLIENT_PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ "${APPFL_DEVICE}" == cuda* ]]; then
    if ! "${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print(f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}", file=sys.stderr)
    print("PyTorch cannot see CUDA in this environment.", file=sys.stderr)
    sys.exit(1)
print(f"CUDA devices: {torch.cuda.device_count()} ({torch.cuda.get_device_name(0)})")
PY
    then
        echo "APPFL_DEVICE=${APPFL_DEVICE} requested, but CUDA is unavailable to PyTorch." >&2
        echo "Install a CUDA-enabled PyTorch build in ${PYTHON_BIN%/bin/python}, or run with APPFL_DEVICE=cpu." >&2
        exit 1
    fi
fi

if ! "${PYTHON_BIN}" - <<'PY'
import torch
import torch_scatter
import torch_geometric

print(
    "PyG stack:",
    f"torch={torch.__version__}",
    f"torch_scatter={torch_scatter.__version__}",
    f"torch_geometric={torch_geometric.__version__}",
)
PY
then
    echo "PyTorch Geometric extensions are not importable." >&2
    echo "Reinstall torch-scatter/torch-geometric wheels matching the installed torch build." >&2
    exit 1
fi

if [[ ! -d "${APPFL_GRIDFM_DATA_PATH}" ]]; then
    echo "GridFM data path does not exist: ${APPFL_GRIDFM_DATA_PATH}" >&2
    echo "Set APPFL_GRIDFM_DATA_PATH or generate the datakit data first." >&2
    exit 1
fi

if [[ "${APPFL_GRIDFM_BUILD_SHARED_NORMALIZER}" == "1" ]]; then
    if [[ -z "${APPFL_GRIDFM_SERVER_CASES}" && -z "${APPFL_GRIDFM_CLIENT_CASES}" ]]; then
        echo "APPFL_GRIDFM_BUILD_SHARED_NORMALIZER=1 requires APPFL_GRIDFM_SERVER_CASES or APPFL_GRIDFM_CLIENT_CASES." >&2
        exit 1
    fi
    if [[ -z "${APPFL_GRIDFM_SCENARIOS:-}" ]]; then
        echo "APPFL_GRIDFM_BUILD_SHARED_NORMALIZER=1 requires APPFL_GRIDFM_SCENARIOS." >&2
        exit 1
    fi
    _fit_cases="${APPFL_GRIDFM_SERVER_CASES:-${APPFL_GRIDFM_CLIENT_CASES}}"
    _apply_cases="${_fit_cases}"
    if [[ -n "${APPFL_GRIDFM_EVAL_CASES}" ]]; then
        _apply_cases="${_apply_cases},${APPFL_GRIDFM_EVAL_CASES}"
    fi
    APPFL_GRIDFM_NORMALIZER_STATS="${LOG_DIR}/shared_normalizer_stats.pt"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_shared_normalizer_stats.py" \
        --data-path "${APPFL_GRIDFM_DATA_PATH}" \
        --fit-cases "${_fit_cases}" \
        --apply-cases "${_apply_cases}" \
        --scenarios "${APPFL_GRIDFM_SCENARIOS}" \
        --output "${APPFL_GRIDFM_NORMALIZER_STATS}" \
        --split-by-load-scenario-idx \
        > "${LOG_DIR}/shared_normalizer.log" 2>&1
fi

IFS=":" read -r SERVER_HOST SERVER_PORT <<< "${APPFL_SERVER_URI}"
if [[ -z "${SERVER_HOST}" || -z "${SERVER_PORT}" ]]; then
    echo "APPFL_SERVER_URI must look like host:port, got: ${APPFL_SERVER_URI}" >&2
    exit 1
fi

cd "${EXAMPLES_DIR}"

echo "APPFL grid run: ${RUN_ID}"
echo "  examples: ${EXAMPLES_DIR}"
echo "  python:   ${PYTHON_BIN}"
echo "  data:     ${APPFL_GRIDFM_DATA_PATH}"
echo "  server:   ${APPFL_SERVER_URI}"
echo "  clients:  ${APPFL_NUM_CLIENTS}"
echo "  global epochs: ${APPFL_NUM_GLOBAL_EPOCHS}"
echo "  local epochs:  ${APPFL_NUM_LOCAL_EPOCHS}"
echo "  start timeout: ${APPFL_SERVER_START_TIMEOUT}s"
echo "  client device: ${APPFL_DEVICE}"
if [[ -n "${APPFL_GRIDFM_SCENARIOS:-}" ]]; then
    echo "  scenarios:     ${APPFL_GRIDFM_SCENARIOS}"
fi
if [[ -n "${APPFL_GRIDFM_CLIENT_CASES:-}" ]]; then
    echo "  client cases:  ${APPFL_GRIDFM_CLIENT_CASES}"
fi
if [[ -n "${APPFL_GRIDFM_SERVER_CASES:-}" ]]; then
    echo "  server cases:  ${APPFL_GRIDFM_SERVER_CASES}"
fi
if [[ -n "${APPFL_GRIDFM_EVAL_CASES:-}" ]]; then
    echo "  eval cases:    ${APPFL_GRIDFM_EVAL_CASES}"
fi
if [[ -n "${APPFL_GRIDFM_NORMALIZER_STATS:-}" ]]; then
    echo "  normalizer:    ${APPFL_GRIDFM_NORMALIZER_STATS}"
fi
if [[ -n "${APPFL_GRIDFM_NUM_LAYERS:-}" ]]; then
    echo "  model layers:  ${APPFL_GRIDFM_NUM_LAYERS}"
fi
echo "  logs:     ${LOG_DIR}"
echo "  final model: ${FINAL_MODEL_PATH}"

APPFL_NUM_CLIENTS="${APPFL_NUM_CLIENTS}" \
APPFL_NUM_GLOBAL_EPOCHS="${APPFL_NUM_GLOBAL_EPOCHS}" \
APPFL_NUM_LOCAL_EPOCHS="${APPFL_NUM_LOCAL_EPOCHS}" \
APPFL_SERVER_URI="${APPFL_SERVER_URI}" \
APPFL_DEVICE="${APPFL_DEVICE}" \
APPFL_GLOBAL_MODEL_PATH="${FINAL_MODEL_PATH}" \
APPFL_GRIDFM_DATA_PATH="${APPFL_GRIDFM_DATA_PATH}" \
APPFL_GRIDFM_SCENARIOS="${APPFL_GRIDFM_SCENARIOS:-}" \
APPFL_GRIDFM_CLIENT_CASES="${APPFL_GRIDFM_CLIENT_CASES}" \
APPFL_GRIDFM_SERVER_CASES="${APPFL_GRIDFM_SERVER_CASES}" \
APPFL_GRIDFM_NORMALIZER_STATS="${APPFL_GRIDFM_NORMALIZER_STATS}" \
APPFL_GRIDFM_NUM_LAYERS="${APPFL_GRIDFM_NUM_LAYERS:-}" \
APPFL_GRIDFM_WORKERS="${APPFL_GRIDFM_WORKERS:-}" \
"${PYTHON_BIN}" notebook_tutorials/grid/run_server_grid.py \
    > "${LOG_DIR}/server.log" 2>&1 &
SERVER_PID="$!"

echo "Started server pid=${SERVER_PID}; waiting for ${APPFL_SERVER_URI}..."
for _ in $(seq 1 "${APPFL_SERVER_START_TIMEOUT}"); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server exited before it became ready. See ${LOG_DIR}/server.log" >&2
        exit 1
    fi
    if (echo > "/dev/tcp/${SERVER_HOST}/${SERVER_PORT}") >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! (echo > "/dev/tcp/${SERVER_HOST}/${SERVER_PORT}") >/dev/null 2>&1; then
    echo "Timed out waiting for server. See ${LOG_DIR}/server.log" >&2
    echo "If this is the first full-data run, preprocessing may still be running." >&2
    echo "Try a larger APPFL_SERVER_START_TIMEOUT, or rerun after preprocessing resumes." >&2
    exit 1
fi

for client_id in $(seq 1 "${APPFL_NUM_CLIENTS}"); do
    APPFL_CLIENT_ID="${client_id}" \
    APPFL_NUM_CLIENTS="${APPFL_NUM_CLIENTS}" \
    APPFL_SERVER_URI="${APPFL_SERVER_URI}" \
    APPFL_DEVICE="${APPFL_DEVICE}" \
    APPFL_GRIDFM_DATA_PATH="${APPFL_GRIDFM_DATA_PATH}" \
    APPFL_GRIDFM_SCENARIOS="${APPFL_GRIDFM_SCENARIOS:-}" \
    APPFL_GRIDFM_CLIENT_CASES="${APPFL_GRIDFM_CLIENT_CASES}" \
    APPFL_GRIDFM_NORMALIZER_STATS="${APPFL_GRIDFM_NORMALIZER_STATS}" \
    APPFL_GRIDFM_NUM_LAYERS="${APPFL_GRIDFM_NUM_LAYERS:-}" \
    APPFL_GRIDFM_WORKERS="${APPFL_GRIDFM_WORKERS:-}" \
    APPFL_CLIENT_HOME_BASE="${APPFL_CLIENT_HOME_BASE}" \
    "${PYTHON_BIN}" notebook_tutorials/grid/run_client_grid.py \
        > "${LOG_DIR}/client_${client_id}.log" 2>&1 &
    CLIENT_PIDS+=("$!")
    echo "Started client ${client_id} pid=${CLIENT_PIDS[-1]}"
done

failed=0
for i in "${!CLIENT_PIDS[@]}"; do
    client_id=$((i + 1))
    if wait "${CLIENT_PIDS[$i]}"; then
        echo "Client ${client_id} finished"
    else
        echo "Client ${client_id} failed. See ${LOG_DIR}/client_${client_id}.log" >&2
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi

if [[ -n "${SERVER_PID}" ]]; then
    if wait "${SERVER_PID}"; then
        echo "Server finished"
        SERVER_PID=""
    else
        echo "Server failed during shutdown. See ${LOG_DIR}/server.log" >&2
        exit 1
    fi
fi

if [[ -n "${APPFL_GRIDFM_EVAL_CASES:-}" ]]; then
    if [[ ! -f "${FINAL_MODEL_PATH}" ]]; then
        echo "Final model was not saved: ${FINAL_MODEL_PATH}" >&2
        exit 1
    fi
    APPFL_GRIDFM_DATA_PATH="${APPFL_GRIDFM_DATA_PATH}" \
    APPFL_GRIDFM_EVAL_CASES="${APPFL_GRIDFM_EVAL_CASES}" \
    APPFL_GRIDFM_SCENARIOS="${APPFL_GRIDFM_SCENARIOS:-}" \
    APPFL_GRIDFM_NORMALIZER_STATS="${APPFL_GRIDFM_NORMALIZER_STATS}" \
    APPFL_GRIDFM_NUM_LAYERS="${APPFL_GRIDFM_NUM_LAYERS:-}" \
    APPFL_GRIDFM_WORKERS="${APPFL_GRIDFM_WORKERS:-}" \
    "${PYTHON_BIN}" resources/configs/grid/evaluate_grid_appfl.py \
        --model-path "${FINAL_MODEL_PATH}" \
        --output-csv "${LOG_DIR}/eval_metrics.csv" \
        > "${LOG_DIR}/eval.log" 2>&1
    echo "Evaluation metrics: ${LOG_DIR}/eval_metrics.csv"
fi

echo "All clients finished. Logs are in ${LOG_DIR}"
