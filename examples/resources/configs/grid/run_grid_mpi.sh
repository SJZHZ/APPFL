#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
APPFL_NUM_CLIENTS="${APPFL_NUM_CLIENTS:-2}"
APPFL_MPI_PROCS="${APPFL_MPI_PROCS:-$((APPFL_NUM_CLIENTS + 1))}"
APPFL_MPI_LAUNCHER="${APPFL_MPI_LAUNCHER:-mpirun}"
APPFL_MPI_EXTRA_ARGS="${APPFL_MPI_EXTRA_ARGS:-}"
APPFL_NUM_GLOBAL_EPOCHS="${APPFL_NUM_GLOBAL_EPOCHS:-10}"
APPFL_NUM_LOCAL_EPOCHS="${APPFL_NUM_LOCAL_EPOCHS:-1}"
APPFL_DEVICE="${APPFL_DEVICE:-cpu}"
APPFL_GRIDFM_CLIENT_CASES="${APPFL_GRIDFM_CLIENT_CASES:-}"
APPFL_GRIDFM_SERVER_CASES="${APPFL_GRIDFM_SERVER_CASES:-${APPFL_GRIDFM_CLIENT_CASES}}"
APPFL_GRIDFM_EVAL_CASES="${APPFL_GRIDFM_EVAL_CASES:-}"
APPFL_GRIDFM_NORMALIZER_STATS="${APPFL_GRIDFM_NORMALIZER_STATS:-}"
APPFL_GRIDFM_BUILD_SHARED_NORMALIZER="${APPFL_GRIDFM_BUILD_SHARED_NORMALIZER:-0}"

# Choose the GridFM datakit output root. Override from the shell, or uncomment
# one of these lines.
# APPFL_GRIDFM_DATA_PATH="/home/sjzhz/tmp/gridfm_multi_case"
# APPFL_GRIDFM_DATA_PATH="/home/sjzhz/tmp/gridfm_case30_full"
# APPFL_GRIDFM_DATA_PATH="/home/sjzhz/tmp/gridfm_case30_smoke"
# APPFL_GRIDFM_DATA_PATH="/home/sjzhz/data"
APPFL_GRIDFM_DATA_PATH="${APPFL_GRIDFM_DATA_PATH:-/home/sjzhz/tmp/gridfm_multi_case}"
APPFL_CLIENT_HOME_BASE="${APPFL_CLIENT_HOME_BASE:-/tmp/appfl_client_homes}"

# Optional quality/runtime knobs.
# APPFL_GRIDFM_SCENARIOS=10000
# APPFL_GRIDFM_CLIENT_CASES=case24_ieee_rts,case30_ieee
# APPFL_GRIDFM_SERVER_CASES=case24_ieee_rts,case30_ieee
# APPFL_GRIDFM_EVAL_CASES=case24_ieee_rts,case30_ieee,case14_ieee,case57_ieee
# APPFL_GRIDFM_NORMALIZER_STATS=/path/to/shared_normalizer_stats.pt
# APPFL_GRIDFM_BUILD_SHARED_NORMALIZER=1
# APPFL_GRIDFM_NUM_LAYERS=4
# APPFL_GRIDFM_WORKERS=0
# APPFL_DEVICE=cuda:0

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${EXAMPLES_DIR}/output/grid_mpi_run_${RUN_ID}"
FINAL_MODEL_PATH="${LOG_DIR}/final_global_model.pt"
mkdir -p "${LOG_DIR}"

if [[ -n "${APPFL_GRIDFM_CLIENT_CASES}" && "${APPFL_NUM_CLIENTS}" == "2" ]]; then
    IFS="," read -r -a _case_array <<< "${APPFL_GRIDFM_CLIENT_CASES}"
    APPFL_NUM_CLIENTS="${#_case_array[@]}"
    APPFL_MPI_PROCS="$((APPFL_NUM_CLIENTS + 1))"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if ! command -v "${APPFL_MPI_LAUNCHER%% *}" >/dev/null 2>&1; then
    echo "MPI launcher not found: ${APPFL_MPI_LAUNCHER}" >&2
    exit 1
fi

if [[ "${APPFL_MPI_PROCS}" -ne "$((APPFL_NUM_CLIENTS + 1))" ]]; then
    echo "APPFL_MPI_PROCS must equal APPFL_NUM_CLIENTS + 1 for this runner." >&2
    echo "Got APPFL_MPI_PROCS=${APPFL_MPI_PROCS}, APPFL_NUM_CLIENTS=${APPFL_NUM_CLIENTS}." >&2
    exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY'
import mpi4py

print(f"mpi4py={mpi4py.__version__}")
PY
then
    echo "mpi4py is not importable from ${PYTHON_BIN}." >&2
    echo "Install mpi4py against the MPI implementation used by ${APPFL_MPI_LAUNCHER}." >&2
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

cd "${EXAMPLES_DIR}"

echo "APPFL GridFM MPI run: ${RUN_ID}"
echo "  examples: ${EXAMPLES_DIR}"
echo "  python:   ${PYTHON_BIN}"
echo "  launcher: ${APPFL_MPI_LAUNCHER}"
echo "  mpi procs: ${APPFL_MPI_PROCS} (1 server + ${APPFL_NUM_CLIENTS} clients)"
echo "  data:     ${APPFL_GRIDFM_DATA_PATH}"
echo "  global epochs: ${APPFL_NUM_GLOBAL_EPOCHS}"
echo "  local epochs:  ${APPFL_NUM_LOCAL_EPOCHS}"
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
echo "  logs:     ${LOG_DIR}"
echo "  final model: ${FINAL_MODEL_PATH}"

read -r -a _mpi_launcher <<< "${APPFL_MPI_LAUNCHER}"
read -r -a _mpi_extra_args <<< "${APPFL_MPI_EXTRA_ARGS}"

APPFL_NUM_CLIENTS="${APPFL_NUM_CLIENTS}" \
APPFL_NUM_GLOBAL_EPOCHS="${APPFL_NUM_GLOBAL_EPOCHS}" \
APPFL_NUM_LOCAL_EPOCHS="${APPFL_NUM_LOCAL_EPOCHS}" \
APPFL_DEVICE="${APPFL_DEVICE}" \
APPFL_LOG_DIR="${LOG_DIR}" \
APPFL_GLOBAL_MODEL_PATH="${FINAL_MODEL_PATH}" \
APPFL_GRIDFM_DATA_PATH="${APPFL_GRIDFM_DATA_PATH}" \
APPFL_GRIDFM_SCENARIOS="${APPFL_GRIDFM_SCENARIOS:-}" \
APPFL_GRIDFM_CLIENT_CASES="${APPFL_GRIDFM_CLIENT_CASES}" \
APPFL_GRIDFM_SERVER_CASES="${APPFL_GRIDFM_SERVER_CASES}" \
APPFL_GRIDFM_NORMALIZER_STATS="${APPFL_GRIDFM_NORMALIZER_STATS}" \
APPFL_GRIDFM_NUM_LAYERS="${APPFL_GRIDFM_NUM_LAYERS:-}" \
APPFL_GRIDFM_WORKERS="${APPFL_GRIDFM_WORKERS:-}" \
APPFL_CLIENT_HOME_BASE="${APPFL_CLIENT_HOME_BASE}" \
"${_mpi_launcher[@]}" -n "${APPFL_MPI_PROCS}" "${_mpi_extra_args[@]}" \
    "${PYTHON_BIN}" notebook_tutorials/grid/run_mpi_grid.py \
    --num-clients "${APPFL_NUM_CLIENTS}" \
    > "${LOG_DIR}/mpi.log" 2>&1

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

echo "MPI run finished. Logs are in ${LOG_DIR}"
