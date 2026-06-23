#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/sjzhz/miniconda3/envs/fl/bin/python}"
APPFL_GRIDFM_DATA_PATH="${APPFL_GRIDFM_DATA_PATH:-/home/sjzhz/tmp/gridfm_multi_case}"
APPFL_GRIDFM_FIT_CASES="${APPFL_GRIDFM_FIT_CASES:-case24_ieee_rts,case30_ieee}"
APPFL_GRIDFM_EVAL_CASES="${APPFL_GRIDFM_EVAL_CASES:-case24_ieee_rts,case30_ieee,case14_ieee,case57_ieee}"
APPFL_GRIDFM_SCENARIOS="${APPFL_GRIDFM_SCENARIOS:-1000}"
APPFL_DEVICE="${APPFL_DEVICE:-cpu}"
APPFL_GRIDFM_NORMALIZER_STATS="${APPFL_GRIDFM_NORMALIZER_STATS:-}"
APPFL_GRIDFM_MODEL_PATH="${APPFL_GRIDFM_MODEL_PATH:-}"

if [[ -z "${APPFL_GRIDFM_MODEL_PATH}" ]]; then
    APPFL_GRIDFM_MODEL_PATH="$(
        find "${EXAMPLES_DIR}/output" -path "*/final_global_model.pt" -type f \
            -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
    )"
fi

if [[ -z "${APPFL_GRIDFM_MODEL_PATH}" || ! -f "${APPFL_GRIDFM_MODEL_PATH}" ]]; then
    echo "Set APPFL_GRIDFM_MODEL_PATH to an existing checkpoint." >&2
    exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$(dirname "${APPFL_GRIDFM_MODEL_PATH}")/eval_existing_${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

if [[ -z "${APPFL_GRIDFM_NORMALIZER_STATS}" ]]; then
    APPFL_GRIDFM_NORMALIZER_STATS="${OUTPUT_DIR}/shared_normalizer_stats.pt"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_shared_normalizer_stats.py" \
        --data-path "${APPFL_GRIDFM_DATA_PATH}" \
        --fit-cases "${APPFL_GRIDFM_FIT_CASES}" \
        --apply-cases "${APPFL_GRIDFM_FIT_CASES},${APPFL_GRIDFM_EVAL_CASES}" \
        --scenarios "${APPFL_GRIDFM_SCENARIOS}" \
        --output "${APPFL_GRIDFM_NORMALIZER_STATS}" \
        --split-by-load-scenario-idx \
        > "${OUTPUT_DIR}/shared_normalizer.log" 2>&1
fi

echo "Evaluating checkpoint:"
echo "  model:      ${APPFL_GRIDFM_MODEL_PATH}"
echo "  data:       ${APPFL_GRIDFM_DATA_PATH}"
echo "  fit cases:  ${APPFL_GRIDFM_FIT_CASES}"
echo "  eval cases: ${APPFL_GRIDFM_EVAL_CASES}"
echo "  scenarios:  ${APPFL_GRIDFM_SCENARIOS}"
echo "  normalizer: ${APPFL_GRIDFM_NORMALIZER_STATS}"
echo "  output:     ${OUTPUT_DIR}"

cd "${EXAMPLES_DIR}"
APPFL_GRIDFM_NORMALIZER_STATS="${APPFL_GRIDFM_NORMALIZER_STATS}" \
APPFL_GRIDFM_DATA_PATH="${APPFL_GRIDFM_DATA_PATH}" \
APPFL_GRIDFM_EVAL_CASES="${APPFL_GRIDFM_EVAL_CASES}" \
APPFL_GRIDFM_SCENARIOS="${APPFL_GRIDFM_SCENARIOS}" \
APPFL_DEVICE="${APPFL_DEVICE}" \
"${PYTHON_BIN}" resources/configs/grid/evaluate_grid_appfl.py \
    --model-path "${APPFL_GRIDFM_MODEL_PATH}" \
    --output-csv "${OUTPUT_DIR}/eval_metrics.csv" \
    > "${OUTPUT_DIR}/eval.log" 2>&1

echo "Evaluation metrics: ${OUTPUT_DIR}/eval_metrics.csv"
