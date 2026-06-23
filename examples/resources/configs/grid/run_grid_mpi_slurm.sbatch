#!/usr/bin/env bash
#SBATCH --job-name=appfl-gridfm
#SBATCH --output=slurm-%j-appfl-gridfm.out
#SBATCH --error=slurm-%j-appfl-gridfm.err
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu

set -euo pipefail

# Cluster-specific environment setup. Uncomment and edit on the Slurm machine.
# module purge
# module load cuda/12.8
# module load openmpi
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate fl
# Ensure `python -c "import mpi4py"` uses the same MPI family as `srun/mpirun`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export APPFL_MPI_LAUNCHER="${APPFL_MPI_LAUNCHER:-srun}"
export APPFL_NUM_CLIENTS="${APPFL_NUM_CLIENTS:-$((SLURM_NTASKS - 1))}"
export APPFL_MPI_PROCS="${APPFL_MPI_PROCS:-$((APPFL_NUM_CLIENTS + 1))}"

# Choose the GridFM datakit output root on the Slurm machine.
# export APPFL_GRIDFM_DATA_PATH="/path/to/gridfm_multi_case"
# export APPFL_GRIDFM_DATA_PATH="/path/to/gridfm_case30_full"
# export APPFL_GRIDFM_DATA_PATH="/path/to/gridfm_case30_smoke"
export APPFL_GRIDFM_DATA_PATH="${APPFL_GRIDFM_DATA_PATH:-/path/to/gridfm_multi_case}"

export APPFL_DEVICE="${APPFL_DEVICE:-cuda:0}"
export APPFL_NUM_GLOBAL_EPOCHS="${APPFL_NUM_GLOBAL_EPOCHS:-30}"
export APPFL_NUM_LOCAL_EPOCHS="${APPFL_NUM_LOCAL_EPOCHS:-1}"
export APPFL_GRIDFM_SCENARIOS="${APPFL_GRIDFM_SCENARIOS:-1000}"
export APPFL_GRIDFM_CLIENT_CASES="${APPFL_GRIDFM_CLIENT_CASES:-case24_ieee_rts,case30_ieee}"
export APPFL_GRIDFM_SERVER_CASES="${APPFL_GRIDFM_SERVER_CASES:-${APPFL_GRIDFM_CLIENT_CASES}}"
export APPFL_GRIDFM_EVAL_CASES="${APPFL_GRIDFM_EVAL_CASES:-case24_ieee_rts,case30_ieee,case14_ieee,case57_ieee}"
export APPFL_GRIDFM_BUILD_SHARED_NORMALIZER="${APPFL_GRIDFM_BUILD_SHARED_NORMALIZER:-1}"
export APPFL_GRIDFM_WORKERS="${APPFL_GRIDFM_WORKERS:-0}"

"${SCRIPT_DIR}/run_grid_mpi.sh"
