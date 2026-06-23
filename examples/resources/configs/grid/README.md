# APPFL GridFM Workflow

This directory contains the current GridFM APPFL workflow. It uses the latest
`/home/sjzhz/FL/gridfm-graphkit` by default; the old vendored
`gridfm-graphkit-v005` directory is no longer required.

## Main FL Run

Single-node gRPC run:

```bash
APPFL_DEVICE=cuda:0 \
APPFL_GRIDFM_DATA_PATH=/home/sjzhz/tmp/gridfm_multi_case \
APPFL_GRIDFM_CLIENT_CASES=case24_ieee_rts,case30_ieee \
APPFL_GRIDFM_SERVER_CASES=case24_ieee_rts,case30_ieee \
APPFL_GRIDFM_EVAL_CASES=case24_ieee_rts,case30_ieee,case14_ieee,case57_ieee \
APPFL_GRIDFM_SCENARIOS=1000 \
APPFL_GRIDFM_BUILD_SHARED_NORMALIZER=1 \
APPFL_NUM_GLOBAL_EPOCHS=30 \
./run_grid_appfl.sh
```

Single-node or interactive MPI run:

```bash
APPFL_MPI_LAUNCHER=mpirun \
APPFL_NUM_CLIENTS=4 \
APPFL_DEVICE=cuda:0 \
APPFL_GRIDFM_DATA_PATH=/home/sjzhz/tmp/gridfm_multi_case \
APPFL_GRIDFM_CLIENT_CASES=case24_ieee_rts,case30_ieee \
APPFL_GRIDFM_SERVER_CASES=case24_ieee_rts,case30_ieee \
APPFL_GRIDFM_EVAL_CASES=case24_ieee_rts,case30_ieee,case14_ieee,case57_ieee \
APPFL_GRIDFM_SCENARIOS=1000 \
APPFL_GRIDFM_BUILD_SHARED_NORMALIZER=1 \
APPFL_NUM_GLOBAL_EPOCHS=30 \
./run_grid_mpi.sh
```

Slurm run:

```bash
sbatch run_grid_mpi_slurm.sbatch
```

Edit the `#SBATCH` resources and the environment setup block in
`run_grid_mpi_slurm.sbatch` for the target cluster. The Slurm environment must
provide `mpi4py` built against the same MPI family used by `srun` or `mpirun`;
the script checks this before launching the job.

With two clients, each client trains one local case:

- Client 1: `case24_ieee_rts`
- Client 2: `case30_ieee`

If `APPFL_NUM_CLIENTS` is larger than the number of entries in
`APPFL_GRIDFM_CLIENT_CASES`, cases are assigned round-robin and each case is
split among the clients assigned to that case. For example, four clients with
`case24_ieee_rts,case30_ieee` gives two case24 shards and two case30 shards.

`case14_ieee` and `case57_ieee` are eval-only unseen cases when they are not
included in `APPFL_GRIDFM_CLIENT_CASES` or `APPFL_GRIDFM_SERVER_CASES`.

## Shared Normalizer

Use `APPFL_GRIDFM_BUILD_SHARED_NORMALIZER=1` to fit one normalizer from the
training cases only. The generated `shared_normalizer_stats.pt` is then applied
to train, validation, and eval cases. This keeps all clients in the same feature
scale before FedAvg.

## Eval Existing Checkpoint

```bash
APPFL_DEVICE=cuda:0 \
APPFL_GRIDFM_MODEL_PATH=/path/to/final_global_model.pt \
./run_grid_eval_existing.sh
```

If `APPFL_GRIDFM_MODEL_PATH` is omitted, the script evaluates the newest
`final_global_model.pt` under `examples/output`.

## Baselines

Local-only baseline:

```bash
APPFL_GRIDFM_NORMALIZER_STATS=/path/to/shared_normalizer_stats.pt \
APPFL_DEVICE=cuda:0 \
python train_grid_baseline.py \
  --data-path /home/sjzhz/tmp/gridfm_multi_case \
  --train-cases case24_ieee_rts \
  --output-dir /home/sjzhz/FL/APPFL/examples/output/baseline_case24
```

Centralized mixed-case baseline:

```bash
APPFL_GRIDFM_NORMALIZER_STATS=/path/to/shared_normalizer_stats.pt \
APPFL_DEVICE=cuda:0 \
python train_grid_baseline.py \
  --data-path /home/sjzhz/tmp/gridfm_multi_case \
  --train-cases case24_ieee_rts,case30_ieee \
  --output-dir /home/sjzhz/FL/APPFL/examples/output/baseline_central_case24_case30
```

Compare each baseline's `eval_metrics.csv` with the FL run's
`eval_metrics.csv`.
