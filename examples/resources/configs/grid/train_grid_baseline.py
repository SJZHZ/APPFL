#!/usr/bin/env python
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader


def _prefer_latest_graphkit():
    appfl_root = Path(__file__).resolve().parents[4]
    default_path = appfl_root.parent / "gridfm-graphkit"
    graphkit_path = Path(os.environ.get("GRIDFM_GRAPHKIT_PATH", default_path))
    if graphkit_path.exists():
        sys.path.insert(0, str(graphkit_path))


_prefer_latest_graphkit()

from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task

from evaluate_grid_appfl import evaluate_case


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/grid/gridfm_graphkit.yaml"


def _split_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _make_config(cases, scenarios, batch_size, workers):
    with open(CONFIG_PATH) as f:
        config_dict = yaml.safe_load(f)
    args = NestedNamespace(**config_dict)
    args.data.networks = cases
    args.data.scenarios = [int(scenarios)] * len(cases)
    args.data.workers = int(workers)
    args.training.batch_size = int(batch_size)

    num_layers = os.environ.get("APPFL_GRIDFM_NUM_LAYERS")
    if num_layers:
        args.model.num_layers = int(num_layers)
    return args


def _run_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        _, loss_dict = model.shared_step(batch)
        loss = loss_dict["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_graphs = int(batch.num_graphs)
        total_loss += _to_float(loss) * n_graphs
        total_graphs += n_graphs
    return total_loss / max(total_graphs, 1)


def _validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, loss_dict = model.shared_step(batch)
            n_graphs = int(batch.num_graphs)
            total_loss += _to_float(loss_dict["loss"]) * n_graphs
            total_graphs += n_graphs
    return total_loss / max(total_graphs, 1)


def train_baseline(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    train_cases = _split_csv(args.train_cases)
    eval_cases = _split_csv(args.eval_cases)
    if not train_cases:
        raise ValueError("--train-cases must contain at least one case.")
    if not eval_cases:
        raise ValueError("--eval-cases must contain at least one case.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "train_history.csv"
    model_path = output_dir / "baseline_model.pt"
    eval_path = output_dir / "eval_metrics.csv"

    config_args = _make_config(
        train_cases,
        args.scenarios,
        args.batch_size,
        args.workers,
    )
    data_module = LitGridHeteroDataModule(
        config_args,
        args.data_path,
        normalizer_stats_path=args.normalizer_stats,
    )
    data_module.setup("fit")

    model = get_task(config_args, data_module.data_normalizers).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config_args.optimizer.learning_rate),
        betas=(float(config_args.optimizer.beta1), float(config_args.optimizer.beta2)),
    )

    train_loader = DataLoader(
        data_module.train_dataset_multi,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        data_module.val_dataset_multi,
        batch_size=args.batch_size,
        shuffle=False,
    )

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "seconds"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss = _run_epoch(model, train_loader, optimizer, args.device)
            val_loss = _validate(model, val_loader, args.device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "seconds": time.time() - start,
            }
            writer.writerow(row)
            f.flush()
            print(row, flush=True)

    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_path)
    print(f"saved_model={model_path}", flush=True)

    old_normalizer = os.environ.get("APPFL_GRIDFM_NORMALIZER_STATS")
    if args.normalizer_stats:
        os.environ["APPFL_GRIDFM_NORMALIZER_STATS"] = args.normalizer_stats
    try:
        rows = [
            evaluate_case(
                model_path,
                case_name,
                args.data_path,
                args.scenarios,
                args.eval_batch_size,
                args.device,
            )
            for case_name in eval_cases
        ]
    finally:
        if old_normalizer is None:
            os.environ.pop("APPFL_GRIDFM_NORMALIZER_STATS", None)
        else:
            os.environ["APPFL_GRIDFM_NORMALIZER_STATS"] = old_normalizer

    fieldnames = sorted({key for row in rows for key in row})
    with open(eval_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"eval_metrics={eval_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train a non-FL GridFM baseline for local-only or centralized comparison.",
    )
    parser.add_argument("--data-path", default="/home/sjzhz/tmp/gridfm_multi_case")
    parser.add_argument("--train-cases", required=True)
    parser.add_argument(
        "--eval-cases",
        default="case24_ieee_rts,case30_ieee,case14_ieee,case57_ieee",
    )
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--normalizer-stats", default=os.environ.get("APPFL_GRIDFM_NORMALIZER_STATS"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=os.environ.get("APPFL_DEVICE", "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_baseline(args)


if __name__ == "__main__":
    main()
