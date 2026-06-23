#!/usr/bin/env python
import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool


def _prefer_latest_graphkit():
    appfl_root = Path(__file__).resolve().parents[4]
    default_path = appfl_root.parent / "gridfm-graphkit"
    graphkit_path = Path(os.environ.get("GRIDFM_GRAPHKIT_PATH", default_path))
    if graphkit_path.exists():
        sys.path.insert(0, str(graphkit_path))


_prefer_latest_graphkit()

from gridfm_graphkit.datasets.globals import VA_H, VM_H, VA_OUT, VM_OUT
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/grid/gridfm_graphkit.yaml"


def _split_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _make_config(case_name, scenarios):
    with open(CONFIG_PATH) as f:
        config_dict = yaml.safe_load(f)
    args = NestedNamespace(**config_dict)
    args.data.networks = [case_name]
    if scenarios is not None:
        args.data.scenarios = [int(scenarios)]
    elif len(args.data.scenarios) != 1:
        args.data.scenarios = [args.data.scenarios[0]]

    num_layers = os.environ.get("APPFL_GRIDFM_NUM_LAYERS")
    if num_layers:
        args.model.num_layers = int(num_layers)

    workers = os.environ.get("APPFL_GRIDFM_WORKERS")
    if workers:
        args.data.workers = int(workers)
    else:
        args.data.workers = 0
    return args


def _batch_pbe(output, batch):
    branch_flow_layer = ComputeBranchFlow()
    node_injection_layer = ComputeNodeInjection()
    node_residuals_layer = ComputeNodeResiduals()

    num_bus = batch.x_dict["bus"].size(0)
    bus_edge_index = batch.edge_index_dict[("bus", "connects", "bus")]
    bus_edge_attr = batch.edge_attr_dict[("bus", "connects", "bus")]
    pft, qft = branch_flow_layer(output["bus"], bus_edge_index, bus_edge_attr)
    p_in, q_in = node_injection_layer(pft, qft, bus_edge_index, num_bus)
    residual_p, residual_q = node_residuals_layer(
        p_in,
        q_in,
        output["bus"],
        batch.x_dict["bus"],
    )
    pbe = torch.sqrt(residual_p**2 + residual_q**2)
    return global_mean_pool(pbe, batch.batch_dict["bus"]).mean(), pbe.max()


def evaluate_case(model_path, case_name, data_path, scenarios, batch_size, device):
    config_args = _make_config(case_name, scenarios)
    data_module = LitGridHeteroDataModule(
        config_args,
        data_path,
        normalizer_stats_path=os.environ.get("APPFL_GRIDFM_NORMALIZER_STATS") or None,
    )
    data_module.setup("fit")

    model = get_task(config_args, data_module.data_normalizers)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    test_dataset = data_module.test_datasets[0]
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    accum = {}
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output, loss_dict = model.shared_step(batch)
            n_graphs = int(batch.num_graphs)
            total_graphs += n_graphs

            for key, value in loss_dict.items():
                accum[key] = accum.get(key, 0.0) + _to_float(value) * n_graphs

            pred = output["bus"][:, [VM_OUT, VA_OUT]]
            target = batch.y_dict["bus"][:, [VM_H, VA_H]]
            mask = batch.mask_dict["bus"][:, [VM_H, VA_H]]
            diff = pred[mask] - target[mask]
            if diff.numel() > 0:
                accum["masked_vm_va_mse"] = (
                    accum.get("masked_vm_va_mse", 0.0)
                    + float((diff**2).mean().detach().cpu()) * n_graphs
                )
                accum["masked_vm_va_mae"] = (
                    accum.get("masked_vm_va_mae", 0.0)
                    + float(diff.abs().mean().detach().cpu()) * n_graphs
                )

            pbe_mean, pbe_max = _batch_pbe(output, batch)
            accum["pbe_mean"] = accum.get("pbe_mean", 0.0) + _to_float(pbe_mean) * n_graphs
            accum["pbe_max"] = max(accum.get("pbe_max", 0.0), _to_float(pbe_max))

    row = {
        "case": case_name,
        "num_test_graphs": total_graphs,
    }
    for key, value in accum.items():
        if key == "pbe_max":
            row[key] = value
        else:
            row[key] = value / max(total_graphs, 1)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--cases",
        default=os.environ.get("APPFL_GRIDFM_EVAL_CASES", ""),
        help="Comma-separated case names to evaluate.",
    )
    parser.add_argument(
        "--data-path",
        default=os.environ.get("APPFL_GRIDFM_DATA_PATH", "/home/sjzhz/tmp/gridfm_case30_smoke"),
    )
    parser.add_argument(
        "--scenarios",
        default=os.environ.get("APPFL_GRIDFM_SCENARIOS"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default=os.environ.get("APPFL_EVAL_DEVICE", os.environ.get("APPFL_DEVICE", "cpu")),
    )
    args = parser.parse_args()

    cases = _split_csv(args.cases)
    if not cases:
        raise ValueError("No eval cases supplied. Set APPFL_GRIDFM_EVAL_CASES or --cases.")

    rows = [
        evaluate_case(
            args.model_path,
            case_name,
            args.data_path,
            args.scenarios,
            args.batch_size,
            args.device,
        )
        for case_name in cases
    ]

    fieldnames = sorted({key for row in rows for key in row})
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
