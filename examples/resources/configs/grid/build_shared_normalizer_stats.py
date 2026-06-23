#!/usr/bin/env python
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _split_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _dedupe(items):
    return list(dict.fromkeys(items))


def _load_num_scenarios(raw_dir):
    n_scenarios_file = raw_dir / "n_scenarios.txt"
    if n_scenarios_file.is_file():
        return int(n_scenarios_file.read_text().strip())
    bus_data = pd.read_parquet(raw_dir / "bus_data.parquet", columns=["scenario"])
    return int(bus_data["scenario"].nunique())


def _select_subset_indices(num_total, requested, seed):
    num_scenarios = min(int(requested), int(num_total))
    all_indices = list(range(num_total))
    random.seed(seed)
    random.shuffle(all_indices)
    return all_indices[:num_scenarios]


def _train_scenario_ids(raw_dir, subset_indices, val_ratio, test_ratio, seed, split_by_load):
    if split_by_load:
        bus_data = pd.read_parquet(
            raw_dir / "bus_data.parquet",
            columns=["scenario", "load_scenario_idx"],
        )
        load_scenarios_by_scenario = (
            bus_data.groupby("scenario", sort=True)["load_scenario_idx"].first()
        )
        load_scenarios = torch.tensor(
            load_scenarios_by_scenario.iloc[subset_indices].values,
        )
        unique_load_scenarios = torch.unique(load_scenarios)
        val_size = int(val_ratio * len(unique_load_scenarios))
        test_size = int(test_ratio * len(unique_load_scenarios))
        train_size = len(unique_load_scenarios) - val_size - test_size

        np.random.seed(seed)
        unique_load_scenarios = torch.tensor(np.random.permutation(unique_load_scenarios))
        train_load_scenarios = unique_load_scenarios[:train_size]
        train_indices = (
            torch.nonzero(torch.isin(load_scenarios, train_load_scenarios))
            .flatten()
            .tolist()
        )
    else:
        val_size = int(val_ratio * len(subset_indices))
        test_size = int(test_ratio * len(subset_indices))
        train_size = len(subset_indices) - val_size - test_size
        np.random.seed(seed)
        train_indices = np.random.permutation(len(subset_indices))[:train_size].tolist()

    return [subset_indices[idx] for idx in train_indices]


def _nonzero_power_values(raw_dir, scenario_ids):
    bus_data = pd.read_parquet(
        raw_dir / "bus_data.parquet",
        columns=["scenario", "Pd", "Qd", "Qg", "vn_kv"],
    )
    gen_data = pd.read_parquet(
        raw_dir / "gen_data.parquet",
        columns=["scenario", "p_mw"],
    )
    bus_data = bus_data[bus_data["scenario"].isin(scenario_ids)]
    gen_data = gen_data[gen_data["scenario"].isin(scenario_ids)]

    values = pd.concat(
        [
            bus_data["Pd"][bus_data["Pd"] != 0],
            bus_data["Qd"][bus_data["Qd"] != 0],
            gen_data["p_mw"][gen_data["p_mw"] != 0],
            bus_data["Qg"][bus_data["Qg"] != 0],
        ],
    )
    return values, float(bus_data["vn_kv"].max())


def build_stats(
    data_path,
    fit_cases,
    apply_cases,
    scenarios,
    seed,
    val_ratio,
    test_ratio,
    split_by_load,
    base_mva_orig,
):
    all_values = []
    vn_kv_max = 0.0
    train_counts = {}

    for case_name in fit_cases:
        raw_dir = Path(data_path) / case_name / "raw"
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"Missing raw data directory: {raw_dir}")

        num_total = _load_num_scenarios(raw_dir)
        subset_indices = _select_subset_indices(num_total, scenarios, seed)
        train_ids = _train_scenario_ids(
            raw_dir,
            subset_indices,
            val_ratio,
            test_ratio,
            seed,
            split_by_load,
        )
        values, case_vn_kv_max = _nonzero_power_values(raw_dir, train_ids)
        all_values.append(values)
        vn_kv_max = max(vn_kv_max, case_vn_kv_max)
        train_counts[case_name] = len(train_ids)

    combined_values = pd.concat(all_values)
    if combined_values.empty:
        raise ValueError("No nonzero power values found for shared normalizer fit.")

    shared_stats = {
        "baseMVA_orig": torch.tensor(float(base_mva_orig), dtype=torch.float),
        "baseMVA": torch.tensor(float(np.percentile(combined_values, 95)), dtype=torch.float),
        "vn_kv_max": torch.tensor(float(vn_kv_max), dtype=torch.float),
    }
    stats = {
        case_name: {key: value.clone() for key, value in shared_stats.items()}
        for case_name in apply_cases
    }
    return stats, train_counts, shared_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--fit-cases", required=True)
    parser.add_argument(
        "--apply-cases",
        default=None,
        help="Cases that should receive the shared stats. Defaults to fit-cases.",
    )
    parser.add_argument("--scenarios", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--base-mva-orig", type=float, default=100.0)
    parser.add_argument("--split-by-load-scenario-idx", action="store_true")
    args = parser.parse_args()

    fit_cases = _split_csv(args.fit_cases)
    apply_cases = _split_csv(args.apply_cases) if args.apply_cases else list(fit_cases)
    fit_cases = _dedupe(fit_cases)
    apply_cases = _dedupe(apply_cases)
    if not fit_cases:
        raise ValueError("--fit-cases must contain at least one case.")
    if not apply_cases:
        raise ValueError("--apply-cases must contain at least one case.")

    stats, train_counts, shared_stats = build_stats(
        args.data_path,
        fit_cases,
        apply_cases,
        args.scenarios,
        args.seed,
        args.val_ratio,
        args.test_ratio,
        args.split_by_load_scenario_idx,
        args.base_mva_orig,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, output)

    print(f"saved={output}")
    print(f"fit_cases={fit_cases}")
    print(f"apply_cases={apply_cases}")
    print(f"train_counts={train_counts}")
    print(
        "shared_stats="
        f"baseMVA_orig={float(shared_stats['baseMVA_orig'])}, "
        f"baseMVA={float(shared_stats['baseMVA'])}, "
        f"vn_kv_max={float(shared_stats['vn_kv_max'])}"
    )


if __name__ == "__main__":
    main()
