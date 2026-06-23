import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml


def _prefer_latest_graphkit():
    appfl_root = Path(__file__).resolve().parents[3]
    default_path = appfl_root.parent / "gridfm-graphkit"
    graphkit_path = Path(os.environ.get("GRIDFM_GRAPHKIT_PATH", default_path))
    if graphkit_path.exists():
        sys.path.insert(0, str(graphkit_path))


_prefer_latest_graphkit()

from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace


CONFIG_PATH = "./resources/configs/grid/gridfm_graphkit.yaml"


def _split_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_data_path(config_args):
    configured_path = getattr(getattr(config_args, "appfl", None), "data_path", None)
    return os.environ.get(
        "APPFL_GRIDFM_DATA_PATH",
        configured_path or os.path.join(os.environ.get("HOME", "."), "data"),
    )


def _get_normalizer_stats_path():
    return os.environ.get("APPFL_GRIDFM_NORMALIZER_STATS") or None


def _apply_env_overrides(config_args):
    networks = _split_csv(os.environ.get("APPFL_GRIDFM_NETWORKS"))
    if networks:
        config_args.data.networks = networks

    scenarios = os.environ.get("APPFL_GRIDFM_SCENARIOS")
    if scenarios:
        config_args.data.scenarios = [int(scenarios)] * len(config_args.data.networks)

    num_layers = os.environ.get("APPFL_GRIDFM_NUM_LAYERS")
    if num_layers:
        config_args.model.num_layers = int(num_layers)

    workers = os.environ.get("APPFL_GRIDFM_WORKERS")
    if workers:
        config_args.data.workers = int(workers)


def _subset_dataset(dataset, indices):
    if isinstance(dataset, np.ndarray):
        return dataset[indices]
    return [dataset[idx] for idx in indices]


def _shard_dataset(dataset, shard_id, shard_count):
    if shard_count <= 1:
        return dataset
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    base_size = len(dataset) // shard_count
    remainder = len(dataset) % shard_count
    start_idx = shard_id * base_size + min(shard_id, remainder)
    subset_size = base_size + (1 if shard_id < remainder else 0)
    return _subset_dataset(dataset, indices[start_idx : start_idx + subset_size])


def _select_client_case(
    config_args,
    client_id,
    num_clients,
    client_case_map=None,
    client_cases=None,
):
    if client_case_map:
        key = str(client_id)
        case_name = client_case_map.get(key, client_case_map.get(client_id))
        if case_name is None:
            raise ValueError(f"No case configured for zero-based client_id={client_id}")
        shard_id = 0
        shard_count = 1
    else:
        cases = (
            list(client_cases)
            if client_cases
            else _split_csv(os.environ.get("APPFL_GRIDFM_CLIENT_CASES"))
        )
        if not cases:
            return None
        case_index = client_id % len(cases)
        case_name = cases[case_index]
        case_group = [
            idx for idx in range(num_clients) if idx % len(cases) == case_index
        ]
        shard_id = case_group.index(client_id)
        shard_count = len(case_group)

    config_args.data.networks = [case_name]
    scenarios = os.environ.get("APPFL_GRIDFM_SCENARIOS")
    if scenarios:
        config_args.data.scenarios = [int(scenarios)]
    elif len(config_args.data.scenarios) != 1:
        config_args.data.scenarios = [config_args.data.scenarios[0]]
    return {
        "case_name": case_name,
        "shard_id": shard_id,
        "shard_count": shard_count,
    }


def get_gridfm_graphkit_dataset(
    num_clients: int,
    client_id: int,
    client_case_map=None,
    client_cases=None,
):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    with open(CONFIG_PATH) as f:
        config_dict = yaml.safe_load(f)

    config_args = NestedNamespace(**config_dict)
    _apply_env_overrides(config_args)
    case_assignment = _select_client_case(
        config_args,
        client_id,
        num_clients,
        client_case_map,
        client_cases,
    )
    data_module = LitGridHeteroDataModule(
        config_args,
        _get_data_path(config_args),
        normalizer_stats_path=_get_normalizer_stats_path(),
    )
    data_module.setup("fit")

    train_dataset = data_module.train_dataset_multi
    val_dataset = data_module.val_dataset_multi

    if case_assignment:
        return (
            _shard_dataset(
                train_dataset,
                case_assignment["shard_id"],
                case_assignment["shard_count"],
            ),
            _shard_dataset(
                val_dataset,
                case_assignment["shard_id"],
                case_assignment["shard_count"],
            ),
        )

    traing_dataset_size = len(train_dataset)
    validation_dataset_size = len(val_dataset)

    train_indices = list(range(traing_dataset_size))
    random.shuffle(train_indices)

    val_indices = list(range(validation_dataset_size))
    random.shuffle(val_indices)

    client_train_dataset_base_size = traing_dataset_size // num_clients
    client_train_dataset_remainder = traing_dataset_size % num_clients
    client_val_dataset_base_size = validation_dataset_size // num_clients
    client_val_dataset_remainder = validation_dataset_size % num_clients

    start_idx = client_id * client_train_dataset_base_size + min(
        client_id, client_train_dataset_remainder
    )
    subset_size = client_train_dataset_base_size + (
        1 if client_id < client_train_dataset_remainder else 0
    )
    train_subset_indices = train_indices[start_idx : start_idx + subset_size]

    client_train_dataset = _subset_dataset(train_dataset, train_subset_indices)

    start_idx = client_id * client_val_dataset_base_size + min(
        client_id, client_val_dataset_remainder
    )
    subset_size = client_val_dataset_base_size + (
        1 if client_id < client_val_dataset_remainder else 0
    )
    val_subset_indices = val_indices[start_idx : start_idx + subset_size]

    client_val_dataset = _subset_dataset(val_dataset, val_subset_indices)

    return client_train_dataset, client_val_dataset
