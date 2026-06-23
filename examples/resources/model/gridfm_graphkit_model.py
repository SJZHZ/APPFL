import os
import sys
from pathlib import Path

import yaml


def _prefer_latest_graphkit():
    appfl_root = Path(__file__).resolve().parents[3]
    default_path = appfl_root.parent / "gridfm-graphkit"
    graphkit_path = Path(os.environ.get("GRIDFM_GRAPHKIT_PATH", default_path))
    if graphkit_path.exists():
        sys.path.insert(0, str(graphkit_path))


_prefer_latest_graphkit()

from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task


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
    networks = (
        _split_csv(os.environ.get("APPFL_GRIDFM_SERVER_CASES"))
        or _split_csv(os.environ.get("APPFL_GRIDFM_NETWORKS"))
        or _split_csv(os.environ.get("APPFL_GRIDFM_CLIENT_CASES"))
    )
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


def get_gridfm_graphkit_model():
    with open(CONFIG_PATH) as f:
        config_dict = yaml.safe_load(f)

    config_args = NestedNamespace(**config_dict)
    _apply_env_overrides(config_args)
    data_module = LitGridHeteroDataModule(
        config_args,
        _get_data_path(config_args),
        normalizer_stats_path=_get_normalizer_stats_path(),
    )
    data_module.setup("fit")

    return get_task(config_args, data_module.data_normalizers)
