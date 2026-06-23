import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


EXAMPLES_DIR = Path(__file__).resolve().parents[2]


def set_seed(seed_value: int = 1) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_grid_run(seed_value: int = 1) -> None:
    warnings.filterwarnings("ignore")
    os.chdir(EXAMPLES_DIR)
    set_seed(seed_value)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def load_server_config(num_clients: int | None = None):
    server_agent_config = OmegaConf.load("./resources/configs/grid/server_fedavg.yaml")
    if num_clients is None:
        num_clients = env_int("APPFL_NUM_CLIENTS", 2)
    server_agent_config.server_configs["num_clients"] = num_clients
    server_agent_config.server_configs["num_global_epochs"] = env_int(
        "APPFL_NUM_GLOBAL_EPOCHS", 10
    )

    num_local_epochs = os.environ.get("APPFL_NUM_LOCAL_EPOCHS")
    if num_local_epochs is not None:
        server_agent_config.client_configs.train_configs["num_local_epochs"] = int(
            num_local_epochs
        )

    appfl_device = os.environ.get("APPFL_DEVICE")
    if appfl_device is not None:
        server_agent_config.client_configs.train_configs["device"] = appfl_device

    log_dir = os.environ.get("APPFL_LOG_DIR")
    if log_dir:
        server_agent_config.server_configs["logging_output_dirname"] = log_dir

    return server_agent_config


def load_client_config(client_id: int, num_clients: int):
    client_agent_config = OmegaConf.load("./resources/configs/grid/client_base.yaml")
    client_agent_config.client_id = f"Client{client_id}"
    client_agent_config.data_configs.dataset_kwargs["num_clients"] = num_clients
    client_agent_config.data_configs.dataset_kwargs["client_id"] = client_id - 1
    client_agent_config.train_configs["logging_output_dirname"] = (
        os.environ.get("APPFL_LOG_DIR") or f"./output/client_{client_id}"
    )

    appfl_device = os.environ.get("APPFL_DEVICE")
    if appfl_device is not None:
        client_agent_config.train_configs["device"] = appfl_device

    return client_agent_config


def isolate_client_home(client_id: int) -> Path:
    client_home_base = os.environ.get("APPFL_CLIENT_HOME_BASE", "/tmp/appfl_client_homes")
    client_home = Path(client_home_base) / f"client_{client_id}"
    client_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(client_home)
    return client_home


def save_final_model(server_agent) -> None:
    final_model_path = os.environ.get("APPFL_GLOBAL_MODEL_PATH")
    if final_model_path is None:
        return
    path = Path(final_model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final_state = server_agent.get_parameters(blocking=True, init_model=False)
    if isinstance(final_state, tuple):
        final_state = final_state[0]
    torch.save(final_state, path)
    print(f"saved_final_model={path}", flush=True)
