import argparse
import os

from appfl.agent import ClientAgent, ServerAgent
from appfl.comm.mpi import MPIClientCommunicator, MPIServerCommunicator
from mpi4py import MPI

from grid_appfl_common import (
    isolate_client_home,
    load_client_config,
    load_server_config,
    prepare_grid_run,
    save_final_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run APPFL GridFM with MPI.")
    parser.add_argument(
        "--num-clients",
        type=int,
        default=None,
        help="Number of FL clients. Defaults to MPI world size minus one.",
    )
    return parser.parse_args()


def run_server(comm, num_clients: int) -> None:
    server_agent_config = load_server_config(num_clients=num_clients)
    server_agent = ServerAgent(server_agent_config=server_agent_config)
    server_communicator = MPIServerCommunicator(
        comm,
        server_agent,
        logger=server_agent.logger,
    )

    print("mpi_role=server", flush=True)
    print(f"mpi_world_size={comm.Get_size()}", flush=True)
    print(f"num_clients={num_clients}", flush=True)
    print(
        f"num_global_epochs={server_agent_config.server_configs.num_global_epochs}",
        flush=True,
    )
    print(
        "num_local_epochs="
        f"{server_agent_config.client_configs.train_configs.num_local_epochs}",
        flush=True,
    )
    print(
        f"client_device={server_agent_config.client_configs.train_configs.device}",
        flush=True,
    )
    if os.environ.get("APPFL_GLOBAL_MODEL_PATH") is not None:
        print(f"final_model_path={os.environ['APPFL_GLOBAL_MODEL_PATH']}", flush=True)

    server_communicator.serve()
    save_final_model(server_agent)


def run_client(comm, rank: int, num_clients: int) -> None:
    client_id = rank
    isolate_client_home(client_id)

    client_agent_config = load_client_config(client_id, num_clients)
    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_communicator = MPIClientCommunicator(
        comm,
        server_rank=0,
        client_id=client_agent.get_id(),
    )

    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)

    init_global_model = client_communicator.get_global_model(init_model=True)
    client_agent.load_parameters(init_global_model)

    sample_size = client_agent.get_sample_size()
    client_communicator.invoke_custom_action(
        action="set_sample_size",
        sample_size=sample_size,
        sync=True,
    )

    round_idx = 0
    while True:
        client_agent.train()
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model
        else:
            metadata = {}
        round_idx += 1
        train_loss = metadata.get("train_loss", [None])[-1]
        val_loss = metadata.get("val_loss", [None])[-1]
        print(
            "mpi_role=client "
            f"client={client_id} round={round_idx} "
            f"sample_size={sample_size} train_loss={train_loss} val_loss={val_loss}",
            flush=True,
        )
        new_global_model, metadata = client_communicator.update_global_model(
            local_model,
            **metadata,
        )
        if metadata["status"] == "DONE":
            print(f"mpi_role=client client={client_id} status=DONE", flush=True)
            break
        client_agent.load_parameters(new_global_model)

    client_communicator.invoke_custom_action(action="close_connection")


def main() -> None:
    args = parse_args()
    prepare_grid_run(seed_value=1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        raise ValueError("MPI GridFM run needs at least 2 ranks: 1 server + 1 client.")

    num_clients = args.num_clients or int(os.environ.get("APPFL_NUM_CLIENTS", size - 1))
    if num_clients != size - 1:
        raise ValueError(
            f"This GridFM MPI runner expects one MPI rank per client. "
            f"Got num_clients={num_clients}, MPI client ranks={size - 1}."
        )
    os.environ["APPFL_NUM_CLIENTS"] = str(num_clients)

    if rank == 0:
        run_server(comm, num_clients)
    else:
        run_client(comm, rank, num_clients)


if __name__ == "__main__":
    main()
