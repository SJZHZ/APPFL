import os

from appfl.agent import ClientAgent
from appfl.comm.grpc import GRPCClientCommunicator

from grid_appfl_common import isolate_client_home, load_client_config, prepare_grid_run


def main() -> None:
    prepare_grid_run(seed_value=1)

    num_clients = int(os.environ.get("APPFL_NUM_CLIENTS", "3"))
    client_id = int(os.environ["APPFL_CLIENT_ID"])
    server_uri = os.environ.get("APPFL_SERVER_URI", "localhost:50051")
    isolate_client_home(client_id)

    client_agent_config = load_client_config(client_id, num_clients)
    client_agent_config.comm_configs.grpc_configs["server_uri"] = server_uri

    client_agent = ClientAgent(client_agent_config=client_agent_config)
    client_communicator = GRPCClientCommunicator(
        client_id=client_agent.get_id(),
        **client_agent_config.comm_configs.grpc_configs,
    )

    client_config = client_communicator.get_configuration()
    client_agent.load_config(client_config)

    init_global_model = client_communicator.get_global_model(init_model=True)
    client_agent.load_parameters(init_global_model)

    round_idx = 0
    while True:
        client_agent.train()
        local_model, metadata = client_agent.get_parameters()
        round_idx += 1
        train_loss = metadata["train_loss"][-1]
        val_loss = metadata["val_loss"][-1]
        print(
            f"client={client_id} round={round_idx} train_loss={train_loss} val_loss={val_loss}",
            flush=True,
        )
        new_global_model, metadata = client_communicator.update_global_model(
            local_model, **metadata
        )
        if metadata["status"] == "DONE":
            print(f"client={client_id} status=DONE", flush=True)
            break
        client_agent.load_parameters(new_global_model)

    client_communicator.invoke_custom_action(action="close_connection")


if __name__ == "__main__":
    main()
