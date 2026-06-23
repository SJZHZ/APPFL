import os
import socket

from appfl.agent import ServerAgent
from appfl.comm.grpc import GRPCServerCommunicator, serve

from grid_appfl_common import load_server_config, prepare_grid_run, save_final_model


def main() -> None:
    prepare_grid_run(seed_value=1)

    num_clients = int(os.environ.get("APPFL_NUM_CLIENTS", "3"))
    server_agent_config = load_server_config(num_clients=num_clients)
    num_global_epochs = server_agent_config.server_configs.num_global_epochs
    num_local_epochs = server_agent_config.client_configs.train_configs.num_local_epochs
    appfl_device = server_agent_config.client_configs.train_configs.device
    server_uri = os.environ.get("APPFL_SERVER_URI", "127.0.0.1:50051")
    server_agent_config.server_configs.comm_configs.grpc_configs["server_uri"] = server_uri

    server_agent = ServerAgent(server_agent_config=server_agent_config)
    communicator = GRPCServerCommunicator(
        server_agent,
        logger=server_agent.logger,
        **server_agent_config.server_configs.comm_configs.grpc_configs,
    )

    hostname = socket.gethostname()
    private_ip = socket.gethostbyname(hostname)
    print(f"server_uri={server_uri}", flush=True)
    print(f"private_ip_hint={private_ip}:50051", flush=True)
    print(f"num_clients={num_clients}", flush=True)
    print(f"num_global_epochs={num_global_epochs}", flush=True)
    print(f"num_local_epochs={num_local_epochs}", flush=True)
    print(f"client_device={appfl_device}", flush=True)
    if os.environ.get("APPFL_GLOBAL_MODEL_PATH") is not None:
        print(f"final_model_path={os.environ['APPFL_GLOBAL_MODEL_PATH']}", flush=True)

    serve(
        communicator,
        **server_agent_config.server_configs.comm_configs.grpc_configs,
    )

    save_final_model(server_agent)


if __name__ == "__main__":
    main()
