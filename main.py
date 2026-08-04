from configs.basic_fl import Config
from torch.utils.data import DataLoader
import torch
from models.mnist_cnn import MNISTCNN
import random

from utils import append_csv, set_seed, get_device
from federated import run_federated_round, evaluate_model
from data import load_mnist, iid_partition, build_client_loader, non_iid_partition

from aggregators import fed_avg


def run_federated_experiment(
    config: Config,
    client_loaders: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    aggregator_name: str
) -> None:
    global_model = MNISTCNN(use_dropout=config.use_dropout).to(device)
    

    num_of_malicious_clients = round(config.num_clients *( config.malicious_fractions))
    malicious_clients = list(range(num_of_malicious_clients))
    selected_clients = list(random.sample(range(config.num_clients), k=config.clients_per_round))

    aggregators = {"fed_avg": fed_avg}

    for round_index in range(config.communication_rounds):
        round_result = run_federated_round(
            global_model,
            selected_clients,
            malicious_clients,
            client_loaders,
            device,
            config.local_epochs,
            config.learning_rate,
            config.momentum,
            aggregators[aggregator_name]
        )

        test_loss, test_accuracy = evaluate_model(global_model, test_loader, device)

        log_row = {
            "round": round_index + 1,
            "global_update_norm": (round_result["global_update_norm"]),
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        }

        append_csv(
            path=config.output_csv,
            row=log_row
        )

        print(
            f"[Federated] "
            f"round={round_index + 1:03d} "
            f"accuracy={test_accuracy:.4f} "
        )

def main() -> None:
    config = Config()

    set_seed(config.seed)
    device = get_device()

    train_dataset, test_dataset = load_mnist(config.data_dir)

    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False) # batch_size might be too large but good for training speed

    #partitions = iid_partition(train_dataset, config.num_clients, config.seed)
    partitions = non_iid_partition(train_dataset, config.num_clients, shards_per_client=2, seed=config.seed)
    client_loaders = build_client_loader(train_dataset, partitions, batch_size=config.batch_size)

    run_federated_experiment(config, client_loaders, test_loader, device, "fed_avg")

if __name__ == "__main__":
    main()
