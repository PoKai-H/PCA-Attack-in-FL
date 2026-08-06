import argparse
from functools import partial

from configs.base import Config
from configs.experiments import CONFIGS
from torch.utils.data import DataLoader
import torch
from models.mnist_cnn import MNISTCNN
import random

from utils import append_csv, set_seed, get_device
from federated import run_federated_round, evaluate_model
from data import load_mnist, iid_partition, build_client_loader, non_iid_partition

from aggregators import coordinate_wise_median, fed_avg, krum
from attacks import OnesAttack, MinMax, MinSum


def run_federated_experiment(
    config: Config,
    client_loaders: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    global_model = MNISTCNN(use_dropout=config.use_dropout).to(device)
    

    num_of_malicious_clients = round(config.num_clients *( config.malicious_fractions))
    malicious_clients = list(range(num_of_malicious_clients))
    selected_clients = list(random.sample(range(config.num_clients), k=config.clients_per_round))

    aggregator = build_aggregator(config, num_of_malicious_clients)
    attack = build_attack(config)

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
            aggregator,
            attack
        )

        test_loss, test_accuracy = evaluate_model(global_model, test_loader, device)

        log_row = {
            "round": round_index + 1,
            "malicious_clients": (round_result["num_malicious_clients"]),
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
            f"malicious_clients={round_result['num_malicious_clients']}"
        )

def build_aggregator(config: Config, num_malicious_clients: int):
    registry = {
        "fed_avg": lambda: fed_avg,
        "coordinate_wise_median": lambda: coordinate_wise_median,
        "krum": lambda: partial(
            krum,
            malicious_clients=num_malicious_clients
        )
    }

    try:
        return registry[config.aggregator]()
    except KeyError:
        raise ValueError(f"Unknown aggregator: {config.aggregator}") from None

def build_attack(config: Config):
    registry = {
        None: lambda: None,
        "ones": lambda: OnesAttack(),
        "min_max": lambda: MinMax(),
        "min_sum": lambda: MinSum()
    }

    try: 
        return registry[config.attack_name]()
    except KeyError:
        raise ValueError(f"Unknown attack: {config.attack_name}") from None

def build_partitions(config: Config, train_dataset):
    registry = {
        "iid": lambda: iid_partition(
            train_dataset,
            config.num_clients,
            config.seed
        ),
        "non_iid": lambda: non_iid_partition(
            train_dataset,
            config.num_clients,
            config.shards_per_client,
            config.seed
        )
    }
    try:
        return registry[config.partition]()
    except KeyError:
        raise ValueError(f"Unknown partition: {config.partition}") from None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=CONFIGS.keys(),
        default="fedavg_iid",
        help="Experiment config to run."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.config]()

    set_seed(config.seed)
    device = get_device()

    train_dataset, test_dataset = load_mnist(config.data_dir)

    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False) # batch_size might be too large but good for training speed

    partitions = build_partitions(config, train_dataset)
    client_loaders = build_client_loader(train_dataset, partitions, batch_size=config.batch_size)

    run_federated_experiment(config, client_loaders, test_loader, device)

if __name__ == "__main__":
    main()
