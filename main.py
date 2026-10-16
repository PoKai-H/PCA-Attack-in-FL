import argparse
import json
from functools import partial

from configs.base import Config

from configs.cifar10.no_attack import CONFIGS as NO_ATTACK_CONFIGS
from configs.cifar10.other_attacks import CONFIGS as OTHER_ATTACKS_CONFIGS
from configs.cifar10.pca_attacks import CONFIGS as GENERATED_PCA_ATTACKS_CONFIGS
from configs.cifar10.extra_seeds import CONFIGS as EXTRA_SEEDS_CONFIGS

from torch.utils.data import DataLoader
import torch

from models.cifar10_net import CIFAR10Net, CIFAR10Simplified
import random

from utils import append_csv, reset_csv, set_seed, get_device
from federated import evaluate_model, run_federated_round
from data import (
    load_mnist,
    load_cifar10,
    find_distribution_map,
    iid_partition,
    build_client_loader,
    non_iid_partition,
)
from sampling import (
    select_stratified_clients,
    count_malicious_clients,
    validate_aggregator_round,
)
from aggregators import (
    bulyan,
    coordinate_wise_median,
    fed_avg,
    krum,
    multi_krum,
    trimmed_mean,
    prodigy,
)
from attacks import ALIE, ALIEOptim, OnesAttack, MinMax, MinSum, SignFlip
from pca_subspace_attack import PCASubspaceAttack

CONFIGS = {
    **NO_ATTACK_CONFIGS,
    **OTHER_ATTACKS_CONFIGS,
    **GENERATED_PCA_ATTACKS_CONFIGS,
    **EXTRA_SEEDS_CONFIGS,
}


def run_federated_experiment(
    config: Config,
    client_loaders: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    global_model = CIFAR10Simplified(use_dropout=config.use_dropout).to(device)

    num_of_malicious_clients = int(config.num_clients * config.malicious_fractions)
    malicious_clients = list(range(num_of_malicious_clients))

    attack = build_attack(config)
    reset_csv(config.output_csv)

    # zero round accuracy check
    zero_round_loss, zero_round_accuracy = evaluate_model(
        global_model, test_loader, device
    )
    log_row = build_log_row(
        round_value=0,
        test_loss=zero_round_loss,
        test_accuracy=zero_round_accuracy,
    )
    append_csv(config.output_csv, row=log_row)

    print(f"[Federated] " f"round={0:03d} " f"accuracy={zero_round_accuracy:.4f} ")

    for round_index in range(config.communication_rounds):
        selected_clients = select_stratified_clients(config, malicious_clients)
        num_selected_malicious_clients = count_malicious_clients(
            selected_clients, malicious_clients
        )
        validate_aggregator_round(
            config, selected_clients, num_selected_malicious_clients
        )
        aggregator_malicious_clients = (
            num_selected_malicious_clients if attack is not None else 0
        )
        aggregator = build_aggregator(config, aggregator_malicious_clients)
        clean_aggregator = build_aggregator(config, 0)

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
            clean_aggregator,
            config.aggregator,
            attack,
            config.include_nnm,
            round_index,
            config.communication_rounds,
        )

        test_loss, test_accuracy = evaluate_model(global_model, test_loader, device)

        log_row = build_log_row(
            round_value=round_index + 1,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            round_result=round_result,
        )

        append_csv(path=config.output_csv, row=log_row)

        print(
            f"[Federated] "
            f"round={round_index + 1:03d} "
            f"accuracy={test_accuracy:.4f} "
            f"malicious_clients={round_result['num_malicious_clients']} "
            f"aggregated_malicious_clients="
            f"{round_result['aggregated_malicious_clients']}"
        )


def build_log_row(
    round_value: int,
    test_loss: float,
    test_accuracy: float,
    round_result: dict | None = None,
) -> dict:
    round_result = round_result or {}
    attack_diagnostics = round_result.get("attack_diagnostics", {})
    round_diagnostics = round_result.get("round_diagnostics", {})

    return {
        "round": round_value,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "learning_rate": serialize_diagnostic(
            round_diagnostics.get("learning_rate", "")
        ),
        "top_k_eigs": serialize_diagnostic(attack_diagnostics.get("top_k_eigs", "")),
        "gamma": serialize_diagnostic(attack_diagnostics.get("gamma", "")),
        "selected_signs": serialize_diagnostic(attack_diagnostics.get("signs", "")),
        "sign_scores": serialize_diagnostic(attack_diagnostics.get("sign_scores", "")),
        "benign_update_norm_mean": serialize_diagnostic(
            round_diagnostics.get("benign_update_norm_mean", "")
        ),
        "malicious_update_norm_mean": serialize_diagnostic(
            round_diagnostics.get("malicious_update_norm_mean", "")
        ),
        "malicious_to_benign_mean_distance": serialize_diagnostic(
            round_diagnostics.get("malicious_to_benign_mean_distance", "")
        ),
        "malicious_to_benign_mean_cosine": serialize_diagnostic(
            round_diagnostics.get("malicious_to_benign_mean_cosine", "")
        ),
        "global_update_norm": serialize_diagnostic(
            round_result.get("global_update_norm", "")
        ),
        "attack_aggregation_shift": serialize_diagnostic(
            round_diagnostics.get("attack_aggregation_shift", "")
        ),
        "relative_attack_shift": serialize_diagnostic(
            round_diagnostics.get("relative_attack_shift", "")
        ),
        "attack_clean_aggregation_cosine": serialize_diagnostic(
            round_diagnostics.get("attack_clean_aggregation_cosine", "")
        ),
        "aggregated_malicious_clients": serialize_diagnostic(
            round_result.get("aggregated_malicious_clients", "")
        ),
        "selected_benign_contribution": serialize_diagnostic(
            round_diagnostics.get("selected_benign_contribution", "")
        ),
        "selected_malicious_contribution": serialize_diagnostic(
            round_diagnostics.get("selected_malicious_contribution", "")
        ),
        "selected_update_to_benign_mean_cosine": serialize_diagnostic(
            round_diagnostics.get("selected_update_to_benign_mean_cosine", "")
        ),
    }


def serialize_diagnostic(value):
    if value == "":
        return ""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)

    return value


def build_aggregator(config: Config, num_malicious_clients: int):
    registry = {
        "fed_avg": lambda: fed_avg,
        "median": lambda: coordinate_wise_median,
        "krum": lambda: partial(krum, malicious_clients=num_malicious_clients),
        "multi_krum": lambda: partial(
            multi_krum, malicious_clients=num_malicious_clients
        ),
        "bulyan": lambda: partial(bulyan, malicious_clients=num_malicious_clients),
        "trimmed_mean": lambda: partial(
            trimmed_mean, malicious_clients=num_malicious_clients
        ),
        "prodigy": lambda: partial(prodigy, malicious_clients=num_malicious_clients),
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
        "min_sum": lambda: MinSum(),
        "alie": lambda: ALIE(z=config.z),
        "alie_optim": lambda: ALIEOptim(
            z_step=config.z_step, z_max=config.z_max, include_nnm=config.include_nnm
        ),
        "sign_flip": lambda: SignFlip(scale=config.sign_flip_scale),
        "pca_subspace": lambda: PCASubspaceAttack(
            k=config.pca_top_k,
            perturbation_mode=config.pca_perturbation_mode,
            sign=config.pca_sign,
        ),
    }

    try:
        return registry[config.attack_name]()
    except KeyError:
        raise ValueError(f"Unknown attack: {config.attack_name}") from None


# def build_partitions(config: Config, train_dataset):
#     registry = {
#         "iid": lambda: iid_partition(train_dataset, config.num_clients, config.seed),
#         "non_iid": lambda: non_iid_partition(
#             train_dataset, config.num_clients, config.shards_per_client, config.seed
#         ),
#     }
#     try:
#         return registry[config.partition]()
#     except KeyError:
#         raise ValueError(f"Unknown partition: {config.partition}") from None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=CONFIGS.keys(),
        default="fedavg_raw_40",
        help="Experiment config to run.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.config]()

    set_seed(config.seed)
    device = get_device()

    train_dataset, test_dataset = load_cifar10(config.data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=False
    )  # batch_size might be too large but good for training speed

    partitions = find_distribution_map(
        train_dataset, config.partition, config.num_clients
    )
    client_loaders = build_client_loader(
        train_dataset, partitions, batch_size=config.batch_size
    )

    run_federated_experiment(config, client_loaders, test_loader, device)


if __name__ == "__main__":
    main()
