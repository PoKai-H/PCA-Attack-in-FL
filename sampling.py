import random

from configs.base import Config

def select_stratified_clients(
    config: Config,
    malicious_clients: list[int],
) -> list[int]:
    """
    Random sampling but making sure the malicious clients per round 
    is the same as the clients_per_round * malicious_fraction
    b.c. direct random sampling clients might select more malicious clients than intended
    
    """
    if config.clients_per_round > config.num_clients:
        raise ValueError(
            "clients_per_round cannot be greater than num_clients."
        )

    malicious_client_set = set(malicious_clients)
    benign_clients = [
        client_id
        for client_id in range(config.num_clients)
        if client_id not in malicious_client_set
    ]

    num_selected_malicious_clients = int(
        config.clients_per_round * config.malicious_fractions
    )
    num_selected_malicious_clients = min(
        num_selected_malicious_clients,
        len(malicious_clients)
    )
    num_selected_benign_clients = (
        config.clients_per_round - num_selected_malicious_clients
    )

    if num_selected_benign_clients > len(benign_clients):
        raise ValueError(
            "Not enough benign clients for stratified sampling. "
            "Decrease malicious_fractions or clients_per_round."
        )

    selected_clients = []

    if num_selected_malicious_clients > 0:
        selected_clients.extend(
            random.sample(
                malicious_clients,
                k=num_selected_malicious_clients
            )
        )

    if num_selected_benign_clients > 0:
        selected_clients.extend(
            random.sample(
                benign_clients,
                k=num_selected_benign_clients
            )
        )

    random.shuffle(selected_clients)
    return selected_clients

def count_malicious_clients(
    selected_clients: list[int],
    malicious_clients: list[int],
) -> int:
    malicious_client_set = set(malicious_clients)
    return sum(
        client_id in malicious_client_set
        for client_id in selected_clients
    )

def validate_aggregator_round(
    config: Config,
    selected_clients: list[int],
    num_malicious_clients: int,
) -> None:
    if config.aggregator == "bulyan":
        if len(selected_clients) < 4 * num_malicious_clients + 3:
            raise ValueError(
                "Bulyan requires selected_clients >= "
                "4 * malicious_clients + 3 for each round, "
                f"but got selected_clients={len(selected_clients)}, "
                f"malicious_clients={num_malicious_clients}."
            )
        return

    if config.aggregator not in {"krum", "multi_krum"}:
        return

    if len(selected_clients) <= 2 * num_malicious_clients + 2:
        raise ValueError(
            f"{config.aggregator} requires selected_clients > "
            "2 * malicious_clients + 2 "
            f"for each round, but got selected_clients={len(selected_clients)}, "
            f"malicious_clients={num_malicious_clients}."
        )
