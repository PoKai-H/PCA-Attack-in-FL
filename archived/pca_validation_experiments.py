import argparse
from pathlib import Path

import torch

from configs.mnist.pca_config import CONFIGS
from data import build_client_loader, iid_partition, load_mnist, non_iid_partition
from federated import train_local_model
from models.mnist_cnn import MNISTCNN
from sampling import select_stratified_clients
from utils import get_device, set_seed


def build_partitions(config, train_dataset):
    if config.partition == "iid":
        return iid_partition(train_dataset, config.num_clients, config.seed)

    if config.partition == "non_iid":
        return non_iid_partition(
            train_dataset,
            config.num_clients,
            config.shards_per_client,
            config.seed,
        )

    raise ValueError(f"Unknown partition: {config.partition}")


def collect_round_update_matrices(config, num_rounds: int = 1) -> list[torch.Tensor]:
    device = get_device()
    train_dataset, _ = load_mnist(config.data_dir)
    partitions = build_partitions(config, train_dataset)
    client_loaders = build_client_loader(
        train_dataset,
        partitions,
        batch_size=config.batch_size,
    )

    global_model = MNISTCNN(use_dropout=config.use_dropout).to(device)
    num_malicious_clients = int(config.num_clients * config.malicious_fractions)
    malicious_clients = list(range(num_malicious_clients))
    malicious_client_set = set(malicious_clients)

    round_updates = []

    for _ in range(num_rounds):
        selected_clients = select_stratified_clients(config, malicious_clients)
        benign_client_ids = [
            client_id
            for client_id in selected_clients
            if client_id not in malicious_client_set
        ]

        updates = []
        for client_id in benign_client_ids:
            update = train_local_model(
                global_model=global_model,
                dataloader=client_loaders[client_id],
                device=device,
                local_epochs=config.local_epochs,
                learning_rate=config.learning_rate,
                momentum=config.momentum,
            )
            updates.append(update.cpu())

        if len(updates) < 2:
            raise ValueError("At least two benign updates are required for PCA.")

        round_updates.append(torch.stack(updates, dim=0))

    return round_updates


def pca_basis_and_variance(updates: torch.Tensor):
    centered = updates - updates.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)

    rank = min(centered.shape[0] - 1, centered.shape[1])
    singular_values = singular_values[:rank]
    basis = vh[:rank].T

    variances = singular_values.square()
    total_variance = variances.sum()
    if total_variance <= 0:
        raise ValueError("PCA variance is zero; collected updates are identical.")

    explained = variances / total_variance
    return centered, basis, explained


def pca_subspace_ratio(direction: torch.Tensor, pca_basis: torch.Tensor) -> float:
    coeffs = pca_basis.T @ direction
    projection = pca_basis @ coeffs  # d_|| = UU^T d

    energy_ratio = (
        torch.linalg.norm(projection) ** 2 / torch.linalg.norm(direction) ** 2
    )
    return energy_ratio.item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS.keys(), required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=50)
    parser.add_argument("--output-dir", default="./results/pca_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.config]()
    set_seed(config.seed)

    round_updates = collect_round_update_matrices(config, args.rounds)

    results = []

    for round_id, updates in enumerate(round_updates):
        # Directions must be computed from ORIGINAL updates.
        mean = updates.mean(dim=0)
        std = updates.std(dim=0, unbiased=False)

        d_sign = -torch.sign(mean)
        d_uv = -mean / torch.linalg.norm(mean)
        d_std = -std

        # PCA is computed from CENTERED updates.
        _, pca_basis, explained = pca_basis_and_variance(updates)

        max_k = min(args.max_k, pca_basis.shape[1])

        for k in range(1, max_k + 1):
            basis_k = pca_basis[:, :k]

            sign_ratio = pca_subspace_ratio(
                d_sign,
                basis_k,
            )
            uv_ratio = pca_subspace_ratio(
                d_uv,
                basis_k,
            )
            std_ratio = pca_subspace_ratio(
                d_std,
                basis_k,
            )

            results.append(
                {
                    "round": round_id,
                    "k": k,
                    "sign": sign_ratio,
                    "uv": uv_ratio,
                    "std": std_ratio,
                    "explained_variance": explained[:k].sum().item(),
                }
            )

            print(
                f"round={round_id:02d}, k={k:02d}: "
                f"sign={sign_ratio:.4f}, "
                f"uv={uv_ratio:.4f}, "
                f"std={std_ratio:.4f}, "
                f"variance={explained[:k].sum().item():.4f}"
            )


if __name__ == "__main__":
    main()
