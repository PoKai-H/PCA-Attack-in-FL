import argparse
import csv
from itertools import combinations
from pathlib import Path

import torch

from configs.experiments import CONFIGS
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


def collect_round_update_matrices(config, num_rounds: int) -> list[torch.Tensor]:
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


def scree_rows(round_updates: list[torch.Tensor], max_k: int) -> list[dict]:
    rows = []
    for round_index, updates in enumerate(round_updates, start=1):
        _, _, explained = pca_basis_and_variance(updates)
        limit = min(max_k, explained.numel())

        for k in range(1, limit + 1):
            rows.append(
                {
                    "round": round_index,
                    "k": k,
                    "explained_variance_ratio": float(explained[k - 1]),
                    "cumulative_explained_variance": float(explained[:k].sum()),
                }
            )

    return rows


def subspace_overlap(basis_a: torch.Tensor, basis_b: torch.Tensor, k: int) -> float:
    subspace_a = basis_a[:, :k]
    subspace_b = basis_b[:, :k]
    singular_values = torch.linalg.svdvals(subspace_a.T @ subspace_b)
    return float(singular_values.square().mean())


def stability_rows(round_updates: list[torch.Tensor], max_k: int) -> list[dict]:
    pca_results = [
        pca_basis_and_variance(updates)
        for updates in round_updates
    ]

    rows = []
    for (i, result_i), (j, result_j) in combinations(enumerate(pca_results), 2):
        _, basis_i, explained_i = result_i
        _, basis_j, explained_j = result_j
        limit = min(max_k, basis_i.shape[1], basis_j.shape[1])

        for k in range(1, limit + 1):
            rows.append(
                {
                    "round_a": i + 1,
                    "round_b": j + 1,
                    "k": k,
                    "subspace_overlap": subspace_overlap(basis_i, basis_j, k),
                    "round_a_cumulative_explained": float(explained_i[:k].sum()),
                    "round_b_cumulative_explained": float(explained_j[:k].sum()),
                }
            )

    return rows


def random_orthonormal_basis(
    dimension: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    random_matrix = torch.randn(dimension, k, device=device)
    basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return basis


def random_baseline_rows(
    round_updates: list[torch.Tensor],
    max_k: int,
    random_trials: int,
) -> list[dict]:
    rows = []
    for round_index, updates in enumerate(round_updates, start=1):
        centered, _, explained = pca_basis_and_variance(updates)
        total_energy = centered.square().sum()
        limit = min(max_k, explained.numel())

        for k in range(1, limit + 1):
            random_explained = []
            for _ in range(random_trials):
                random_basis = random_orthonormal_basis(
                    dimension=centered.shape[1],
                    k=k,
                    device=centered.device,
                )
                projected = centered @ random_basis
                random_explained.append(float(projected.square().sum() / total_energy))

            random_values = torch.tensor(random_explained)
            rows.append(
                {
                    "round": round_index,
                    "k": k,
                    "pca_cumulative_explained": float(explained[:k].sum()),
                    "random_cumulative_explained_mean": float(random_values.mean()),
                    "random_cumulative_explained_std": float(
                        random_values.std(unbiased=False)
                    ),
                    "random_trials": random_trials,
                }
            )

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    with path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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

    output_dir = Path(args.output_dir)
    round_updates = collect_round_update_matrices(config, args.rounds)

    scree = scree_rows(round_updates, args.max_k)
    stability = stability_rows(round_updates, args.max_k)
    baseline = random_baseline_rows(
        round_updates,
        args.max_k,
        args.random_trials,
    )

    prefix = args.config
    write_csv(output_dir / f"{prefix}_scree.csv", scree)
    write_csv(output_dir / f"{prefix}_stability.csv", stability)
    write_csv(output_dir / f"{prefix}_random_baseline.csv", baseline)

    print(f"Collected {len(round_updates)} round update matrices.")
    print(f"Benign updates per round: {round_updates[0].shape[0]}")
    print(f"Update dimension: {round_updates[0].shape[1]}")
    print(f"Wrote PCA validation CSVs to {output_dir}")


if __name__ == "__main__":
    main()
