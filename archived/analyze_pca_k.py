import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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


def collect_benign_updates(config, num_sampling_rounds: int) -> torch.Tensor:
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

    updates = []

    for _ in range(num_sampling_rounds):
        selected_clients = select_stratified_clients(config, malicious_clients)
        benign_client_ids = [
            client_id
            for client_id in selected_clients
            if client_id not in malicious_client_set
        ]

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

    return torch.stack(updates, dim=0)


def analyze_pca(updates: torch.Tensor, max_k: int) -> list[dict]:
    centered = updates - updates.mean(dim=0, keepdim=True)
    _, singular_values, right_vectors = torch.linalg.svd(
        centered,
        full_matrices=False,
    )

    variances = singular_values.square()
    total_variance = variances.sum()

    if total_variance <= 0:
        raise ValueError("PCA variance is zero; collected updates are identical.")

    explained_ratios = variances / total_variance
    max_k = min(max_k, right_vectors.shape[0])

    rows = []
    total_norm = centered.norm()

    for k in range(1, max_k + 1):
        top_k = right_vectors[:k]
        projected = centered @ top_k.T
        reconstructed = projected @ top_k
        reconstruction_error = (centered - reconstructed).norm() / total_norm

        rows.append(
            {
                "k": k,
                "explained_variance_ratio": float(explained_ratios[k - 1]),
                "cumulative_explained_variance": float(
                    explained_ratios[:k].sum()
                ),
                "relative_reconstruction_error": float(reconstruction_error),
            }
        )

    return rows


def write_rows(path: str, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS.keys(), required=True)
    parser.add_argument("--sampling-rounds", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument(
        "--output-csv",
        default="./results/pca_k_analysis.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.config]()

    set_seed(config.seed)

    updates = collect_benign_updates(
        config=config,
        num_sampling_rounds=args.sampling_rounds,
    )
    rows = analyze_pca(updates, args.max_k)
    write_rows(args.output_csv, rows)

    print(f"Collected benign updates: {updates.shape[0]}")
    print(f"Update dimension: {updates.shape[1]}")
    print(f"Wrote PCA k analysis to {args.output_csv}")
    print("k,cumulative_explained_variance,relative_reconstruction_error")

    for row in rows:
        print(
            f"{row['k']},"
            f"{row['cumulative_explained_variance']:.4f},"
            f"{row['relative_reconstruction_error']:.4f}"
        )


if __name__ == "__main__":
    main()
