import argparse
import csv
from copy import deepcopy
from pathlib import Path

import torch

from configs.experiments import CONFIGS
from data import build_client_loader, iid_partition, load_mnist, non_iid_partition
from federated import evaluate_model, train_local_model
from main import build_aggregator, build_attack
from models.mnist_cnn import MNISTCNN
from preprocessing import nearest_neighbor_indices, nearest_neighbor_mixxing
from sampling import (
    count_malicious_clients,
    select_stratified_clients,
    validate_aggregator_round,
)
from utils import get_device, model_to_vector, set_seed, vector_to_model


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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    with path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clone_model(model, config, device):
    cloned = MNISTCNN(use_dropout=config.use_dropout).to(device)
    cloned.load_state_dict(deepcopy(model.state_dict()))
    return cloned


def collect_submitted_updates(
    model,
    selected_clients,
    malicious_clients,
    client_loaders,
    device,
    config,
    attack,
    aggregator,
):
    malicious_client_set = set(malicious_clients)
    benign_client_ids = [
        client_id
        for client_id in selected_clients
        if client_id not in malicious_client_set
    ]
    malicious_client_ids = [
        client_id
        for client_id in selected_clients
        if client_id in malicious_client_set
    ]

    if not benign_client_ids:
        raise ValueError("At least one benign client must participate.")

    benign_updates = []
    for client_id in benign_client_ids:
        update = train_local_model(
            global_model=model,
            dataloader=client_loaders[client_id],
            device=device,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
        )
        benign_updates.append(update)

    benign_updates = torch.stack(benign_updates, dim=0)
    num_malicious_clients = len(malicious_client_ids)

    if attack is None or num_malicious_clients == 0:
        return benign_updates, benign_client_ids, malicious_client_ids

    malicious_updates = attack.generate(
        benign_updates,
        num_malicious_clients,
        aggregator,
        model,
    )
    submitted_updates = torch.cat([benign_updates, malicious_updates], dim=0)
    return submitted_updates, benign_client_ids, malicious_client_ids


def labels_for_updates(benign_client_ids, malicious_client_ids):
    labels = [f"benign:{client_id}" for client_id in benign_client_ids]
    labels.extend(f"malicious:{client_id}" for client_id in malicious_client_ids)
    return labels


def validate_nnm_neighbors(
    updates,
    mixed_updates,
    labels,
    num_malicious_clients,
    round_index,
    include_self,
):
    n = updates.shape[0]
    k = n - num_malicious_clients
    distances = torch.cdist(updates, updates)
    nearest_indices = nearest_neighbor_indices(
        updates,
        num_malicious_clients,
        include_self=include_self,
    )

    rows = []
    for update_index in range(n):
        neighbor_ids = nearest_indices[update_index].tolist()
        neighbor_labels = [labels[neighbor_id] for neighbor_id in neighbor_ids]
        neighbor_distances = distances[update_index, nearest_indices[update_index]]
        recomputed = updates[nearest_indices[update_index]].mean(dim=0)

        rows.append(
            {
                "round": round_index,
                "update_index": update_index,
                "label": labels[update_index],
                "is_malicious": labels[update_index].startswith("malicious:"),
                "k": k,
                "self_included": update_index in neighbor_ids,
                "neighbor_indices": "|".join(str(index) for index in neighbor_ids),
                "neighbor_labels": "|".join(neighbor_labels),
                "neighbor_malicious_count": sum(
                    label.startswith("malicious:") for label in neighbor_labels
                ),
                "nearest_distance_mean": float(neighbor_distances.mean()),
                "nearest_distance_max": float(neighbor_distances.max()),
                "raw_update_norm": float(updates[update_index].norm()),
                "mixed_update_norm": float(mixed_updates[update_index].norm()),
                "mix_displacement_norm": float(
                    (mixed_updates[update_index] - updates[update_index]).norm()
                ),
                "matches_recomputed_mean": bool(
                    torch.allclose(mixed_updates[update_index], recomputed)
                ),
            }
        )

    return rows


def apply_update_to_clone(model, update, config, device):
    cloned = clone_model(model, config, device)
    vector_to_model(model_to_vector(model) + update.cpu(), cloned)
    return cloned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS.keys(), required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output-dir", default="./results/nnm_validation")
    parser.add_argument(
        "--advance-with",
        choices=["raw", "nnm"],
        default="nnm",
        help="Which branch becomes the next round's base model.",
    )
    parser.add_argument(
        "--include-self",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether each update itself is included in its nearest-neighbor set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.config]()
    set_seed(config.seed)

    device = get_device()
    train_dataset, test_dataset = load_mnist(config.data_dir)
    partitions = build_partitions(config, train_dataset)
    client_loaders = build_client_loader(
        train_dataset,
        partitions,
        batch_size=config.batch_size,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
    )

    model = MNISTCNN(use_dropout=config.use_dropout).to(device)
    attack = build_attack(config)
    total_malicious_clients = int(config.num_clients * config.malicious_fractions)
    malicious_clients = list(range(total_malicious_clients))

    summary_rows = []
    neighbor_rows = []

    for round_index in range(1, args.rounds + 1):
        selected_clients = select_stratified_clients(config, malicious_clients)
        num_selected_malicious_clients = count_malicious_clients(
            selected_clients,
            malicious_clients,
        )
        validate_aggregator_round(
            config,
            selected_clients,
            num_selected_malicious_clients,
        )
        aggregator = build_aggregator(config, num_selected_malicious_clients)

        submitted_updates, benign_client_ids, malicious_client_ids = (
            collect_submitted_updates(
                model,
                selected_clients,
                malicious_clients,
                client_loaders,
                device,
                config,
                attack,
                aggregator,
            )
        )
        labels = labels_for_updates(benign_client_ids, malicious_client_ids)

        raw_aggregated_update = aggregator(submitted_updates)
        nnm_updates = nearest_neighbor_mixxing(
            submitted_updates,
            num_selected_malicious_clients,
            include_self=args.include_self,
        )
        nnm_aggregated_update = aggregator(nnm_updates)

        raw_model = apply_update_to_clone(
            model,
            raw_aggregated_update,
            config,
            device,
        )
        nnm_model = apply_update_to_clone(
            model,
            nnm_aggregated_update,
            config,
            device,
        )
        raw_loss, raw_accuracy = evaluate_model(raw_model, test_loader, device)
        nnm_loss, nnm_accuracy = evaluate_model(nnm_model, test_loader, device)

        submitted_mean = submitted_updates.mean(dim=0)
        nnm_mean = nnm_updates.mean(dim=0)
        round_neighbor_rows = validate_nnm_neighbors(
            submitted_updates,
            nnm_updates,
            labels,
            num_selected_malicious_clients,
            round_index,
            args.include_self,
        )
        summary_rows.append(
            {
                "round": round_index,
                "config": args.config,
                "aggregator": config.aggregator,
                "attack_name": config.attack_name,
                "selected_clients": "|".join(
                    str(client) for client in selected_clients
                ),
                "benign_clients": "|".join(
                    str(client) for client in benign_client_ids
                ),
                "malicious_clients": "|".join(
                    str(client) for client in malicious_client_ids
                ),
                "num_updates": submitted_updates.shape[0],
                "num_malicious_updates": num_selected_malicious_clients,
                "nnm_k": submitted_updates.shape[0] - num_selected_malicious_clients,
                "include_self": args.include_self,
                "self_included_count": sum(
                    row["self_included"] for row in round_neighbor_rows
                ),
                "all_rows_match_recomputed_mean": all(
                    row["matches_recomputed_mean"] for row in round_neighbor_rows
                ),
                "malicious_neighbor_count_mean": sum(
                    row["neighbor_malicious_count"] for row in round_neighbor_rows
                )
                / len(round_neighbor_rows),
                "raw_update_norm": float(raw_aggregated_update.norm()),
                "nnm_update_norm": float(nnm_aggregated_update.norm()),
                "aggregation_delta_norm": float(
                    (nnm_aggregated_update - raw_aggregated_update).norm()
                ),
                "submitted_mean_norm": float(submitted_mean.norm()),
                "nnm_mean_norm": float(nnm_mean.norm()),
                "mean_delta_norm": float((nnm_mean - submitted_mean).norm()),
                "raw_test_loss": raw_loss,
                "nnm_test_loss": nnm_loss,
                "raw_test_accuracy": raw_accuracy,
                "nnm_test_accuracy": nnm_accuracy,
                "accuracy_delta_nnm_minus_raw": nnm_accuracy - raw_accuracy,
                "advance_with": args.advance_with,
            }
        )
        neighbor_rows.extend(round_neighbor_rows)

        model = raw_model if args.advance_with == "raw" else nnm_model

        print(
            f"[NNM validation] round={round_index:03d} "
            f"raw_acc={raw_accuracy:.4f} nnm_acc={nnm_accuracy:.4f} "
            f"delta={nnm_accuracy - raw_accuracy:+.4f} "
            f"agg_delta_norm={summary_rows[-1]['aggregation_delta_norm']:.4f}"
        )

    output_dir = Path(args.output_dir)
    self_mode = "include_self" if args.include_self else "exclude_self"
    prefix = f"{args.config}_{args.advance_with}_{self_mode}"
    write_csv(output_dir / f"{prefix}_raw_vs_nnm.csv", summary_rows)
    write_csv(output_dir / f"{prefix}_neighbors.csv", neighbor_rows)

    print(f"Wrote raw vs NNM log to {output_dir / f'{prefix}_raw_vs_nnm.csv'}")
    print(f"Wrote NNM neighbor log to {output_dir / f'{prefix}_neighbors.csv'}")


if __name__ == "__main__":
    main()
