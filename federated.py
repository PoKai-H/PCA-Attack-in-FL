from copy import deepcopy
import os
from typing import Callable, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from aggregators import (
    count_aggregated_malicious_clients,
    krum_selected_index,
    multi_krum_selected_indices,
)
from utils import model_to_vector, vector_to_model
from preprocessing import apply_nnm_with_contributions


STRICT_FINITE = os.getenv("STRICT_FINITE", "0") == "1"
_WARNED_NONFINITE_LABELS = set()


def _ensure_finite(tensor: torch.Tensor, label: str) -> None:
    if torch.isfinite(tensor).all():
        return

    finite_values = tensor[torch.isfinite(tensor)]
    if finite_values.numel() == 0:
        message = f"{label} contains only NaN/Inf values."
    else:
        message = (
            f"{label} contains NaN/Inf values. "
            f"finite_min={finite_values.min().item()} "
            f"finite_max={finite_values.max().item()}"
        )

    if STRICT_FINITE:
        raise FloatingPointError(message)

    if label not in _WARNED_NONFINITE_LABELS:
        print(f"[Warning] {message}")
        _WARNED_NONFINITE_LABELS.add(label)


def train_local_model(
    global_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    local_epochs: int,
    learning_rate: float,
    momentum: float,
    client_id: int | None = None,
    round_idx: int | None = None,
) -> torch.Tensor:
    local_model = deepcopy(global_model).to(device)
    local_model.train()

    optimizer = torch.optim.SGD(
        local_model.parameters(), lr=learning_rate, momentum=momentum
    )

    criterion = nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = local_model(images)
            _ensure_finite(
                logits,
                f"local logits round={round_idx} client={client_id}",
            )

            loss = criterion(logits, labels)
            _ensure_finite(
                loss,
                f"local loss round={round_idx} client={client_id}",
            )

            loss.backward()
            optimizer.step()

    global_vector = model_to_vector(global_model)
    local_vector = model_to_vector(local_model)

    model_update = local_vector - global_vector
    _ensure_finite(
        model_update,
        f"local model update round={round_idx} client={client_id}",
    )

    return model_update


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")  # ?

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            total_loss += criterion(
                logits,
                labels,
            ).item()

            prediction = logits.argmax(dim=1)

            total_correct += (prediction == labels).sum().item()

            total_examples += labels.numel()  # ??

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def _update_norm_mean(updates: torch.Tensor) -> float:
    return float(updates.norm(dim=1).mean())


def _safe_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = a.norm() * b.norm()
    if denominator.item() == 0:
        return 0.0
    return float(torch.dot(a, b) / denominator)


def _aggregator_selection_diagnostics(
    updates: torch.Tensor,
    malicious_clients: int,
    aggregator_name: str,
) -> list[int]:
    if malicious_clients == 0:
        return []

    if aggregator_name == "krum":
        selected_indices = [int(krum_selected_index(updates, malicious_clients).item())]
    elif aggregator_name == "multi_krum":
        selected_indices = [
            int(index.item())
            for index in multi_krum_selected_indices(updates, malicious_clients)
        ]
    else:
        return []

    return selected_indices


def _selected_update_diagnostics(
    updates: torch.Tensor,
    selected_indices: list[int],
    benign_mean_update: torch.Tensor,
    row_contributions: torch.Tensor | None,
    num_benign: int,
) -> dict[str, float | str]:
    if not selected_indices:
        return {
            "selected_benign_contribution": "",
            "selected_malicious_contribution": "",
            "selected_update_to_benign_mean_cosine": "",
        }

    selected_index_tensor = torch.tensor(selected_indices, device=updates.device)
    selected_updates = updates[selected_index_tensor]
    selected_update = selected_updates.mean(dim=0)

    if row_contributions is None:
        row_contributions = torch.eye(
            updates.shape[0],
            device=updates.device,
            dtype=updates.dtype,
        )

    selected_contributions = row_contributions[selected_index_tensor].mean(dim=0)
    selected_benign_contribution = selected_contributions[:num_benign].sum().item()
    selected_malicious_contribution = selected_contributions[num_benign:].sum().item()

    return {
        "selected_benign_contribution": selected_benign_contribution,
        "selected_malicious_contribution": selected_malicious_contribution,
        "selected_update_to_benign_mean_cosine": _safe_cosine_similarity(
            selected_update,
            benign_mean_update,
        ),
    }


def run_federated_round(
    global_model: nn.Module,
    selected_clients: list,
    malicious_clients: list,
    client_loaders: Dict[int, DataLoader],
    device: torch.device,
    local_epochs: int,
    learning_rate: float,
    momentum: float,
    aggregator: Callable[[torch.tensor], torch.tensor],
    clean_aggregator: Callable[[torch.tensor], torch.tensor],
    aggregator_name: str,
    attack,
    include_nnm: bool,
    round_idx: int,
    communication_rounds: int,
):

    previous_global_vector = model_to_vector(global_model)

    benign_client_ids = [id for id in selected_clients if id not in malicious_clients]

    if not benign_client_ids:
        raise ValueError("At least one benign client must participate.")

    benign_updates = []
    switch_round = int(2 * communication_rounds / 3)

    if round_idx < switch_round:
        learning_rate = 0.05
    else:
        learning_rate = 0.005

    for client_id in benign_client_ids:
        client_update = train_local_model(
            global_model=global_model,
            dataloader=client_loaders[client_id],
            device=device,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            client_id=client_id,
            round_idx=round_idx + 1,
        )

        benign_updates.append(client_update)

    benign_updates_tensor = torch.stack(benign_updates, dim=0)
    benign_update_norm_mean = _update_norm_mean(benign_updates_tensor)
    benign_mean_update = benign_updates_tensor.mean(dim=0)

    malicious_client_ids = [
        client_id for client_id in selected_clients if client_id in malicious_clients
    ]
    num_malicious_client = len(malicious_client_ids)
    attack_diagnostics = {}

    if attack is not None and num_malicious_client > 0:
        attack_result = attack.generate(
            benign_updates_tensor,
            num_malicious_client,
            aggregator,
            clean_aggregator,
            global_model,
        )
        if isinstance(attack_result, tuple):
            malicious_update_tensor = attack_result[0]
            if len(attack_result) > 1 and isinstance(attack_result[1], dict):
                attack_diagnostics = attack_result[1]
        else:
            malicious_update_tensor = attack_result

        submitted_updates = torch.cat([benign_updates_tensor, malicious_update_tensor])
        num_submitted_malicious_clients = num_malicious_client

        malicious_mean_update = malicious_update_tensor.mean(dim=0)
        attack_debug = {
            "malicious_update_norm_mean": _update_norm_mean(malicious_update_tensor),
            "malicious_to_benign_mean_distance": float(
                torch.linalg.norm(malicious_mean_update - benign_mean_update)
            ),
            "malicious_to_benign_mean_cosine": _safe_cosine_similarity(
                malicious_mean_update,
                benign_mean_update,
            ),
        }
    else:
        submitted_updates = benign_updates_tensor
        num_submitted_malicious_clients = 0
        attack_debug = {
            "malicious_update_norm_mean": "",
            "malicious_to_benign_mean_distance": "",
            "malicious_to_benign_mean_cosine": "",
        }

    row_contributions = None
    if include_nnm:
        submitted_updates, row_contributions = apply_nnm_with_contributions(
            submitted_updates,
            num_submitted_malicious_clients,
        )
    _ensure_finite(submitted_updates, f"submitted updates round={round_idx + 1}")

    clean_updates = benign_updates_tensor
    if include_nnm and num_submitted_malicious_clients > 0:
        clean_updates, _ = apply_nnm_with_contributions(
            benign_updates_tensor,
            num_submitted_malicious_clients,
        )

    clean_aggregated_update = clean_aggregator(clean_updates)

    selected_indices = _aggregator_selection_diagnostics(
        submitted_updates,
        num_submitted_malicious_clients,
        aggregator_name,
    )

    aggregated_updates = aggregator(submitted_updates)
    _ensure_finite(aggregated_updates, f"aggregated update round={round_idx + 1}")
    if num_submitted_malicious_clients > 0:
        attack_aggregation_shift = float(
            torch.linalg.norm(aggregated_updates - clean_aggregated_update)
        )
        relative_attack_shift = float(
            torch.linalg.norm(aggregated_updates - clean_aggregated_update)
            / clean_aggregated_update.norm().clamp_min(1e-12)
        )
        attack_clean_aggregation_cosine = _safe_cosine_similarity(
            aggregated_updates,
            clean_aggregated_update,
        )
    else:
        attack_aggregation_shift = ""
        relative_attack_shift = ""
        attack_clean_aggregation_cosine = ""

    selected_update_debug = _selected_update_diagnostics(
        submitted_updates,
        selected_indices,
        benign_mean_update,
        row_contributions,
        len(benign_client_ids),
    )

    aggregated_malicious_clients = count_aggregated_malicious_clients(
        submitted_updates,
        num_submitted_malicious_clients,
        aggregator_name,
        row_contributions=row_contributions,
    )

    new_global_vector = (
        previous_global_vector + aggregated_updates
    )  # ??? 為什麼這裡可以直接相加在一個 set 內
    _ensure_finite(new_global_vector, f"new global vector round={round_idx + 1}")

    vector_to_model(new_global_vector, global_model)

    return {
        "global_update_norm": float(aggregated_updates.norm()),
        "aggregated_malicious_clients": aggregated_malicious_clients,
        "attack_diagnostics": attack_diagnostics,
        "round_diagnostics": {
            "learning_rate": learning_rate,
            "benign_update_norm_mean": benign_update_norm_mean,
            "malicious_update_norm_mean": attack_debug["malicious_update_norm_mean"],
            "malicious_to_benign_mean_distance": attack_debug[
                "malicious_to_benign_mean_distance"
            ],
            "malicious_to_benign_mean_cosine": attack_debug[
                "malicious_to_benign_mean_cosine"
            ],
            "attack_aggregation_shift": attack_aggregation_shift,
            "relative_attack_shift": relative_attack_shift,
            "attack_clean_aggregation_cosine": attack_clean_aggregation_cosine,
            **selected_update_debug,
        },
        "num_benign_clients": len(benign_client_ids),
        "num_malicious_clients": len(malicious_client_ids),
    }
