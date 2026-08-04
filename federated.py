from copy import deepcopy
from typing import Callable, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import model_to_vector, vector_to_model

def train_local_model(
        global_model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        local_epochs: int,
        learning_rate: float,
        momentum: float
) -> torch.Tensor:
    local_model = deepcopy(global_model).to(device)
    local_model.train()

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr = learning_rate,
        momentum=momentum
    )

    criterion = nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = local_model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

    global_vector = model_to_vector(global_model)
    local_vector = model_to_vector(local_model)

    model_update = local_vector - global_vector

    return model_update

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()

    criterion = nn.CrossEntropyLoss(
        reduction="sum" #?
    )

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

            total_correct += (
                prediction == labels
            ).sum().item()

            total_examples += labels.numel() #??

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy

def run_federated_round(global_model: nn.Module, 
                        selected_clients: list, 
                        malicious_clients: list,
                        client_loaders: Dict[int, DataLoader],
                        device: torch.device,
                        local_epochs: int,
                        learning_rate: float,
                        momentum: float,
                        aggregator: Callable[
                            [torch.tensor],
                            torch.tensor
                        ]):
    
    previous_global_vector = model_to_vector(global_model)

    benign_client_ids = [id for id in selected_clients if id not in malicious_clients]

    if not benign_client_ids:
        raise ValueError("At least one benign client must participate.")

    benign_updates = []

    for client_id in benign_client_ids:
        client_update = train_local_model(
            global_model=global_model,
            dataloader=client_loaders[client_id],
            device=device,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            momentum=momentum
        )

        benign_updates.append(client_update)

    benign_updates_tensor = torch.stack(benign_updates, dim=0)

    # malicious_client_ids = [id for id in selected_clients if id in malicious_clients]
    # malicious_updates_tensor = benign_updates_tensor.new_ones(
    #     (len(malicious_client_ids), *benign_updates[0].shape)
    # ) # use torch.ones as a place holder now for attacks 

    submited_updates = torch.cat([
        benign_updates_tensor, 
        # malicious_updates_tensor
    ])

    aggregated_updates = aggregator(
        submited_updates
    )

    new_global_vector = (previous_global_vector + aggregated_updates) #??? 為什麼這裡可以直接相加在一個 set 內

    vector_to_model(
        new_global_vector,
        global_model

    )

    return {
        "global_update_norm": float(aggregated_updates.norm()),
        "num_benign_clients": len(benign_client_ids),
        # "num_malicious_clients": len(malicious_client_ids)
    }
