import torch

def nearest_neighbor_mixxing(updates: torch.Tensor, malicious_clients: int) -> torch.Tensor:
    n = updates.shape[0]
    k = n - malicious_clients

    distances = torch.cdist(updates, updates)

    nearest_indecies = torch.topk(distances, k=k, largest=False, dim=1).indices

    nearest_updates = updates[nearest_indecies]

    return nearest_updates.mean(dim=1)


            