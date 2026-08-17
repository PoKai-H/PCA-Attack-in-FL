import torch


def nearest_neighbor_indices(
    updates: torch.Tensor,
    malicious_clients: int,
    include_self: bool = True,
) -> torch.Tensor:
    n = updates.shape[0]
    k = n - malicious_clients

    if k <= 0:
        raise ValueError("NNM requires k = n - malicious_clients to be positive.")

    neighborhood = k - 1 if include_self else k
    if neighborhood > n - 1:
        raise ValueError("exclude-self nearest neighbors cannot exceed n - 1.")

    distances = torch.cdist(updates, updates)
    distances.fill_diagonal_(float("inf"))
    sorted_indices = torch.argsort(distances, dim=1)
    nearest_indices = sorted_indices[:, :neighborhood]

    if include_self:
        self_indices = torch.arange(n, device=updates.device).unsqueeze(1)
        nearest_indices = torch.cat([nearest_indices, self_indices], dim=1)

    return nearest_indices


def nearest_neighbor_mixxing(
    updates: torch.Tensor,
    malicious_clients: int,
    include_self: bool = True,
) -> torch.Tensor:
    nearest_indices = nearest_neighbor_indices(
        updates,
        malicious_clients,
        include_self=include_self,
    )
    nearest_updates = updates[nearest_indices]

    return nearest_updates.mean(dim=1)
