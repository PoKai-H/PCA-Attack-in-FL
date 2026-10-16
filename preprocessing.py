import torch


def nearest_neighbor_indices(
    updates: torch.Tensor,
    malicious_clients: int,
) -> torch.Tensor:
    n = updates.shape[0]
    f = malicious_clients

    if f < 0:
        raise ValueError("malicious_clients must be non-negative.")

    if f >= n / 2:
        raise ValueError(f"NNM requires f < n/2, got n={n}, f={f}.")

    k = n - f

    distances = torch.cdist(updates, updates)

    # Self-distance is 0, so each update is naturally included
    # in its own nearest-neighbor set.
    nearest_indices = torch.argsort(
        distances,
        dim=1,
    )[:, :k]

    return nearest_indices


def apply_nnm_with_contributions(
    updates: torch.Tensor,
    malicious_clients: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    nearest_indices = nearest_neighbor_indices(
        updates,
        malicious_clients,
    )

    nearest_updates = updates[nearest_indices]

    mixed_updates = nearest_updates.mean(dim=1)

    row_contributions = torch.eye(
        updates.shape[0],
        device=updates.device,
        dtype=updates.dtype,
    )[nearest_indices].mean(dim=1)

    return mixed_updates, row_contributions
