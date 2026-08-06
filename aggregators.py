import torch

def fed_avg(updates: torch.Tensor) -> torch.Tensor:
    """
    Args:
        updates:
            Tensor shaped
            [num_clients, num_parameters].

    Returns:
        Aggregated update shaped
        [num_parameters].
    """
    if updates.ndim != 2:
        raise ValueError(
            "updates must have shape "
            "[num_clients, num_parameters]."
        )

    if updates.shape[0] == 0:
        raise ValueError(
            "At least one client update is required."
        )

    return updates.mean(dim=0)


def krum(updates: torch.Tensor, malicious_clients: int) -> torch.Tensor:
    if updates.ndim != 2:
        raise ValueError("Updates must have shape [num_clients, num_parameters].")

    n = updates.shape[0]

    if n == 0:
        raise ValueError("At least one client update is required.")

    if malicious_clients < 0: 
        raise ValueError("malicious_clients must be non-negative.")

    if n <= 2 * malicious_clients + 2:
        raise ValueError(f"Krum requires n > 2 * malicious_clients + 2, but got n={n}, malicious_clients={malicious_clients}.")

    num_neighbors = n - malicious_clients - 2
    scores = []

    for candidate_index, candidate in enumerate(updates):
        distances = []

        for other_index, other in enumerate(updates):
            if candidate_index == other_index:
                continue

            distances.append(torch.sum((candidate - other) ** 2))

        nearest_distance = torch.stack(distances).sort().values[:num_neighbors]
        scores.append(nearest_distance.sum())

    best_index = torch.stack(scores).argmin()
    
    return updates[best_index]


def coordinate_wise_median(updates: torch.Tensor) -> torch.Tensor:
    if updates.ndim != 2:
            raise ValueError(
                "updates must have shape "
                "[num_clients, num_parameters]."
            )
    
    if updates.shape[0] == 0:
            raise ValueError(
                "At least one client update is required."
            )
    return torch.median(updates, dim=0).values