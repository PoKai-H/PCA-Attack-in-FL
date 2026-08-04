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