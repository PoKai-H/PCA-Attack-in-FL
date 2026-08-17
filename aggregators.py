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
        raise ValueError("updates must have shape " "[num_clients, num_parameters].")

    if updates.shape[0] == 0:
        raise ValueError("At least one client update is required.")

    return updates.mean(dim=0)


def _compute_krum_score(updates: torch.Tensor, malicious_clients: int) -> torch.Tensor:
    n = updates.shape[0]
    if updates.ndim != 2:
        raise ValueError("Updates must have shape [num_clients, num_parameters].")

    if n == 0:
        raise ValueError("At least one client update is required.")

    if malicious_clients < 0:
        raise ValueError("malicious_clients must be non-negative.")

    if n <= 2 * malicious_clients + 2:
        raise ValueError(
            f"Krum requires n > 2 * malicious_clients + 2, but got n={n}, malicious_clients={malicious_clients}."
        )
    num_neighbors = n - malicious_clients - 2

    distances = torch.cdist(updates, updates).pow(2)
    # Exclude candidate itself
    distances.fill_diagonal_(float("inf"))

    nearest_distances = torch.topk(
        distances, k=num_neighbors, largest=False, dim=1
    ).values

    return nearest_distances.sum(dim=1)


def krum(updates: torch.Tensor, malicious_clients: int) -> torch.Tensor:
    # scores = []

    # for candidate_index, candidate in enumerate(updates):
    #     distances = []

    #     for other_index, other in enumerate(updates):
    #         if candidate_index == other_index:
    #             continue

    #         distances.append(torch.sum((candidate - other) ** 2))

    #     nearest_distance = torch.stack(distances).sort().values[:num_neighbors]
    #     scores.append(nearest_distance.sum())

    # best_index = torch.stack(scores).argmin()

    # Krum score
    scores = _compute_krum_score(updates, malicious_clients)

    best_index = scores.argmin()

    return updates[best_index]


def _multi_krum_selection(
    updates: torch.Tensor, malicious_clients: int, num_selected: int
) -> torch.Tensor:
    n = updates.shape[0]

    if num_selected <= 0:
        raise ValueError("num_selected must be positive")
    if num_selected > n:
        raise ValueError("num_selected cannot exceed num_clients")

    # Krum score
    scores = _compute_krum_score(updates, malicious_clients)

    selected_indicies = torch.topk(scores, k=num_selected, largest=False).indices

    return updates[selected_indicies]


def multi_krum(updates: torch.Tensor, malicious_clients: int) -> torch.Tensor:
    n = updates.shape[0]
    num_selected = n - malicious_clients - 2

    return _multi_krum_selection(updates, malicious_clients, num_selected).mean(dim=0)


def bulyan(updates: torch.Tensor, malicious_clients: int) -> torch.Tensor:
    n = updates.shape[0]

    if updates.ndim != 2:
        raise ValueError("updates must have shape [num_clients, num_parameters].")

    if malicious_clients < 0:
        raise ValueError("malicious_clients must be non-negative.")

    if n < 4 * malicious_clients + 3:
        raise ValueError(
            "Bulyan requires n >= 4 * malicious_clients + 3, "
            f"but got n={n}, malicious_clients={malicious_clients}."
        )

    # Stage 1: Multi-Krum selection
    num_selected = n - 2 * malicious_clients

    selected = _multi_krum_selection(updates, malicious_clients, num_selected)

    # Stage 2: coordinate-wise filtering
    beta = n - 4 * malicious_clients

    med = coordinate_wise_median(selected)

    coordinate_distances = torch.abs(selected - med)

    closest_indices = torch.topk(
        coordinate_distances, k=beta, largest=False, dim=0
    ).indices

    closest_values = torch.gather(selected, dim=0, index=closest_indices)

    return closest_values.mean(dim=0)


def coordinate_wise_median(updates: torch.Tensor) -> torch.Tensor:
    if updates.ndim != 2:
        raise ValueError("updates must have shape " "[num_clients, num_parameters].")

    if updates.shape[0] == 0:
        raise ValueError("At least one client update is required.")
    return torch.median(updates, dim=0).values


def trimmed_mean(updates: torch.Tensor, malicious_clients: int) -> torch.Tensor:
    n = updates.shape[0]

    if updates.ndim != 2:
        raise ValueError("Updates must have shape [num_clients, num_parameters].")

    if malicious_clients < 0:
        raise ValueError("malicious_clients must be non-negative.")

    if n <= 2 * malicious_clients:
        raise ValueError(
            f"Trimmed mean requires n > 2 * malicious_clients, but got n={n}, malicious_clients={malicious_clients}."
        )

    number_to_consider = n - 2 * malicious_clients
    med = torch.median(updates, dim=0).values
    diff = updates - med
    indices = torch.argsort(diff.abs(), dim=0)
    sorted_diff = torch.gather(diff, 0, indices)
    mean_values = sorted_diff[:number_to_consider].mean(dim=0)

    return mean_values + med


def _pairwise_sq_distances(updates: torch.Tensor):
    # Squared Euclidean distance matrix [n, n]; diagonal set to + inf so a client
    # is never treated as its own neighbour (mirros the old dict that skipped j == i)
    d = torch.cdist(updates, updates)
    d = d**2
    d.fill_diagonal_(float("inf"))
    return d


def _neighbor_stats(updates: torch.Tensor, neighborhood: int):
    # Returns, per client, the (neighborhood nearest + self) index groups and the sorted
    # squared distances (self excluded), shared by the prodigy/gatshca/peel family.
    n = updates.shape[0]
    D = _pairwise_sq_distances(updates)
    sorted_vals, sorted_idx = torch.sort(
        D, dim=1
    )  # last column per row is the +inf self
    self_idx = torch.arange(n, device=updates.device).unsqueeze(1)
    group_idx = torch.cat(
        [sorted_idx[:, :neighborhood], self_idx], dim=1
    )  # [n, neighborhood + 1]
    errors = sorted_vals[:, : n - 1]  # drop the +inf self column
    return group_idx, errors


def _weighted_average(updates: torch.Tensor, score):
    sumscore = score.sum()
    if sumscore.abs() < 1e-8:
        return fed_avg(updates)
    return ((score.unsqueeze(1) / sumscore) * updates).sum(dim=0)


def prodigy(updates: torch.Tensor, malicious_clients: int) -> torch.Tensor:
    if malicious_clients == 0:
        return fed_avg(updates)
    n = updates.shape[0]
    neighborhood = malicious_clients - 1
    group_idx, errors = _neighbor_stats(updates, neighborhood)

    score1 = torch.empty(n, device=updates.device, dtype=updates.dtype)
    for client in range(n):
        grp = updates[group_idx[client]]
        mean_grads = grp.mean(dim=0)
        std_dev_grads = grp.var(dim=0, unbiased=False).sqrt()
        score1[client] = torch.linalg.norm(std_dev_grads) / (
            torch.linalg.norm(mean_grads) + 1e-8
        )

    score2 = errors[:, neighborhood : n - malicious_clients - 1].sum(dim=1)

    score = score1 / (score2 + 1e-8)
    zero_idx = torch.argsort(score)[:malicious_clients]
    score[zero_idx] = 0
    return _weighted_average(updates, score)
