from typing import Dict

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def load_mnist(data_dir: str) -> tuple[Dataset, Dataset]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
    )

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


def iid_partition(
    dataset: Dataset, num_clients: int, seed: int
) -> Dict[int, list[int]]:
    rng = np.random.default_rng(seed)

    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    client_spilts = np.array_split(indices, num_clients)

    return {client_id: split.tolist() for client_id, split in enumerate(client_spilts)}


def non_iid_partition(
    dataset: Dataset, num_clients: int, shards_per_client: int, seed: int
):
    """

    Label-skew non-IID partition.

    Sort examples by label, split them into shards, then assign a small
    number of shards to each client.

    """

    rng = np.random.default_rng(seed)

    labels = np.array(dataset.targets)
    indicies = np.arange(len(dataset))

    sorted_indeicies = indicies[np.argsort(labels)]
    num_shards = num_clients * shards_per_client
    shards = np.array_split(sorted_indeicies, num_shards)

    shard_ids = np.arange(num_shards)
    rng.shuffle(shard_ids)

    partitions = {client_id: [] for client_id in range(num_clients)}

    for client_id in range(num_clients):
        assinged_shards = shard_ids[
            client_id * shards_per_client : (client_id + 1) * shards_per_client
        ]

        for shard_id in assinged_shards:
            partitions[client_id].extend(shards[shard_id].tolist())

    return partitions


def build_client_loader(
    dataset: Dataset, partitions: Dict[int, list[int]], batch_size: int
) -> Dict[int, DataLoader]:
    client_loaders = {}

    for client_id, indcies in partitions.items():
        client_dataset = Subset(dataset, indcies)

        client_loaders[client_id] = DataLoader(
            client_dataset, batch_size=batch_size, shuffle=True
        )

    return client_loaders


def get_transform(is_train: bool):
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )


def load_cifar10(data_dir: str):
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=get_transform(is_train=True),
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=get_transform(is_train=False),
    )

    return train_dataset, test_dataset


def find_distribution_map(
    dataset,
    partition,
    clients_count,
    alpha=0.5,
    batch_size=256,
):
    n_train = len(dataset)

    if partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, clients_count)
        return {i: batch_idxs[i] for i in range(clients_count)}

    elif partition == "non_iid":
        min_size = 0
        min_require_size = batch_size
        num_classes = 10

        labels = np.array(dataset.targets)

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(clients_count)]

            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(alpha, clients_count))

                proportions = np.array(
                    [
                        p * (len(idx_j) < n_train / clients_count)
                        for p, idx_j in zip(
                            proportions,
                            idx_batch,
                        )
                    ]
                )

                proportions = proportions / proportions.sum()

                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                splits = np.split(
                    idx_k,
                    split_points,
                )

                idx_batch = [
                    idx_j + split.tolist()
                    for idx_j, split in zip(
                        idx_batch,
                        splits,
                    )
                ]

            min_size = min(len(idx_j) for idx_j in idx_batch)

        for indices in idx_batch:
            np.random.shuffle(indices)

        return {i: list(idx_batch[i]) for i in range(clients_count)}

    else:
        raise ValueError(f"Unknown partition: {partition}")
