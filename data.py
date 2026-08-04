from typing import Dict

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

def load_mnist(data_dir: str) -> tuple[Dataset, Dataset]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ]
    )

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def iid_partition(dataset: Dataset, num_clients: int, seed: int) -> Dict[int, list[int]]:
    rng = np.random.default_rng(seed)

    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    client_spilts = np.array_split(indices, num_clients)

    return {client_id: split.tolist() for client_id, split in enumerate(client_spilts)}

def non_iid_partition(dataset: Dataset, num_clients: int, shards_per_client: int, seed: int):
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

def build_client_loader(dataset: Dataset, partitions: Dict[int, list[int]], batch_size: int) -> Dict[int, DataLoader]:
    client_loaders = {}

    for client_id, indcies in partitions.items():
        client_dataset = Subset(dataset, indcies)

        client_loaders[client_id] = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)

    return client_loaders
