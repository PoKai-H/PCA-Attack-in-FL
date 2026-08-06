from abc import ABC, abstractmethod

import torch


class Attack(ABC):
    @abstractmethod
    def generate(
        self, benign_updates: torch.Tensor, num_malicious_clients: int
    ) -> torch.Tensor:
        pass


class OnesAttack(Attack):
    def generate(
        self, benign_updates: torch.Tensor, num_malicious_clients: int
    ) -> torch.Tensor:
        return benign_updates.new_ones((num_malicious_clients, benign_updates.shape[1]))


class MinMax(Attack):
    def generate(
        self, benign_updates: torch.Tensor, num_malicious_clients: int
    ) -> torch.Tensor:
        benign_mean = benign_updates.mean(dim=0)
        perturbation_direction = -torch.sign(benign_mean)
        
        boundary = benign_updates.new_tensor(0.0)
        for candidate_id, candidate_update in enumerate(benign_updates):
            for other_id, other_update in enumerate(benign_updates):
                if candidate_id == other_id:
                    continue
                distance = torch.norm(candidate_update - other_update)
                if distance > boundary:
                    boundary = distance

        # gamma binary search
        high = 10
        low = 0

        while high - low > 1e-5:
            gamma = (high + low) / 2
            malicious = benign_mean + gamma * perturbation_direction

            distance = torch.norm(benign_updates - malicious, dim=1)

            if distance.max() < boundary:
                low = gamma
            else:
                high = gamma

        malicious_update = benign_mean + low * perturbation_direction

        return malicious_update.unsqueeze(0).repeat(num_malicious_clients, 1)


class MinSum(Attack):
    def generate(
        self, benign_updates: torch.Tensor, num_malicious_clients: int
    ) -> torch.Tensor:
        benign_mean = benign_updates.mean(dim=0)
        perturbation_direction = -torch.sign(benign_mean)

        boundary = benign_updates.new_tensor(0.0)
        for candidate_id, candidate_update in enumerate(benign_updates):
            distance_sum = 0
            for other_id, other_update in enumerate(benign_updates):
                if candidate_id == other_id:
                    continue
                distance = torch.norm(candidate_update - other_update)

                distance_sum += distance

            if distance_sum > boundary:
                boundary = distance_sum

        # gamma binary search
        high = 10
        low = 0

        while high - low > 1e-5:
            gamma = (high + low) / 2
            malicious = benign_mean + gamma * perturbation_direction

            distance = torch.norm(benign_updates - malicious, dim=1)

            if distance.sum() <= boundary:
                low = gamma
            else:
                high = gamma

        malicious_update = benign_mean + low * perturbation_direction
        return malicious_update.unsqueeze(0).repeat(num_malicious_clients, 1)
