from abc import ABC, abstractmethod

import torch


class Attack(ABC):
    @abstractmethod
    def generate(
        self,
        benign_updates: torch.Tensor,
        num_malicious_clients: int,
        aggregator=None,
        global_model=None,
    ) -> torch.Tensor:
        pass


class OnesAttack(Attack):
    def generate(
        self,
        benign_updates: torch.Tensor,
        num_malicious_clients: int,
        aggregator=None,
        global_model=None,
    ) -> torch.Tensor:
        return benign_updates.new_ones((num_malicious_clients, benign_updates.shape[1]))


class MinMax(Attack):
    def generate(
        self,
        benign_updates: torch.Tensor,
        num_malicious_clients: int,
        aggregator=None,
        global_model=None,
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
        self,
        benign_updates: torch.Tensor,
        num_malicious_clients: int,
        aggregator=None,
        global_model=None,
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


class ALIE(Attack):
    def __init__(self, num_std: float):
        self.num_std = num_std
        self.grads_mean = None
        self.grads_stdev = None

    def generate(
        self,
        benign_updates: torch.Tensor,
        num_malicious_clients: int,
        aggregator=None,
        global_model=None,
    ) -> torch.Tensor:
        self.grads_mean = benign_updates.mean(dim=0)
        self.grads_stdev = benign_updates.var(dim=0, unbiased=False).sqrt()

        malicious_update = self.grads_mean - self.num_std * self.grads_stdev
        return malicious_update.unsqueeze(0).repeat(num_malicious_clients, 1)


class ALIEOptim(Attack):
    def __init__(self, num_std: float):
        self.num_std = num_std
        self.grads_mean = None
        self.grads_stdev = None

    def generate(
        self,
        benign_updates: torch.Tensor,
        num_malicious_clients: int,
        aggregator=None,
        global_model=None,
    ) -> torch.Tensor:
        if aggregator is None:
            raise ValueError("ALIEOptim requires an aggregator.")

        self.grads_mean = benign_updates.mean(dim=0)
        self.grads_stdev = benign_updates.var(dim=0, unbiased=False).sqrt()

        if self.num_std == 0:
            malicious_update = self.grads_mean
            return malicious_update.unsqueeze(0).repeat(num_malicious_clients, 1)

        step = 0.25 * self.num_std
        z_values = torch.arange(
            step,
            2.0 + 1e-12,
            step,
            device=benign_updates.device,
            dtype=benign_updates.dtype,
        )

        max_dist = benign_updates.new_tensor(-1.0)
        best_z = benign_updates.new_tensor(self.num_std)

        for z in z_values:
            malicious_update = self.grads_mean - z * self.grads_stdev
            submitted_updates = torch.cat(
                [
                    benign_updates,
                    malicious_update.unsqueeze(0).repeat(num_malicious_clients, 1),
                ],
                dim=0,
            )
            aggregated_update = aggregator(submitted_updates)
            dist = torch.linalg.norm(aggregated_update - self.grads_mean)

            if dist > max_dist:
                max_dist = dist
                best_z = z

        malicious_update = self.grads_mean - best_z * self.grads_stdev
        return malicious_update.unsqueeze(0).repeat(num_malicious_clients, 1)


class SignFlip(Attack):
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def generate(
        self,
        benign_updates: torch.Tensor,
        num_malicious_clients: int,
        aggregator=None,
        global_model=None,
    ) -> torch.Tensor:
        malicious_update = -self.scale * benign_updates.mean(dim=0)
        return malicious_update.unsqueeze(0).repeat(num_malicious_clients, 1)
