from attacks import Attack
from typing import Callable
import torch
from torch import nn


class PCASubspaceAttack(Attack):
    """
    Knows benign updates
    Knows aggregation method
    """

    def __init__(
        self,
        k: int = 3,
        m: int = 3,
        noise_distribution: str = "low_eigenvalue",
        noise_scale: float = 1e-3,
        gamma_search: str = "binary",
    ):
        self.k = k
        self.m = m
        self.noise_scale = noise_scale

        registry = {
            "low_eigenvalue": self._sample_low_eigenvalue_noise,
            "gaussian": self._sample_gaussian_noise,
        }

        try:
            self.sample_noise = registry[noise_distribution]
        except KeyError:
            raise ValueError(
                f"Unknown PCA noise distribution: {noise_distribution}"
            ) from None

        search_registry = {
            "binary": self._binary_search_gamma,
        }

        try:
            self.search_gamma = search_registry[gamma_search]
        except KeyError:
            raise ValueError(f"Unknown gamma search strategy: {gamma_search}") from None

    def generate(
        self,
        benign_updates: torch.Tensor,
        num_malicious_clients: int,
        aggregator,
        global_model,
    ):
        benign_mean = benign_updates.mean(dim=0)
        centered = benign_updates - benign_mean

        top_k_dirs, noise_dirs, noise_eigs = self._pca(centered)
        gamma = self.search_gamma(benign_updates, top_k_dirs)

        return torch.stack(
            [
                self._build_malicious_update(
                    benign_mean, top_k_dirs, gamma, noise_dirs, noise_eigs
                )
                for _ in range(num_malicious_clients)
            ]
        )

    def _pca(self, centered: torch.Tensor):
        n = centered.shape[0]
        if n < 2:
            raise ValueError("At least two benign updates are required for PCA.")
        q = min(n - 1, centered.shape[1])
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)

        # keep only the theoretically meaningful centerd-data subspace,
        # X was 8 x D , the 8th singular value S8 should be \approx 0
        S = S[:q]
        V = Vh[:q, :].T

        k = min(self.k, q)
        m = min(self.m, max(q - k, 0))
        top_k_dirs = V[:, :k]

        if m > 0:
            noise_dirs = V[:, -m:]
            noise_eigs = S[-m:] ** 2 / max(n - 1, 1)
        else:
            noise_dirs = V[:, :0]
            noise_eigs = S[:0]

        return top_k_dirs, noise_dirs, noise_eigs

    def _binary_search_gamma(
        self,
        benign_updates: torch.Tensor,
        p_dir: torch.Tensor,
        num_candidates: int = 20,
        high: float = 10.0,
        low: float = 0.0,
    ) -> torch.Tensor:
        boundary = self._max_pairwise_distance(benign_updates)
        benign_mean = benign_updates.mean(dim=0)

        k = p_dir.shape[1]
        candidates = torch.randn(num_candidates, k, device=p_dir.device)
        candidates = candidates / (candidates.norm(dim=1, keepdim=True) + 1e-8)

        best_gamma = torch.zeros(k, device=p_dir.device)
        best_t = 0.0

        for z in candidates:
            direction = (
                p_dir @ z
            )  # p_dir 是正交欄，z 是單位向量，direction 自動也是單位向量
            high = self._find_infeasible_high(
                benign_updates, benign_mean, direction, boundary
            )
            lo, hi = low, high
            while hi - lo > 1e-5:
                t = (hi + lo) / 2
                malicious = (
                    benign_mean - t * direction
                )  # 跟 _build_malicious_update 的 benign_mean - step 對齊
                distance = torch.norm(benign_updates - malicious, dim=1)
                if distance.max() < boundary:
                    lo = t
                else:
                    hi = t
            if lo > best_t:  # 挑能推得最遠、又還在邊界內的那個候選方向
                best_t = lo
                best_gamma = lo * z

        return best_gamma

    def _max_pairwise_distance(self, updates: torch.Tensor) -> torch.Tensor:
        return torch.cdist(updates, updates).max()

    def _sample_low_eigenvalue_noise(self, noise_dirs, noise_eigs, reference):
        if noise_dirs.shape[1] == 0:
            return torch.zeros_like(reference)

        std = noise_eigs.clamp_min(0).sqrt()
        coeffs = torch.randn(noise_dirs.shape[1], device=noise_dirs.device) * std
        return self.noise_scale * (noise_dirs @ coeffs)

    def _sample_gaussian_noise(self, noise_dirs, noise_eigs, reference):
        return self.noise_scale * torch.randn_like(reference)

    def _build_malicious_update(
        self, benign_mean, directions, gamma, noise_dirs, noise_eigs
    ):
        step = directions @ gamma
        noise = self.sample_noise(noise_dirs, noise_eigs, benign_mean)
        return benign_mean - step + noise

    def _find_infeasible_high(
        self,
        benign_updates,
        benign_mean,
        direction,
        boundary,
        start=1.0,
        max_high=1e6,
    ):
        t = start
        while (
            torch.norm(benign_updates - (benign_mean - t * direction), dim=1).max()
            < boundary
        ):
            t *= 2
            if t > max_high:
                break
        return t
