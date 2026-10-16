from attacks import Attack
import torch
import itertools


class PCASubspaceAttack(Attack):
    """
    PCA-based Byzantine attack.

    The attack constructs a perturbation direction from the principal
    components of benign updates, then searches for the largest scalar
    gamma satisfying the Min-Max distance constraint:

        max_i ||g_m - g_i||_2
            <=
        max_{i,j} ||g_i - g_j||_2

    where

        g_m = g_b - gamma * d

    and d is a normalized PCA-based perturbation direction.
    """

    def __init__(
        self,
        k: int = 1,
        perturbation_mode: str = "pc1",
        tolerance: float = 1e-5,
        initial_high: float = 1.0,
        max_high: float = 1e6,
        sign: int = 0,
    ):
        self.k = k
        self.perturbation_mode = perturbation_mode
        self.tolerance = tolerance
        self.initial_high = initial_high
        self.max_high = max_high
        self.sign = sign

        valid_modes = {
            "pc1",
            "pc2",
            "pc3",
            "std_weighted",
        }

        if perturbation_mode not in valid_modes:
            raise ValueError(
                f"Unknown perturbation mode: {perturbation_mode}. "
                f"Available modes: {sorted(valid_modes)}"
            )

        if sign not in {-1, 0, 1}:
            raise ValueError(f"sign must be -1, 0, or 1, got {sign}")

    def generate(
        self,
        benign_updates: torch.Tensor,
        num_malicious_clients: int,
        aggregator=None,
        clean_aggregator=None,
        global_model=None,
    ) -> torch.Tensor:
        """
        Generate identical malicious updates for all malicious clients.

        Args:
            benign_updates:
                Tensor with shape
                [num_benign_clients, num_parameters].

            num_malicious_clients:
                Number of malicious clients.

            aggregator:
                Aggregation function used to evaluate candidate PCA
                sign patterns.

            global_model:
                Unused.

        Returns:
            Tensor with shape
            [num_malicious_clients, num_parameters].
        """

        self._validate_inputs(
            benign_updates,
            num_malicious_clients,
        )

        benign_mean = benign_updates.mean(dim=0)
        centered = benign_updates - benign_mean

        top_k_dirs, top_k_eigs = self._pca(centered)

        weights = self._build_pc_weights(
            top_k_dirs=top_k_dirs,
            top_k_eigs=top_k_eigs,
        )
        if self.sign == 0:
            direction, gamma, signs, sign_scores = self._select_pc_sign_pattern(
                aggregator=aggregator,
                benign_updates=benign_updates,
                benign_mean=benign_mean,
                pc_dirs=top_k_dirs,
                weights=weights,
                num_malicious_clients=num_malicious_clients,
            )
        else:
            direction, gamma, signs, sign_scores = self._build_fixed_sign_direction(
                aggregator=aggregator,
                benign_updates=benign_updates,
                benign_mean=benign_mean,
                pc_dirs=top_k_dirs,
                weights=weights,
                num_malicious_clients=num_malicious_clients,
                sign=self.sign,
            )

        malicious_update = benign_mean - gamma * direction

        malicious_updates = malicious_update.unsqueeze(0).repeat(
            num_malicious_clients,
            1,
        )
        diagnostics = {
            "top_k_eigs": top_k_eigs.detach().cpu().tolist(),
            "gamma": float(gamma),
            "pca_sign": self.sign,
            "signs": signs.detach().cpu().tolist(),
            "sign_scores": self._serialize_sign_scores(sign_scores),
        }

        return malicious_updates, diagnostics

    def _build_fixed_sign_direction(
        self,
        aggregator,
        benign_updates: torch.Tensor,
        benign_mean: torch.Tensor,
        pc_dirs: torch.Tensor,
        weights: torch.Tensor,
        num_malicious_clients: int,
        sign: int,
    ):
        signs = torch.full_like(weights, float(sign))
        direction = pc_dirs @ (weights * signs)
        direction = self._normalize(direction)

        gamma = self._binary_search_gamma(
            benign_updates=benign_updates,
            benign_mean=benign_mean,
            direction=direction,
        )

        score = self._evaluate_attack_direction(
            aggregator=aggregator,
            benign_updates=benign_updates,
            benign_mean=benign_mean,
            gamma=gamma,
            direction=direction,
            num_malicious_clients=num_malicious_clients,
        )

        sign_scores = {tuple(signs.detach().cpu().tolist()): score}

        return direction, gamma, signs, sign_scores

    def _select_pc_sign_pattern(
        self,
        aggregator,
        benign_updates: torch.Tensor,
        benign_mean: torch.Tensor,
        pc_dirs: torch.Tensor,
        weights: torch.Tensor,
        num_malicious_clients: int,
    ):
        active = weights != 0
        num_active = int(active.sum().item())

        best_score = benign_updates.new_tensor(float("-inf"))
        best_direction = None
        best_gamma = None
        best_signs = None
        sign_scores = {}
        for sign_tuple in itertools.product(
            [-1.0, 1.0],
            repeat=num_active,
        ):
            signs = torch.ones_like(weights)

            signs[active] = weights.new_tensor(sign_tuple)

            # d = sum_i s_i * w_i * PC_i
            direction = pc_dirs @ (weights * signs)

            direction_norm = torch.linalg.norm(direction)

            if direction_norm <= 1e-12:
                continue

            direction = direction / direction_norm

            gamma = self._binary_search_gamma(
                benign_updates=benign_updates,
                benign_mean=benign_mean,
                direction=direction,
            )

            score = self._evaluate_attack_direction(
                aggregator=aggregator,
                benign_updates=benign_updates,
                benign_mean=benign_mean,
                gamma=gamma,
                direction=direction,
                num_malicious_clients=num_malicious_clients,
            )

            sign_scores[sign_tuple] = score

            if score > best_score:
                best_score = score
                best_direction = direction
                best_gamma = gamma
                best_signs = signs.clone()

        return (best_direction, best_gamma, best_signs, sign_scores)

    @staticmethod
    def _serialize_sign_scores(sign_scores: dict) -> dict[str, float]:
        return {
            ",".join(f"{sign:g}" for sign in sign_tuple): float(score.detach().cpu())
            for sign_tuple, score in sign_scores.items()
        }

    def _evaluate_attack_direction(
        self,
        aggregator,
        benign_updates: torch.Tensor,
        benign_mean: torch.Tensor,
        gamma: float,
        direction: torch.Tensor,
        num_malicious_clients: int,
    ):
        malicious_update = benign_mean - gamma * direction
        malicious_updates = malicious_update.unsqueeze(0).repeat(
            num_malicious_clients,
            1,
        )

        attacked_updates = torch.cat(
            [benign_updates, malicious_updates],
            dim=0,
        )

        clean_aggregate = aggregator(benign_updates)

        attacked_aggregate = aggregator(attacked_updates)

        return torch.linalg.norm(attacked_aggregate - clean_aggregate)

    def _pca(
        self,
        centered: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the PCA basis from centered benign updates.

        Returns:
            top_k_dirs:
                Principal directions with shape [D, k].

            top_k_eigs:
                Corresponding covariance eigenvalues with shape [k].
        """

        n, d = centered.shape

        if n < 2:
            raise ValueError("At least two benign updates are required for PCA.")

        # Centered data has rank at most n - 1.
        q = min(
            n - 1,
            d,
        )

        _, singular_values, vh = torch.linalg.svd(
            centered,
            full_matrices=False,
        )

        singular_values = singular_values[:q]

        # Right singular vectors are stored row-wise in Vh.
        directions = vh[:q, :].T

        k = min(
            self.k,
            q,
        )

        if k <= 0:
            raise ValueError("No valid PCA direction is available.")

        top_k_dirs = directions[:, :k]

        # Eigenvalues of the sample covariance matrix:
        #
        # lambda_i = S_i^2 / (n - 1)
        top_k_eigs = singular_values[:k] ** 2 / (n - 1)

        return (
            top_k_dirs,
            top_k_eigs,
        )

    def _build_pc_weights(
        self,
        top_k_dirs: torch.Tensor,
        top_k_eigs: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.perturbation_mode

        weights = torch.zeros(
            top_k_dirs.shape[1],
            device=top_k_dirs.device,
            dtype=top_k_dirs.dtype,
        )

        if mode == "pc1":
            self._require_k(top_k_dirs, 1)
            weights[0] = 1.0

        elif mode == "pc2":
            self._require_k(top_k_dirs, 2)
            weights[1] = 1.0

        elif mode == "pc3":
            self._require_k(top_k_dirs, 3)
            weights[2] = 1.0

        elif mode == "std_weighted":
            weights = top_k_eigs.sqrt()

        else:
            raise RuntimeError(f"Unsupported perturbation mode: {mode}")

        return weights

    def _binary_search_gamma(
        self,
        benign_updates: torch.Tensor,
        benign_mean: torch.Tensor,
        direction: torch.Tensor,
    ) -> float:
        """
        Search for the largest feasible scalar gamma.

        Feasibility condition:

            max_i ||g_m - g_i||_2
                <=
            max_{i,j} ||g_i - g_j||_2

        where

            g_m = g_b - gamma * direction
        """

        boundary = self._max_pairwise_distance(benign_updates)

        high = self._find_infeasible_high(
            benign_updates=benign_updates,
            benign_mean=benign_mean,
            direction=direction,
            boundary=boundary,
        )

        low = 0.0

        while high - low > self.tolerance:
            gamma = (high + low) / 2.0

            malicious_update = benign_mean - gamma * direction

            if self._is_feasible(
                benign_updates=benign_updates,
                malicious_update=malicious_update,
                boundary=boundary,
            ):
                low = gamma
            else:
                high = gamma

        return low

    def _find_infeasible_high(
        self,
        benign_updates: torch.Tensor,
        benign_mean: torch.Tensor,
        direction: torch.Tensor,
        boundary: torch.Tensor,
    ) -> float:
        """
        Dynamically find an upper bound on gamma that violates
        the Min-Max constraint.
        """

        high = self.initial_high

        while high <= self.max_high:
            malicious_update = benign_mean - high * direction

            if not self._is_feasible(
                benign_updates=benign_updates,
                malicious_update=malicious_update,
                boundary=boundary,
            ):
                return high

            high *= 2.0

        raise RuntimeError(
            "Could not find an infeasible gamma before " f"max_high={self.max_high}."
        )

    def _is_feasible(
        self,
        benign_updates: torch.Tensor,
        malicious_update: torch.Tensor,
        boundary: torch.Tensor,
    ) -> bool:
        """
        Check the Min-Max distance constraint.
        """

        distances = torch.linalg.norm(
            benign_updates - malicious_update,
            dim=1,
        )

        max_distance = distances.max()

        return bool((max_distance <= boundary).item())

    def _max_pairwise_distance(
        self,
        updates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute:

            max_{i,j} ||g_i - g_j||_2
        """

        return torch.cdist(
            updates,
            updates,
            p=2,
        ).max()

    @staticmethod
    def _normalize(
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize the perturbation direction to unit L2 norm.
        """

        norm = torch.linalg.norm(direction)

        if norm <= 1e-12:
            raise ValueError("Perturbation direction has zero norm.")

        return direction / norm

    @staticmethod
    def _require_k(
        top_k_dirs: torch.Tensor,
        required: int,
    ):
        """
        Ensure that enough principal components are available.
        """

        available = top_k_dirs.shape[1]

        if available < required:
            raise ValueError(
                f"This perturbation mode requires at least "
                f"{required} principal components, "
                f"but only {available} are available."
            )

    @staticmethod
    def _validate_inputs(
        benign_updates: torch.Tensor,
        num_malicious_clients: int,
    ):
        if benign_updates.ndim != 2:
            raise ValueError(
                "benign_updates must have shape " "[num_clients, num_parameters]."
            )

        if benign_updates.shape[0] < 2:
            raise ValueError("At least two benign updates are required.")

        if num_malicious_clients <= 0:
            raise ValueError("num_malicious_clients must be positive.")
