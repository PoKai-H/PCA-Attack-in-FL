from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    data_dir: str = "./data"
    output_csv: str = "./results/basic_mnist.csv"

    num_clients: int = 10
    clients_per_round: int = 10
    malicious_fractions: float = 0.3

    communication_rounds: int = 50
    local_epochs: int = 1  # set to 1 to simplified the system
    batch_size: int = 256

    learning_rate: float = 0.01
    momentum: float = 0.9

    centralized_epochs: int = 3

    use_dropout: bool = False

    attack_name: str | None = None
    attack_optimizer_name: str | None = None
    # param for alie
    z: float = 0.41
    # params for alie optim
    z_max: float = 2.0
    z_step: float = 0.125

    sign_flip_scale: float = 1.0
    pca_top_k: int = 1
    pca_perturbation_mode: str = "pc1"
    pca_sign: int = 1
    pca_noise_dims: int = 1
    pca_noise_distribution: str = "low_eigenvalue"
    pca_noise_scale: float = 1.5
    include_noise: bool = False

    aggregator: str = "fed_avg"

    partition: str = "non_iid"
    shards_per_client: int = 2

    include_nnm: bool = False
