from dataclasses import dataclass
from configs.base import Config


@dataclass
class PCABasic(Config):
    partition: str = "non_iid"
    attack_name: str | None = "pca_subspace"
    pca_top_k: int = 3
    pca_perturbation_mode: str = "pc1"
    include_nnm: bool = True


@dataclass
class KrumPC1P1Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_pc1_p1_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "pc1"
    pca_sign: int = 1
    pca_top_k: int = 1


@dataclass
class KrumPC1M1Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_pc1_m1_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "pc1"
    pca_sign: int = -1
    pca_top_k: int = 1


@dataclass
class KrumPC1P0Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_pc1_p0_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "pc1"
    pca_sign: int = 0
    pca_top_k: int = 1


class KrumPC1P01(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_pc1_p0_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = False
    pca_perturbation_mode: str = "pc1"
    pca_sign: int = 0
    pca_top_k: int = 1


@dataclass
class TrimmedmeanPC1P0Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/trimmedmean_pc1_p0_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "pc1"
    pca_sign: int = 0
    pca_top_k: int = 1


@dataclass
class KrumPC2P1Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_pc2_p1_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "pc2"
    pca_sign: int = 1
    pca_top_k: int = 1


@dataclass
class KrumPC2M1Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_pc2_m1_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "pc1"
    pca_sign: int = -1
    pca_top_k: int = 2


@dataclass
class KrumPC2P0Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_pc2_p0_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "pc1"
    pca_sign: int = 0
    pca_top_k: int = 2


@dataclass
class KrumPC2P1Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_pc2_p1_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "pc2"
    pca_sign: int = 1
    pca_top_k: int = 1


@dataclass
class KrumPC2M1Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_pc2_m1_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "pc1"
    pca_sign: int = -1
    pca_top_k: int = 2


@dataclass
class KrumPC2P0Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_pc2_p0_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "pc1"
    pca_sign: int = 0
    pca_top_k: int = 2


@dataclass
class KrumPC2P0Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_pc2_p0_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "pc1"
    pca_sign: int = 0
    pca_top_k: int = 2


@dataclass
class KrumStdK3P0Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_stdk3_p0_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "std_weighted"
    pca_sign: int = 0
    pca_top_k: int = 3


@dataclass
class KrumStdK3P01(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/krum_stdk3_p0_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "std_weighted"
    pca_sign: int = 0
    pca_top_k: int = 3


@dataclass
class TrimmedmeanStdK3P0Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/trimmedmean_stdk3_p0_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "std_weighted"
    pca_sign: int = 0
    pca_top_k: int = 3


@dataclass
class MedianStdK3P0Nnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/median_stdk3_p0_nnm_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "median"
    partition: str = "non_iid"
    include_nnm: bool = True
    pca_perturbation_mode: str = "std_weighted"
    pca_sign: int = 0
    pca_top_k: int = 3


@dataclass
class TrimmedmeanStdK3P01(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/trimmedmean_stdk3_p0_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    include_nnm: bool = False
    pca_perturbation_mode: str = "std_weighted"
    pca_sign: int = 0
    pca_top_k: int = 3


@dataclass
class MultikrumStdK3P01(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/multikrum_stdk3_p0_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "multi_krum"
    partition: str = "non_iid"
    include_nnm: bool = False
    pca_perturbation_mode: str = "std_weighted"
    pca_sign: int = 0
    pca_top_k: int = 3


@dataclass
class FedavgStdK3P01(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/pca_attack/fedavg_stdk3_p0_1.csv"
    attack_name: str | None = "pca_subspace"
    aggregator: str = "fed_avg"
    partition: str = "non_iid"
    include_nnm: bool = False
    pca_perturbation_mode: str = "std_weighted"
    pca_sign: int = 0
    pca_top_k: int = 3


CONFIGS = {
    "krum_pc1_p1_nnm_1": KrumPC1P1Nnm1,
    "krum_pc1_m1_nnm_1": KrumPC1M1Nnm1,
    "krum_pc1_p0_nnm_1": KrumPC1P0Nnm1,
    "krum_pc1_p0_1": KrumPC1P01,
    "krum_stdk3_p0_nnm_1": KrumStdK3P0Nnm1,
    "krum_stdk3_p0_1": KrumPC1P01,
    "trimmedmean_pc1_p0_nnm_1": TrimmedmeanPC1P0Nnm1,
    "trimmedmean_stdk3_p0_nnm_1": TrimmedmeanStdK3P0Nnm1,
    "trimmedmean_stdk3_p0_1": TrimmedmeanStdK3P01,
    "median_stdk3_p0_nnm_1": MedianStdK3P0Nnm1,
    "multikrum_stdk3_p0_1": MultikrumStdK3P01,
    "fedavg_stdk3_p0_1": FedavgStdK3P01,
}
