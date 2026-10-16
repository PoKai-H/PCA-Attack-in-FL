from dataclasses import dataclass
from configs.base import Config


@dataclass
class KrumSignflipNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/krum_signflip_nnm_1.csv"
    attack_name: str | None = "sign_flip"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True


@dataclass
class KrumMinmaxNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/krum_minmax_nnm_1.csv"
    attack_name: str | None = "min_max"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True


@dataclass
class KrumMinsumNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/krum_minsum_nnm_1.csv"
    attack_name: str | None = "min_sum"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True


@dataclass
class KrumAlieoptimNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/krum_alieoptim_nnm_1.csv"
    attack_name: str | None = "alie_optim"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True


@dataclass
class KrumAlieoptim1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/krum_alieoptim_1.csv"
    attack_name: str | None = "alie_optim"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = False


@dataclass
class KrumAlieNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/krum_alie_nnm_1.csv"
    attack_name: str | None = "alie"
    aggregator: str = "krum"
    partition: str = "non_iid"
    include_nnm: bool = True
    z: float = 0.5  # n = 10, f = 3 , s = 5+1-3 = 3, phi(z) = 0.7, z = 0.52


@dataclass
class MultikrumSignflip1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/multikrum_signflip_1.csv"
    attack_name: str | None = "sign_flip"
    aggregator: str = "multi_krum"
    partition: str = "non_iid"
    include_nnm: bool = False


@dataclass
class MultikrumMinmax1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/multikrum_minmax_1.csv"
    attack_name: str | None = "min_max"
    aggregator: str = "multi_krum"
    partition: str = "non_iid"
    include_nnm: bool = False


@dataclass
class MultikrumMinsum1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/multikrum_minsum_1.csv"
    attack_name: str | None = "min_sum"
    aggregator: str = "multi_krum"
    partition: str = "non_iid"
    include_nnm: bool = False


@dataclass
class MultikrumAlieoptim1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/multikrum_alieoptim_1.csv"
    attack_name: str | None = "alie_optim"
    aggregator: str = "multi_krum"
    partition: str = "non_iid"
    include_nnm: bool = False


@dataclass
class MultikrumAlie1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/multikrum_alie_1.csv"
    attack_name: str | None = "alie"
    aggregator: str = "multi_krum"
    partition: str = "non_iid"
    include_nnm: bool = False
    z: float = 0.5


@dataclass
class MedianSignflipNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/median_signflip_nnm_1.csv"
    attack_name: str | None = "sign_flip"
    aggregator: str = "median"
    partition: str = "non_iid"
    include_nnm: bool = True


@dataclass
class MedianMinmaxNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/median_minmax_nnm_1.csv"
    attack_name: str | None = "min_max"
    aggregator: str = "median"
    partition: str = "non_iid"
    include_nnm: bool = True


@dataclass
class MedianMinsumNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/median_minsum_nnm_1.csv"
    attack_name: str | None = "min_sum"
    aggregator: str = "multi_krum"
    partition: str = "non_iid"
    include_nnm: bool = True


@dataclass
class MedianAlieoptimNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/median_alieoptim_nnm_1.csv"
    attack_name: str | None = "alie_optim"
    aggregator: str = "median"
    partition: str = "non_iid"
    include_nnm: bool = True


@dataclass
class MedianAlieNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/median_alie_nnm_1.csv"
    attack_name: str | None = "alie"
    aggregator: str = "median"
    partition: str = "non_iid"
    include_nnm: bool = True
    z: float = 0.5


@dataclass
class TrimmedmeanAlieoptim1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/trimmedmean_alieoptim_1.csv"
    attack_name: str | None = "alie_optim"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    include_nnm: bool = False


@dataclass
class TrimmedmeanAlie1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/treammedmean_alie_1.csv"
    attack_name: str | None = "alie"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    include_nnm: bool = False
    z: float = 0.5


@dataclass
class TrimmedmeanAlieoptimNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/trimmedmean_alieoptim_nnm_1.csv"
    attack_name: str | None = "alie_optim"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    include_nnm: bool = True


@dataclass
class TrimmedmeanAlieNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/trimmedmean_alie_nnm_1.csv"
    attack_name: str | None = "alie"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    include_nnm: bool = True
    z: float = 0.5


@dataclass
class TrimmedmeanSignflipNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/trimmedmean_signflip_nnm_1.csv"
    attack_name: str | None = "sign_flip"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    include_nnm: bool = True


@dataclass
class TrimmedmeanMinSumNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/trimmedmean_minsum_nnm_1.csv"
    attack_name: str | None = "min_sum"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    include_nnm: bool = True


@dataclass
class TrimmedmeanMinMaxNnm1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/trimmedmean_minmax_nnm_1.csv"
    attack_name: str | None = "min_sum"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    include_nnm: bool = True


@dataclass
class FedavgSignflip1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/fedavg_signflip_1.csv"
    attack_name: str | None = "sign_flip"
    aggregator: str = "fed_avg"
    partition: str = "non_iid"
    include_nnm: bool = False


@dataclass
class FedavgMinmax1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/fedavg_minmax_1.csv"
    attack_name: str | None = "min_max"
    aggregator: str = "fed_avg"
    partition: str = "non_iid"
    include_nnm: bool = False


@dataclass
class FedavgMinsum1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/fedavg_minsum_1.csv"
    attack_name: str | None = "min_sum"
    aggregator: str = "fed_avg"
    partition: str = "non_iid"
    include_nnm: bool = False


@dataclass
class FedavgAlieoptim1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/fedavg_alieoptim_1.csv"
    attack_name: str | None = "alie_optim"
    aggregator: str = "fed_avg"
    partition: str = "non_iid"
    include_nnm: bool = False


@dataclass
class FedavgAlie1(Config):
    seed: int = 40
    output_csv: str = "./results/cifar10/classic_attack/fedavg_alie_1.csv"
    attack_name: str | None = "alie"
    aggregator: str = "fed_avg"
    partition: str = "non_iid"
    include_nnm: bool = False
    z: float = 0.5


CONFIGS = {
    "trimmedmean_alieoptim_nnm_1": TrimmedmeanAlieoptimNnm1,
    "trimmedmean_alieoptim_1": TrimmedmeanAlieoptim1,
    "trimmedmean_alie_nnm_1": TrimmedmeanAlieNnm1,
    "trimmedmean_alie_1": TrimmedmeanAlie1,
    "trimmedmean_signflip_nnm_1": TrimmedmeanSignflipNnm1,
    "trimmedmean_minmax_nnm_1": TrimmedmeanMinMaxNnm1,
    "trimmedmean_minsum_nnm_1": TrimmedmeanMinSumNnm1,
    "krum_alieoptim_nnm_1": KrumAlieoptimNnm1,
    "krum_alieoptim_1": KrumAlieoptim1,
    "krum_alie_nnm_1": KrumAlieNnm1,
    "krum_signflip_nnm_1": KrumSignflipNnm1,
    "krum_minmax_nnm_1": KrumMinmaxNnm1,
    "krum_minsum_nnm_1": KrumMinsumNnm1,
    "multikrum_alieoptim_1": MultikrumAlieoptim1,
    "multikrum_alie_1": MultikrumAlie1,
    "multikrum_signflip_1": MultikrumSignflip1,
    "multikrum_minmax_1": MultikrumMinmax1,
    "multikrum_minsum_1": MultikrumMinsum1,
    "fedavg_alieoptim_1": FedavgAlieoptim1,
    "fedavg_alie_1": FedavgAlie1,
    "fedavg_signflip_1": FedavgSignflip1,
    "fedavg_minmax_1": FedavgMinmax1,
    "fedavg_minsum_1": FedavgMinsum1,
    "median_alieoptim_nnm_1": MedianAlieoptimNnm1,
    "median_alie_nnm_1": MedianAlieNnm1,
    "median_signflip_nnm_1": MedianSignflipNnm1,
    "median_minmax_nnm_1": MedianMinmaxNnm1,
    "median_minsum_nnm_1": MedianMinsumNnm1,
}
