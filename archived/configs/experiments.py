from dataclasses import dataclass
from configs.base import Config


@dataclass
class FedAvgIID(Config):
    output_csv: str = "./results/fedavg_iid.csv"


@dataclass
class FedAvgNonIID(Config):
    partition: str = "non_iid"
    output_csv: str = "./results/fedavg_noniid.csv"


@dataclass
class FedAvgIIDOnes(Config):
    output_csv: str = "./results/fedavg_iid_ones.csv"
    attack_name: str | None = "ones"
    malicious_fractions: float = 0.2


@dataclass
class FedAvgNonIIDOnes(Config):
    output_csv: str = "./results/fedavg_noniid_ones.csv"
    aggregator: str = "fed_avg"
    partition: str = "non_iid"
    attack_name: str | None = "ones"
    malicious_fractions: float = 0.2


# alie
@dataclass
class FedAvgNonIIDALIE(Config):
    output_csv: str = "./results/fedavg_noniid_alie.csv"
    aggregator: str = "fed_avg"
    partition: str = "non_iid"
    attack_name: str | None = "alie"
    malicious_fractions: float = 0.2
    alie_num_std: float = 0.2014
    include_nnm: bool = False


# alie optim
@dataclass
class FedAvgNonIIDALIEOptim(Config):
    output_csv: str = "./results/temp/fedavg_noniid_alie_optim.csv"
    aggregator: str = "fed_avg"
    partition: str = "non_iid"
    attack_name: str | None = "alie_optim"
    malicious_fractions: float = 0.2
    alie_num_std: float = 1.0
    include_nnm: bool = True


# sign filp
@dataclass
class FedAvgNonIIDSignFlip(Config):
    output_csv: str = "./results/fedavg_noniid_sign_flip.csv"
    aggregator: str = "fed_avg"
    partition: str = "non_iid"
    attack_name: str | None = "sign_flip"
    malicious_fractions: float = 0.2
    sign_flip_scale: float = 1.0


# Krum
@dataclass
class KrumIIDOnes(Config):
    output_csv: str = "./results/krum_iid_ones.csv"
    attack_name: str | None = "ones"
    aggregator: str = "krum"
    malicious_fractions: float = 0.2


@dataclass
class KrumNonIID(Config):
    output_csv: str = "./results/krum_noniid.csv"
    aggregator: str = "krum"
    partition: str = "non_iid"
    malicious_fractions: float = 0.0


@dataclass
class KrumNonIIDOnes(Config):
    output_csv: str = "./results/krum_noniid_ones.csv"
    attack_name: str | None = "ones"
    aggregator: str = "krum"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class KrumNonIIDOnesNNM(Config):
    output_csv: str = "./results/krum_noniid_ones_nnm.csv"
    attack_name: str | None = "ones"
    aggregator: str = "krum"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


# Median
@dataclass
class MedianNonIIDOnes(Config):
    output_csv: str = "./results/median_noniid_ones.csv"
    attack_name: str | None = "ones"
    aggregator: str = "coordinate_wise_median"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class MedianNonIIDMinMax(Config):
    output_csv: str = "./results/median_noniid_min_max.csv"
    attack_name: str | None = "min_max"
    aggregator: str = "coordinate_wise_median"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class MedianNonIIDMinSum(Config):
    output_csv: str = "./results/median_noniid_min_sum.csv"
    attack_name: str | None = "min_sum"
    aggregator: str = "coordinate_wise_median"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class MedianNonIIDMinMax_NNM(Config):
    output_csv: str = "./results/median_noniid_min_max_nnm.csv"
    attack_name: str | None = "min_max"
    aggregator: str = "coordinate_wise_median"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


@dataclass
class MedianNonIIDMinSum_NNM(Config):
    output_csv: str = "./results/median_noniid_min_sum_nnm.csv"
    attack_name: str | None = "min_sum"
    aggregator: str = "coordinate_wise_median"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


# Multi-krum
@dataclass
class MultiKrumNonIIDOnes(Config):
    output_csv: str = "./results/multikrum_noniid_ones.csv"
    attack_name: str | None = "ones"
    aggregator: str = "multi_krum"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class MultiKrumNonIIDMinMax(Config):
    output_csv: str = "./results/multikrum_noniid_min_max.csv"
    attack_name: str | None = "min_max"
    aggregator: str = "multi_krum"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class MultiKrumNonIIDMinSum(Config):
    output_csv: str = "./results/multikrum_noniid_min_sum.csv"
    attack_name: str | None = "min_sum"
    aggregator: str = "multi_krum"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class MultiKrumNonIIDMinMaxNNM(Config):
    output_csv: str = "./results/multikrum_noniid_min_max_nnm.csv"
    attack_name: str | None = "min_max"
    aggregator: str = "multi_krum"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


@dataclass
class MultiKrumNonIIDMinSumNNM(Config):
    output_csv: str = "./results/multikrum_noniid_min_sum_nnm.csv"
    attack_name: str | None = "min_sum"
    aggregator: str = "multi_krum"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


# Bulyan
@dataclass
class BulyanNonIIDOnes(Config):
    output_csv: str = "./results/bulyan_noniid_ones.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "ones"
    aggregator: str = "bulyan"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class BulyanNonIIDMinMax(Config):
    output_csv: str = "./results/bulyan_noniid_min_max.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "min_max"
    aggregator: str = "bulyan"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class BulyanNonIIDMinSum(Config):
    output_csv: str = "./results/bulyan_noniid_min_sum.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "min_sum"
    aggregator: str = "bulyan"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class BulyanNonIIDMinMaxNNM(Config):
    output_csv: str = "./results/bulyan_noniid_min_max_nnm.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "min_max"
    aggregator: str = "bulyan"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


@dataclass
class BulyanNonIIDMinSumNNM(Config):
    output_csv: str = "./results/bulyan_noniid_min_sum_nnm.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "min_sum"
    aggregator: str = "bulyan"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


# Trimmed mean
@dataclass
class TrimmedMeanNonIID(Config):
    output_csv: str = "./results/trimmed_mean_noniid.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    malicious_fractions: float = 0.0


@dataclass
class TrimmedMeanNonIIDOnes(Config):
    output_csv: str = "./results/trimmed_mean_noniid_ones.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "ones"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class TrimmedMeanNonIIDMinMax(Config):
    output_csv: str = "./results/trimmed_mean_noniid_min_max.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "min_max"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class TrimmedMeanNonIIDMinSum(Config):
    output_csv: str = "./results/trimmed_mean_noniid_min_sum.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "min_sum"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class TrimmedMeanNonIIDMinMaxNNM(Config):
    output_csv: str = "./results/trimmed_mean_noniid_min_max_nnm.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "min_max"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


@dataclass
class TrimmedMeanNonIIDMinSumNNM(Config):
    output_csv: str = "./results/trimmed_mean_noniid_min_sum_nnm.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "min_sum"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


@dataclass
class TrimmedMeanNonIIDAlieOptimNNM(Config):
    output_csv: str = (
        "./results/cifar10/classic_attack/trimmed_mean_noniid_alie_optim.csv"
    )
    num_clients: int = 30
    clients_per_round: int = 30
    attack_name: str | None = "alie_optim"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True
    communication_rounds: int = 20


@dataclass
class TrimmedMeanNonIIDAlieNNM(Config):
    output_csv: str = "./results/temp/trimmed_mean_noniid_alie.csv"
    num_clients: int = 30
    clients_per_round: int = 30
    attack_name: str | None = "alie"
    aggregator: str = "trimmed_mean"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True
    communication_rounds: int = 100
    alie_num_std: float = 1.5


# Prodigy
@dataclass
class ProdigyNonIID(Config):
    output_csv: str = "./results/prodigy_noniid.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    aggregator: str = "prodigy"
    partition: str = "non_iid"
    malicious_fractions: float = 0.0


@dataclass
class ProdigyNonIIDOnes(Config):
    output_csv: str = "./results/prodigy_noniid_ones.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "ones"
    aggregator: str = "prodigy"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class ProdigyNonIIDMinMax(Config):
    output_csv: str = "./results/prodigy_noniid_min_max.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "min_max"
    aggregator: str = "prodigy"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class ProdigyNonIIDMinSum(Config):
    output_csv: str = "./results/prodigy_noniid_min_sum.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "min_sum"
    aggregator: str = "prodigy"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2


@dataclass
class ProdigyNonIIDMinMaxNNM(Config):
    output_csv: str = "./results/prodigy_noniid_min_max_nnm.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "min_max"
    aggregator: str = "prodigy"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


@dataclass
class ProdigyNonIIDMinSumNNM(Config):
    output_csv: str = "./results/prodigy_noniid_min_sum_nnm.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "min_sum"
    aggregator: str = "prodigy"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


@dataclass
class ProdigyNonIIDALIENNM(Config):
    output_csv: str = "./results/prodigy_noniid_alie_nnm.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "alie"
    aggregator: str = "prodigy"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


@dataclass
class ProdigyNonIIDSingFlip(Config):
    output_csv: str = "./results/prodigy_noniid_sign_flip.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "sign_flip"
    aggregator: str = "prodigy"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = False


@dataclass
class ProdigyNonIIDSingFlipNNM(Config):
    output_csv: str = "./results/prodigy_noniid_sign_flip_nnm.csv"
    num_clients: int = 11
    clients_per_round: int = 11
    attack_name: str | None = "sign_flip"
    aggregator: str = "prodigy"
    partition: str = "non_iid"
    malicious_fractions: float = 0.2
    include_nnm: bool = True


@dataclass
class AnalyzePca(Config):
    num_clients: int = 30
    clients_per_round: int = 30
    malicious_fractions: float = 0.2


CONFIGS = {
    "analyze_pca": AnalyzePca,
    "fedavg_iid": FedAvgIID,
    "fedavg_noniid": FedAvgNonIID,
    "fedavg_iid_ones": FedAvgIIDOnes,
    "fedavg_noniid_ones": FedAvgNonIIDOnes,
    "fedavg_noniid_alie": FedAvgNonIIDALIE,
    "fedavg_noniid_alie_optim": FedAvgNonIIDALIEOptim,
    "fedavg_noniid_sign_flip": FedAvgNonIIDSignFlip,
    "krum_iid_ones": KrumIIDOnes,
    "krum_noniid": KrumNonIID,
    "krum_noniid_ones": KrumNonIIDOnes,
    "krum_noniid_ones_nnm": KrumNonIIDOnesNNM,
    "median_noniid_ones": MedianNonIIDOnes,
    "median_noniid_min_max": MedianNonIIDMinMax,
    "median_noniid_min_sum": MedianNonIIDMinSum,
    "median_noniid_min_max_nnm": MedianNonIIDMinMax_NNM,
    "median_noniid_min_sum_nnm": MedianNonIIDMinSum_NNM,
    "multikrum_noniid_ones": MultiKrumNonIIDOnes,
    "multikrum_noniid_min_max": MultiKrumNonIIDMinMax,
    "multikrum_noniid_min_sum": MultiKrumNonIIDMinSum,
    "multikrum_noniid_min_max_nnm": MultiKrumNonIIDMinMaxNNM,
    "multikrum_noniid_min_sum_nnm": MultiKrumNonIIDMinSumNNM,
    "bulyan_noniid_ones": BulyanNonIIDOnes,
    "bulyan_noniid_min_max": BulyanNonIIDMinMax,
    "bulyan_noniid_min_sum": BulyanNonIIDMinSum,
    "bulyan_noniid_min_max_nnm": BulyanNonIIDMinMaxNNM,
    "bulyan_noniid_min_sum_nnm": BulyanNonIIDMinSumNNM,
    "trimmed_mean_noniid": TrimmedMeanNonIID,
    "trimmed_mean_noniid_ones": TrimmedMeanNonIIDOnes,
    "trimmed_mean_noniid_min_max": TrimmedMeanNonIIDMinMax,
    "trimmed_mean_noniid_min_sum": TrimmedMeanNonIIDMinSum,
    "trimmed_mean_noniid_min_max_nnm": TrimmedMeanNonIIDMinMaxNNM,
    "trimmed_mean_noniid_min_sum_nnm": TrimmedMeanNonIIDMinSumNNM,
    "trimmed_mean_noniid_alie_optim_nnm": TrimmedMeanNonIIDAlieOptimNNM,
    "trimmed_mean_noniid_alie_nnm": TrimmedMeanNonIIDAlieNNM,
    "prodigy_noniid": ProdigyNonIID,
    "prodigy_noniid_ones": ProdigyNonIIDOnes,
    "prodigy_noniid_min_max": ProdigyNonIIDMinMax,
    "prodigy_noniid_min_sum": ProdigyNonIIDMinSum,
    "prodigy_noniid_min_max_nnm": ProdigyNonIIDMinMaxNNM,
    "prodigy_noniid_min_sum_nnm": ProdigyNonIIDMinSumNNM,
    "prodigy_noniid_alie_nnm": ProdigyNonIIDALIENNM,
    "prodigy_noniid_sign_flip": ProdigyNonIIDSingFlip,
    "prodigy_noniid_sign_flip_nnm": ProdigyNonIIDSingFlipNNM,
}
