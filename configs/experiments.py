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




CONFIGS = {
    "fedavg_iid": FedAvgIID,
    "fedavg_noniid": FedAvgNonIID,
    "fedavg_iid_ones": FedAvgIIDOnes,
    "fedavg_noniid_ones": FedAvgNonIIDOnes,
    "krum_iid_ones": KrumIIDOnes,
    "krum_noniid": KrumNonIID,
    "krum_noniid_ones": KrumNonIIDOnes,
    "krum_noniid_ones_nnm": KrumNonIIDOnesNNM,
    "median_noniid_ones": MedianNonIIDOnes,
    "median_noniid_min_max": MedianNonIIDMinMax,
    "median_noniid_min_sum": MedianNonIIDMinSum,

}
