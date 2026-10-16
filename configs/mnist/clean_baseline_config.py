from dataclasses import dataclass

from configs.mnist.pca_config import PCABasic


@dataclass
class CleanBaselineBasic(PCABasic):
    attack_name: str | None = None
    include_nnm: bool = True


@dataclass
class FedAvg_CleanBaseline(CleanBaselineBasic):
    output_csv: str = "./results/clean_baseline/fedavg_noniid_clean_baseline.csv"
    aggregator: str = "fed_avg"


@dataclass
class Median_CleanBaseline(CleanBaselineBasic):
    output_csv: str = "./results/clean_baseline/median_noniid_clean_baseline.csv"
    aggregator: str = "coordinate_wise_median"


@dataclass
class Krum_CleanBaseline(CleanBaselineBasic):
    output_csv: str = "./results/clean_baseline/krum_noniid_clean_baseline.csv"
    aggregator: str = "krum"


@dataclass
class MultiKrum_CleanBaseline(CleanBaselineBasic):
    output_csv: str = "./results/clean_baseline/multikrum_noniid_clean_baseline.csv"
    aggregator: str = "multi_krum"


@dataclass
class Bulyan_CleanBaseline(CleanBaselineBasic):
    output_csv: str = "./results/clean_baseline/bulyan_noniid_clean_baseline.csv"
    aggregator: str = "bulyan"


@dataclass
class TrimmedMean_CleanBaseline(CleanBaselineBasic):
    output_csv: str = "./results/clean_baseline/trimmed_mean_noniid_clean_baseline.csv"
    aggregator: str = "trimmed_mean"


@dataclass
class Prodigy_CleanBaseline(CleanBaselineBasic):
    output_csv: str = "./results/clean_baseline/prodigy_noniid_clean_baseline.csv"
    aggregator: str = "prodigy"


CONFIGS = {
    "fedavg_clean_baseline": FedAvg_CleanBaseline,
    "median_clean_baseline": Median_CleanBaseline,
    "krum_clean_baseline": Krum_CleanBaseline,
    "multikrum_clean_baseline": MultiKrum_CleanBaseline,
    "bulyan_clean_baseline": Bulyan_CleanBaseline,
    "trimmed_mean_clean_baseline": TrimmedMean_CleanBaseline,
    "prodigy_clean_baseline": Prodigy_CleanBaseline,
}
