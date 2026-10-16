from configs.base import Config
from dataclasses import dataclass


@dataclass
class PCABasic(Config):
    num_clients: int = 30
    clients_per_round: int = 30
    malicious_fractions: float = 0.2
    partition: str = "non_iid"
    attack_name: str | None = "pca_subspace"
    pca_top_k: int = 3
    pca_perturbation_mode: str = "pc1"
    include_nnm: bool = False


@dataclass
class FedAvg_PCA_pc1(PCABasic):
    output_csv: str = "./results/pca/fedavg_noniid_pca_pc1.csv"
    aggregator: str = "fed_avg"
    pca_perturbation_mode: str = "pc1"


@dataclass
class FedAvg_PCA_pc2(PCABasic):
    output_csv: str = "./results/pca/fedavg_noniid_pca_pc2.csv"
    aggregator: str = "fed_avg"
    pca_perturbation_mode: str = "pc2"


@dataclass
class FedAvg_PCA_pc3(PCABasic):
    output_csv: str = "./results/pca/fedavg_noniid_pca_pc3.csv"
    aggregator: str = "fed_avg"
    pca_perturbation_mode: str = "pc3"


@dataclass
class FedAvg_PCA_equal_pc12(PCABasic):
    output_csv: str = "./results/pca/fedavg_noniid_pca_equal_pc12.csv"
    aggregator: str = "fed_avg"
    pca_perturbation_mode: str = "equal_pc12"


@dataclass
class FedAvg_PCA_equal_pc123(PCABasic):
    output_csv: str = "./results/pca/fedavg_noniid_pca_equal_pc123.csv"
    aggregator: str = "fed_avg"
    pca_perturbation_mode: str = "equal_pc123"


@dataclass
class FedAvg_PCA_eigen_weighted(PCABasic):
    output_csv: str = "./results/pca/fedavg_noniid_pca_eigen_weighted.csv"
    aggregator: str = "fed_avg"
    pca_perturbation_mode: str = "eigen_weighted"


@dataclass
class Median_PCA_pc1(PCABasic):
    output_csv: str = "./results/pca/median_noniid_pca_pc1.csv"
    aggregator: str = "coordinate_wise_median"
    pca_perturbation_mode: str = "pc1"


@dataclass
class Median_PCA_pc2(PCABasic):
    output_csv: str = "./results/pca/median_noniid_pca_pc2.csv"
    aggregator: str = "coordinate_wise_median"
    pca_perturbation_mode: str = "pc2"


@dataclass
class Median_PCA_pc3(PCABasic):
    output_csv: str = "./results/pca/median_noniid_pca_pc3.csv"
    aggregator: str = "coordinate_wise_median"
    pca_perturbation_mode: str = "pc3"


@dataclass
class Median_PCA_equal_pc12(PCABasic):
    output_csv: str = "./results/pca/median_noniid_pca_equal_pc12.csv"
    aggregator: str = "coordinate_wise_median"
    pca_perturbation_mode: str = "equal_pc12"


@dataclass
class Median_PCA_equal_pc123(PCABasic):
    output_csv: str = "./results/pca/median_noniid_pca_equal_pc123.csv"
    aggregator: str = "coordinate_wise_median"
    pca_perturbation_mode: str = "equal_pc123"


@dataclass
class Median_PCA_eigen_weighted(PCABasic):
    output_csv: str = "./results/pca/median_noniid_pca_eigen_weighted.csv"
    aggregator: str = "coordinate_wise_median"
    pca_perturbation_mode: str = "eigen_weighted"


@dataclass
class Krum_PCA_pc1(PCABasic):
    output_csv: str = "./results/pca/krum_noniid_pca_pc1.csv"
    aggregator: str = "krum"
    pca_perturbation_mode: str = "pc1"


@dataclass
class Krum_PCA_pc2(PCABasic):
    output_csv: str = "./results/pca/krum_noniid_pca_pc2.csv"
    aggregator: str = "krum"
    pca_perturbation_mode: str = "pc2"


@dataclass
class Krum_PCA_pc3(PCABasic):
    output_csv: str = "./results/pca/krum_noniid_pca_pc3.csv"
    aggregator: str = "krum"
    pca_perturbation_mode: str = "pc3"


@dataclass
class Krum_PCA_equal_pc12(PCABasic):
    output_csv: str = "./results/pca/krum_noniid_pca_equal_pc12.csv"
    aggregator: str = "krum"
    pca_perturbation_mode: str = "equal_pc12"


@dataclass
class Krum_PCA_equal_pc123(PCABasic):
    output_csv: str = "./results/pca/krum_noniid_pca_equal_pc123.csv"
    aggregator: str = "krum"
    pca_perturbation_mode: str = "equal_pc123"


@dataclass
class Krum_PCA_eigen_weighted(PCABasic):
    output_csv: str = "./results/pca/krum_noniid_pca_eigen_weighted.csv"
    aggregator: str = "krum"
    pca_perturbation_mode: str = "eigen_weighted"


@dataclass
class MultiKrum_PCA_pc1(PCABasic):
    output_csv: str = "./results/pca/multikrum_noniid_pca_pc1.csv"
    aggregator: str = "multi_krum"
    pca_perturbation_mode: str = "pc1"


@dataclass
class MultiKrum_PCA_pc2(PCABasic):
    output_csv: str = "./results/pca/multikrum_noniid_pca_pc2.csv"
    aggregator: str = "multi_krum"
    pca_perturbation_mode: str = "pc2"


@dataclass
class MultiKrum_PCA_pc3(PCABasic):
    output_csv: str = "./results/pca/multikrum_noniid_pca_pc3.csv"
    aggregator: str = "multi_krum"
    pca_perturbation_mode: str = "pc3"


@dataclass
class MultiKrum_PCA_equal_pc12(PCABasic):
    output_csv: str = "./results/pca/multikrum_noniid_pca_equal_pc12.csv"
    aggregator: str = "multi_krum"
    pca_perturbation_mode: str = "equal_pc12"


@dataclass
class MultiKrum_PCA_equal_pc123(PCABasic):
    output_csv: str = "./results/pca/multikrum_noniid_pca_equal_pc123.csv"
    aggregator: str = "multi_krum"
    pca_perturbation_mode: str = "equal_pc123"


@dataclass
class MultiKrum_PCA_eigen_weighted(PCABasic):
    output_csv: str = "./results/pca/multikrum_noniid_pca_eigen_weighted.csv"
    aggregator: str = "multi_krum"
    pca_perturbation_mode: str = "eigen_weighted"


@dataclass
class Bulyan_PCA_pc1(PCABasic):
    output_csv: str = "./results/pca/bulyan_noniid_pca_pc1.csv"
    aggregator: str = "bulyan"
    pca_perturbation_mode: str = "pc1"


@dataclass
class Bulyan_PCA_pc2(PCABasic):
    output_csv: str = "./results/pca/bulyan_noniid_pca_pc2.csv"
    aggregator: str = "bulyan"
    pca_perturbation_mode: str = "pc2"


@dataclass
class Bulyan_PCA_pc3(PCABasic):
    output_csv: str = "./results/pca/bulyan_noniid_pca_pc3.csv"
    aggregator: str = "bulyan"
    pca_perturbation_mode: str = "pc3"


@dataclass
class Bulyan_PCA_equal_pc12(PCABasic):
    output_csv: str = "./results/pca/bulyan_noniid_pca_equal_pc12.csv"
    aggregator: str = "bulyan"
    pca_perturbation_mode: str = "equal_pc12"


@dataclass
class Bulyan_PCA_equal_pc123(PCABasic):
    output_csv: str = "./results/pca/bulyan_noniid_pca_equal_pc123.csv"
    aggregator: str = "bulyan"
    pca_perturbation_mode: str = "equal_pc123"


@dataclass
class Bulyan_PCA_eigen_weighted(PCABasic):
    output_csv: str = "./results/pca/bulyan_noniid_pca_eigen_weighted.csv"
    aggregator: str = "bulyan"
    pca_perturbation_mode: str = "eigen_weighted"


@dataclass
class TrimmedMean_PCA_pc1(PCABasic):
    output_csv: str = "./results/pca/trimmed_mean_noniid_pca_pc1.csv"
    aggregator: str = "trimmed_mean"
    pca_perturbation_mode: str = "pc1"


@dataclass
class TrimmedMean_PCA_pc2(PCABasic):
    output_csv: str = "./results/pca/trimmed_mean_noniid_pca_pc2.csv"
    aggregator: str = "trimmed_mean"
    pca_perturbation_mode: str = "pc2"


@dataclass
class TrimmedMean_PCA_pc3(PCABasic):
    output_csv: str = "./results/pca/trimmed_mean_noniid_pca_pc3.csv"
    aggregator: str = "trimmed_mean"
    pca_perturbation_mode: str = "pc3"


@dataclass
class TrimmedMean_PCA_equal_pc12(PCABasic):
    output_csv: str = "./results/pca/trimmed_mean_noniid_pca_equal_pc12.csv"
    aggregator: str = "trimmed_mean"
    pca_perturbation_mode: str = "equal_pc12"


@dataclass
class TrimmedMean_PCA_equal_pc123(PCABasic):
    output_csv: str = "./results/pca/trimmed_mean_noniid_pca_equal_pc123.csv"
    aggregator: str = "trimmed_mean"
    pca_perturbation_mode: str = "equal_pc123"


@dataclass
class TrimmedMean_PCA_eigen_weighted(PCABasic):
    output_csv: str = "./results/pca/trimmed_mean_noniid_pca_eigen_weighted.csv"
    aggregator: str = "trimmed_mean"
    pca_perturbation_mode: str = "eigen_weighted"


@dataclass
class Prodigy_PCA_pc1(PCABasic):
    output_csv: str = "./results/pca/prodigy_noniid_pca_pc1.csv"
    aggregator: str = "prodigy"
    pca_perturbation_mode: str = "pc1"


@dataclass
class Prodigy_PCA_pc2(PCABasic):
    output_csv: str = "./results/pca/prodigy_noniid_pca_pc2.csv"
    aggregator: str = "prodigy"
    pca_perturbation_mode: str = "pc2"


@dataclass
class Prodigy_PCA_pc3(PCABasic):
    output_csv: str = "./results/pca/prodigy_noniid_pca_pc3.csv"
    aggregator: str = "prodigy"
    pca_perturbation_mode: str = "pc3"


@dataclass
class Prodigy_PCA_equal_pc12(PCABasic):
    output_csv: str = "./results/pca/prodigy_noniid_pca_equal_pc12.csv"
    aggregator: str = "prodigy"
    pca_perturbation_mode: str = "equal_pc12"


@dataclass
class Prodigy_PCA_equal_pc123(PCABasic):
    output_csv: str = "./results/pca/prodigy_noniid_pca_equal_pc123.csv"
    aggregator: str = "prodigy"
    pca_perturbation_mode: str = "equal_pc123"


@dataclass
class Prodigy_PCA_eigen_weighted(PCABasic):
    output_csv: str = "./results/pca/prodigy_noniid_pca_eigen_weighted.csv"
    aggregator: str = "prodigy"
    pca_perturbation_mode: str = "eigen_weighted"


CONFIGS = {
    "pca_basic": PCABasic,
    "fedavg_pca_pc1": FedAvg_PCA_pc1,
    "fedavg_pca_pc2": FedAvg_PCA_pc2,
    "fedavg_pca_pc3": FedAvg_PCA_pc3,
    "fedavg_pca_equal_pc12": FedAvg_PCA_equal_pc12,
    "fedavg_pca_equal_pc123": FedAvg_PCA_equal_pc123,
    "fedavg_pca_eigen_weighted": FedAvg_PCA_eigen_weighted,
    "median_pca_pc1": Median_PCA_pc1,
    "median_pca_pc2": Median_PCA_pc2,
    "median_pca_pc3": Median_PCA_pc3,
    "median_pca_equal_pc12": Median_PCA_equal_pc12,
    "median_pca_equal_pc123": Median_PCA_equal_pc123,
    "median_pca_eigen_weighted": Median_PCA_eigen_weighted,
    "krum_pca_pc1": Krum_PCA_pc1,
    "krum_pca_pc2": Krum_PCA_pc2,
    "krum_pca_pc3": Krum_PCA_pc3,
    "krum_pca_equal_pc12": Krum_PCA_equal_pc12,
    "krum_pca_equal_pc123": Krum_PCA_equal_pc123,
    "krum_pca_eigen_weighted": Krum_PCA_eigen_weighted,
    "multikrum_pca_pc1": MultiKrum_PCA_pc1,
    "multikrum_pca_pc2": MultiKrum_PCA_pc2,
    "multikrum_pca_pc3": MultiKrum_PCA_pc3,
    "multikrum_pca_equal_pc12": MultiKrum_PCA_equal_pc12,
    "multikrum_pca_equal_pc123": MultiKrum_PCA_equal_pc123,
    "multikrum_pca_eigen_weighted": MultiKrum_PCA_eigen_weighted,
    "bulyan_pca_pc1": Bulyan_PCA_pc1,
    "bulyan_pca_pc2": Bulyan_PCA_pc2,
    "bulyan_pca_pc3": Bulyan_PCA_pc3,
    "bulyan_pca_equal_pc12": Bulyan_PCA_equal_pc12,
    "bulyan_pca_equal_pc123": Bulyan_PCA_equal_pc123,
    "bulyan_pca_eigen_weighted": Bulyan_PCA_eigen_weighted,
    "trimmed_mean_pca_pc1": TrimmedMean_PCA_pc1,
    "trimmed_mean_pca_pc2": TrimmedMean_PCA_pc2,
    "trimmed_mean_pca_pc3": TrimmedMean_PCA_pc3,
    "trimmed_mean_pca_equal_pc12": TrimmedMean_PCA_equal_pc12,
    "trimmed_mean_pca_equal_pc123": TrimmedMean_PCA_equal_pc123,
    "trimmed_mean_pca_eigen_weighted": TrimmedMean_PCA_eigen_weighted,
    "prodigy_pca_pc1": Prodigy_PCA_pc1,
    "prodigy_pca_pc2": Prodigy_PCA_pc2,
    "prodigy_pca_pc3": Prodigy_PCA_pc3,
    "prodigy_pca_equal_pc12": Prodigy_PCA_equal_pc12,
    "prodigy_pca_equal_pc123": Prodigy_PCA_equal_pc123,
    "prodigy_pca_eigen_weighted": Prodigy_PCA_eigen_weighted,
}
