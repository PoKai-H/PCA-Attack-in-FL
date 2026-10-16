from dataclasses import dataclass

from configs.mnist.pca_config import PCABasic


@dataclass
class PCASignBasic(PCABasic):
    pca_top_k: int = 1
    pca_perturbation_mode: str = "pc1"
    pca_sign: int = 0


AGGREGATORS = {
    "fedavg": ("FedAvg", "fed_avg"),
    "median": ("Median", "coordinate_wise_median"),
    "krum": ("Krum", "krum"),
    "multikrum": ("MultiKrum", "multi_krum"),
    "bulyan": ("Bulyan", "bulyan"),
    "trimmed_mean": ("TrimmedMean", "trimmed_mean"),
    "prodigy": ("Prodigy", "prodigy"),
}

PCA_MODES = {
    "pc1": ("pc1", 1, "PC1"),
    "pc2": ("pc2", 2, "PC2"),
    "pc3": ("pc3", 3, "PC3"),
    "std_weighted_k2": ("std_weighted", 2, "StdWeightedK2"),
    "std_weighted_k3": ("std_weighted", 3, "StdWeightedK3"),
}

PCA_SIGNS = {
    "sign_m1": (-1, "SignM1"),
    "sign_0": (0, "Sign0"),
    "sign_p1": (1, "SignP1"),
}


def _make_config_class(
    config_name: str,
    class_name: str,
    aggregator: str,
    pca_perturbation_mode: str,
    pca_top_k: int,
    pca_sign: int,
):
    namespace = {
        "__annotations__": {
            "output_csv": str,
            "aggregator": str,
            "pca_perturbation_mode": str,
            "pca_top_k": int,
            "pca_sign": int,
        },
        "output_csv": f"./results/pca_sign/{config_name}.csv",
        "aggregator": aggregator,
        "pca_perturbation_mode": pca_perturbation_mode,
        "pca_top_k": pca_top_k,
        "pca_sign": pca_sign,
    }
    return dataclass(type(class_name, (PCASignBasic,), namespace))


CONFIGS = {}

for aggregator_slug, (aggregator_class, aggregator_name) in AGGREGATORS.items():
    for mode_slug, (pca_mode, pca_top_k, mode_class) in PCA_MODES.items():
        for sign_slug, (pca_sign, sign_class) in PCA_SIGNS.items():
            config_name = f"{aggregator_slug}_noniid_pca_sign_{mode_slug}_{sign_slug}"
            class_name = f"{aggregator_class}_PCASign_{mode_class}_{sign_class}"
            config_class = _make_config_class(
                config_name=config_name,
                class_name=class_name,
                aggregator=aggregator_name,
                pca_perturbation_mode=pca_mode,
                pca_top_k=pca_top_k,
                pca_sign=pca_sign,
            )
            globals()[class_name] = config_class
            CONFIGS[f"{aggregator_slug}_pca_sign_{mode_slug}_{sign_slug}"] = (
                config_class
            )
