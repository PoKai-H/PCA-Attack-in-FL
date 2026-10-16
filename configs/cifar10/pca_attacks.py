from dataclasses import dataclass

from configs.base import Config


@dataclass
class PCABasic(Config):
    attack_name: str | None = "pca_subspace"
    partition: str = "non_iid"
    pca_perturbation_mode: str = "std_weighted"


AGGREGATORS = {
    "fedavg": ("FedAvg", "fed_avg"),
    "median": ("Median", "median"),
    "krum": ("Krum", "krum"),
    "multikrum": ("Multikrum", "multi_krum"),
    "trimmedmean": ("TrimmedMean", "trimmed_mean"),
}

PCA_TOP_K = [2, 3]
SEEDS = [40, 60, 11]


def _make_config_class(
    config_name: str,
    class_name: str,
    aggregator_name: str,
    pca_top_k: int,
    pca_sign: int,
    seed: int,
    include_nnm: bool,
):
    namespace = {
        "__annotations__": {
            "output_csv": str,
            "aggregator": str,
            "seed": int,
            "include_nnm": bool,
            "pca_top_k": int,
            "pca_sign": int,
        },
        "output_csv": f"./results/cifar10/pca_attacks/{config_name}.csv",
        "aggregator": aggregator_name,
        "seed": seed,
        "include_nnm": include_nnm,
        "pca_top_k": pca_top_k,
        "pca_sign": pca_sign,
    }

    return dataclass(type(class_name, (PCABasic,), namespace))


CONFIGS = {}

for aggregator_slug, (aggregator_class, aggregator_name) in AGGREGATORS.items():
    for pca_top_k in PCA_TOP_K:
        for include_nnm in [False, True]:
            for seed in SEEDS:
                nnm_slug = "nnm" if include_nnm else "raw"
                config_name = f"{aggregator_slug}_stdk{pca_top_k}_p0_{nnm_slug}_{seed}"
                class_name = (
                    f"{aggregator_class}StdK{pca_top_k}"
                    f"P0{'NNM' if include_nnm else 'Raw'}{seed}"
                )

                config_class = _make_config_class(
                    class_name=class_name,
                    config_name=config_name,
                    aggregator_name=aggregator_name,
                    pca_top_k=pca_top_k,
                    pca_sign=0,
                    seed=seed,
                    include_nnm=include_nnm,
                )

                globals()[class_name] = config_class
                CONFIGS[config_name] = config_class
