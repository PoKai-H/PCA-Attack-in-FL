from dataclasses import dataclass

from configs.base import Config


@dataclass
class NoAttackBasic(Config):
    attack_name: str | None = None
    partition: str = "non_iid"


AGGREGATORS = {
    "fedavg": ("FedAvg", "fed_avg"),
    "median": ("Median", "median"),
    "krum": ("Krum", "krum"),
    "multikrum": ("Multikrum", "multi_krum"),
    "trimmedmean": ("TrimmedMean", "trimmed_mean"),
}

SEEDS = [40, 60, 11]


def _make_config_class(
    config_name: str,
    class_name: str,
    aggregator_name: str,
    seed: int,
    include_nnm: bool,
):
    namespace = {
        "__annotations__": {
            "output_csv": str,
            "aggregator": str,
            "seed": int,
            "include_nnm": bool,
        },
        "output_csv": f"./results/cifar10/no_attacks/{config_name}.csv",
        "aggregator": aggregator_name,
        "seed": seed,
        "include_nnm": include_nnm,
    }

    return dataclass(type(class_name, (NoAttackBasic,), namespace))


CONFIGS = {}

for aggregator_slug, (aggregator_class, aggregator_name) in AGGREGATORS.items():
    for include_nnm in [False, True]:
        for seed in SEEDS:
            nnm_slug = "nnm" if include_nnm else "raw"
            config_name = f"{aggregator_slug}_{nnm_slug}_{seed}"
            class_name = f"{aggregator_class}{'NNM' if include_nnm else 'Raw'}{seed}"

            config_class = _make_config_class(
                class_name=class_name,
                config_name=config_name,
                aggregator_name=aggregator_name,
                seed=seed,
                include_nnm=include_nnm,
            )

            globals()[class_name] = config_class
            CONFIGS[config_name] = config_class
