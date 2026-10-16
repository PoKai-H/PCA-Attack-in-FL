from dataclasses import dataclass
from configs.base import Config


@dataclass
class OtherAttackBasic(Config):
    attack_name: str | None = None


AGGREGATORS = {
    "fedavg": ("FedAvg", "fed_avg"),
    "median": ("Median", "median"),
    "krum": ("Krum", "krum"),
    "multikrum": ("Multikrum", "multi_krum"),
    "trimmedmean": ("TrimmedMean", "trimmed_mean"),
}

ATTACKS = {
    "min_max": "Minmax",
    "min_sum": "Minsum",
    "alie_optim": "Alieoptim",
    "sign_flip": "Signflip",
}


SEED = [40, 60, 11]


def _make_config_class(
    config_name: str,
    class_name: str,
    aggregator: str,
    attack_name: str,
    seed: int,
    include_nnm: bool,
):
    namespace = {
        "__annotations__": {
            "output_csv": str,
            "aggregator": str,
            "attack_name": str | None,
            "seed": int,
            "include_nnm": bool,
        },
        "output_csv": f"./results/cifar10/classic_attacks/{config_name}.csv",
        "aggregator": aggregator,
        "attack_name": attack_name,
        "seed": seed,
        "include_nnm": include_nnm,
    }
    return dataclass(type(class_name, (OtherAttackBasic,), namespace))


CONFIGS = {}

for aggregator_slug, (aggregator_class, aggregator_name) in AGGREGATORS.items():
    for attack_name, attack_class in ATTACKS.items():
        for include_nnm in [False, True]:
            for seed in SEED:
                nnm_slug = "nnm" if include_nnm else "raw"

                config_name = f"{aggregator_slug}_{attack_name}_{nnm_slug}_{seed}"
                class_name = f"{aggregator_class}_{attack_class}_{'NNM' if include_nnm else 'Raw'}_{seed}"

                config_class = _make_config_class(
                    class_name=class_name,
                    config_name=config_name,
                    aggregator=aggregator_name,
                    attack_name=attack_name,
                    seed=seed,
                    include_nnm=include_nnm,
                )

                globals()[class_name] = config_class
                CONFIGS[f"{aggregator_slug}_{attack_name}_{nnm_slug}_{seed}"] = (
                    config_class
                )
