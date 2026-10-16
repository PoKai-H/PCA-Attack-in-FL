from dataclasses import dataclass

from configs.mnist.pca_config import PCABasic


@dataclass
class OtherAttackBasic(PCABasic):
    attack_name: str | None = "min_max"


AGGREGATORS = {
    "fedavg": ("FedAvg", "fed_avg"),
    "median": ("Median", "coordinate_wise_median"),
    "krum": ("Krum", "krum"),
    "multikrum": ("MultiKrum", "multi_krum"),
    "bulyan": ("Bulyan", "bulyan"),
    "trimmed_mean": ("TrimmedMean", "trimmed_mean"),
    "prodigy": ("Prodigy", "prodigy"),
}

ATTACKS = {
    "min_max": "MinMax",
    "min_sum": "MinSum",
    "alie": "ALIE",
    "alie_optim": "ALIEOptim",
    "sign_flip": "SignFlip",
}


def _make_config_class(
    config_name: str,
    class_name: str,
    aggregator: str,
    attack_name: str,
):
    namespace = {
        "__annotations__": {
            "output_csv": str,
            "aggregator": str,
            "attack_name": str | None,
        },
        "output_csv": f"./results/other_attacks/{config_name}.csv",
        "aggregator": aggregator,
        "attack_name": attack_name,
    }
    return dataclass(type(class_name, (OtherAttackBasic,), namespace))


CONFIGS = {}

for aggregator_slug, (aggregator_class, aggregator_name) in AGGREGATORS.items():
    for attack_name, attack_class in ATTACKS.items():
        config_name = f"{aggregator_slug}_noniid_{attack_name}"
        class_name = f"{aggregator_class}_{attack_class}"
        config_class = _make_config_class(
            config_name=config_name,
            class_name=class_name,
            aggregator=aggregator_name,
            attack_name=attack_name,
        )
        globals()[class_name] = config_class
        CONFIGS[f"{aggregator_slug}_other_{attack_name}"] = config_class
