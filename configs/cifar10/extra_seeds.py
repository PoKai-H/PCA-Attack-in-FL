from dataclasses import dataclass
from configs.cifar10.other_attacks import CONFIGS as OTHER_ATTACKS_CONFIGS


@dataclass
class FedavgMinmaxRaw62(OTHER_ATTACKS_CONFIGS["fedavg_min_max_raw_60"]):
    seed: int = 62
    output_csv: str = "./results/cifar10/extra_seeds/fedavg_min_max_raw_62.csv"


CONFIGS = {"fedavg_min_max_raw_62": FedavgMinmaxRaw62}
