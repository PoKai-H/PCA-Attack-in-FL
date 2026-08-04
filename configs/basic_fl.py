from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    data_dir: str = "./data"
    output_csv: str = "./results/basic_mnist.csv"

    num_clients: int = 10
    clients_per_round: int = 10
    malicious_fractions: float = 0.0

    communication_rounds: int = 20
    local_epochs: int = 1 # set to 1 to simplified the system
    batch_size: int = 64
 
    learning_rate: float = 0.01
    momentum: float = 0.9

    centralized_epochs: int = 3

    use_dropout: bool = False

    attack_name: str | None = None
    attack_optimizer_name: str | None = None

    aggregator = "fed_avg"

