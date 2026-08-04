import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.manual_seed(seed):
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def model_to_vector(model: nn.Module) -> torch.Tensor:
    parameters = [
        parameter.detach().reshape(-1).cpu() for parameter in model.parameters()
    ]

    return torch.cat(parameters)

def vector_to_model(vector: torch.Tensor, model: nn.Module):
    offset = 0

    with torch.no_grad():
        for param in model.parameters():
            number_of_elements = param.numel()

            param_vector = vector[offset: offset + number_of_elements]

            param_vector = param_vector.view_as(param)

            param.copy_(param_vector.to(param.device))

            offset += number_of_elements

    if offset !=vector.numel():
        raise ValueError("The vector length does not match the model parameter count")

def append_csv(path: str, row: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    should_write_header = not output_path.exists()

    with output_path.open(mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if should_write_header:
            writer.writeheader()

        writer.writerow(row)