from dataclasses import dataclass
from typing import Any


@dataclass
class Hyperparameter:
    name: str
    value: Any


def generate_hyperparameter(name: str, value: Any) -> Hyperparameter:
    return Hyperparameter(name=name, value=value)
