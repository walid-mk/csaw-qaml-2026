"""Explicit architecture and training contracts for the Road Signs study."""

import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Architecture:
    name: str
    family: str
    block: str = "heisenberg"
    topology: str = "open"
    output_qubits: int = 1
    readout: str = "xyz"


ARCHITECTURES = (
    Architecture("qnn_ring", "qnn"),
    Architecture("qnn_qasm_medium", "qnn"),
    Architecture("qnn_qasm_primary", "qnn"),
    Architecture("qnn_basic", "qnn"),
    Architecture("qnn_strong", "qnn"),
    Architecture("qnn_two_design", "qnn"),
    Architecture("qcnn_baseline", "qcnn"),
    Architecture("qcnn_light", "qcnn", block="light"),
    Architecture("qcnn_su4", "qcnn", block="su4"),
    Architecture("qcnn_periodic", "qcnn", topology="periodic"),
    Architecture("qcnn_probability2", "qcnn", output_qubits=2, readout="probabilities"),
)


def two_wire(architecture):
    if architecture.output_qubits == 2 and architecture.readout == "probabilities":
        return architecture
    return replace(
        architecture,
        name=architecture.name + "_probability2",
        output_qubits=2,
        readout="probabilities",
    )


@dataclass(frozen=True)
class Config:
    matrix_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    confirmation_seeds: tuple[int, ...] = (601, 701, 809, 907, 1009)
    epochs: int = 30
    minimum_epochs: int = 15
    patience: int = 8
    learning_rate: float = 0.02
    weight_decay: float = 0.001
    batch_size: int = 32
    evaluation_batch_size: int = 128
    clip_norm: float = 5.0
    bootstrap_samples: int = 2000
    workers: int = 12
    threads: int = 1
    probability_floor: float = 1e-12
    pca_seed: int = 20261020

    def __post_init__(self):
        for name in ("matrix_seeds", "confirmation_seeds"):
            value = tuple(getattr(self, name))
            object.__setattr__(self, name, value)
            if (
                not value
                or len(set(value)) != len(value)
                or any(type(v) is not int or not 0 <= v < 2**32 for v in value)
            ):
                raise ValueError("Seeds must be distinct uint32 integers")
        if type(self.pca_seed) is not int or not 0 <= self.pca_seed < 2**32:
            raise ValueError("PCA seed must be a uint32 integer")
        if set(self.matrix_seeds) & set(self.confirmation_seeds):
            raise ValueError("Matrix and confirmation seeds must be disjoint")
        for name in (
            "epochs",
            "minimum_epochs",
            "patience",
            "batch_size",
            "evaluation_batch_size",
            "bootstrap_samples",
            "workers",
            "threads",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer training budgets required")
        if self.minimum_epochs > self.epochs or self.bootstrap_samples < 2:
            raise ValueError("Invalid stopping/bootstrap budget")
        if not (
            0 < self.learning_rate <= 1
            and 0 <= self.weight_decay < 1
            and 0 < self.clip_norm < 100
            and 0 < self.probability_floor < 1e-3
        ):
            raise ValueError("Invalid numerical hyperparameter")

    def to_dict(self):
        return asdict(self)


def load_config(path):
    with Path(path).open("rb") as handle:
        return Config(**tomllib.load(handle))
