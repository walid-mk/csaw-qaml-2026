"""Strict, versioned configuration for the noiseless Phase 0 pilot."""

import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class PilotConfig:
    schema_version: int = 1
    datasets: tuple[str, ...] = ("digits",)
    families: tuple[str, ...] = ("qnn", "qcnn")
    classes: tuple[int, ...] = (0, 1, 2)
    seeds: tuple[int, ...] = (11, 29, 47)
    split_seed: int = 2026
    train_per_class: int = 90
    dev_per_class: int = 30
    final_per_class: int = 30
    qubits: int = 4
    layers: int = 2
    encoding: str = "ry_rz"
    preprocessing: str = "pca"
    epochs: int = 25
    batch_size: int = 30
    learning_rate: float = 0.025
    threads: int = 1
    bootstrap_samples: int = 2000
    probe_epsilons: tuple[float, ...] = (0.025, 0.05, 0.1)
    probe_samples: int = 90

    def __post_init__(self):
        for key in ("datasets", "families", "classes", "seeds", "probe_epsilons"):
            value = getattr(self, key)
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{key} must be an array")
            object.__setattr__(self, key, tuple(value))
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported configuration schema")
        if not self.datasets or set(self.datasets) - {
            "digits",
            "mnist",
            "fashion_mnist",
            "bloodmnist",
            "az23",
            "road_signs",
        }:
            raise ValueError("Unsupported dataset")
        if not self.families or set(self.families) - {"qnn", "qcnn"}:
            raise ValueError("Unsupported model family")
        if len(set(self.datasets)) != len(self.datasets) or len(
            set(self.families)
        ) != len(self.families):
            raise ValueError(
                "Duplicate datasets or families would invalidate independent run counts"
            )
        if len(self.classes) < 3 or len(set(self.classes)) != len(self.classes):
            raise ValueError("At least three distinct classes required")
        if any(type(c) is not int or c < 0 for c in self.classes):
            raise ValueError("Classes must be nonnegative integers")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Distinct training seeds required")
        if any(
            type(s) is not int or not 0 <= s < 2**32
            for s in (*self.seeds, self.split_seed)
        ):
            raise ValueError("Seeds must be integers in [0, 2**32)")
        for key in (
            "train_per_class",
            "dev_per_class",
            "final_per_class",
            "layers",
            "epochs",
            "batch_size",
            "threads",
            "bootstrap_samples",
            "probe_samples",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.bootstrap_samples < 2:
            raise ValueError("At least two bootstrap samples required")
        if type(self.qubits) is not int or self.qubits not in (4, 8, 10, 12):
            raise ValueError("Supported qubit counts are 4, 8, 10 and 12")
        if self.encoding not in (
            "ry",
            "ry_rz",
            "amplitude",
        ) or self.preprocessing not in (
            "pca",
            "resize",
        ):
            raise ValueError("Unsupported encoding/preprocessing")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate < 1:
            raise ValueError("Learning rate must be finite in (0, 1)")
        if not self.probe_epsilons or any(
            not math.isfinite(e) or not 0 < e <= 1 for e in self.probe_epsilons
        ):
            raise ValueError("Pixel probe epsilon must lie in (0, 1]")

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> PilotConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    unknown = set(raw) - {f.name for f in fields(PilotConfig)}
    if unknown:
        raise ValueError(
            f"Unknown or out-of-scope configuration keys: {sorted(unknown)}"
        )
    for key in ("datasets", "families", "classes", "seeds", "probe_epsilons"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return PilotConfig(**raw)
