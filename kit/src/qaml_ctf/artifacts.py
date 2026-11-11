"""Deterministic manifests and safe numeric checkpoint storage."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def save_weights(path: str | Path, model: torch.nn.Module) -> None:
    """NPZ only; never deserialize participant Python/pickle payloads."""
    np.savez_compressed(
        path, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    )


def load_weights(path: str | Path, model: torch.nn.Module) -> None:
    with np.load(path, allow_pickle=False) as weights:
        expected = model.state_dict()
        if set(weights.files) != set(expected):
            raise ValueError("Checkpoint keys do not match architecture")
        state = {}
        for name, prototype in expected.items():
            value = weights[name]
            if value.shape != tuple(prototype.shape) or not np.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint array: {name}")
            state[name] = torch.as_tensor(value.copy(), dtype=prototype.dtype)
    model.load_state_dict(state, strict=True)
