"""Original-pixel dataset adapters with reproducible, disjoint split manifests."""

import gzip
import hashlib
import shutil
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits

from qaml_ctf.artifacts import file_hash
from qaml_ctf.config import PilotConfig


class DatasetUnavailable(RuntimeError):
    """An actual dataset could not be obtained; never substitute synthetic data."""


@dataclass
class ImageSplit:
    images: torch.Tensor
    labels: torch.Tensor
    ids: list[str]


@dataclass
class DatasetBundle:
    train: ImageSplit
    dev: ImageSplit
    final: ImageSplit
    manifest: dict


def _obtain(path: Path, url: str, download: bool):
    if path.is_file():
        return
    if not download:
        raise DatasetUnavailable(
            f"Missing {path}; supply source data or enable --download"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        with (
            urllib.request.urlopen(url, timeout=30) as response,
            temporary.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        temporary.replace(path)
    except (OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise DatasetUnavailable(f"Download failed for {url}: {exc}") from exc


def read_idx(path: Path) -> np.ndarray:
    """Read IDX integers and check magic, dimensions and exact payload length."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        blob = stream.read()
    if len(blob) < 4 or blob[:3] != b"\x00\x00\x08" or blob[3] not in (1, 3):
        raise ValueError(f"Invalid IDX header: {path}")
    ndim = blob[3]
    if len(blob) < 4 + 4 * ndim:
        raise ValueError("Truncated IDX shape")
    shape = struct.unpack(">" + "I" * ndim, blob[4 : 4 + 4 * ndim])
    array = np.frombuffer(blob[4 + 4 * ndim :], dtype=np.uint8)
    if not all(shape) or int(np.prod(shape)) != array.size:
        raise ValueError("IDX dimensions/payload mismatch")
    return array.reshape(shape).copy()


def _source(name: str, root: Path, download: bool):
    if name == "digits":
        ds = load_digits()
        images = ds.images[:, None].astype(np.float64) / 16
        return {"pool": (images, ds.target)}, {
            "source": "sklearn.load_digits",
            "normalization": 16,
        }
    if name in ("mnist", "fashion_mnist"):
        base = (
            "https://ossci-datasets.s3.amazonaws.com/mnist/"
            if name == "mnist"
            else "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
        )
        arrays, hashes = {}, {}
        for split, prefix in (("train", "train"), ("test", "t10k")):
            for kind, suffix in (("images", "idx3"), ("labels", "idx1")):
                filename = f"{prefix}-{kind}-{suffix}-ubyte"
                path = root / name / filename
                if not path.is_file():
                    path = path.with_suffix(".gz")
                _obtain(path, base + filename + ".gz", download)
                arrays[split, kind] = read_idx(path)
                hashes[path.name] = file_hash(path)
        pools = {
            split: (
                arrays[split, "images"][:, None].astype(np.float64) / 255,
                arrays[split, "labels"].astype(np.int64),
            )
            for split in ("train", "test")
        }
        return pools, {"source": base, "sha256": hashes, "normalization": 255}
    if name == "bloodmnist":
        path = root / name / "bloodmnist.npz"
        url = "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1"
        _obtain(path, url, download)
        # Published MedMNIST checksum; SHA256 additionally recorded in the manifest.
        if (
            hashlib.md5(path.read_bytes()).hexdigest()
            != "7053d0359d879ad8a5505303e11de1dc"
        ):
            raise ValueError("BloodMNIST source checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            pools = {
                s: (
                    data[f"{s}_images"].transpose(0, 3, 1, 2).astype(np.float64) / 255,
                    data[f"{s}_labels"].reshape(-1).astype(np.int64),
                )
                for s in ("train", "val", "test")
            }
        return pools, {
            "source": url,
            "sha256": {path.name: file_hash(path)},
            "normalization": 255,
        }
    raise ValueError(f"Unknown dataset {name}")


def prepare_dataset(
    name: str, config: PilotConfig, root: str | Path, download: bool = False
) -> DatasetBundle:
    pools, provenance = _source(name, Path(root), download)
    rng = np.random.default_rng(config.split_seed)
    assignments = {"train": [], "dev": [], "final": []}
    sizes = (config.train_per_class, config.dev_per_class, config.final_per_class)
    for new_label, original in enumerate(config.classes):
        if "pool" in pools:
            pool_names = ("pool", "pool", "pool")
        elif "val" in pools:
            pool_names = ("train", "val", "test")
        else:
            pool_names = ("train", "train", "test")
        orders = {
            p: rng.permutation(np.flatnonzero(pools[p][1] == original))
            for p in sorted(set(pool_names))
        }
        offsets = {p: 0 for p in orders}
        for split, size, pool in zip(assignments, sizes, pool_names, strict=True):
            start = offsets[pool]
            chosen = orders[pool][start : start + size]
            if len(chosen) != size:
                raise ValueError(
                    f"Insufficient {name} class {original} samples in {pool}"
                )
            offsets[pool] += size
            assignments[split].extend((pool, int(i), new_label) for i in chosen)
    splits, split_ids = {}, {}
    for split, records in assignments.items():
        # Independent permutation, identical across model families and training seeds.
        records = [records[i] for i in rng.permutation(len(records))]
        images = np.stack([pools[p][0][i] for p, i, _ in records])
        if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Invalid source image pixels")
        ids = [f"{name}:{p}:{i}" for p, i, _ in records]
        splits[split] = ImageSplit(
            torch.from_numpy(images),
            torch.tensor([label for _, _, label in records]),
            ids,
        )
        split_ids[split] = ids
    flat_ids = [sample_id for ids in split_ids.values() for sample_id in ids]
    if len(set(flat_ids)) != len(flat_ids):
        raise ValueError("Dataset split leakage detected")
    # Hash all selected original pixels and labels as an additional provenance guard.
    digest = hashlib.sha256()
    for split in splits.values():
        digest.update(split.images.numpy().tobytes())
        digest.update(split.labels.numpy().tobytes())
    manifest = {
        "dataset": name,
        "classes": list(config.classes),
        "split_seed": config.split_seed,
        "ids": split_ids,
        "selected_pixels_sha256": digest.hexdigest(),
        **provenance,
    }
    return DatasetBundle(**splits, manifest=manifest)
