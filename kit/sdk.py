"""Load approved public models and input arrays using QNN/QCNN names."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def verify_assets():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Invalid kit member")
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Kit integrity check failed: {relative}")
    return manifest


def load_model(family, run=0, *, kind="hardened"):
    if family not in ("qnn", "qcnn") or type(run) is not int or run not in range(5):
        raise ValueError("Use qnn/qcnn and run 0–4")
    if kind not in ("original", "hardened"):
        raise ValueError("Use original or hardened public models")
    verify_assets()
    sys.path.insert(0, str(ROOT / "src"))
    import qaml_ctf
    if not Path(qaml_ctf.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError("Use a clean participant process; another backend is already imported")
    from qaml_ctf.artifacts import load_weights
    from qaml_ctf.expansion.preprocessing import ExpansionPreprocessor
    from qaml_ctf.qnn_depth.models import DepthModel
    from qaml_ctf.road_architecture.config import Architecture
    from qaml_ctf.road_architecture.models import ArchitectureModel

    preprocessing = ExpansionPreprocessor((3, 32, 32), 64, 8, "ry")
    load_weights(ROOT / "preprocessing.npz", preprocessing)
    model = (DepthModel("primary", 3, 4, preprocessing) if family == "qnn" else
             ArchitectureModel(Architecture("qcnn_su4", "qcnn", block="su4"), 4, preprocessing))
    load_weights(ROOT / "models" / family / kind / f"run-{run}.npz", model)
    return model.eval()


def load_inputs(family, access="whitebox"):
    if family not in ("qnn", "qcnn") or access not in ("whitebox", "blackbox"):
        raise ValueError("Unknown input variant")
    verify_assets()
    with np.load(ROOT / "inputs" / f"{family}-{access}.npz", allow_pickle=False) as values:
        return {key: values[key].copy() for key in ("ids", "images", "targets")}


def write_submission(path, ids, images):
    array = images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Use a .npz output filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        np.savez_compressed(output, ids=np.asarray(ids, dtype=str), images=array.astype(np.float64))
