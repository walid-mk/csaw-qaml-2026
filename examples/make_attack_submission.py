"""Copy approved inputs into a numeric submission; format example, not a solve."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.inputs, allow_pickle=False) as data:
        ids, images = data["ids"], data["images"]
        if ids.dtype.kind != "U" or ids.ndim != 1 or images.shape != (len(ids), 3, 32, 32):
            parser.error("Expected canonical RGB inputs and Unicode IDs")
        if not np.isfinite(images).all() or np.any(images < 0) or np.any(images > 1):
            parser.error("Expected finite pixels in [0,1]")
        if args.output.suffix != ".npz":
            parser.error("Use a .npz output filename")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            np.savez_compressed(handle, ids=ids, images=images.astype(np.float64))


if __name__ == "__main__":
    main()
