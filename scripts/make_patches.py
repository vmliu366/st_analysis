#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from zarrnii import ZarrNii
import zarr


def load_shape_2d(input_zarr_zip: Path, level: int, channel_index: int):
    store = zarr.storage.ZipStore(input_zarr_zip, mode="r")
    znimg = ZarrNii.from_ome_zarr(store_or_path=store, level=level)
    arr = znimg.darr

    # mirror structure_tensor.py logic
    if arr.ndim >= 3:
        try:
            arr = arr[:, channel_index, ...]
        except Exception:
            pass
    arr = arr.squeeze()

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D after squeeze; got shape {arr.shape} (ndim={arr.ndim}).")

    return arr.shape  # (Y, X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-zarr-zip", required=True, type=Path)
    ap.add_argument("--zarr-level", required=True, type=int)
    ap.add_argument("--channel-index", required=True, type=int)

    ap.add_argument("--patch-height", required=True, type=int)
    ap.add_argument("--patch-width", required=True, type=int)
    ap.add_argument("--overlap-y", default=0, type=int)
    ap.add_argument("--overlap-x", default=0, type=int)
    ap.add_argument("--edge-mode", choices=["drop"], default="drop")

    ap.add_argument("--out-tsv", required=True, type=Path)
    ap.add_argument("--out-shape-json", required=True, type=Path)
    args = ap.parse_args()

    H, W = load_shape_2d(args.input_zarr_zip, args.zarr_level, args.channel_index)

    step_y = args.patch_height - args.overlap_y
    step_x = args.patch_width - args.overlap_x
    if step_y <= 0 or step_x <= 0:
        raise ValueError("Overlap must be smaller than patch size.")

    # only "drop" implemented here
    ys = list(range(0, H - args.patch_height + 1, step_y))
    xs = list(range(0, W - args.patch_width + 1, step_x))

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w") as f:
        f.write("py\tpx\ty0\ty1\tx0\tx1\n")
        for py, y0 in enumerate(ys):
            for px, x0 in enumerate(xs):
                y1 = y0 + args.patch_height
                x1 = x0 + args.patch_width
                f.write(f"{py}\t{px}\t{y0}\t{y1}\t{x0}\t{x1}\n")

    args.out_shape_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_shape_json.write_text(json.dumps({"H": int(H), "W": int(W)}))


if __name__ == "__main__":
    main()