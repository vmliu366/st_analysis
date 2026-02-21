#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches-tsv", required=True, type=Path)
    ap.add_argument("--patch-root", required=True, type=Path,
                    help="Directory that contains per-patch outputs, e.g. results/.../patches")
    ap.add_argument("--figure-name", required=True,
                    help="e.g. theta_hist.png, lobes.png")
    ap.add_argument("--out-png", required=True, type=Path)
    args = ap.parse_args()

    rows = []
    with args.patches_tsv.open("r") as f:
        header = f.readline().strip().split("\t")
        idx_py = header.index("py")
        idx_px = header.index("px")
        for line in f:
            parts = line.strip().split("\t")
            py = int(parts[idx_py])
            px = int(parts[idx_px])
            rows.append((py, px))

    if not rows:
        raise RuntimeError("No patches found in TSV.")

    max_py = max(py for py, _ in rows)
    max_px = max(px for _, px in rows)

    # Load first tile to get tile size
    first = args.patch_root / f"py{rows[0][0]}_px{rows[0][1]}" / "figures" / args.figure_name
    if not first.exists():
        raise FileNotFoundError(f"Missing first tile: {first}")
    tile0 = Image.open(first).convert("RGBA")
    tw, th = tile0.size

    mosaic = Image.new("RGBA", ((max_px + 1) * tw, (max_py + 1) * th), (0, 0, 0, 0))

    for py, px in rows:
        tile_path = args.patch_root / f"py{py}_px{px}" / "figures" / args.figure_name
        if not tile_path.exists():
            raise FileNotFoundError(f"Missing tile: {tile_path}")
        tile = Image.open(tile_path).convert("RGBA")
        if tile.size != (tw, th):
            raise ValueError(f"Tile size mismatch at {tile_path}: got {tile.size}, expected {(tw, th)}")
        mosaic.paste(tile, (px * tw, py * th))

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(args.out_png)

if __name__ == "__main__":
    main()