#!/usr/bin/env python
#
# Produces 5 full-size HxW PNGs (exact pixel dimensions of the plane).
#
# Requirements / assumptions:
# 1) patches.tsv has columns: py px y0 y1 x0 x1
# 2) shape.json has keys: H, W
# 3) patch-root contains directories: py{py}_px{px}/...
# 4) For each patch:
#    - summary/peaks.json exists with:
#         {"peak_angles":[a1,a2], "peak_amps":[p1,p2], "peak_ratio": r}
#    - figures/orientation_polar.png exists and is EXACTLY (patch_width x patch_height) pixels
#    - arrays/theta.npy and arrays/AI.npy exist (for ellipsoids map)
#
# Output encodings (RGB):
# - theta_hist.png: per-patch, hue=peak1 angle, value=normalized peak1 amp
# - theta_polar.png: per-patch, hue=peak2 angle, value=normalized peak2 amp
# - orientation_polar.png: per-pixel orientation RGB copied from patch PNG
# - lobes.png: per-patch, hue=peak1 angle, value=normalized peak1 amp, saturation=clamped peak_ratio
# - ellipsoids.png: per-pixel, hue=theta/pi, value=AI, saturation=1.0

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_patches(tsv_path: Path):
    patches = []
    with tsv_path.open("r") as f:
        header = f.readline().strip().split("\t")
        i_py = header.index("py")
        i_px = header.index("px")
        i_y0 = header.index("y0")
        i_y1 = header.index("y1")
        i_x0 = header.index("x0")
        i_x1 = header.index("x1")
        for line in f:
            parts = line.strip().split("\t")
            patches.append({
                "py": int(parts[i_py]),
                "px": int(parts[i_px]),
                "y0": int(parts[i_y0]),
                "y1": int(parts[i_y1]),
                "x0": int(parts[i_x0]),
                "x1": int(parts[i_x1]),
            })
    return patches


def hsv_to_rgb_uint8(H, S, V):
    """
    Vectorized HSV (0..1 floats) -> RGB uint8.
    H,S,V can be broadcastable arrays with last dim absent.
    Returns uint8 array (...,3).
    """
    H = np.mod(H, 1.0)
    S = np.clip(S, 0.0, 1.0)
    V = np.clip(V, 0.0, 1.0)

    h6 = H * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = V * (1.0 - S)
    q = V * (1.0 - S * f)
    t = V * (1.0 - S * (1.0 - f))

    i = i % 6

    r = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [V, q, p, p, t, V],
        default=V,
    )
    g = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [t, V, V, q, p, p],
        default=V,
    )
    b = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [p, p, t, V, V, q],
        default=V,
    )

    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def load_peaks_json(path: Path):
    obj = json.loads(path.read_text())
    peak_angles = obj.get("peak_angles", [0.0, 0.0])
    peak_amps = obj.get("peak_amps", [0.0, 0.0])
    ratio = obj.get("peak_ratio", 0.0)
    # sanitize
    a1 = float(peak_angles[0]) if len(peak_angles) > 0 else 0.0
    a2 = float(peak_angles[1]) if len(peak_angles) > 1 else 0.0
    p1 = float(peak_amps[0]) if len(peak_amps) > 0 else 0.0
    p2 = float(peak_amps[1]) if len(peak_amps) > 1 else 0.0
    ratio = float(ratio)
    return a1, a2, p1, p2, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches-tsv", required=True, type=Path)
    ap.add_argument("--shape-json", required=True, type=Path)
    ap.add_argument("--patch-root", required=True, type=Path)

    ap.add_argument("--out-theta-hist", required=True, type=Path)
    ap.add_argument("--out-theta-polar", required=True, type=Path)
    ap.add_argument("--out-orientation-polar", required=True, type=Path)
    ap.add_argument("--out-lobes", required=True, type=Path)
    ap.add_argument("--out-ellipsoids", required=True, type=Path)
    args = ap.parse_args()

    patches = read_patches(args.patches_tsv)
    if not patches:
        raise RuntimeError("No patches found in TSV.")

    shape = json.loads(args.shape_json.read_text())
    H = int(shape["H"])
    W = int(shape["W"])

    # Full-size canvases (RGB uint8)
    theta_hist_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    theta_polar_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    orientation_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    lobes_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    ellipsoids_canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # First pass: find global max amps to normalize value channel consistently
    p1_all = []
    p2_all = []
    for p in patches:
        tag = f"py{p['py']}_px{p['px']}"
        peaks_path = args.patch_root / tag / "summary" / "peaks.json"
        if not peaks_path.exists():
            raise FileNotFoundError(f"Missing peaks.json: {peaks_path}")
        _a1, _a2, p1, p2, _r = load_peaks_json(peaks_path)
        p1_all.append(p1)
        p2_all.append(p2)

    p1_max = max(p1_all) if p1_all else 1.0
    p2_max = max(p2_all) if p2_all else 1.0
    if p1_max <= 1e-12:
        p1_max = 1.0
    if p2_max <= 1e-12:
        p2_max = 1.0

    # Second pass: fill canvases
    for p in patches:
        py, px = p["py"], p["px"]
        y0, y1, x0, x1 = p["y0"], p["y1"], p["x0"], p["x1"]
        tag = f"py{py}_px{px}"

        # ---- Patch-level peak encodings ----
        peaks_path = args.patch_root / tag / "summary" / "peaks.json"
        a1, a2, p1, p2, ratio = load_peaks_json(peaks_path)

        # normalize to 0..1
        v1 = np.clip(p1 / p1_max, 0.0, 1.0)
        v2 = np.clip(p2 / p2_max, 0.0, 1.0)
        s_ratio = np.clip(ratio, 0.0, 1.0)

        # hue is angle/pi
        h1 = (a1 / np.pi) % 1.0
        h2 = (a2 / np.pi) % 1.0

        # make solid-color tiles (patch size)
        ph = y1 - y0
        pw = x1 - x0

        tile_hist = hsv_to_rgb_uint8(
            H=np.full((ph, pw), h1, dtype=np.float32),
            S=np.ones((ph, pw), dtype=np.float32),
            V=np.full((ph, pw), v1, dtype=np.float32),
        )
        tile_polar = hsv_to_rgb_uint8(
            H=np.full((ph, pw), h2, dtype=np.float32),
            S=np.ones((ph, pw), dtype=np.float32),
            V=np.full((ph, pw), v2, dtype=np.float32),
        )
        tile_lobes = hsv_to_rgb_uint8(
            H=np.full((ph, pw), h1, dtype=np.float32),
            S=np.full((ph, pw), s_ratio, dtype=np.float32),
            V=np.full((ph, pw), v1, dtype=np.float32),
        )

        theta_hist_canvas[y0:y1, x0:x1, :] = tile_hist
        theta_polar_canvas[y0:y1, x0:x1, :] = tile_polar
        lobes_canvas[y0:y1, x0:x1, :] = tile_lobes

        # ---- Per-pixel orientation_polar.png copied exactly ----
        ori_path = args.patch_root / tag / "figures" / "orientation_polar.png"
        if not ori_path.exists():
            raise FileNotFoundError(f"Missing orientation_polar.png: {ori_path}")
        ori_img = Image.open(ori_path).convert("RGB")
        if ori_img.size != (pw, ph):
            raise ValueError(
                f"Patch image size mismatch for {ori_path}: got {ori_img.size}, expected {(pw, ph)}. "
                f"Ensure patch PNGs are saved pixel-exact (no matplotlib tight bbox)."
            )
        orientation_canvas[y0:y1, x0:x1, :] = np.asarray(ori_img, dtype=np.uint8)

        # ---- Ellipsoids map (per-pixel theta+AI) ----
        theta_path = args.patch_root / tag / "arrays" / "theta.npy"
        ai_path = args.patch_root / tag / "arrays" / "AI.npy"
        if not theta_path.exists():
            raise FileNotFoundError(f"Missing theta.npy: {theta_path}")
        if not ai_path.exists():
            raise FileNotFoundError(f"Missing AI.npy: {ai_path}")

        theta = np.load(theta_path)  # (ph, pw)
        AI = np.load(ai_path)        # (ph, pw)
        if theta.shape != (ph, pw) or AI.shape != (ph, pw):
            raise ValueError(
                f"Array shape mismatch in {tag}: theta {theta.shape}, AI {AI.shape}, expected {(ph, pw)}"
            )

        # Hue=theta/pi, Value=AI, Saturation=1
        ell_tile = hsv_to_rgb_uint8(
            H=(theta / np.pi).astype(np.float32),
            S=np.ones((ph, pw), dtype=np.float32),
            V=np.clip(AI, 0.0, 1.0).astype(np.float32),
        )
        ellipsoids_canvas[y0:y1, x0:x1, :] = ell_tile

    # Save outputs (RGB PNG)
    args.out_theta_hist.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(theta_hist_canvas, mode="RGB").save(args.out_theta_hist)

    args.out_theta_polar.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(theta_polar_canvas, mode="RGB").save(args.out_theta_polar)

    args.out_orientation_polar.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(orientation_canvas, mode="RGB").save(args.out_orientation_polar)

    args.out_lobes.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(lobes_canvas, mode="RGB").save(args.out_lobes)

    args.out_ellipsoids.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ellipsoids_canvas, mode="RGB").save(args.out_ellipsoids)


if __name__ == "__main__":
    main()