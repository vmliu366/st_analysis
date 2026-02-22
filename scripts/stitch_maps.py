#!/usr/bin/env python
"""
Stitch full-size HxW PNGs from per-patch outputs.

Per patch (pixel-exact PNGs):
  - qc/orientation.png         (RGB)
  - qc/lobes.png               (RGB)
  - qc/AI.png                  (L / grayscale)

Per patch (arrays):
  - st_outputs/theta.npy                (ph, pw)
  - st_outputs/AI.npy                   (ph, pw)

Outputs:
  - out-orientation (RGB)  [tile-copy]
  - out-lobes       (RGB)  [tile-copy]
  - out-ai          (L)    [tile-copy] (optional)
  - out-ellipsoids  (RGB)  [computed HSV from theta+AI arrays]
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import numpy as np
from matplotlib import cm

def roi_to_rgb_background_uint8(roi_np: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(roi_np, [1, 99])
    bg01 = np.clip((roi_np - p1) / (p99 - p1 + 1e-12), 0, 1)
    bg8 = (bg01 * 255.0).astype(np.uint8)
    return np.stack([bg8, bg8, bg8], axis=-1)  # (H,W,3)

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

    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [V, q, p, p, t, V], default=V)
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [t, V, V, q, p, p], default=V)
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [p, p, t, V, V, q], default=V)

    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

def render_ellipsoids(
    roi_np: np.ndarray,
    eigenvals_np: np.ndarray,
    peak_angle_rad: float,
    step: int = 8,
    alpha: float = 0.6,
    qmin: float = 0.02,
    qmax: float = 0.98,
    min_w_frac: float = 0.15,
    max_w_frac: float = 0.80,
    min_h_frac: float = 0.05,
    max_h_frac: float = 0.40,
) -> np.ndarray:
    """
    roi_np: (H,W) float/uint
    eigenvals_np: (H,W,2) float, eigenvals[...,0]=lam1 (minor), eigenvals[...,1]=lam2 (major)
    peak_angle_rad: scalar in [0, pi)
    Returns RGB uint8 image (H,W,3) exactly.
    """
    H, W = roi_np.shape
    out = roi_to_rgb_background_uint8(roi_np).astype(np.float32)

    # grid centers
    Y, X = np.mgrid[step//2:H:step, step//2:W:step]
    ys = Y.ravel().astype(int)
    xs = X.ravel().astype(int)

    # sample eigenvalues at centers
    lam0 = eigenvals_np[..., 0]  # minor
    lam1 = eigenvals_np[..., 1]  # major
    lam0_s = lam0[ys, xs]
    lam1_s = lam1[ys, xs]

    # robust normalize eigenvalues -> [0,1]
    both = np.stack([lam0_s, lam1_s], axis=-1)
    ev_min = np.quantile(both, qmin)
    ev_max = np.quantile(both, qmax)
    eps = 1e-12
    def scale(a):
        return np.clip((a - ev_min) / (ev_max - ev_min + eps), 0, 1)

    lam0_n = scale(lam0_s)
    lam1_n = scale(lam1_s)

    # ellipse size ranges (pixels)
    min_w, max_w = step * min_w_frac, step * max_w_frac  # major axis uses lam1_n
    min_h, max_h = step * min_h_frac, step * max_h_frac  # minor axis uses lam0_n

    widths  = min_w + lam1_n * (max_w - min_w)  # major axis length
    heights = min_h + lam0_n * (max_h - min_h)  # minor axis length

    # color: constant hue from primary direction
    hue = (peak_angle_rad / np.pi) % 1.0
    color_rgba = cm.hsv(hue)  # (r,g,b,a) with a=1
    col = np.array(color_rgba[:3], dtype=np.float32) * 255.0  # RGB in 0..255

    # rotation: your plotting used angle_deg = degrees(theta) + 90.
    # Keep the same convention:
    ang = peak_angle_rad + 0.5 * np.pi
    ca = np.cos(ang)
    sa = np.sin(ang)

    # draw each ellipse in a local bounding box (fast enough for patch-sized images)
    for (y0, x0, w, h) in zip(ys, xs, widths, heights):
        a = 0.5 * float(w)  # semi-major
        b = 0.5 * float(h)  # semi-minor
        if a <= 0.5 or b <= 0.5:
            continue

        # bounding box in pixel coords
        r = int(np.ceil(max(a, b))) + 1
        y_min = max(0, y0 - r)
        y_max = min(H, y0 + r + 1)
        x_min = max(0, x0 - r)
        x_max = min(W, x0 + r + 1)

        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        dy = (yy - y0).astype(np.float32)
        dx = (xx - x0).astype(np.float32)

        # rotate coords into ellipse frame
        xpr =  ca * dx + sa * dy
        ypr = -sa * dx + ca * dy

        mask = (xpr * xpr) / (a * a + 1e-12) + (ypr * ypr) / (b * b + 1e-12) <= 1.0
        if not np.any(mask):
            continue

        # alpha blend
        sub = out[y_min:y_max, x_min:x_max, :]
        sub[mask] = alpha * col + (1.0 - alpha) * sub[mask]
        out[y_min:y_max, x_min:x_max, :] = sub

    return np.clip(out, 0, 255).astype(np.uint8)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches-tsv", required=True, type=Path)
    ap.add_argument("--shape-json", required=True, type=Path)
    ap.add_argument("--patch-root", required=True, type=Path)

    ap.add_argument("--out-orientation", required=True, type=Path)
    ap.add_argument("--out-lobes", required=True, type=Path)
    ap.add_argument("--out-ellipsoids", required=True, type=Path)
    ap.add_argument("--out-ai", required=False, type=Path, default=None)
    args = ap.parse_args()

    patches = read_patches(args.patches_tsv)
    if not patches:
        raise RuntimeError("No patches found in TSV.")

    shape = json.loads(args.shape_json.read_text())
    H = int(shape["H"])
    W = int(shape["W"])

    orientation_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    lobes_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    ellipsoids_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    ai_canvas = None
    if args.out_ai is not None:
        ai_canvas = np.zeros((H, W), dtype=np.uint8)

    for p in patches:
        py, px = p["py"], p["px"]
        y0, y1, x0, x1 = p["y0"], p["y1"], p["x0"], p["x1"]
        ph = y1 - y0
        pw = x1 - x0
        tag = f"py{py}_px{px}"

        # ---- Copy orientation.png ----
        ori_path = args.patch_root / tag / "qc" / "orientation.png"
        if not ori_path.exists():
            raise FileNotFoundError(f"Missing orientation.png: {ori_path}")
        ori_img = Image.open(ori_path).convert("RGB")
        if ori_img.size != (pw, ph):
            raise ValueError(
                f"Patch orientation size mismatch for {ori_path}: got {ori_img.size}, expected {(pw, ph)}."
            )
        orientation_canvas[y0:y1, x0:x1, :] = np.asarray(ori_img, dtype=np.uint8)

        # ---- Copy lobes.png ----
        lobes_path = args.patch_root / tag / "qc" / "lobes.png"
        if not lobes_path.exists():
            raise FileNotFoundError(f"Missing lobes.png: {lobes_path}")
        lobes_img = Image.open(lobes_path).convert("RGB")
        if lobes_img.size != (pw, ph):
            raise ValueError(
                f"Patch lobes size mismatch for {lobes_path}: got {lobes_img.size}, expected {(pw, ph)}."
            )
        lobes_canvas[y0:y1, x0:x1, :] = np.asarray(lobes_img, dtype=np.uint8)

        # ---- Copy AI.png (optional) ----
        if ai_canvas is not None:
            ai_path = args.patch_root / tag / "qc" / "AI.png"
            if not ai_path.exists():
                raise FileNotFoundError(f"Missing AI.png: {ai_path}")
            ai_img = Image.open(ai_path).convert("L")
            if ai_img.size != (pw, ph):
                raise ValueError(
                    f"Patch AI size mismatch for {ai_path}: got {ai_img.size}, expected {(pw, ph)}."
                )
            ai_canvas[y0:y1, x0:x1] = np.asarray(ai_img, dtype=np.uint8)

        # ---- Copy ellipsoids.png (RGB) ----
        ell_path = args.patch_root / tag / "qc" / "ellipsoids.png"
        if not ell_path.exists():
            raise FileNotFoundError(f"Missing ellipsoids.png: {ell_path}")
        ell_img = Image.open(ell_path).convert("RGB")
        if ell_img.size != (pw, ph):
            raise ValueError(
                f"Patch ellipsoids size mismatch for {ell_path}: got {ell_img.size}, expected {(pw, ph)}."
            )
        ellipsoids_canvas[y0:y1, x0:x1, :] = np.asarray(ell_img, dtype=np.uint8)

    args.out_orientation.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(orientation_canvas, mode="RGB").save(args.out_orientation)

    args.out_lobes.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(lobes_canvas, mode="RGB").save(args.out_lobes)

    args.out_ellipsoids.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ellipsoids_canvas, mode="RGB").save(args.out_ellipsoids)

    if args.out_ai is not None:
        args.out_ai.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ai_canvas, mode="L").save(args.out_ai)


if __name__ == "__main__":
    main()