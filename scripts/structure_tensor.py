#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import dask.array as da
import dask_image.ndfilters
from dask.array.gufunc import apply_gufunc

from PIL import Image
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm

from zarrnii import ZarrNii


def construct_tensor(roi, sigma_g, sigma_w, truncate):
    gx = dask_image.ndfilters.gaussian_filter(
        roi, sigma=sigma_g, order=(1, 0), mode="nearest", truncate=truncate
    )
    gy = dask_image.ndfilters.gaussian_filter(
        roi, sigma=sigma_g, order=(0, 1), mode="nearest", truncate=truncate
    )

    fxx = dask_image.ndfilters.gaussian_filter(gx * gx, sigma=sigma_w, mode="constant", truncate=truncate)
    fxy = dask_image.ndfilters.gaussian_filter(gx * gy, sigma=sigma_w, mode="constant", truncate=truncate)
    fyy = dask_image.ndfilters.gaussian_filter(gy * gy, sigma=sigma_w, mode="constant", truncate=truncate)

    # construct J tensor 
    y, x = fxx.shape
    J = da.stack([fxx, fxy, fxy, fyy], axis=2)
    J = da.reshape(J, (y, x, 2, 2))

    Jc = J.rechunk((1, x, 2, 2))
    w, _v = apply_gufunc(np.linalg.eigh, "(i,j)->(i),(i,j)", Jc)# eigen decomposition, we ony use eigenvalues here for AI
    lam1 = w[..., 0]
    lam2 = w[..., 1]

    # anisotropy index [0,1]
    AI = (lam2 - lam1) / da.maximum(lam2 + lam1, 1e-12)
    AI = da.clip(AI, 0, 1)

    fyy_ = J[:, :, 1, 1]
    fxx_ = J[:, :, 0, 0]
    fxy_ = J[:, :, 0, 1]
    theta = 0.5 * da.angle((fyy_ - fxx_) + 1j * 2 * fxy_) # theta here is calculated by the closed-form formula of structure tensor -> avoids per-pixel eigendecomposition (numerically stable)
    mask = theta < 0
    adjusted = da.angle(da.exp(1j * theta) * da.exp(1j * np.pi))
    theta = da.where(mask, adjusted, theta)  # radians

    return J, theta, AI, w


def load_znimg_2d(input_zarr_zip: Path, level: int, channel_index: int):
    znimg = ZarrNii.from_ome_zarr_zip(path=input_zarr_zip, level=level)
    arr = znimg.darr

    # Same intent as your previous logic
    if arr.ndim >= 3:
        try:
            arr = arr[:, channel_index, ...]
        except Exception:
            pass
    arr = arr.squeeze()

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D after squeeze; got shape {arr.shape} (ndim={arr.ndim}).")
    return arr


def compute_orientation_rgb(roi_np: np.ndarray, theta_np: np.ndarray, AI_np: np.ndarray) -> np.ndarray:
    # Hue encodes theta; saturation encodes AI; value encodes inverted intensity (as before)
    H = (theta_np / np.pi)  # normalized from [0, π) to [0, 1)
    S = np.clip(AI_np, 0, 1) 

    # # percentile normalization 
    # p1, p99 = np.percentile(roi_np, [1, 99])
    # roi_n = np.clip((roi_np - p1) / (p99 - p1 + 1e-12), 0, 1)
    # V = np.clip((1.0 - roi_n) ** 0.99, 0, 1)
    V = np.ones_like(roi_np, dtype=np.float32)


    rgb = hsv_to_rgb(np.stack([H, S, V], axis=-1))  # float in [0,1]
    rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return rgb8


def save_rgb_png(rgb8: np.ndarray, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb8, mode="RGB").save(out_png)

def save_grayscale_png(img: np.ndarray, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-zarr-zip", required=True, type=Path)
    ap.add_argument("--zarr-level", required=True, type=int)
    ap.add_argument("--channel-index", required=True, type=int)

    ap.add_argument("--patch", required=True, nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"))

    ap.add_argument("--sigma-g", required=True, type=float)
    ap.add_argument("--sigma-w", required=True, type=float)
    ap.add_argument("--truncate", required=True, type=float)

    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    out_arrays = args.out_dir / "st_outputs"
    out_figs = args.out_dir / "qc"
    out_arrays.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    zn2d = load_znimg_2d(args.input_zarr_zip, args.zarr_level, args.channel_index)

    y0, y1, x0, x1 = args.patch
    patch = zn2d[y0:y1, x0:x1]

    # right now this does local normalization per patch (fast), but seams may appear in full mosaics (!!)
    patch = patch.astype(np.float32)
    patch = patch / (da.max(patch) + 1e-12) * 255.0

    J, theta, AI, eigenvals = construct_tensor(
        patch,
        sigma_g=args.sigma_g,
        sigma_w=args.sigma_w,
        truncate=args.truncate,
    )

    patch_np = patch.compute()
    J_np = J.compute()
    theta_np = theta.compute()
    AI_np = AI.compute()
    eigenvals_np = eigenvals.compute()

    np.save(out_arrays / "roi.npy", patch_np)
    np.save(out_arrays / "J.npy", J_np)
    np.save(out_arrays / "theta.npy", theta_np)
    np.save(out_arrays / "AI.npy", AI_np)
    np.save(out_arrays / "eigenvals.npy", eigenvals_np)

    rgb8 = compute_orientation_rgb(patch_np, theta_np, AI_np)
    np.save(out_arrays / "orientation.npy", rgb8)
    save_rgb_png(rgb8, out_figs / "orientation.png") 
    save_grayscale_png(AI_np, out_figs / "AI.png")


if __name__ == "__main__":
    main()