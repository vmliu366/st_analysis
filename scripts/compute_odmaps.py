#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute optical density (OD) maps from OME-Zarr histology chunks and export as NIfTI.
"""

import argparse
import copy
import json
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import dask.array as da
from tqdm import tqdm
from skimage.filters import threshold_otsu
from scipy.signal import find_peaks

from zarrnii import ZarrNii


def extract_background(tmp_np, smooth_factor=5):
    """Compute the background mean intensity from a grayscale image."""
    _ = threshold_otsu(tmp_np)  # kept for continuity; not used directly

    hist, bin_edges = np.histogram(tmp_np.ravel(), bins=256)
    smoothed = np.convolve(hist, np.ones(smooth_factor) / smooth_factor, mode="same")
    peaks, _ = find_peaks(smoothed)

    last_peak = None
    for i in reversed(peaks):
        v = bin_edges[i]
        if v < 200:
            last_peak = v
            break

    if last_peak is None:
        # fallback: high percentile as background proxy
        last_peak = np.percentile(tmp_np, 95)

    background_mask = tmp_np >= last_peak
    background_img = np.where(background_mask, tmp_np, np.nan)
    background_mean = np.nanmean(background_img)

    # fallback if nanmean fails (rare but possible)
    if not np.isfinite(background_mean) or background_mean <= 0:
        background_mean = np.nanpercentile(tmp_np, 95)

    return float(background_mean)


def calc_odmaps(numpy_array, background_mean, bins=50, min_count=50):
    """Calculate normalized optical density map."""
    od_map = -np.log10((numpy_array + 1e-12) / (background_mean + 1e-12))
    mask = np.isfinite(od_map) & (od_map > 0) & (od_map < 6)
    valid = od_map[mask]

    if valid.size == 0:
        return np.zeros_like(numpy_array, dtype=np.float32)

    n, bin_edges = np.histogram(valid, bins=bins)

    good = np.where(n > min_count)[0]
    if good.size == 0:
        tail_end = float(np.percentile(valid, 99))
        tail_end = max(tail_end, 1e-6)
    else:
        idx = good[-1]
        tail_end = float(bin_edges[idx + 1])
        tail_end = max(tail_end, 1e-6)

    clipped = np.where(mask, np.minimum(od_map, tail_end), 0.0)
    od_map_norm = (clipped / tail_end).astype(np.float32)

    return od_map_norm


def to_nifti(zarrnii_obj, filename: Path):
    """Convert a ZarrNii instance to a NIfTI-1 image and save."""
    darr = zarrnii_obj.darr
    if darr.ndim == 5:
        darr = darr[0, 0, ...]

    if getattr(zarrnii_obj, "axes_order", None) == "ZYX":
        data = da.moveaxis(darr, (0, 1, 2, 3), (0, 3, 2, 1)).compute()
        affine = zarrnii_obj.reorder_affine_xyz_zyx(zarrnii_obj.affine.matrix)
    else:
        data = darr.compute()
        affine = np.array(
            [
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    nii_img = nib.Nifti1Image(data[0, :, :, :], affine)
    filename.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii_img, str(filename))


def parse_int_list(x: str):
    if x.strip() == "":
        return []
    return [int(v) for v in x.split(",")]


def process_images(
    input_dir: Path,
    output_dir: Path,
    level: int,
    chunk_start: int,
    chunk_end: int,
    chunk_step: int,
    skip_list,
    file_template: str,
    out_prefix: str,
):
    """Main processing loop for OD map computation."""
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="distributed.client",
        message="Sending large graph of size",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids = list(range(chunk_start, chunk_end + 1, chunk_step))
    skip = set(skip_list)

    znimgs = []
    original_chunk_numbers = []

    # Step 1 — load images
    for i in tqdm(chunk_ids, desc="loading chunks"):
        if i in skip:
            continue
        fname = file_template.format(chunk=i)
        path = input_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
        zarr_img = ZarrNii.from_ome_zarr_zip(path=path, level=level)

        tmp_darr = zarr_img.darr.squeeze(axis=0)  # (T,C,Z,Y,X) -> (C,Z,Y,X) if present
        zarr_img.darr = tmp_darr
        znimgs.append(zarr_img)
        original_chunk_numbers.append(i)

    if not znimgs:
        raise RuntimeError("No chunks loaded (check chunk range, skip list, and template).")

    # Step 2 — normalize shape & contrast (per chunk)
    znimgs_corrshape = copy.deepcopy(znimgs)
    for img in tqdm(range(len(znimgs)), desc="normalizing"):
        tmp_darr = znimgs[img].darr
        tmp_darr = tmp_darr.squeeze()

        # assume channel first -> move to last and take channel 0
        if tmp_darr.ndim == 4:
            tmp_darr = da.moveaxis(tmp_darr, 0, -1)  # (C,Z,Y,X) -> (Z,Y,X,C)
            tmp_darr = tmp_darr[..., 0]              # (Z,Y,X)
        elif tmp_darr.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected array ndim after squeeze: {tmp_darr.ndim}")

        tmp_darr = (tmp_darr / (da.max(tmp_darr) + 1e-12)) * 255.0
        tmp_darr = tmp_darr.astype(np.uint8)
        znimgs_corrshape[img].darr = tmp_darr

    # Step 3 — compute OD maps
    od_maps_znimgs = copy.deepcopy(znimgs)
    for img in tqdm(range(len(znimgs_corrshape)), desc="computing OD maps"):
        tmp_np = znimgs_corrshape[img].darr.compute()
        tmp_background = extract_background(tmp_np)
        od_map_np = calc_odmaps(tmp_np, tmp_background)
        od_maps_znimgs[img].darr = da.from_array(
            od_map_np[None, None, ...], chunks=(1, 1, *od_map_np.shape)
        )

    # Step 4 — save as NIfTI
    out_files = []
    for idx, _img in tqdm(list(enumerate(od_maps_znimgs)), desc="saving NIfTIs"):
        chunk = original_chunk_numbers[idx]
        nifti_name = output_dir / f"{out_prefix}_chunk-{chunk:03d}_desc-level{level}_ODmaps.nii.gz"
        to_nifti(od_maps_znimgs[idx], filename=nifti_name)
        out_files.append(str(nifti_name))

    # also write a small manifest for provenance
    (output_dir / "odmaps_manifest.json").write_text(
        json.dumps({"level": level, "outputs": out_files}, indent=2)
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute OD maps from OME-Zarr chunks and export as NIfTI.")
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--level", required=True, type=int)

    ap.add_argument("--chunk-start", type=int, required=True)
    ap.add_argument("--chunk-end", type=int, required=True)
    ap.add_argument("--chunk-step", type=int, default=1)

    ap.add_argument("--skip", type=str, default="", help="Comma-separated chunk ids to skip, e.g. '34,112,114'")
    ap.add_argument(
        "--file-template",
        type=str,
        required=True,
        help="Filename template under input-dir, use '{chunk}' placeholder, e.g. '..._chunk-{chunk:03d}_BF.ome.zarr.zip'",
    )
    ap.add_argument(
        "--out-prefix",
        type=str,
        default="sub-T20_ses-01",
        help="Prefix for NIfTI filenames before chunk-XXX...",
    )

    args = ap.parse_args()

    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        level=args.level,
        chunk_start=args.chunk_start,
        chunk_end=args.chunk_end,
        chunk_step=args.chunk_step,
        skip_list=parse_int_list(args.skip),
        file_template=args.file_template,
        out_prefix=args.out_prefix,
    )