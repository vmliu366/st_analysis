#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import nibabel as nib
from zarrnii import ZarrNii


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

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


def get_affine_from_zarr(zn: ZarrNii):
    """
    Mirror your working OD script logic.
    """
    if getattr(zn, "axes_order", None) == "ZYX":
        return zn.reorder_affine_xyz_zyx(zn.affine.matrix)
    else:
        return np.array([
            [0,  0,  1,  0],
            [-1, 0,  0,  0],
            [0, -1, 0,  0],
            [0,  0,  0,  1]
        ], dtype=float)

def load_mask_2d_from_nifti(mask_path: Path) -> np.ndarray:
    """
    Load a 2D binary mask saved as NIfTI.

    Expected layout matches this pipeline's convention:
      - data stored as (X, Y, 1) or (X, Y) in NIfTI space
      - returns mask as (Y, X) boolean
    """
    nii = nib.load(str(mask_path))
    data = np.asanyarray(nii.dataobj)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"Mask NIfTI must be 2D after squeeze; got shape {data.shape}")

    # Convert (X,Y) -> (Y,X) to match internal canvases
    mask_yx = data.T
    return mask_yx > 0.5

def save_float_map_as_nifti(vol_yx: np.ndarray, zn_template: ZarrNii, out_path: Path):
    """
    Save stitched (H,W) float array as (X,Y,1) NIfTI float32.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert (Y,X) -> (X,Y,1)
    data_xyz1 = np.asarray(vol_yx, dtype=np.float32).T[:, :, None]

    affine = get_affine_from_zarr(zn_template)

    nii = nib.Nifti1Image(data_xyz1, affine)
    nii.set_data_dtype(np.float32)
    nii.set_sform(affine, code=1)
    nii.set_qform(affine, code=1)

    nib.save(nii, str(out_path))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--patches-tsv", required=True, type=Path)
    ap.add_argument("--shape-json", required=True, type=Path)
    ap.add_argument("--patch-root", required=True, type=Path)

    ap.add_argument("--input-zarr-zip", required=True, type=Path)
    ap.add_argument("--zarr-level", required=True, type=int)
    ap.add_argument("--mask-nifti", required=False, type=Path, default=None)

    ap.add_argument("--out-orientation", required=True, type=Path)
    ap.add_argument("--out-lobes", required=True, type=Path)
    ap.add_argument("--out-ellipsoids", required=True, type=Path)
    ap.add_argument("--out-ai", required=False, type=Path)

    ap.add_argument("--out-theta-nifti", required=True, type=Path)
    ap.add_argument("--out-ai-nifti", required=True, type=Path)

    args = ap.parse_args()

    patches = read_patches(args.patches_tsv)
    if not patches:
        raise RuntimeError("No patches found.")

    shape = json.loads(args.shape_json.read_text())
    H = int(shape["H"])
    W = int(shape["W"])

    # QC PNG canvases
    orientation_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    lobes_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    ellipsoids_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    ai_canvas = None
    if args.out_ai is not None:
        ai_canvas = np.zeros((H, W), dtype=np.uint8)

    # stitched arrays
    theta_full = np.zeros((H, W), dtype=np.float32)
    ai_full = np.zeros((H, W), dtype=np.float32)

    for p in patches:
        py, px = p["py"], p["px"]
        y0, y1, x0, x1 = p["y0"], p["y1"], p["x0"], p["x1"]
        ph = y1 - y0
        pw = x1 - x0
        tag = f"py{py}_px{px}"

        # ---- QC PNGs ----
        for name, canvas in [
            ("orientation.png", orientation_canvas),
            ("lobes.png", lobes_canvas),
            ("ellipsoids.png", ellipsoids_canvas),
        ]:
            path = args.patch_root / tag / "qc" / name
            if not path.exists():
                raise FileNotFoundError(f"Missing {name}: {path}")
            img = Image.open(path).convert("RGB")
            if img.size != (pw, ph):
                raise ValueError(f"Size mismatch in {path}")
            canvas[y0:y1, x0:x1] = np.asarray(img, dtype=np.uint8)

        if ai_canvas is not None:
            ai_path = args.patch_root / tag / "qc" / "AI.png"
            if not ai_path.exists():
                raise FileNotFoundError(f"Missing AI.png: {ai_path}")
            img = Image.open(ai_path).convert("L")
            if img.size != (pw, ph):
                raise ValueError(f"Size mismatch in {ai_path}")
            ai_canvas[y0:y1, x0:x1] = np.asarray(img, dtype=np.uint8)

        # ---- arrays ----
        theta_path = args.patch_root / tag / "st_outputs" / "theta.npy"
        ai_path = args.patch_root / tag / "st_outputs" / "AI.npy"

        if not theta_path.exists():
            raise FileNotFoundError(f"Missing theta.npy: {theta_path}")
        if not ai_path.exists():
            raise FileNotFoundError(f"Missing AI.npy: {ai_path}")

        theta = np.load(theta_path).astype(np.float32)
        AI = np.load(ai_path).astype(np.float32)

        if theta.shape != (ph, pw) or AI.shape != (ph, pw):
            raise ValueError(f"Array shape mismatch in {tag}")

        theta_full[y0:y1, x0:x1] = theta
        ai_full[y0:y1, x0:x1] = AI

    # ---- Apply mask (post-stitch) ----
    if args.mask_nifti is not None:
        mask = load_mask_2d_from_nifti(args.mask_nifti)
        if mask.shape != (H, W):
            raise ValueError(f"Mask shape {mask.shape} does not match stitched shape {(H, W)}")
        inv = ~mask

        # RGB canvases: set outside mask to black
        orientation_canvas[inv] = 0
        lobes_canvas[inv] = 0
        ellipsoids_canvas[inv] = 0

        # grayscale AI canvas (if requested): set outside mask to 0
        if ai_canvas is not None:
            ai_canvas[inv] = 0

        # scientific arrays: set outside mask to 0
        theta_full[inv] = 0.0
        ai_full[inv] = 0.0
        
    # ---- Save stitched PNGs ----
    args.out_orientation.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(orientation_canvas).save(args.out_orientation)

    args.out_lobes.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(lobes_canvas).save(args.out_lobes)

    args.out_ellipsoids.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ellipsoids_canvas).save(args.out_ellipsoids)

    if ai_canvas is not None:
        args.out_ai.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ai_canvas).save(args.out_ai)

    # ---- Save scientific NIfTI outputs ----
    zn_template = ZarrNii.from_ome_zarr_zip(
        path=args.input_zarr_zip,
        level=args.zarr_level
    )

    save_float_map_as_nifti(theta_full, zn_template, args.out_theta_nifti)
    save_float_map_as_nifti(ai_full, zn_template, args.out_ai_nifti)


if __name__ == "__main__":
    main()