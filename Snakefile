import zarr

configfile: "config.yml"

OUTDIR = config["out_dir"]

PATCH_H = int(config.get("patch_height", 100))
PATCH_W = int(config.get("patch_width", 100))
OVY = int(config.get("patch_overlap_y", 0))
OVX = int(config.get("patch_overlap_x", 0))
EDGE_MODE = config.get("edge_mode", "drop")

PATCH_DIR   = f"{OUTDIR}/patches"
PATCHES_TSV = f"{PATCH_DIR}/patches.tsv"
SHAPE_JSON  = f"{PATCH_DIR}/shape.json"

MASKS_DIR  = f"{OUTDIR}/masks"
MASK_NII   = f"{MASKS_DIR}/mask_level{config['zarr_level']}.nii.gz"

rule all:
    input:
        f"{OUTDIR}/stitched/orientation.png",
        f"{OUTDIR}/stitched/AI.png",
        f"{OUTDIR}/stitched/lobes.png",
        f"{OUTDIR}/stitched/ellipsoids.png",
        f"{OUTDIR}/stitched/theta.nii.gz",
        f"{OUTDIR}/stitched/AI.nii.gz",


checkpoint make_patches:
    output:
        tsv   = PATCHES_TSV,
        shape = SHAPE_JSON
    params:
        input_zarr_zip = config["input_zarr_zip"],
        zarr_level     = config["zarr_level"],
        channel_index  = config["channel_index"],
        patch_h        = PATCH_H,
        patch_w        = PATCH_W,
        ovy            = OVY,
        ovx            = OVX,
        edge_mode      = EDGE_MODE,
    shell:
        r"""
        python scripts/make_patches.py \
          --input-zarr-zip "{params.input_zarr_zip}" \
          --zarr-level {params.zarr_level} \
          --channel-index {params.channel_index} \
          --patch-height {params.patch_h} \
          --patch-width {params.patch_w} \
          --overlap-y {params.ovy} \
          --overlap-x {params.ovx} \
          --edge-mode {params.edge_mode} \
          --out-tsv "{output.tsv}" \
          --out-shape-json "{output.shape}"
        """


def read_patches(tsv_path):
    patches = []
    with open(tsv_path, "r") as f:
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


def patch_targets(wildcards):
    ck = checkpoints.make_patches.get(**wildcards)
    patches = read_patches(ck.output.tsv)

    outs = []
    for p in patches:
        tag = f"py{p['py']}_px{p['px']}"
        outs += [
            # required for ellipsoids
            f"{PATCH_DIR}/{tag}/st_outputs/theta.npy",
            f"{PATCH_DIR}/{tag}/st_outputs/AI.npy",

            # required for tile-stitched outputs
            f"{PATCH_DIR}/{tag}/qc/orientation.png",
            f"{PATCH_DIR}/{tag}/qc/AI.png",
            f"{PATCH_DIR}/{tag}/qc/lobes.png",
            f"{PATCH_DIR}/{tag}/qc/ellipsoids.png",

            f"{PATCH_DIR}/{tag}/qc/theta_hist.json",
        ]
    return outs

def load_mask_2d(mask_nii: str, input_zarr_zip: str, zarr_level: int, channel_index: int, out_mask_nii: str):
    """
    Load a binary NIfTI mask that was created on the level-5 OME-Zarr grid, resample it
    to the target OME-Zarr level (config['zarr_level']), and save to out_mask_nii.

    Conventions:
      - NIfTI mask is expected to be stored as (X,Y,1) or (X,Y) and will be converted
        internally to (Y,X) to match the patch/stitch code.
      - Resampling uses nearest-neighbour to preserve binariness.
    """
    import numpy as np
    import nibabel as nib
    from pathlib import Path
    from skimage.transform import resize
    from zarrnii import ZarrNii

    # --- target shape from OME-Zarr at requested level ---
    store = zarr.storage.ZipStore(input_zarr_zip, mode="r")
    zn = ZarrNii.from_ome_zarr(store_or_path=store, level=int(zarr_level))
    arr = zn.darr

    # mirror make_patches/structure_tensor squeeze logic
    if arr.ndim >= 3:
        try:
            arr = arr[:, int(channel_index), ...]
        except Exception:
            pass
    arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D OME-Zarr slice after squeeze; got shape {arr.shape}")
    Ht, Wt = map(int, arr.shape)  # (Y,X)

    # --- load source mask (level-5 grid) ---
    nii = nib.load(mask_nii)
    src = np.asanyarray(nii.dataobj)
    src = np.squeeze(src)
    if src.ndim != 2:
        raise ValueError(f"Mask NIfTI must be 2D after squeeze; got shape {src.shape}")

    # --- resample to target (Y,X) with nearest neighbour ---
    m_rs = resize(
        src.astype(np.float32),
        (Ht, Wt),
        order=0,              # nearest
        mode="edge",
        preserve_range=True,
        anti_aliasing=False,
    )
    m_bin = (m_rs > 0.5).astype(np.uint8)

    # --- save as NIfTI in pipeline convention (X,Y,1) ---
    out_p = Path(out_mask_nii)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out = m_bin.T # Convert (X,Y) -> (Y,X)
    out_xyz1 = out[:, :, None]  # (Y,X,1)

    out_nii = nib.Nifti1Image(out_xyz1, affine=nii.affine)
    out_nii.set_data_dtype(np.uint8)
    nib.save(out_nii, str(out_p))

rule compute_structure_tensor:
    input:
        tsv = PATCHES_TSV
    output:
        roi      = f"{PATCH_DIR}/py{{py}}_px{{px}}/st_outputs/roi.npy",
        J        = f"{PATCH_DIR}/py{{py}}_px{{px}}/st_outputs/J.npy",
        theta    = f"{PATCH_DIR}/py{{py}}_px{{px}}/st_outputs/theta.npy",
        AI       = f"{PATCH_DIR}/py{{py}}_px{{px}}/st_outputs/AI.npy",
        eigenvals= f"{PATCH_DIR}/py{{py}}_px{{px}}/st_outputs/eigenvals.npy",
        ori_png  = f"{PATCH_DIR}/py{{py}}_px{{px}}/qc/orientation.png",
        ai_png   = f"{PATCH_DIR}/py{{py}}_px{{px}}/qc/AI.png",
    params:
        input_zarr_zip = config["input_zarr_zip"],
        zarr_level     = config["zarr_level"],
        channel_index  = config["channel_index"],
        sigma_g        = config["sigma_g"],
        sigma_w        = config["sigma_w"],
        truncate       = config["truncate"],
        out_dir        = lambda wc: f"{PATCH_DIR}/py{wc.py}_px{wc.px}",
    run:
        patches = read_patches(input.tsv)
        py = int(wildcards.py)
        px = int(wildcards.px)
        m = [p for p in patches if p["py"] == py and p["px"] == px]
        if not m:
            raise ValueError(f"Patch not found in {input.tsv}: py={py}, px={px}")
        p = m[0]

        shell(
            r"""
            python scripts/structure_tensor.py \
              --input-zarr-zip "{params.input_zarr_zip}" \
              --zarr-level {params.zarr_level} \
              --channel-index {params.channel_index} \
              --patch {y0} {y1} {x0} {x1} \
              --sigma-g {params.sigma_g} \
              --sigma-w {params.sigma_w} \
              --truncate {params.truncate} \
              --out-dir "{params.out_dir}"
            """.format(params=params, **p)
        )


rule qc_theta_vis:
    input:
        roi   = f"{PATCH_DIR}/py{{py}}_px{{px}}/st_outputs/roi.npy",
        theta = f"{PATCH_DIR}/py{{py}}_px{{px}}/st_outputs/theta.npy",
        AI    = f"{PATCH_DIR}/py{{py}}_px{{px}}/st_outputs/AI.npy",
        eigenvals= f"{PATCH_DIR}/py{{py}}_px{{px}}/st_outputs/eigenvals.npy",
    output:
        theta_hist_json = f"{PATCH_DIR}/py{{py}}_px{{px}}/qc/theta_hist.json",
        theta_hist_png = f"{PATCH_DIR}/py{{py}}_px{{px}}/qc/theta_hist.png",
        lobes_png      = f"{PATCH_DIR}/py{{py}}_px{{px}}/qc/lobes.png",
        ellipsoids_png = f"{PATCH_DIR}/py{{py}}_px{{px}}/qc/ellipsoids.png",
    params:
        bins          = config["bins"],
        AI_power      = config["AI_power"],
        AI_thresh     = config["AI_thresh"],
        harmonic_M    = config["harmonic_M"],
        peak_distance = config["peak_distance"],
    run:
        cmd = [
            "python", "scripts/peak_vis.py",
            "--roi-npy", input.roi,
            "--theta-npy", input.theta,
            "--ai-npy", input.AI,
            "--eigenvals-npy", input.eigenvals,
            "--out-theta-hist-json", output.theta_hist_json,
            "--out-theta-hist-png", output.theta_hist_png,
            "--out-lobes-png", output.lobes_png,
            "--out-ellipsoids-png", output.ellipsoids_png,
            "--bins", str(params.bins),
            "--ai-power", str(params.AI_power),
            "--harmonic-m", str(params.harmonic_M),
            "--peak-distance", str(params.peak_distance),
        ]
        if params.AI_thresh is not None:
            cmd += ["--ai-thresh", str(params.AI_thresh)]
        shell(" ".join(cmd))

rule resample_mask:
    output:
        mask = MASK_NII
    params:
        mask_nii = lambda wc: config.get("mask_nii", None),
        input_zarr_zip    = config["input_zarr_zip"],
        zarr_level        = config["zarr_level"],
        channel_index     = config["channel_index"],
    run:
        if params.mask_nii is None:
            raise ValueError("config.yml is missing 'mask_nii' (path to your level-5 NIfTI mask).")

        load_mask_2d(
            mask_nii=params.mask_nii,
            input_zarr_zip=params.input_zarr_zip,
            zarr_level=int(params.zarr_level),
            channel_index=int(params.channel_index),
            out_mask_nii=output.mask,
        )

rule stitch_maps:
    input:
        tsv      = PATCHES_TSV,
        shape    = SHAPE_JSON,
        allpatch = patch_targets,
        mask = MASK_NII,
    output:
        orientation = f"{OUTDIR}/stitched/orientation.png",
        AI_png      = f"{OUTDIR}/stitched/AI.png",
        lobes       = f"{OUTDIR}/stitched/lobes.png",
        ellipsoids  = f"{OUTDIR}/stitched/ellipsoids.png",
        theta_nii   = f"{OUTDIR}/stitched/theta.nii.gz",
        AI_nii      = f"{OUTDIR}/stitched/AI.nii.gz",
    params:
        patch_root     = PATCH_DIR,
        input_zarr_zip = config["input_zarr_zip"],
        zarr_level     = config["zarr_level"],
    shell:
        r"""
        python scripts/stitch_maps.py \
          --patches-tsv "{input.tsv}" \
          --shape-json "{input.shape}" \
          --patch-root "{params.patch_root}" \
          --input-zarr-zip "{params.input_zarr_zip}" \
          --zarr-level {params.zarr_level} \
          --mask-nifti "{input.mask}" \
          --out-orientation "{output.orientation}" \
          --out-ai "{output.AI_png}" \
          --out-lobes "{output.lobes}" \
          --out-ellipsoids "{output.ellipsoids}" \
          --out-theta-nifti "{output.theta_nii}" \
          --out-ai-nifti "{output.AI_nii}"
        """