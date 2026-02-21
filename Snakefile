configfile: "config.yml"

from pathlib import Path

OUTDIR = config["out_dir"]

PATCH_H = int(config.get("patch_height", 100))
PATCH_W = int(config.get("patch_width", 100))
OVY = int(config.get("patch_overlap_y", 0))
OVX = int(config.get("patch_overlap_x", 0))
EDGE_MODE = config.get("edge_mode", "drop")

PATCH_DIR   = f"{OUTDIR}/patches"
PATCHES_TSV = f"{PATCH_DIR}/patches.tsv"
SHAPE_JSON  = f"{PATCH_DIR}/shape.json"


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


def patch_targets(wildcards):
    # This function is evaluated after the checkpoint completes
    ck = checkpoints.make_patches.get(**wildcards)
    patches = read_patches(ck.output.tsv)

    outs = []
    for p in patches:
        tag = f"py{p['py']}_px{p['px']}"
        outs += [
            f"{PATCH_DIR}/{tag}/arrays/theta.npy",
            f"{PATCH_DIR}/{tag}/arrays/AI.npy",
            f"{PATCH_DIR}/{tag}/arrays/eigenvals.npy",
            f"{PATCH_DIR}/{tag}/figures/orientation_polar.png",
            f"{PATCH_DIR}/{tag}/summary/peaks.json",
        ]
    return outs


rule all:
    input:
        f"{OUTDIR}/stitched/theta_hist.png",
        f"{OUTDIR}/stitched/theta_polar.png",
        f"{OUTDIR}/stitched/orientation_polar.png",
        f"{OUTDIR}/stitched/lobes.png",
        f"{OUTDIR}/stitched/ellipsoids.png",


# Ensures all per-patch outputs exist (dynamic after checkpoint)
rule per_patch_all:
    input:
        patch_targets


rule compute_structure_tensor:
    input:
        tsv = PATCHES_TSV
    output:
        roi2      = f"{PATCH_DIR}/py{{py}}_px{{px}}/arrays/roi2.npy",
        J         = f"{PATCH_DIR}/py{{py}}_px{{px}}/arrays/J.npy",
        theta     = f"{PATCH_DIR}/py{{py}}_px{{px}}/arrays/theta.npy",
        AI        = f"{PATCH_DIR}/py{{py}}_px{{px}}/arrays/AI.npy",
        eigenvals = f"{PATCH_DIR}/py{{py}}_px{{px}}/arrays/eigenvals.npy",
        ori_png   = f"{PATCH_DIR}/py{{py}}_px{{px}}/figures/orientation_polar.png",
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


rule peaks_summary_patch:
    input:
        theta = f"{PATCH_DIR}/py{{py}}_px{{px}}/arrays/theta.npy",
        AI    = f"{PATCH_DIR}/py{{py}}_px{{px}}/arrays/AI.npy",
    output:
        js = f"{PATCH_DIR}/py{{py}}_px{{px}}/summary/peaks.json",
    params:
        bins          = config["bins"],
        AI_power      = config["AI_power"],
        AI_thresh     = config["AI_thresh"],
        harmonic_M    = config["harmonic_M"],
        peak_distance = config["peak_distance"],
    run:
        cmd = [
            "python", "scripts/peak_vis.py",
            "--theta-npy", input.theta,
            "--ai-npy", input.AI,
            "--out-json", output.js,
            "--bins", str(params.bins),
            "--ai-power", str(params.AI_power),
            "--harmonic-m", str(params.harmonic_M),
            "--peak-distance", str(params.peak_distance),
        ]
        if params.AI_thresh is not None:
            cmd += ["--ai-thresh", str(params.AI_thresh)]
        shell(" ".join(cmd))


# Final stitch step: produces full-size HxW outputs
rule stitch_fullsize:
    input:
        tsv   = PATCHES_TSV,
        shape = SHAPE_JSON,
        # force all per-patch outputs to exist first
        allpatch = rules.per_patch_all.input,
    output:
        theta_hist       = f"{OUTDIR}/stitched/theta_hist.png",
        theta_polar      = f"{OUTDIR}/stitched/theta_polar.png",
        orientation_polar= f"{OUTDIR}/stitched/orientation_polar.png",
        lobes            = f"{OUTDIR}/stitched/lobes.png",
        ellipsoids       = f"{OUTDIR}/stitched/ellipsoids.png",
    params:
        patch_root = PATCH_DIR,
    shell:
        r"""
        python scripts/stitch_maps.py \
          --patches-tsv "{input.tsv}" \
          --shape-json "{input.shape}" \
          --patch-root "{params.patch_root}" \
          --out-theta-hist "{output.theta_hist}" \
          --out-theta-polar "{output.theta_polar}" \
          --out-orientation-polar "{output.orientation_polar}" \
          --out-lobes "{output.lobes}" \
          --out-ellipsoids "{output.ellipsoids}"
        """