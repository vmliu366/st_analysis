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


rule all:
    input:
        f"{OUTDIR}/stitched/orientation.png",
        f"{OUTDIR}/stitched/AI.png",
        f"{OUTDIR}/stitched/lobes.png",
        f"{OUTDIR}/stitched/ellipsoids.png",


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


rule qc_lobe_vis:
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


rule stitch_maps:
    input:
        tsv      = PATCHES_TSV,
        shape    = SHAPE_JSON,
        allpatch = patch_targets,
    output:
        orientation = f"{OUTDIR}/stitched/orientation.png",
        AI          = f"{OUTDIR}/stitched/AI.png",
        lobes       = f"{OUTDIR}/stitched/lobes.png",
        ellipsoids  = f"{OUTDIR}/stitched/ellipsoids.png",
    params:
        patch_root = PATCH_DIR,
    shell:
        r"""
        python scripts/stitch_maps.py \
          --patches-tsv "{input.tsv}" \
          --shape-json "{input.shape}" \
          --patch-root "{params.patch_root}" \
          --out-orientation "{output.orientation}" \
          --out-ai "{output.AI}" \
          --out-lobes "{output.lobes}" \
          --out-ellipsoids "{output.ellipsoids}"
        """