#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline


def hist_theta(K, theta, AI, AI_power=1.0, AI_thresh=None):
    """
    Convert the pixelwise orientation field into a distribution over angle through [0,pi]
    Generates orientation histogram for a given patch
    """
    edges = np.linspace(0.0, np.pi, K + 1, endpoint=True)
    th = np.asarray(theta).ravel()
    ai = np.asarray(AI).ravel()

    # optional: filter out pixels with low AI
    if AI_thresh is not None: 
        keep = ai >= AI_thresh
        th, ai = th[keep], ai[keep]

    # optional: add weight by AI 
    w = np.power(ai, AI_power) if AI_power is not None else np.ones_like(ai, dtype=float) 
    
    # raw histogram 
    H, _ = np.histogram(th, bins=edges, weights=w)
    H = H.astype(float)

    # smooth so peaks are less sensitive to noise 
    H_smooth = gaussian_filter1d(H, 3, mode="wrap")
    H_smooth /= (H_smooth.sum() + 1e-12) # normalize histogram so it's a probability distribution over angles 
    alpha = 0.5 * (edges[:-1] + edges[1:]) # computes bin centre 
    return H_smooth, alpha, edges


# def circular_harmonics(alpha, H, M=10):
#     # model: f(α) = c0 + Σ_{k=1..M} [a_k cos(2kα) + b_k sin(2kα)]
#     cols = [np.ones_like(alpha)]
#     for k in range(1, M + 1):
#         cols += [np.cos(2 * k * alpha), np.sin(2 * k * alpha)]
#     X = np.column_stack(cols)  # (K, 2M+1)

#     coef, *_ = np.linalg.lstsq(X, H, rcond=None)
#     H_fit = X @ coef
#     H_fit = np.clip(H_fit, 0, None)
#     H_fit /= (H_fit.sum() + 1e-12)
#     return H_fit


def top_peaks(alpha, H_smooth, peak_distance, peak_num=5):
    idx, _ = find_peaks(H_smooth, distance=int(peak_distance))
    if idx.size == 0:
        return np.zeros(peak_num, dtype=float), np.zeros(peak_num, dtype=float)

    amps = H_smooth[idx]
    angs = alpha[idx]

    order = np.argsort(amps)[::-1]
    amps = amps[order][:peak_num]
    angs = angs[order][:peak_num]

    # pad to length peak_num
    if amps.size < peak_num:
        amps = np.pad(amps, (0, peak_num - amps.size), constant_values=0.0)
        angs = np.pad(angs, (0, peak_num - angs.size), constant_values=0.0)

    return angs.astype(float), amps.astype(float)


def roi_to_rgb_background_uint8(roi_np: np.ndarray) -> np.ndarray:
    """
    Background for overlays. Uses robust percentiles for display only.
    Output is always (H,W,3) uint8 and pixel-exact.
    """
    roi = np.asarray(roi_np, dtype=np.float32)
    p1, p99 = np.percentile(roi, [1, 99])
    bg01 = np.clip((roi - p1) / (p99 - p1 + 1e-12), 0, 1)
    bg8 = (bg01 * 255.0).astype(np.uint8)
    return np.stack([bg8, bg8, bg8], axis=-1)


def ref_colourwheel(theta_np: np.ndarray, alpha: np.ndarray):
    theta_vis = np.asarray(theta_np).ravel()
    mu2 = np.angle(np.mean(np.exp(1j * 2.0 * theta_vis)))
    theta_ref = (mu2 % (2 * np.pi)) / 2.0  # [0, π)

    h_alpha = ((alpha - theta_ref) % np.pi) / np.pi  # [0,1)
    h_alpha = (h_alpha - 0.25) % 1.0
    h_alpha = (1.0 - h_alpha) % 1.0

    rgb_alpha = hsv_to_rgb(np.stack([h_alpha, np.ones_like(h_alpha), np.ones_like(h_alpha)], axis=-1))
    return rgb_alpha, h_alpha


def save_theta_hist_figure(roi_np, H, alpha, edges, out_png: Path,theta_np=None):
    width = float(edges[1] - edges[0])
    custom_ticks = [0, np.pi / 6, np.pi / 3, np.pi / 2, 4 * np.pi / 6, 5 * np.pi / 6, np.pi]
    custom_labels = [
        "0",
        r"$\frac{1}{6}\pi$",
        r"$\frac{1}{3}\pi$",
        r"$\frac{1}{2}\pi$",
        r"$\frac{4}{6}\pi$",
        r"$\frac{5}{6}\pi$",
        r"$\pi$",
    ]

    fig = plt.figure(figsize=(12, 4), dpi=150)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.bar(alpha, H, width=0.9 * width, align="center")
    ax1.set_title("histogram of local theta")
    ax1.set_xlabel("θ (rad)")
    ax1.set_ylabel("probability")
    ax1.set_xticks(custom_ticks)
    ax1.set_xticklabels(custom_labels)

    ax2 = fig.add_subplot(1, 3, 2, projection="polar")
    ax2.bar(alpha, H, width=0.9 * width, bottom=0.0)
    ax2.set_title("polar histogram")
    ax2.set_xticks(custom_ticks)
    ax2.set_xticklabels(custom_labels)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(roi_np, cmap="gray", origin="upper")
    ax3.axis("off")
    ax3.set_title("patch preview")
    # --- overlay ---
    if theta_np is not None:
        rgb_alpha, _h_alpha = ref_colourwheel(theta_np, alpha)
        Hh, Wh = roi_np.shape
        yc, xc = Hh // 2, Wh // 2

        scale = 3 * float(min(Hh, Wh))

        for a, r_amp, col in zip(alpha, H, rgb_alpha):
            L = scale * float(r_amp)
            dx = L * np.cos(a)
            dy = -L * np.sin(a)  # minus to match image coordinates
            ax3.plot([xc - dx, xc + dx], [yc - dy, yc + dy], lw=2, color=col)
    # ----------------------------------------

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def render_lobes(
    roi_np: np.ndarray,
    alpha: np.ndarray,
    H_smooth: np.ndarray,
    peak_angles: np.ndarray,
    peak_amps: np.ndarray,
    h_alpha: np.ndarray,
    lobe_halfwidth_deg: float = 10.0,
) -> np.ndarray:
    """
    Pixel-exact overlay (H,W,3) uint8. Uses ROI as background and overlays a
    lobe-like shape using the fitted histogram, weighted around the top peaks.
    """
    H_, W_ = roi_np.shape
    out = roi_to_rgb_background_uint8(roi_np).astype(np.float32)

    K = alpha.size
    a2 = np.r_[alpha, alpha + np.pi]
    H2 = np.r_[H_smooth, H_smooth]

    w = np.deg2rad(lobe_halfwidth_deg)
    H_mix = np.zeros_like(H2, dtype=float)

    wamps = np.asarray(peak_amps, float)
    if wamps.size == 0 or np.all(wamps <= 1e-12):
        return out.astype(np.uint8)
    wamps = wamps / (wamps.max() + 1e-12)

    # Only use the top 2 peaks for the lobe mix (pseudo-FOD)
    use_n = min(2, len(peak_angles))
    for ang, wk in zip(np.mod(peak_angles[:use_n], np.pi), wamps[:use_n]):
        d1 = np.abs(np.angle(np.exp(1j * (a2 - ang))))
        d2 = np.abs(np.angle(np.exp(1j * (a2 - (ang + np.pi)))))
        m = (d1 <= w) | (d2 <= w)
        H_mix += wk * np.where(m, H2, 0.0)

    H_ax = 0.5 * (H_mix[:K] + H_mix[K:])
    H_ax = gaussian_filter1d(H_ax, 1, mode="wrap")
    H_ax = H_ax / (H_ax.max() + 1e-12)
    r = np.clip(H_ax, 0, 1)

    # π-periodic radial spline r(θ)
    alpha_s = np.r_[0.0, alpha, np.pi]
    r_s = np.r_[r[0], r, r[0]]
    order = np.argsort(alpha_s)
    spl_r = CubicSpline(alpha_s[order], r_s[order], bc_type="periodic")

    # π-periodic hue spline from h_alpha
    a_h = np.r_[0.0, alpha, np.pi]
    h_h = np.r_[h_alpha[0], h_alpha, h_alpha[0]]
    u_h = np.exp(1j * 2 * np.pi * h_h)
    ord_h = np.argsort(a_h)
    spl_h_re = CubicSpline(a_h[ord_h], u_h.real[ord_h], bc_type="periodic")
    spl_h_im = CubicSpline(a_h[ord_h], u_h.imag[ord_h], bc_type="periodic")

    yy, xx = np.mgrid[0:H_, 0:W_]
    yc, xc = (H_ - 1) / 2.0, (W_ - 1) / 2.0
    phi = (np.arctan2(-(yy - yc), (xx - xc)) + 2 * np.pi) % (2 * np.pi)
    phi_m = phi % np.pi

    r_img = np.clip(spl_r(phi_m), 0, 1)
    rad = np.hypot(xx - xc, yy - yc)
    scale = 0.5 * min(H_, W_)
    rad_n = rad / (scale + 1e-12)
    mask = rad_n <= r_img

    u_img = spl_h_re(phi_m) + 1j * spl_h_im(phi_m)
    h_img = (np.angle(u_img) % (2 * np.pi)) / (2 * np.pi)
    s_img = np.ones_like(h_img)
    v_img = np.clip(0.25 + 0.75 * r_img, 0, 1)

    rgb = hsv_to_rgb(np.stack([h_img, s_img, v_img], axis=-1))
    rgb8 = (np.clip(rgb, 0, 1) * 255.0).astype(np.float32)

    a = (mask.astype(np.float32) * 0.9)[..., None]
    out = a * rgb8 + (1.0 - a) * out
    return np.clip(out, 0, 255).astype(np.uint8)


def render_ellipsoid(
    roi_np: np.ndarray,
    eigenvals_np: np.ndarray,
    peak_angle_rad: float,
    alpha: float = 0.6,
    qmin: float = 0.02,
    qmax: float = 0.98,
    major_min_frac: float = 0.25,
    major_max_frac: float = 0.95,
    minor_min_frac: float = 0.08,
    minor_max_frac: float = 0.45,
) -> np.ndarray:
    """
    One centered DTI-like ellipsoid:
      - orientation = primary peak angle
      - axis lengths = robust summary of eigenvalues over patch
    Output is pixel-exact (H,W,3) uint8.
    """
    H, W = roi_np.shape
    out = roi_to_rgb_background_uint8(roi_np).astype(np.float32)

    ev = np.asarray(eigenvals_np, dtype=np.float32)
    if ev.ndim != 3 or ev.shape[2] != 2:
        raise ValueError(f"Expected eigenvals shape (H,W,2); got {ev.shape}")
    ev = np.nan_to_num(ev, nan=0.0, posinf=0.0, neginf=0.0)

    lam0 = ev[..., 0].ravel()
    lam1 = ev[..., 1].ravel()

    both = np.concatenate([lam0, lam1], axis=0)
    lo = float(np.quantile(both, qmin))
    hi = float(np.quantile(both, qmax))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
        lo = float(np.min(both))
        hi = float(np.max(both))
        if (hi - lo) < 1e-6:
            lo, hi = 0.0, 1.0

    def scale(x):
        return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    l0n = float(np.median(scale(lam0)))  # minor
    l1n = float(np.median(scale(lam1)))  # major

    base = float(min(H, W))
    major = (major_min_frac + l1n * (major_max_frac - major_min_frac)) * base
    minor = (minor_min_frac + l0n * (minor_max_frac - minor_min_frac)) * base
    major = max(2.0, major)
    minor = max(2.0, minor)

    hue = (float(peak_angle_rad) / np.pi) % 1.0
    col = (np.array(cm.hsv(hue)[:3], dtype=np.float32) * 255.0)

    yc = (H - 1) / 2.0
    xc = (W - 1) / 2.0

    ang = float(peak_angle_rad) + 0.5 * np.pi
    ca = np.cos(ang)
    sa = np.sin(ang)

    a = 0.5 * major  # semi-major
    b = 0.5 * minor  # semi-minor
    r = int(np.ceil(max(a, b))) + 2

    y_min = max(0, int(yc - r))
    y_max = min(H, int(yc + r + 1))
    x_min = max(0, int(xc - r))
    x_max = min(W, int(xc + r + 1))

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    dy = (yy - yc).astype(np.float32)
    dx = (xx - xc).astype(np.float32)

    xpr = ca * dx + sa * dy
    ypr = -sa * dx + ca * dy

    mask = (xpr * xpr) / (a * a + 1e-12) + (ypr * ypr) / (b * b + 1e-12) <= 1.0
    if np.any(mask):
        sub = out[y_min:y_max, x_min:x_max, :]
        sub[mask] = alpha * col + (1.0 - alpha) * sub[mask]
        out[y_min:y_max, x_min:x_max, :] = sub

    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-npy", required=True, type=Path)
    ap.add_argument("--theta-npy", required=True, type=Path)
    ap.add_argument("--ai-npy", required=True, type=Path)
    ap.add_argument("--eigenvals-npy", required=True, type=Path)

    ap.add_argument("--out-theta-hist-json", required=True, type=Path)
    ap.add_argument("--out-theta-hist-png", required=True, type=Path)
    ap.add_argument("--out-lobes-png", required=True, type=Path)
    ap.add_argument("--out-ellipsoids-png", required=True, type=Path)

    ap.add_argument("--bins", required=True, type=int)
    ap.add_argument("--ai-power", required=True, type=float)
    ap.add_argument("--ai-thresh", type=float, default=None)
    ap.add_argument("--harmonic-m", required=True, type=int)
    ap.add_argument("--peak-distance", required=True, type=int)

    args = ap.parse_args()

    roi = np.load(args.roi_npy)
    theta = np.load(args.theta_npy)
    AI = np.load(args.ai_npy)
    eigenvals = np.load(args.eigenvals_npy)

    H_smooth, alpha, edges = hist_theta(
        K=args.bins,
        theta=theta,
        AI=AI,
        AI_power=args.ai_power,
        AI_thresh=args.ai_thresh,
    )
#     H_smooth = circular_harmonics(alpha=alpha, H=H, M=args.harmonic_m)

    peak_angles, peak_amps = top_peaks(
        alpha=alpha,
        H_smooth=H_smooth,
        peak_distance=args.peak_distance,
        peak_num=5,
    )

    # theta_hist.json (top 5 peaks)
    theta_hist_dict = {
        "peak_angles": peak_angles.tolist(),
        "peak_amps": peak_amps.tolist(),
    }
    args.out_theta_hist_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_theta_hist_json.write_text(json.dumps(theta_hist_dict, indent=2))

    # theta_hist.png (3-panel QC; not pixel-exact required)
    save_theta_hist_figure(roi, H_smooth, alpha, edges, args.out_theta_hist_png, theta_np=theta)

    # lobes.png (pixel-exact overlay; use top-2 peaks for pseudo-FOD)
    _rgb_alpha, h_alpha = ref_colourwheel(theta, alpha)
    lobes_rgb = render_lobes(
        roi_np=roi,
        alpha=alpha,
        H_smooth=H_smooth,
        peak_angles=peak_angles,
        peak_amps=peak_amps,
        h_alpha=h_alpha,
    )
    args.out_lobes_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(lobes_rgb, mode="RGB").save(args.out_lobes_png)

    # ellipsoids.png (pixel-exact overlay; primary peak only)
    ell_rgb = render_ellipsoid(
        roi_np=roi,
        eigenvals_np=eigenvals,
        peak_angle_rad=float(peak_angles[0]),
        alpha=0.6,
    )
    args.out_ellipsoids_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ell_rgb, mode="RGB").save(args.out_ellipsoids_png)


if __name__ == "__main__":
    main()