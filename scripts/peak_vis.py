#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


def hist_theta(K, theta, AI, AI_power=1.0, AI_thresh=None):
    edges = np.linspace(0.0, np.pi, K + 1, endpoint=True)
    th = theta.ravel()
    ai = AI.ravel()

    if AI_thresh is not None:
        keep = ai >= AI_thresh
        th, ai = th[keep], ai[keep]

    w = np.power(ai, AI_power)
    H, _ = np.histogram(th, bins=edges, weights=w)
    H = H.astype(float)
    H /= (H.sum() + 1e-12)
    alpha = 0.5 * (edges[:-1] + edges[1:])
    return H, alpha


def circular_harmonics(alpha, H, M=10):
    cols = [np.ones_like(alpha)]
    for k in range(1, M + 1):
        cols += [np.cos(2 * k * alpha), np.sin(2 * k * alpha)]
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, H, rcond=None)
    H_fit = X @ coef
    H_fit = np.clip(H_fit, 0, None)
    H_fit /= (H_fit.sum() + 1e-12)
    return H_fit


def top_peaks(alpha, H_fit, peak_distance, peak_num=2):
    idx, _ = find_peaks(H_fit, distance=int(peak_distance))
    if idx.size == 0:
        return [0.0] * peak_num, [0.0] * peak_num

    amps = H_fit[idx]
    angs = alpha[idx]

    order = np.argsort(amps)[::-1]
    amps = amps[order][:peak_num]
    angs = angs[order][:peak_num]

    # pad to fixed length
    amps = np.pad(amps, (0, max(0, peak_num - len(amps))), constant_values=0.0)
    angs = np.pad(angs, (0, max(0, peak_num - len(angs))), constant_values=0.0)
    return angs.tolist(), amps.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-npy", required=True, type=Path)
    ap.add_argument("--ai-npy", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)

    ap.add_argument("--bins", required=True, type=int)
    ap.add_argument("--ai-power", required=True, type=float)
    ap.add_argument("--ai-thresh", type=float, default=None)

    ap.add_argument("--harmonic-m", required=True, type=int)
    ap.add_argument("--peak-distance", required=True, type=int)

    args = ap.parse_args()

    theta = np.load(args.theta_npy)
    AI = np.load(args.ai_npy)

    H, alpha = hist_theta(
        K=args.bins,
        theta=theta,
        AI=AI,
        AI_power=args.ai_power,
        AI_thresh=args.ai_thresh,
    )
    H_fit = circular_harmonics(alpha=alpha, H=H, M=args.harmonic_m)

    peak_angles, peak_amps = top_peaks(alpha, H_fit, peak_distance=args.peak_distance, peak_num=2)

    p1, p2 = float(peak_amps[0]), float(peak_amps[1])
    ratio = (p2 / p1) if p1 > 1e-12 else 0.0

    out = {
        "peak_angles": [float(peak_angles[0]), float(peak_angles[1])],  # radians, [0, pi)
        "peak_amps": [p1, p2],
        "peak_ratio": float(ratio),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out))


if __name__ == "__main__":
    main()