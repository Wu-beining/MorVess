
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dispersion & Sparsity analysis for 3D medical segmentation datasets (AIIB & PARSE).

What it does
------------
- Scans AIIB2023 and/or PARSE2022 dataset folders to find label (.nii.gz) files
- Computes per-case metrics:
    * Sparsity Index (SI)              = foreground_voxels / total_voxels
    * Dispersion Index (DI)            = mean distance of FG voxels to centroid,
                                         normalized by mean distance of a same-volume sphere (3R/4)
    * PCA eigenvalue-based shape stats = linearity, planarity, sphericity
    * Slice coverage ratios            = fraction of slices containing FG along (H, W, D)
    * Fragmentation (components)       = number of 26-connected components
    * Physical volume (mm^3)
- Saves metrics to CSV and makes a scatter plot (log10 SI vs DI).

Assumptions
-----------
- AIIB2023 layout: base_dir/img/*.nii.gz + base_dir/gt/*.nii.gz with matching names
- PARSE2022 layout: base_dir/<case_id>/{image,label}/<case_id>.nii.gz
- Labels are binary or multi-class; we binarize with (label > 0)

Usage
-----
python dispersion_analysis.py \
  --aiib /home/ET/bnwu/MA-SAM/data/AIIB23_Train_T1 \
  --parse /home/ET/bnwu/MA-SAM/data/parse2022/train \
  --outdir ./dispersion_out \
  --sample-max 250000
"""
import os
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import label as cc_label

def read_mask_and_spacing(nii_path: str):
    """Read NIfTI, return (mask HxWxD np.uint8), spacing tuple (sy, sx, sz) in mm.
       Note: we transpose SimpleITK's (D,H,W) array -> (H,W,D).
             Spacing from sitk is (sx, sy, sz) mapping to (sy, sx, sz) after transpose.
    """
    img = sitk.ReadImage(str(nii_path))
    arr = sitk.GetArrayFromImage(img)  # (D,H,W)
    arr = arr.transpose(1, 2, 0)       # (H,W,D)
    # binarize: foreground > 0
    mask = (arr > 0).astype(np.uint8)

    sx, sy, sz = img.GetSpacing()      # (x,y,z) in mm
    spacing = (sy, sx, sz)             # align with (H,W,D)
    return mask, spacing

def compute_metrics(mask: np.ndarray, spacing, sample_max:int=250000):
    """Compute metrics for one 3D binary mask (H,W,D) with physical spacing."""
    H, W, D = mask.shape
    total = H * W * D
    fg_inds = np.argwhere(mask > 0)  # (N,3) -> (y,x,z) index

    si = float(fg_inds.shape[0]) / float(total) if total > 0 else 0.0
    if fg_inds.shape[0] == 0:
        return dict(si=0.0, di=np.nan, volume_mm3=0.0, eig1=np.nan, eig2=np.nan, eig3=np.nan,
                    linearity=np.nan, planarity=np.nan, sphericity=np.nan,
                    covH=0.0, covW=0.0, covD=0.0, components=0)

    # Physical coordinates (mm)
    sy, sx, sz = spacing
    coords_mm = fg_inds.astype(np.float64)
    coords_mm[:, 0] *= sy
    coords_mm[:, 1] *= sx
    coords_mm[:, 2] *= sz

    # Physical volume (mm^3)
    voxel_vol = sy * sx * sz
    volume_mm3 = fg_inds.shape[0] * voxel_vol

    # Dispersion Index (DI): mean radial distance / (0.75 * R_sphere) with same volume
    centroid = coords_mm.mean(axis=0, keepdims=True)
    r = np.linalg.norm(coords_mm - centroid, axis=1)
    r_mean = float(r.mean())

    # Sphere radius with same volume: V = 4/3*pi*R^3 => R = (3V/4pi)^(1/3)
    R_sphere = (3.0 * volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    mean_r_sphere = 0.75 * R_sphere if R_sphere > 0 else np.nan
    di = float(r_mean / mean_r_sphere) if mean_r_sphere and mean_r_sphere > 0 else np.nan

    # PCA eigenvalues (shape): sample if too many voxels
    N = coords_mm.shape[0]
    if sample_max is not None and N > sample_max:
        sel = np.random.choice(N, size=sample_max, replace=False)
        X = coords_mm[sel] - centroid
    else:
        X = coords_mm - centroid
    # Covariance eigenvalues (descending)
    C = np.cov(X.T)
    evals, _ = np.linalg.eigh(C)
    evals = np.sort(evals)[::-1]  # λ1 ≥ λ2 ≥ λ3
    eig1, eig2, eig3 = evals.tolist()

    # Linearity / Planarity / Sphericity (cf. tensor voting style descriptors)
    linearity = (eig1 - eig2) / eig1 if eig1 > 0 else np.nan
    planarity = (eig2 - eig3) / eig1 if eig1 > 0 else np.nan
    sphericity = eig3 / eig1 if eig1 > 0 else np.nan

    # Slice coverage ratios
    covH = float(np.count_nonzero(mask.sum(axis=(1,2)) > 0)) / H
    covW = float(np.count_nonzero(mask.sum(axis=(0,2)) > 0)) / W
    covD = float(np.count_nonzero(mask.sum(axis=(0,1)) > 0)) / D

    # Fragmentation: number of 26-connected components
    struct = np.ones((3,3,3), dtype=np.uint8)
    comps, ncomp = cc_label(mask, structure=struct)

    return dict(si=si, di=di, volume_mm3=volume_mm3,
                eig1=eig1, eig2=eig2, eig3=eig3,
                linearity=linearity, planarity=planarity, sphericity=sphericity,
                covH=covH, covW=covW, covD=covD, components=int(ncomp))

def scan_aiib(base_dir: str):
    """Return list of (case_name, label_path) for AIIB2023 layout."""
    gt_dir = Path(base_dir) / 'gt'
    if not gt_dir.exists():
        return []
    label_paths = sorted([p for p in gt_dir.glob('*.nii.gz')])
    return [(p.name, str(p)) for p in label_paths]

def scan_parse(base_dir: str):
    """Return list of (case_name, label_path) for PARSE2022 layout."""
    base = Path(base_dir)
    out = []
    if not base.exists():
        return out
    for case_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        case_id = case_dir.name
        lp = case_dir / 'label' / f'{case_id}.nii.gz'
        if lp.exists():
            out.append((case_id, str(lp)))
    return out

def make_plot(df: pd.DataFrame, out_png: str):
    plt.figure(figsize=(8,6), dpi=160)
    for ds, sub in df.groupby('dataset'):
        plt.scatter(sub['si'], sub['di'], label=ds)
    plt.xscale('log')
    plt.xlabel('Sparsity Index (log scale)')
    plt.ylabel('Dispersion Index (normalized mean radius)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aiib', type=str, default='', help='Path to AIIB23_Train_T1 base dir (optional)')
    ap.add_argument('--parse', type=str, default='', help='Path to PARSE2022/train base dir (optional)')
    ap.add_argument('--outdir', type=str, default='./dispersion_out')
    ap.add_argument('--sample-max', type=int, default=250000, help='Max voxels to sample for PCA (speed)')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.aiib:
        for case_name, lp in scan_aiib(args.aiib):
            mask, spacing = read_mask_and_spacing(lp)
            m = compute_metrics(mask, spacing, sample_max=args.sample_max)
            m.update(dict(dataset='AIIB', case=case_name))
            rows.append(m)

    if args.parse:
        for case_name, lp in scan_parse(args.parse):
            mask, spacing = read_mask_and_spacing(lp)
            m = compute_metrics(mask, spacing, sample_max=args.sample_max)
            m.update(dict(dataset='PARSE', case=case_name))
            rows.append(m)

    if not rows:
        print("No data found. Please pass --aiib and/or --parse with valid paths.")
        return

    df = pd.DataFrame(rows)
    csv_path = outdir / 'dataset_metrics.csv'
    df.to_csv(csv_path, index=False)

    png_path = outdir / 'sparsity_vs_dispersion.png'
    make_plot(df, str(png_path))

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")
    print("Columns:", list(df.columns))

if __name__ == '__main__':
    main()
