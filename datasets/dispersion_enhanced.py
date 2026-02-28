
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced dispersion visualization & HTML report.

Features
--------
1) Size-encoded scatter plots:
   - scatter_size_by_volume.png     (point size ~ volume_mm3)
   - scatter_size_by_components.png (point size ~ connected components)

2) Vessel-like auto-flagging (quantile or manual thresholds).
   Reuses an existing 'vessel_like' column if present; otherwise computes.

3) HTML case report:
   - Generates overlays (three orthogonal mid-slices) for top-K vessel-like cases
     or top-K by highest DI if --vessel-first is not used.
   - Writes report.html with thumbnails and key metrics.

Dataset layouts (same as analysis script)
----------------------------------------
AIIB:  <AIIB_BASE>/img/*.nii.gz  and  <AIIB_BASE>/gt/*.nii.gz
PARSE: <PARSE_BASE>/<case>/image/<case>.nii.gz  and  .../label/<case>.nii.gz

Usage
-----
python dispersion_enhanced.py \
  --csv ./dispersion_out/dataset_metrics.csv \
  --aiib /home/ET/bnwu/MA-SAM/data/AIIB23_Train_T1 \
  --parse /home/ET/bnwu/MA-SAM/data/parse2022/train \
  --outdir ./dispersion_out \
  --k 24 --auto --vessel-first
"""
import os, math, io
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk
from typing import Tuple, Optional

# ------------ I/O helpers ------------
def load_metrics(csv_path:str)->pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def ensure_flag_column(df: pd.DataFrame, auto: bool, si_max=None, di_min=None, lin_min=None, sph_max=None):
    if 'vessel_like' in df.columns and df['vessel_like'].notna().any():
        return df, dict(si_max=si_max, di_min=di_min, lin_min=lin_min, sph_max=sph_max, source='existing')
    # Otherwise compute thresholds
    sub = df[['si','di','linearity','sphericity']].copy()
    if auto:
        q = sub.quantile([0.2, 0.8]).to_dict()
        si_max = float(q['si'][0.2]) if si_max is None else si_max
        di_min = float(q['di'][0.8]) if di_min is None else di_min
        lin_min = float(q['linearity'][0.8]) if lin_min is None and 'linearity' in q else lin_min
        sph_max = float(q['sphericity'][0.2]) if sph_max is None and 'sphericity' in q else sph_max
    else:
        if si_max is None: si_max = sub['si'].quantile(0.3)
        if di_min is None: di_min = sub['di'].quantile(0.7)
        if lin_min is None and 'linearity' in sub: lin_min = sub['linearity'].quantile(0.7)
        if sph_max is None and 'sphericity' in sub: sph_max = sub['sphericity'].quantile(0.3)
    crit = (df['si'] <= si_max) & (df['di'] >= di_min)
    if 'linearity' in df.columns and lin_min is not None:
        crit = crit & (df['linearity'] >= lin_min)
    if 'sphericity' in df.columns and sph_max is not None:
        crit = crit & (df['sphericity'] <= sph_max)
    out = df.copy()
    out['vessel_like'] = crit.astype(int)
    return out, dict(si_max=si_max, di_min=di_min, lin_min=lin_min, sph_max=sph_max, source='computed')

# ------------ Plots ------------
def _norm_sizes(x, smin=10, smax=120):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=np.nanmax(x[np.isfinite(x)]) if np.isfinite(x).any() else 1.0)
    if x.size == 0: return np.array([smin])
    lo, hi = np.percentile(x[x>0], 5), np.percentile(x[x>0], 95) if (x>0).any() else (1,1)
    lo = lo if lo>0 else 1e-12
    y = (x - lo) / (hi - lo + 1e-12)
    y = np.clip(y, 0, 1)
    return smin + y * (smax - smin)

def scatter_size(df: pd.DataFrame, by: str, out_png: str):
    plt.figure(figsize=(8,6), dpi=160)
    for ds, g in df.groupby('dataset'):
        sizes = _norm_sizes(g[by].values)
        plt.scatter(g['si'], g['di'], s=sizes, label=ds)
    plt.xscale('log')
    plt.xlabel('Sparsity Index (log scale)')
    plt.ylabel('Dispersion Index (normalized mean radius)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)

# ------------ NIfTI reading & overlay ------------
def read_nifti_label_path(dataset: str, case: str, aiib_base: Optional[str], parse_base: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if dataset.upper().startswith('AIIB') and aiib_base:
        label_path = Path(aiib_base) / 'label' / f'{case}'  # fallback
        gt_path = Path(aiib_base) / 'gt' / f'{case}'
        if gt_path.exists(): label_path = gt_path
        img_path = Path(aiib_base) / 'img' / f'{case}'
        return (str(img_path) if img_path.exists() else None, str(label_path) if label_path.exists() else None)
    if dataset.upper().startswith('PARSE') and parse_base:
        case_id = case.split('.')[0]
        img_path = Path(parse_base) / case_id / 'image' / f'{case_id}.nii.gz'
        label_path = Path(parse_base) / case_id / 'label' / f'{case_id}.nii.gz'
        return (str(img_path) if img_path.exists() else None, str(label_path) if label_path.exists() else None)
    return (None, None)

def load_nii(path: str):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).transpose(1,2,0)  # (H,W,D)
    return arr, img

def auto_window(img_arr: np.ndarray):
    # robust windowing using percentiles to [0,1]
    lo, hi = np.percentile(img_arr, 2), np.percentile(img_arr, 98)
    if hi <= lo: hi = lo + 1e-6
    win = (img_arr - lo) / (hi - lo)
    win = np.clip(win, 0, 1)
    return win

def save_overlays(img_arr: np.ndarray, lab_arr: np.ndarray, out_prefix: Path):
    H, W, D = img_arr.shape
    mids = (H//2, W//2, D//2)
    planes = [
        ('axial',   ('z', mids[2])),
        ('sagittal',('x', mids[1])),
        ('coronal', ('y', mids[0])),
    ]
    img_arr = auto_window(img_arr)
    lab = (lab_arr > 0).astype(np.uint8)

    paths = []
    for name, (axis, idx) in planes:
        plt.figure(figsize=(4,4), dpi=150)
        if axis=='z':
            plt.imshow(img_arr[:,:,idx])
            plt.contour(lab[:,:,idx], levels=[0.5], linewidths=1)
        elif axis=='x':
            plt.imshow(img_arr[:,idx,:])
            plt.contour(lab[:,idx,:], levels=[0.5], linewidths=1)
        else:
            plt.imshow(img_arr[idx,:,:])
            plt.contour(lab[idx,:,:], levels=[0.5], linewidths=1)
        plt.axis('off')
        p = Path(f"{out_prefix}_{name}.png")
        plt.tight_layout(pad=0)
        plt.savefig(p, bbox_inches='tight', pad_inches=0)
        plt.close()
        paths.append(str(p))
    return paths

# ------------ HTML report ------------
def write_html(cases, out_html: str):
    html = io.StringIO()
    html.write("<html><head><meta charset='utf-8'><title>Dispersion Report</title></head><body>")
    html.write("<h2>Vessel-like / High-DI Cases</h2>")
    html.write("<table border='1' cellpadding='6' cellspacing='0'>")
    html.write("<tr><th>Dataset</th><th>Case</th><th>SI</th><th>DI</th><th>Linearity</th><th>Sphericity</th><th>Components</th><th>Overlays</th></tr>")
    for row in cases:
        imgs = "".join([f"<img src='{Path(p).name}' width='160'/>" for p in row['pngs']])
        html.write(
            f"<tr><td>{row['dataset']}</td><td>{row['case']}</td>"
            f"<td>{row['si']:.4g}</td><td>{row['di']:.4g}</td>"
            f"<td>{row.get('linearity', float('nan')):.3g}</td>"
            f"<td>{row.get('sphericity', float('nan')):.3g}</td>"
            f"<td>{row.get('components', 0)}</td><td>{imgs}</td></tr>"
        )
    html.write("</table></body></html>")
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html.getvalue())

# ------------ Main ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--aiib', type=str, default='')
    ap.add_argument('--parse', type=str, default='')
    ap.add_argument('--outdir', type=str, default='./dispersion_out')
    ap.add_argument('--auto', action='store_true')
    ap.add_argument('--si-max', type=float, default=None)
    ap.add_argument('--di-min', type=float, default=None)
    ap.add_argument('--lin-min', type=float, default=None)
    ap.add_argument('--sph-max', type=float, default=None)
    ap.add_argument('--k', type=int, default=24, help='Top-K cases to render in HTML')
    ap.add_argument('--vessel-first', action='store_true', help='Prefer vessel_like=1 for report selection')
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(args.csv)
    df, thr = ensure_flag_column(df, args.auto, args.si_max, args.di_min, args.lin_min, args.sph_max)

    # 1) size-encoded scatters
    if 'volume_mm3' in df.columns:
        scatter_size(df, 'volume_mm3', str(outdir / 'scatter_size_by_volume.png'))
    if 'components' in df.columns:
        scatter_size(df, 'components', str(outdir / 'scatter_size_by_components.png'))

    # 2) HTML report selection
    if args.vessel_first and 'vessel_like' in df.columns:
        pool = df[df['vessel_like']==1].copy()
        if len(pool) < args.k:
            pool = pd.concat([pool, df.sort_values('di', ascending=False)]).drop_duplicates(subset=['dataset','case'])
        pool = pool.sort_values(['vessel_like','di'], ascending=[False, False]).head(args.k)
    else:
        pool = df.sort_values('di', ascending=False).head(args.k)

    # 3) Generate overlays
    rendered = []
    for _, r in pool.iterrows():
        img_p, lab_p = read_nifti_label_path(str(r['dataset']), str(r['case']), args.aiib, args.parse)
        if not (lab_p and Path(lab_p).exists()):
            continue
        try:
            lab_arr, _ = load_nii(lab_p)
            if img_p and Path(img_p).exists():
                img_arr, _ = load_nii(img_p)
                # shape guards
                if img_arr.shape != lab_arr.shape:
                    # naive center crop/pad to match label
                    H,W,D = lab_arr.shape
                    img_arr = img_arr[:H,:W,:D]
            else:
                img_arr = lab_arr.astype(np.float32)
            prefix = outdir / f"{Path(str(r['dataset']).replace(' ','_'))}_{Path(str(r['case'])).stem}"
            pngs = save_overlays(img_arr, lab_arr, prefix)
            rendered.append(dict(dataset=r['dataset'], case=r['case'], pngs=pngs,
                                 si=float(r['si']), di=float(r['di']),
                                 linearity=float(r.get('linearity', np.nan)) if 'linearity' in r else np.nan,
                                 sphericity=float(r.get('sphericity', np.nan)) if 'sphericity' in r else np.nan,
                                 components=int(r.get('components', 0)) if 'components' in r else 0))
        except Exception as e:
            # Skip problematic case but continue
            print("Skip", r['dataset'], r['case'], "->", e)

    # 4) HTML
    out_html = outdir / 'report.html'
    write_html(rendered, str(out_html))

    print("Thresholds:", thr)
    print("Saved plots to:", outdir)
    print("Rendered cases:", len(rendered), "->", out_html)

if __name__ == '__main__':
    main()
