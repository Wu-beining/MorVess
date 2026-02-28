
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-visualizations for dispersion analysis.

Inputs
------
- --csv: path to metrics CSV produced by dispersion_analysis.py
- --outdir: directory to save figures
- Heuristic thresholds (optional overrides):
  --si-max, --di-min, --lin-min, --sph-max
  or use quantile-based auto thresholds:
  --auto (uses si<=q20, di>=q80, linearity>=q80, sphericity<=q20 per-dataset pooled)

Outputs
-------
- annotated_scatter.png
- violin_SI.png, violin_DI.png, violin_linearity.png, violin_sphericity.png
- radar_medians.png
- vessel_like_cases.csv
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_df(csv_path:str)->pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Guard against zeros/NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['si','di'])
    return df

def flag_vessel_like(df: pd.DataFrame, si_max=None, di_min=None, lin_min=None, sph_max=None, auto=False):
    sub = df.copy()
    if auto:
        # auto thresholds from pooled data (exclude NaNs)
        q = sub[['si','di','linearity','sphericity']].quantile([0.2, 0.8]).to_dict()
        si_max = si_max if si_max is not None else float(q['si'][0.2])
        di_min = di_min if di_min is not None else float(q['di'][0.8])
        lin_min = lin_min if lin_min is not None else float(q['linearity'][0.8]) if 'linearity' in q else None
        sph_max = sph_max if sph_max is not None else float(q['sphericity'][0.2]) if 'sphericity' in q else None
    else:
        # conservative defaults if none provided
        if si_max is None: si_max = sub['si'].quantile(0.3)
        if di_min is None: di_min = sub['di'].quantile(0.7)
        if ('linearity' in sub) and (lin_min is None): lin_min = sub['linearity'].quantile(0.7)
        if ('sphericity' in sub) and (sph_max is None): sph_max = sub['sphericity'].quantile(0.3)

    crit = (sub['si'] <= si_max) & (sub['di'] >= di_min)
    if 'linearity' in sub.columns and lin_min is not None:
        crit = crit & (sub['linearity'] >= lin_min)
    if 'sphericity' in sub.columns and sph_max is not None:
        crit = crit & (sub['sphericity'] <= sph_max)
    sub['vessel_like'] = crit.astype(int)
    return sub, dict(si_max=si_max, di_min=di_min, lin_min=lin_min, sph_max=sph_max)

def annotated_scatter(df: pd.DataFrame, out_png: str):
    # Base scatter
    plt.figure(figsize=(8,6), dpi=160)
    for ds, g in df.groupby('dataset'):
        plt.scatter(g['si'], g['di'], label=ds)
    # Highlight vessel_like with larger markers
    vh = df[df['vessel_like'] == 1]
    if len(vh):
        plt.scatter(vh['si'], vh['di'], s=80, marker='^')
        # draw dashed rectangle around cluster
        pad_x = (vh['si'].max() - vh['si'].min()) * 0.2 if len(vh)>1 else 1e-3
        pad_y = (vh['di'].max() - vh['di'].min()) * 0.2 if len(vh)>1 else 1e-3
        x0, x1 = vh['si'].min() - pad_x, vh['si'].max() + pad_x
        y0, y1 = vh['di'].min() - pad_y, vh['di'].max() + pad_y
        plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],'--')
        # add a simple arrow annotation to the first point
        p = vh.iloc[0]
        plt.annotate('Vessel-like', xy=(p['si'], p['di']), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='->'))
    plt.xscale('log')
    plt.xlabel('Sparsity Index (log scale)')
    plt.ylabel('Dispersion Index')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)

def violin_metric(df: pd.DataFrame, metric: str, out_png: str):
    plt.figure(figsize=(8,5), dpi=160)
    data = [g[metric].dropna().values for _, g in df.groupby('dataset')]
    labels = [name for name, _ in df.groupby('dataset')]
    plt.violinplot(data, showmeans=True, showextrema=True)
    plt.xticks(range(1, len(labels)+1), labels, rotation=20)
    plt.xlabel('Dataset')
    plt.ylabel(metric)
    if metric == 'si':
        plt.xscale('linear')  # keep linear axis; SI may span orders but violin in log is awkward
    plt.tight_layout()
    plt.savefig(out_png)

def radar_medians(df: pd.DataFrame, metrics, out_png: str):
    # Compute per-dataset medians
    med = df.groupby('dataset')[metrics].median()
    # Min-max normalize across datasets for each metric
    norm = (med - med.min()) / (med.max() - med.min() + 1e-12)
    labels = list(norm.columns)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    plt.figure(figsize=(7,7), dpi=160)
    ax = plt.subplot(111, polar=True)

    for ds in norm.index:
        vals = norm.loc[ds].values
        vals = np.concatenate([vals, vals[:1]])
        ax.plot(angles, vals, label=ds)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_ylim(0,1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True, help='Path to dataset_metrics.csv')
    ap.add_argument('--outdir', type=str, default='./dispersion_out')
    ap.add_argument('--auto', action='store_true', help='Use quantile-based auto thresholds')
    ap.add_argument('--si-max', type=float, default=None)
    ap.add_argument('--di-min', type=float, default=None)
    ap.add_argument('--lin-min', type=float, default=None)
    ap.add_argument('--sph-max', type=float, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_df(args.csv)
    df2, thr = flag_vessel_like(df, args.si_max, args.di_min, args.lin_min, args.sph_max, auto=args.auto)

    # Save flagged list
    flagged = df2[df2['vessel_like']==1][['dataset','case','si','di','linearity','sphericity','components']]
    flagged.to_csv(outdir / 'vessel_like_cases.csv', index=False)

    # Plots
    annotated_scatter(df2, str(outdir / 'annotated_scatter.png'))
    for m in ['si','di','linearity','sphericity']:
        if m in df2.columns:
            violin_metric(df2, m, str(outdir / f'violin_{m}.png'))
    radar_metrics = [m for m in ['si','di','linearity','sphericity','components'] if m in df2.columns]
    if radar_metrics:
        radar_medians(df2, radar_metrics, str(outdir / 'radar_medians.png'))

    print('Thresholds used:', thr)
    print('Saved:', outdir)

if __name__ == '__main__':
    main()
