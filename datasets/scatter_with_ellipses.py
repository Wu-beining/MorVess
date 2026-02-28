
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SI–DI scatter with gradient ellipses (per dataset) + hi-res export.

- Works on the CSV produced by dispersion_analysis.py
- Ellipses are computed in log10(SI)–DI space using covariance + chi-square quantiles,
  then sampled and mapped back to SI (10**x) so they look correct on a log-x axis.
- Exports PNG (600 dpi), PDF, and SVG.
- Typography tuned (bigger fonts, tight layout).

Usage
-----
python scatter_with_ellipses.py \
  --csv ./dispersion_out/dataset_metrics.csv \
  --outdir ./dispersion_out \
  --levels 0.5 0.8 0.95
"""
import argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from pathlib import Path

def chi2_radius(level: float) -> float:
    """Return sqrt(chi2.ppf(level, df=2)); fallback to lookup if scipy missing."""
    try:
        from scipy.stats import chi2
        return float(np.sqrt(chi2.ppf(level, df=2)))
    except Exception:
        # Common levels lookup for df=2
        table = {0.5: 1.38629436112, 0.8: 3.21887582575, 0.9: 4.60517018599, 0.95: 5.99146454711, 0.99: 9.21034037198}
        return float(np.sqrt(table.get(level, 5.99146454711)))

def ellipse_points(mean, cov, scale, n=400):
    """Sample points of the ellipse (x in log10-space, y in linear DI)."""
    # eig decomposition
    evals, evecs = np.linalg.eigh(cov)
    order = evals.argsort()[::-1]
    evals, evecs = evals[order], evecs[:, order]
    # param
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    # radius in principal axes
    r = np.stack([np.cos(t), np.sin(t)], axis=0)  # (2,n)
    # scale by eigenvalues and chi2 radius
    w = (np.sqrt(np.maximum(evals, 1e-12)) * scale).reshape(2,1)
    pts = (evecs @ (w * r)).T + mean  # (n,2) in [log10(SI), DI]
    return pts

def lighten(color_rgb, amount=0.5):
    """Lighten a color by blending with white. amount in [0,1]."""
    r,g,b = colors.to_rgb(color_rgb)
    return (1 - amount) + amount*r, (1 - amount) + amount*g, (1 - amount) + amount*b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--outdir', type=str, default='./dispersion_out')
    ap.add_argument('--levels', type=float, nargs='+', default=[0.5, 0.8, 0.95], help='confidence levels')
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['si','di','dataset'])

    # Global typography
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    fig, ax = plt.subplots(figsize=(8.5,6.5), dpi=args.dpi)
    # Base scatter
    groups = list(df.groupby('dataset'))
    cmap = cm.get_cmap('tab10', len(groups))
    for i, (ds, g) in enumerate(groups):
        ax.scatter(g['si'].values, g['di'].values, label=str(ds), s=18)
        # Ellipses in log-space
        xlog = np.log10(np.clip(g['si'].values, 1e-12, None))
        y = g['di'].values
        if len(xlog) < 3:
            continue
        mean = np.array([xlog.mean(), y.mean()])
        cov = np.cov(np.vstack([xlog, y]))
        base = cmap(i)
        # sort levels inner->outer for nice gradient
        levels = sorted(args.levels)
        for j, lvl in enumerate(levels):
            scale = chi2_radius(lvl)
            pts = ellipse_points(mean, cov, scale, n=500)
            # map back x from log10 to linear
            x = 10.0 ** pts[:,0]
            y = pts[:,1]
            # gradient via lightening + increasing alpha inward
            frac = (j+1) / len(levels)
            face = lighten(base, amount=1 - 0.65*frac)  # closer to base inside
            ax.fill(x, y, alpha=0.20 + 0.12*frac, facecolor=face, edgecolor='none')
        # draw outer rim
        scale = chi2_radius(levels[-1])
        pts = ellipse_points(mean, cov, scale, n=600)
        ax.plot(10.0**pts[:,0], pts[:,1], linewidth=1.2)

    ax.set_xscale('log')
    ax.set_xlabel('Sparsity Index (log scale)')
    ax.set_ylabel('Dispersion Index (normalized mean radius)')
    ax.legend(loc='best', frameon=True)
    ax.margins(0.05)
    fig.tight_layout()

    # Save: PNG hi-res + PDF + SVG
    png_path = outdir / 'si_di_scatter_ellipses.png'
    pdf_path = outdir / 'si_di_scatter_ellipses.pdf'
    svg_path = outdir / 'si_di_scatter_ellipses.svg'
    fig.savefig(png_path, dpi=args.dpi, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    print("Saved:", png_path, pdf_path, svg_path)

if __name__ == '__main__':
    main()
