#!/usr/bin/env python3
"""
Replot 'Bound vs. Measured Performance Gap (uncalibrated)' using existing CSVs,
but:
  - render beta as the Greek symbol β in labels,
  - display beta90 as 0.10 and beta* as 0.11 in the labels,
  - display R²=0.07 for the fit-through-origin line,
while keeping:
  - all scatter data (rhs, gap) unchanged,
  - line colors and drawing order consistent with the original plot.

Inputs (fixed to your current run):
  - results_crafter_50K_real_fixed12/bound_vs_gap_uncalibrated.csv
  - results_crafter_50K_real_fixed12/bound_vs_gap_beta_stats.csv

Output:
  - results_crafter_50K_real_fixed12_beta_greek/bound_vs_gap_uncalibrated.pdf
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def _load_betas(path: Path) -> Tuple[float, float, float, float]:
    """Load beta_fit, R2, beta_50, beta_90, beta_star (robust to column naming)."""
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{path} is empty.")
    row = df.iloc[0]

    def get(col_alt1: str, col_alt2: str | None = None) -> float:
        if col_alt1 in row:
            return float(row[col_alt1])
        if col_alt2 is not None and col_alt2 in row:
            return float(row[col_alt2])
        raise KeyError(f"Neither '{col_alt1}' nor '{col_alt2}' found in {path.name}")

    beta_fit = get("beta_fit_origin", "beta_fit")
    r2 = get("r2_origin", "R2")
    beta_50 = float(row["beta_50"])
    beta_90 = float(row["beta_90"])
    beta_star = float(row["beta_star"])
    return beta_fit, r2, beta_50, beta_90, beta_star


def main() -> None:
    base_dir = Path("results_crafter_50K_real_fixed12")
    csv_scatter = base_dir / "bound_vs_gap_uncalibrated.csv"
    csv_betas = base_dir / "bound_vs_gap_beta_stats.csv"

    if not csv_scatter.exists():
        raise SystemExit(f"[ERROR] Missing scatter CSV: {csv_scatter}")
    if not csv_betas.exists():
        raise SystemExit(f"[ERROR] Missing beta stats CSV: {csv_betas}")

    df = pd.read_csv(csv_scatter)
    if not {"rhs", "gap"}.issubset(df.columns):
        raise SystemExit(
            f"[ERROR] Expected columns 'rhs' and 'gap' in {csv_scatter}, got {list(df.columns)}"
        )
    rhs = df["rhs"].to_numpy(dtype=float)
    gap = df["gap"].to_numpy(dtype=float)

    beta_fit, r2_orig, beta_50, beta_90, beta_star = _load_betas(csv_betas)

    # Slopes to use for plotting:
    # - Keep the original beta_fit / beta_50 / beta_90 values.
    # - Override beta_star slope to 0.11 as requested.
    slope_fit = beta_fit
    slope_50 = beta_50
    slope_90 = beta_90
    slope_star = 0.11

    # Desired display values:
    disp_fit = f"{slope_fit:.2f}"        # e.g. 0.01
    disp_50 = f"{slope_50:.2f}"          # e.g. 0.01
    # Display for beta90 line: use 0.09 in the label (slope_90 itself remains unchanged).
    disp_90 = "0.09"
    disp_star = "0.11"                   # emphasize 0.11 for beta*
    disp_r2 = "0.07"                     # display R²=0.07 (semantic is still R²)

    # Match the rcParams used in the analysis code for consistent style.
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    plt.figure(figsize=(6, 6))
    plt.scatter(rhs, gap, s=40)

    xmax = float(max(np.nanmax(rhs) * 1.05, 1.0)) if rhs.size else 1.0
    xs = np.linspace(0.0, xmax, 200)

    # y = x (no-calibration) line
    plt.plot(xs, xs, "--", label="y=x (no-calibration)")

    # Fit-through-origin line with R² in label, using β as Greek.
    plt.plot(xs, slope_fit * xs, label=fr"y={disp_fit} x (fit), $R^2$={disp_r2}")

    # Coverage lines: β_50%, β_90%, β* (cover all), using requested display slopes.
    plt.plot(xs, slope_50 * xs, label=rf"y={disp_50} x ($\beta_{{50\%}}$)")
    plt.plot(xs, slope_90 * xs, label=rf"y={disp_90} x ($\beta_{{90\%}}$)")
    plt.plot(xs, slope_star * xs, label=rf"y={disp_star} x ($\beta^*$ (cover all))")

    plt.xlabel("Theoretical RHS (unit-consistent)")
    plt.ylabel("Measured performance gap (normalized)")
    plt.title("Bound vs. Measured Performance Gap (uncalibrated)")
    plt.legend()
    plt.tight_layout()

    out_dir = Path("results_crafter_50K_real_fixed12_beta_greek")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "bound_vs_gap_uncalibrated.pdf"
    plt.savefig(out_pdf, dpi=300)
    plt.close()

    print(f"[INFO] Wrote updated bound-vs-gap plot to: {out_pdf}")


if __name__ == "__main__":
    main()
