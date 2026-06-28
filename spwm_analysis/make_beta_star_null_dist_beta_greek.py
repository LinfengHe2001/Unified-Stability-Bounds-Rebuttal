#!/usr/bin/env python3
"""
Replot the 'beta* null distribution' figure using existing CSVs, but:
  1) Change the title suffix from '(random pseudo-RHS)' to '(retagged)'.
  2) Render beta as the Greek symbol β in the title and y-axis label.
  3) Change the legend text from 'true beta* = 0.01' to 'true β* = 0.11',
     while keeping the true beta* line at its original numeric value.

All data, colors, and line styles follow the original implementation in
spwm_analysis.spwm_metrics_system._beta_stats_and_plots.

Inputs (fixed to your current run):
  - results_crafter_50K_real_fixed9/beta_star_null_dist.csv
  - results_crafter_50K_real_fixed9/bound_vs_gap_note.txt

Output:
  - results_crafter_50K_real_fixed9_beta_greek/beta_star_null_dist.pdf
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def _parse_true_beta_star(note_path: Path) -> float:
    """
    Parse true beta* from a bound_vs_gap_note.txt line like:
      'beta* (100% 覆盖) = 0.1177; beta50% = ...'
    """
    txt = note_path.read_text(encoding="utf-8")
    line = next((ln for ln in txt.splitlines() if "beta*" in ln), "")
    if not line:
        raise RuntimeError(f"No line containing 'beta*' found in {note_path}")
    # Find the first 'beta* =' pattern
    # Example segment: 'beta* (100% 覆盖) = 0.1177;'
    seg = line.split("beta*")[1]
    # Look for '=' and take the numeric that follows
    if "=" not in seg:
        raise RuntimeError(f"Cannot parse beta* from: {line}")
    after_eq = seg.split("=", 1)[1]
    # Numeric ends at first semicolon or end of string
    num_str = after_eq.split(";")[0].strip()
    return float(num_str)


def main() -> None:
    base_dir = Path("results_crafter_50K_real_fixed9")
    null_csv = base_dir / "beta_star_null_dist.csv"
    note_path = base_dir / "bound_vs_gap_note.txt"

    if not null_csv.exists():
        raise SystemExit(f"[ERROR] Missing null distribution CSV: {null_csv}")
    if not note_path.exists():
        raise SystemExit(f"[ERROR] Missing note file: {note_path}")

    nd = pd.read_csv(null_csv)
    if "beta_star" not in nd.columns:
        raise SystemExit(f"[ERROR] Expected 'beta_star' column in {null_csv}, got {list(nd.columns)}")
    beta_null = nd["beta_star"].to_numpy(dtype=float)

    beta_true = _parse_true_beta_star(note_path)

    # Match style from _beta_stats_and_plots for consistency.
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
            # Legend one step smaller for better balance in this figure.
            "legend.fontsize": 10,
        }
    )

    plt.figure(figsize=(5, 4))
    parts = plt.violinplot([beta_null], showmeans=True, showextrema=True)
    # Preserve the original body color and alpha.
    for pc in parts["bodies"]:
        pc.set_facecolor("#1f77b4")
        pc.set_alpha(0.6)

    # True beta* line: keep numeric value beta_true, and show it explicitly in the legend.
    plt.axhline(beta_true, color="r", linestyle="--", label=f"true β* = {beta_true:.4f}")

    # Axis labels and title with Greek β.
    plt.ylabel(r"$\beta^*$")
    plt.xticks([1], ["null"])
    plt.title("Null distribution of β* (retagged)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_dir = Path("results_crafter_50K_real_fixed9_beta_greek")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "beta_star_null_dist.pdf"
    plt.savefig(out_pdf, dpi=300)
    plt.close()

    print(f"[INFO] Wrote β*-null distribution plot to: {out_pdf}")


if __name__ == "__main__":
    main()
