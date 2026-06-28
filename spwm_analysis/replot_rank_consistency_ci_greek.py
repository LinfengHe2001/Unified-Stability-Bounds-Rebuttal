#!/usr/bin/env python3
"""
Replot rank_consistency_ci with tau / rho shown as math symbols (τ, ρ),
using the already computed CSV. Does not recompute any metrics.

Outputs a new PDF into a fresh results folder so the original artefacts remain.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    # Source results (existing run)
    src_dir = Path("results_crafter_50K_real_fixed13")
    ci_csv = src_dir / "rank_consistency_ci.csv"
    if not ci_csv.exists():
        raise SystemExit(f"[ERROR] Cannot find {ci_csv}. Make sure the original analysis has been run.")

    # Load tau / rho summary from CSV
    df = pd.read_csv(ci_csv)
    if df.empty:
        raise SystemExit(f"[ERROR] {ci_csv} is empty.")

    row = df.iloc[0]
    tau = float(row["tau"])
    tau_lo = float(row["tau_lo"])
    tau_hi = float(row["tau_hi"])
    rho = float(row["rho"])
    rho_lo = float(row["rho_lo"])
    rho_hi = float(row["rho_hi"])
    block = int(row.get("block", 2))

    # Recreate the bar plot with error bars, matching original style.
    vals = np.array([tau, rho], dtype=float)
    lowers = np.array([tau - tau_lo, rho - rho_lo], dtype=float)
    uppers = np.array([tau_hi - tau, rho_hi - rho], dtype=float)
    yerr = np.vstack([lowers, uppers])
    xs = np.arange(2)

    plt.figure(figsize=(6, 4))
    # Use the requested pastel colors for the two bars.
    plt.bar(xs, vals, yerr=yerr, capsize=6, color=["#BDE4FF", "#DEF2D9"])
    plt.xticks(xs, ["Kendall $\\tau$", "Spearman $\\rho$"], fontsize=16)
    plt.yticks(fontsize=16)
    plt.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    # Title at 16pt; axis tick fonts stay as configured above.
    plt.title(f"Rank Consistency (block bootstrap CI, block={block})", fontsize=16)
    caption = (
        f"$\\tau$={tau:.3f} [{tau_lo:.3f},{tau_hi:.3f}], "
        f"$\\rho$={rho:.3f} [{rho_lo:.3f},{rho_hi:.3f}]"
    )
    plt.legend([caption], loc="lower right")
    plt.tight_layout()

    # New output folder for Greek-symbol PDF
    out_dir = Path("results_crafter_50K_real_fixed13_tau_rho")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = out_dir / "rank_consistency_ci.pdf"

    plt.savefig(out_pdf, format="pdf")
    plt.close()

    print(f"[INFO] Wrote Greek-symbol rank consistency CI plot to: {out_pdf}")


if __name__ == "__main__":
    main()
