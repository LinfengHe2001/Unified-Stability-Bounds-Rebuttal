#!/usr/bin/env python3
"""
Replot the 'Bound vs Gap — late point covered under beta*' figure using the
existing CSV outputs, but render beta as the Greek symbol β in the title and
legend. All data, colors, and line styles follow the original overlay script.

Inputs (hard-coded to match your current layout):
  - Base data dir: results_crafter_50K_real_fixed12
    * bound_vs_gap_uncalibrated.csv (rhs, gap, yhat)
    * bound_vs_gap_with_late_note.txt (step, x_late, y_late, beta*, covered)

Outputs:
  - New folder: results_crafter_50K_real_fixed12_beta_greek
    * test_overlay_with_annot.pdf (beta rendered as β)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def _parse_late_note(path: Path) -> Dict[str, str]:
    """Parse a single-line note like 'step=..., x_late=..., beta*=..., covered=True'."""
    txt = path.read_text(encoding="utf-8").strip()
    # Take the first non-empty line
    line = next((ln for ln in txt.splitlines() if ln.strip()), "")
    out: Dict[str, str] = {}
    for token in line.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> None:
    base_dir = Path("results_crafter_50K_real_fixed12")
    out_dir = Path("results_crafter_50K_real_fixed12_beta_greek")
    out_dir.mkdir(parents=True, exist_ok=True)

    scat_csv = base_dir / "bound_vs_gap_uncalibrated.csv"
    note_path = base_dir / "bound_vs_gap_with_late_note.txt"

    if not scat_csv.exists():
        raise SystemExit(f"[ERROR] Missing scatter CSV: {scat_csv}")
    if not note_path.exists():
        raise SystemExit(f"[ERROR] Missing late-point note: {note_path}")

    # Load base scatter: rhs, gap, yhat
    base = pd.read_csv(scat_csv)
    if not {"rhs", "gap"}.issubset(base.columns):
        raise SystemExit(f"[ERROR] Expected columns 'rhs' and 'gap' in {scat_csv}, got {list(base.columns)}")

    rhs = base["rhs"].to_numpy(dtype=float)
    gap = base["gap"].to_numpy(dtype=float)

    # Load late point and beta* from note (no recomputation to keep numbers identical).
    note = _parse_late_note(note_path)
    try:
        x_late = float(note.get("x_late"))
        y_late = float(note.get("y_late"))
        beta_star = float(note.get("beta*"))
        step = int(note.get("step"))
        covered_str = note.get("covered", "True")
        covered = covered_str.lower() == "true"
    except Exception as e:
        raise SystemExit(f"[ERROR] Failed to parse late note {note_path}: {note}") from e

    # Match rcParams from the original overlay script for consistent style.
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

    # Late-step formatting (e.g., 3.5M), copied from overlay script logic.
    def _fmt_step(v: int) -> str:
        try:
            v_int = int(v)
        except Exception:
            return str(v)
        if v_int >= 1_000_000:
            val = v_int / 1_000_000.0
            return f"{val:.1f}M" if abs(val - int(val)) > 1e-9 else f"{int(val)}M"
        if v_int >= 1_000:
            val = v_int / 1_000.0
            return f"{val:.1f}K" if abs(val - int(val)) > 1e-9 else f"{int(val)}K"
        return f"{v_int}"

    # Prepare grid for y=x and beta* x.
    xmax = max(float(np.nanmax(rhs)), x_late) * 1.05
    xmax = max(xmax, 1.0)
    grid = np.linspace(0.0, xmax, 200)

    # Plot: reuse colors and ordering from the original overlay script.
    plt.figure(figsize=(7.0, 7.0))
    plt.scatter(rhs, gap, s=30, alpha=0.7)
    plt.plot(grid, grid, "--", label="y=x")
    # Use Greek beta in the label, numeric value unchanged.
    plt.plot(grid, beta_star * grid, "-", label=f"y={beta_star:.2f} x (β*)")
    plt.scatter(
        [x_late],
        [y_late],
        marker="*",
        s=200.0,
        color="red",
        label=f"late {_fmt_step(step)} (covered={covered})",
    )

    # Title with β* instead of 'beta*'.
    if covered:
        title = "Bound vs Gap — late point covered under β*"
    else:
        title = "Bound vs Gap — late point not covered"
    plt.title(title, fontsize=mpl.rcParams["axes.titlesize"])

    plt.xlabel("Theoretical RHS (unit-consistent)", fontsize=mpl.rcParams["axes.labelsize"])
    plt.ylabel("Measured performance gap (normalized)", fontsize=mpl.rcParams["axes.labelsize"])
    plt.xticks(fontsize=mpl.rcParams["xtick.labelsize"])
    plt.yticks(fontsize=mpl.rcParams["ytick.labelsize"])
    # Enlarge legend font while keeping all line colors and data unchanged.
    plt.legend(fontsize=16)
    plt.tight_layout()

    out_pdf = out_dir / "test_overlay_with_annot.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"[INFO] Wrote β-symbol overlay plot to: {out_pdf}")


if __name__ == "__main__":
    main()
