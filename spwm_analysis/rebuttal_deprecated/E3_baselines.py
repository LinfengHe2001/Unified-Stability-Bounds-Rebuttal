#!/usr/bin/env python3
"""
E3_baselines.py — Rebuttal E3：one-step MSE 基线走同一 protocol
=====================================================================
回答 2q75 的 W4：若 one-step MSE 单独当 RHS 也能同样覆盖/区分，
"三通道揭示原因"的说法就站不住。

流程（与 E1 完全同 protocol，防 cherry-pick）：
 1. 在 E1 的全部 checkpoint 上取 one-step MSE / recon MSE / reward MAE
    （E1 collect_run 时 include_base=True 已算好）。
 2. 每个基线指标单独当 RHS：同一标定 run（baseline_main 50k–500k）上
    求 β*，冻结后在全部测试 runs 上算覆盖率/紧度/AUROC。
 3. 加两个组合对照：三通道 + one-step MSE；只用 one-step MSE。
 4. 定位差异案例：三通道某通道异常 vs one-step MSE 正常（或反之）。

输出：
  E3_baselines.csv : 每 RHS 一行（覆盖率/紧度/AUROC/CI）
  E3_localize.csv  : 定位差异案例清单
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys_path_1 = str(Path(__file__).resolve().parent)
import sys  # noqa: E402
sys.path.insert(0, sys_path_1)


def coverage_stats(rhs, gap, beta):
    covered = beta * rhs >= gap - 1e-12
    pos = gap > 0
    tight = beta * rhs[pos] / gap[pos] if pos.any() else np.array([])
    return {"coverage": float(covered.mean()),
             "tight_med": float(np.median(tight)) if len(tight) else np.nan,
             "n_violations": int((~covered).sum())}


def auroc(rhs, fail, run_ids, B=1000, seed=42):
    from sklearn.metrics import roc_auc_score
    runs = np.unique(run_ids)
    rng = np.random.default_rng(seed)

    def _a(idx):
        sel = np.isin(run_ids, idx)
        if fail[sel].sum() == 0 or (~fail[sel].astype(bool)).sum() == 0:
            return np.nan
        return roc_auc_score(fail[sel].astype(int), rhs[sel])

    pt = _a(runs)
    boots = [ _a(rng.choice(runs, len(runs), True)) for _ in range(B) ]
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    return {"auroc": pt,
              "ci_lo": float(np.percentile(boots, 2.5)) if len(boots) else np.nan,
              "ci_hi": float(np.percentile(boots, 97.5)) if len(boots) else np.nan}


def frozen_beta(cal_rhs, cal_gap):
    ratio = cal_gap / np.maximum(cal_rhs, 1e-12)
    return float(ratio.max())


def minmax_fit(cal_vals):
    lo, hi = np.nanpercentile(cal_vals, 5), np.nanpercentile(cal_vals, 95)
    return lo, hi


def apply_mm(v, lo, hi):
    return np.clip((v - lo) / max(hi - lo, 1e-8), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1_csv", default="rebuttal/E1_all_checkpoints.csv")
    ap.add_argument("--freeze_json", default="rebuttal/E1/freeze_beta.json")
    ap.add_argument("--calib_run", default="baseline_main")
    ap.add_argument("--calib_lo", type=int, default=50000)
    ap.add_argument("--calib_hi", type=int, default=500000)
    ap.add_argument("--output_dir", default="rebuttal/E3")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.e1_csv)
    fr = json.loads(Path(args.freeze_json).read_text())

    cal_mask = (df["run"] == args.calib_run) & \
                 (df["step"] >= args.calib_lo) & \
                 (df["step"] <= args.calib_hi)

    # RHS 候选（任务书：三通道 / one-step / recon / reward / 组合 / 单 MSE）
    rhs_defs = {
        "three_channels": df["kappa_mean"] + df["tc"] + df["lee"],
        "one_step_mse": df["one_step_mse"],
        "recon_mse": df["recon_mse"],
        "reward_mae": df["reward_mae"],
        "combo_channels_plus_mse":
            df["kappa_mean"] + df["tc"] + df["lee"] + df["one_step_mse"],
        "one_step_mse_only": df["one_step_mse"],
    }

    gap = df["gap_norm"].values
    fail = df["fail"].values.astype(float)
    run_ids = df["run"].values

    rows = []
    scaled = {}
    for name, raw in rhs_defs.items():
        raw = np.asarray(raw, dtype=float)
        lo, hi = minmax_fit(raw[cal_mask.values])
        rhs = apply_mm(raw, lo, hi)
        scaled[name] = rhs
        beta = frozen_beta(rhs[cal_mask.values], gap[cal_mask.values])
        st = coverage_stats(rhs, gap, beta)
        au = auroc(rhs, fail, run_ids)
        rows.append({"rhs": name, **st, **au})
    pd.DataFrame(rows).to_csv(out / "E3_baselines.csv", index=False)

    # ---------- 定位差异案例 ----------
    # 异常 = 归一化值超出标定集 95 分位（每通道独立判定）
    ch_norm = {}
    for ch in ("kappa_mean", "tc", "lee", "one_step_mse"):
        lo, hi = minmax_fit(df[ch][cal_mask.values].values.astype(float))
        ch_norm[ch] = apply_mm(df[ch].values.astype(float), lo, hi)
    thres = {ch: np.nanpercentile(v[cal_mask.values], 95)
               for ch, v in ch_norm.items()}
    rows = []
    for i, r in df.iterrows():
        tri_bad = any(ch_norm[c][i] > thres[c]
                           for c in ("kappa_mean", "tc", "lee"))
        mse_bad = ch_norm["one_step_mse"][i] > thres["one_step_mse"]
        if tri_bad != mse_bad:
            rows.append({"run": r["run"], "step": r["step"],
                          "tri_channel_bad": tri_bad,
                          "mse_bad": mse_bad,
                          "gap_norm": r["gap_norm"]})
    loc = pd.DataFrame(rows)
    loc.to_csv(out / "E3_localize.csv", index=False)

    # 汇总打印（rows 变量已被 localize 复用，重新读回 CSV）
    summ = pd.read_csv(out / "E3_baselines.csv")
    print(f"""
✅ E3 完成 → {out}/E3_baselines.csv
{summ[['rhs','coverage','auroc','ci_lo','ci_hi']].to_string(index=False)}
   定位差异案例数: {len(loc)}  → E3_localize.csv
   若 one_step_mse_only 的 AUROC/覆盖率 ≈ 三通道，"揭示原因"表述需弱化。
""")


if __name__ == "__main__":
    main()
