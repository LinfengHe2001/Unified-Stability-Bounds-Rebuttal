#!/usr/bin/env python3
"""
E2_leadlag.py — Rebuttal E2：过度压缩 runs 的 lead-lag 分析
=====================================================================
回答 2q75 的 Q2/W2：哪个通道先于回报下降变化。
"flag failures before return drops" 能否保留取决于本项。

直接复用 E1 的 E1_all_checkpoints.csv（不重算指标）。

事先固定定义（任务书）：
  t_drop  : 回报比同 seed baseline 低超过 1 个 seed 间标准差，
            且之后连续 2 个 checkpoint 保持该状态的第 1 个 step。
  t_alarm : 对每个通道（κ/TC/LEE/RHS），以 baseline 同 step 值为参照，
            偏离超过 baseline seed 间 2 个标准差的第 1 个 step。
            双方向都检测，记下方向。
  lead    = t_drop − t_alarm（step 单位；分辨率 = checkpoint 间隔 50k）
  同样对 one-step MSE / recon MSE / reward MAE 计算（与 E3 共用）。

输出：
  E2_leadlag.csv  : run, lambda_tc, channel, t_alarm, t_drop, lead, direction
  E2_<run>.png    : 每 run 一张 4 行图（回报/κ/TC/LEE + 基线叠加）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT_GAP = 50_000  # 事先固定：分辨率 = 50k


def baseline_ref(df: pd.DataFrame, seed: int,
                    cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """同 seed 的 baseline（λ_TC=0 且非 aug），按 step 平均（跨 seed）。

    cols 缺失的列自动跳过（例如 rhs_norm 不在 E1 CSV 时）。
    """
    all_cols = ["J_mean", "kappa_mean", "tc", "lee",
                 "one_step_mse", "recon_mse", "reward_mae", "rhs_norm"]
    want = cols or all_cols
    use = [c for c in want if c in df.columns]
    m = (df["lambda_tc"] == 0) & (df["seed"] == seed) & \
        (~df["run"].str.contains("aug", regex=False))
    base = df[m].groupby("step")[use].mean()
    base_std = df[m].groupby("step")[use].std()
    return base, base_std


def t_drop_of(run_js: pd.Series, base_js: pd.Series,
                 base_j_std: pd.Series) -> int | None:
    """第一个满足：低于 baseline 超过 1σ，且连续 2 个 checkpoint 保持。

    σ 处理：若某 step 的 baseline 间 std 为 NaN（该 step 只有 1 个
    baseline run），回退到用全 step 的 baseline std 中位数（保守且
    非事后可调——中位数在标定前即可确定）。
    """
    # NaN σ 回退：全 step 中位数
    std_med = base_j_std.median()
    if pd.isna(std_med) or std_med == 0:
        std_med = float(np.abs(base_js.diff()).median() or 0.0)
    steps = sorted(set(run_js.index) & set(base_js.index))
    for i, s in enumerate(steps[:-2]):
        s1, s2, s3 = steps[i], steps[i + 1], steps[i + 2]
        cond = all(
            run_js.loc[sx] < base_js.loc[sx] - (
                base_j_std.loc[sx] if pd.notna(base_j_std.loc[sx])
                else std_med)
            for sx in (s1, s2, s3))
        if cond:
            return int(s1)
    return None


def t_alarm_of(run_v: pd.Series, base_v: pd.Series,
                  base_v_std: pd.Series, n_sigma: float = 2.0):
    """第一个偏离 baseline 超过 nσ 的 step + 方向。

    σ NaN（单 baseline）时同样回退到中位数。
    """
    std_med = base_v_std.median()
    if pd.isna(std_med) or std_med == 0:
        std_med = float(np.abs(base_v.diff()).median() or 0.0)
    steps = sorted(set(run_v.index) & set(base_v.index))
    for s in steps:
        mu, sd = base_v.loc[s], base_v_std.loc[s]
        if pd.isna(mu):
            continue
        if pd.isna(sd) or sd == 0:
            sd = std_med
            if sd == 0:
                continue
        v = run_v.loc[s]
        if v > mu + n_sigma * sd:
            return int(s), "up"
        if v < mu - n_sigma * sd:
            return int(s), "down"
    return None, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1_csv", default="rebuttal/E1_all_checkpoints.csv")
    ap.add_argument("--output_dir", default="rebuttal/E2")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.e1_csv)

    channels = {
        "kappa": "kappa_mean", "tc": "tc", "lee": "lee",
        "one_step_mse": "one_step_mse",
        "recon_mse": "recon_mse",
        "reward_mae": "reward_mae",
    }
    # RHS 总和通道：优先用 rhs_norm（E1 产物），缺失时由自然尺度合成
    if "rhs_norm" in df.columns:
        channels["rhs"] = "rhs_norm"
    elif {"kappa_mean", "tc", "lee"}.issubset(df.columns):
        df["rhs_nat"] = df["kappa_mean"] + df["tc"] + df["lee"]
        channels["rhs"] = "rhs_nat"
    rows = []
    sweeps = df[df["lambda_tc"] > 0]
    for (run, seed), g in sweeps.groupby(["run", "seed"]):
        base, base_std = baseline_ref(df, seed)
        run_g = g.set_index("step").sort_index()
        td = t_drop_of(run_g["J_mean"], base["J_mean"], base_std["J_mean"])
        if td is None:
            rows.append({"run": run, "seed": seed, "lambda_tc":
                          g["lambda_tc"].iloc[0], "channel": "ALL",
                          "t_alarm": None, "t_drop": None, "lead": None,
                          "direction": "no-drop"})
            continue
        for ch, col in channels.items():
            ta, direction = t_alarm_of(run_g[col], base[col],
                                          base_std[col])
            lead = (td - ta) // CKPT_GAP if ta is not None else None
            rows.append({"run": run, "seed": seed,
                          "lambda_tc": g["lambda_tc"].iloc[0],
                          "channel": ch, "t_alarm": ta, "t_drop": td,
                          "lead": lead, "direction": direction})
    res = pd.DataFrame(rows)
    res.to_csv(out / "E2_leadlag.csv", index=False)

    # ---------- 每 run 四行图 ----------
    for (run, seed), g in sweeps.groupby(["run", "seed"]):
        base, _ = baseline_ref(df, seed)
        fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        specs = [("J_mean", "Episode return"),
                  ("kappa_mean", "κ"), ("tc", "TC (nats)"),
                  ("lee", "LEE (mean)")]
        td = t_drop_of(g.set_index("step")["J_mean"], base["J_mean"],
                          baseline_ref(df, seed)[1]["J_mean"])
        for ax, (col, label) in zip(axes, specs):
            ax.plot(base.index, base[col], color="gray",
                     label="baseline (same seed)")
            ax.plot(g["step"], g[col], color="tab:red",
                     label=f"{run} (λ={g['lambda_tc'].iloc[0]})")
            if td:
                ax.axvline(td, color="k", ls=":", alpha=0.7)
                ax.text(td, ax.get_ylim()[1] * 0.9, " t_drop",
                          fontsize=8)
            # 标注各通道 t_alarm
            ch_map = {"kappa_mean": "kappa", "tc": "tc", "lee": "lee"}
            if col in ch_map:
                ta, _ = t_alarm_of(g.set_index("step")[col],
                                      base[col], baseline_ref(df, seed)[1][col])
                if ta:
                    ax.axvline(ta, color="tab:blue", ls="--", alpha=0.6)
                    ax.text(ta, ax.get_ylim()[0] * 1.05, " t_alarm",
                              color="tab:blue", fontsize=8)
            ax.set_ylabel(label)
            ax.legend(fontsize=7)
        axes[-1].set_xlabel("Training step")
        plt.suptitle(f"E2 lead-lag: {run} (seed={seed})")
        plt.tight_layout()
        safe = f"{run}_seed{seed}".replace("/", "_")
        plt.savefig(out / f"E2_{safe}.png", dpi=180)
        plt.close()

    # ---------- 结论底稿 ----------
    pos = res[res["lead"].notna() & (res["channel"] != "ALL")]
    summ = pos.groupby("channel")["lead"].agg(["count", "mean", "max"])
    summ.to_csv(out / "E2_channel_summary.csv")
    print(f"""
✅ E2 完成 → {out}
   lead>0 的通道（先于回报下降报警）：
{summ.to_string()}
   one-step MSE 的 lead 对比：见 E2_leadlag.csv 中 channel=one_step_mse 行。
   分辨率 = 50k（checkpoint 间隔）。
""")


if __name__ == "__main__":
    main()
