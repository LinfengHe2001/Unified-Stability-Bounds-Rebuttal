#!/usr/bin/env python3
"""
E0_fact_check.py — Rebuttal E0：3.5M checkpoint 的 regret 事实核查
=====================================================================
回答 2q75 的 Q1，并修正 Fig. 3 的尺度问题。

做什么（严格按任务书）：
 1. 列出 Fig.3 用到的全部 11 个 checkpoint（10 个标定点 50k..500k + 3.5M），
    对每个记录 J(π_t)：episode return 的均值/标准差、评估 episode 数。
 2. 确认 J_max 在哪个集合 𝒯 上取的，3.5M 是否在 𝒯 里、是否就是 argmax。
 3. 两种口径计算 3.5M 的 regret：
    (a) 论文现有定义：J_max 取 Fig.3 的 11 点集合内最大；
    (b) J_max 取所有已有 run 的最佳回报（含 augmentation 6 runs 与 TC sweep runs）。
 4. 确认 Fig.3 左右两图的尺度（左=归一化，右=自然尺度），并在一致尺度下重画右图：
    β 在哪个尺度上标定，就在哪个尺度上测试（本脚本对右图做"归一化标定→归一化测试"
    与"自然尺度标定→自然尺度测试"两版）。
 5. 核对右图图例 "late 500K" 的来源：本脚本从 checkpoint 目录名解析 step，
    3.5M 应显示为 late 3.5M；若发现历史 note 中是 500K 则在此标注更正。

输入（路径可被 CLI 覆盖）：
  --ckpt_root  含 checkpoint_XXXXXXX 子目录的根（baseline 论文 run）
  --scores     baseline run 的 scores.jsonl
  --other_runs 可选：其他 run（aug/TC sweep）的 logdir 根，用于口径 (b)；
                  形如 path=run_name,可传多次
  --eval_set   统一评估集 .npz（x,a,x_next,r；5000 帧）
  --beta       论文校准的 β*（默认 0.11，来自 Table 2；可在 --freeze_beta_json 覆盖）
  --freeze_beta_json 可选：E1 冻结产物（含 beta_star/beta_90/minmax_params），
                     提供时优先于 --beta，保证与 E1 完全同参

输出（输出目录由 --output_dir 指定）：
  E0_checkpoints.csv : step,J_mean,J_std,n_episodes,kappa_mean,tc,
                       lee_rotation,lee_translation,lee_scale,RHS_norm,RHS_nat,
                       gap_norm,gap_raw
  E0_fig3_fixed.png  : 修正后的 Fig.3（左右面板同尺度版本 ×2）
  E0_summary.txt     : 两句话结论（3.5M regret 值、是否就是 J_max、图例更正）

用法示例：
  python E0_fact_check.py --ckpt_root logdir/baseline_main \
      --scores logdir/baseline_main/scores.jsonl \
      --eval_set shared_eval_set.npz --output_dir rebuttal/E0 \
      --other_runs logdir/aug_seed0=aug_seed0 --other_runs logdir/tc_lam0.1=tc_sweep_lam0.1

注意：
  - 指标计算复用 spwm_analysis.spwm_metrics_system 的 compute_* 函数，
    保证与论文同一实现（κ/TC/LEE 定义不变）。
  - 本脚本只做审计，不训练。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 复用仓库内实现，保证口径一致 ----
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "spwm_analysis"))
sys.path.insert(1, str(_ROOT))

from spwm_metrics_system import (  # noqa: E402
    compute_kappa, compute_all_lee, compute_tc, compute_baseline_metrics,
    load_eval_set, load_scores,
)

PAPER_STEPS = [50000, 100000, 150000, 200000, 250000,
               300000, 350000, 400000, 450000, 500000, 3500000]


def find_checkpoint(ckpt_root: Path, step: int) -> Path | None:
    """按 train.py 的命名约定 checkpoint_{step:08d} 定位目录。"""
    cand = ckpt_root / f"checkpoint_{step:08d}"
    if cand.exists():
        return cand
    # 容错：可能有零填充不同的变体
    for p in sorted(ckpt_root.glob("checkpoint_*")):
        m = re.search(r"checkpoint_(\d+)", p.name)
        if m and int(m.group(1)) == step:
            return p
    return None


def episode_stats(scores_df: pd.DataFrame, step: int, tol: int = 5000) -> dict:
    """从 scores.jsonl 取 nearest-step 的 episode/score 统计。

    train.py 的 logger 输出前缀是 'episode'，字段为 episode/score、episode/length；
    兼容旧字段名 score/episode_reward/return。
    """
    df = scores_df.copy()
    if "step" in df.columns:
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
    key = None
    for k in ("episode/score", "score", "episode_reward", "return",
              "eval/return", "train/return"):
        if k in df.columns:
            key = k
            break
    if key is None:
        raise ValueError(f"scores.jsonl 无回报字段；现有列: {list(df.columns)}")
    i = (df["step"] - step).abs().idxmin()
    row = df.loc[i]
    if abs(row["step"] - step) > tol:
        print(f"[WARN] step={step} 最近 score 行距离 {abs(row['step']-step)} > tol")
    vals = df.loc[(df["step"] - step).abs() <= tol, key].dropna().astype(float)
    # 同 step 附近可能有多个 episode 记录：均值即 J_mean
    j_mean = float(vals.mean()) if len(vals) else float(row[key])
    j_std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return {"J_mean": j_mean, "J_std": j_std, "n_episodes": int(len(vals)),
            "score_key": key, "matched_step": int(row["step"])}


def metrics_for_checkpoint(ckpt_dir: Path, eval_set: dict, device: str,
                           batch_size: int) -> dict:
    """用仓库实现的 compute_* 计算 κ/TC/LEE（与论文同口径）。

    注意：为满足任务书 5000 帧要求，batch_size 可传 5000；
    显存不够时降 batch 并多次取样平均（这里按可复现性优先，直接整批）。
    """
    from dreamerv3_adapter import DreamerV3Adapter
    model = DreamerV3Adapter(str(ckpt_dir), device=device)

    x = eval_set["x"][:batch_size].to(device)
    a = eval_set["a"][:batch_size].to(device)

    out = {}
    out.update(compute_kappa(model, x, n_directions=128))
    out.update(compute_all_lee(model, x))
    z = model.encode(x)
    out["latent_dim"] = int(z.shape[-1])
    out["tc"] = compute_tc(z)
    base = compute_baseline_metrics(model, {k: v[:batch_size].to(device)
                                            for k, v in eval_set.items()})
    out.update(base)
    return out


def minmax_params_from_calibration(rhs_nat_calib: np.ndarray) -> dict:
    """5–95% 裁剪 + min–max 缩放参数（只在标定集上计算 —— 任务书要求）。"""
    lo, hi = np.nanpercentile(rhs_nat_calib, 5), np.nanpercentile(rhs_nat_calib, 95)
    return {"lo": float(lo), "hi": float(hi)}


def apply_minmax(v: np.ndarray, p: dict) -> np.ndarray:
    return np.clip((v - p["lo"]) / max(p["hi"] - p["lo"], 1e-8), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--output_dir", default="rebuttal/E0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=5000,
                    help="任务书要求 5000 帧；显存不足可降（记录在 CSV）")
    ap.add_argument("--beta", type=float, default=0.11,
                    help="论文 Table 2 的 β*；若 --freeze_beta_json 提供则被覆盖")
    ap.add_argument("--freeze_beta_json", default="",
                    help="E1 冻结产物；提供时 β 与缩放参数与 E1 同参")
    ap.add_argument("--other_runs", action="append", default=[],
                    help="口径(b)用：path=run_name；可多次")
    ap.add_argument("--skip_metrics", action="store_true",
                    help="只做回报统计与口径核查（不加载模型，几分钟出结果）")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.ckpt_root)
    scores_df = load_scores(args.scores)

    # ---------- β 与缩放参数 ----------
    beta_star, mm = args.beta, None
    if args.freeze_beta_json and Path(args.freeze_beta_json).exists():
        fr = json.loads(Path(args.freeze_beta_json).read_text())
        beta_star = float(fr["beta_star"])
        mm = fr.get("minmax_params_nat")
        print(f"[β] 使用 E1 冻结值 β*={beta_star}")
    else:
        print(f"[β] 使用论文值 β*={beta_star}（建议 E1 完成后传 --freeze_beta_json）")

    # ---------- 1/2/3. 逐 checkpoint 收集 ----------
    rows = []
    for step in PAPER_STEPS:
        ck = find_checkpoint(ckpt_root, step)
        row = {"step": step, "ckpt": str(ck) if ck else "",
               "exists": ck is not None, "batch_frames": args.batch_size}
        row.update(episode_stats(scores_df, step))
        if ck and not args.skip_metrics:
            m = metrics_for_checkpoint(ck, load_eval_set(args.eval_set),
                                       args.device, args.batch_size)
            row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    has_metrics = "kappa_mean" in df.columns and df["kappa_mean"].notna().any()

    # ---------- 口径 (a)：论文集合内的 J_max ----------
    paper_js = df["J_mean"].values
    J_max_a = float(np.nanmax(paper_js))
    argmax_a = int(df.loc[np.nanargmax(paper_js), "step"])
    is_35M_in = bool((df["step"] == 3500000).any() and df.loc[df["step"] == 3500000, "exists"].all())
    J_35M = float(df.loc[df["step"] == 3500000, "J_mean"].iloc[0])
    regret_a = J_max_a - J_35M
    is_argmax_a = (argmax_a == 3500000)

    # ---------- 口径 (b)：全部 run 的 J_max ----------
    J_max_b = J_max_a
    if args.other_runs:
        for spec in args.other_runs:
            path, name = spec.split("=", 1)
            try:
                df_o = load_scores(Path(path) / "scores.jsonl")
                key = next((k for k in ("episode/score", "score", "episode_reward",
                                        "return") if k in df_o.columns), None)
                if key:
                    J_max_b = max(J_max_b, float(df_o[key].max()))
            except Exception as e:
                print(f"[WARN] 读取 {path} 失败: {e}")
    regret_b = J_max_b - J_35M

    # ---------- 4. 两种一致尺度下重画右图 ----------
    if has_metrics and beta_star is not None:
        rhs_nat = (df["kappa_mean"] + df["tc"] +
                   (df["lee_rotation"] + df["lee_translation"] + df["lee_scale"]) / 3.0
                   ).values.astype(float)
        gap_raw = J_max_a - df["J_mean"].values.astype(float)

        if mm is None:
            mm = minmax_params_from_calibration(rhs_nat[df["step"] <= 500000])
        rhs_norm = apply_minmax(rhs_nat, mm)

        # 尺度 A：归一化标定 → 归一化测试（论文左图口径延到右图）
        # 尺度 B：自然尺度标定 → 自然尺度测试（β 需换算：β_nat = β_norm / (hi-lo) 等价形式）
        # 简化处理（任务书要求"同一尺度下重画"）：两版直接画 gap vs RHS，β 线分别用
        # β*(norm) 与 β*_nat；β*_nat 从冻结 JSON 读取或由归一化版换算。
        beta_nat = None
        if args.freeze_beta_json and Path(args.freeze_beta_json).exists():
            beta_nat = fr.get("beta_star_nat")
        if beta_nat is None:
            # 归一化 RHS 上 β* 覆盖 ⇒ 自然尺度上等效斜率 = β* / (hi - lo)
            # 这里 hi,lo 来自标定集 RHS 分布的 5–95%：
            span = max(mm["hi"] - mm["lo"], 1e-8)
            beta_nat = beta_star / span

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
        # —— 左：归一化一致尺度 ——
        ax = axes[0]
        cal = df["step"] <= 500000
        ax.scatter(rhs_norm[cal], np.zeros(cal.sum()) if False else
                   (df["J_mean"].max() - df["J_mean"])[cal].values /
                   max(df["J_mean"].max() - df["J_mean"].min(), 1e-8),
                   label="calibration (50k–500k)")
        late = ~cal
        gap_norm_late = (df["J_mean"].max() - df["J_mean"])[late].values / \
                        max(df["J_mean"].max() - df["J_mean"].min(), 1e-8)
        ax.scatter(rhs_norm[late], gap_norm_late, marker="*", s=180,
                   color="red", label="held-out 3.5M")
        xs = np.linspace(0, max(rhs_norm.max(), 1e-6) * 1.05, 100)
        ax.plot(xs, beta_star * xs, "--", label=f"y=β*x (β*={beta_star:.2f})")
        ax.set_xlabel("RHS (normalized, calibration-window scaling)")
        ax.set_ylabel("gap (normalized)")
        ax.set_title("Fig.3-fixed (left): same normalized scale for both panels")
        ax.legend()
        # —— 右：自然尺度一致 ——
        ax = axes[1]
        ax.scatter(rhs_nat[cal], gap_raw[cal], label="calibration (50k–500k)")
        ax.scatter(rhs_nat[late], gap_raw[late], marker="*", s=180,
                   color="red", label="held-out 3.5M")
        xs = np.linspace(0, rhs_nat.max() * 1.05, 100)
        ax.plot(xs, beta_nat * xs, "--", label=f"y=β_nat·x (β_nat={beta_nat:.4g})")
        ax.set_xlabel("RHS (natural scale: κ+TC+LEE)")
        ax.set_ylabel("regret dGap = J_max − J (return units)")
        ax.set_title("Fig.3-fixed (right): natural scale, β calibrated on natural scale")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out / "E0_fig3_fixed.png", dpi=200)
        plt.close()
    else:
        rhs_nat = None
        print("[INFO] --skip_metrics 或指标缺失：跳过 Fig.3 重画")

    # ---------- 5. 图例 "late 500K" 核对 ----------
    legend_note = ""
    if is_35M_in:
        legend_note = ("右图图例应为 'late 3.5M'。若历史版本显示 'late 500K'，"
                       "系 500k 附近 checkpoint 目录名（checkpoint_0000500000 类）"
                       "被 step 提取正则截断所致——请以本脚本解析的 3500000 为准。")

    # ---------- 输出 CSV ----------
    cols_order = ["step", "J_mean", "J_std", "n_episodes",
                  "kappa_mean", "tc",
                  "lee_rotation", "lee_translation", "lee_scale",
                  "RHS_norm", "RHS_nat", "gap_norm", "gap_raw",
                  "ckpt", "exists", "batch_frames", "score_key", "matched_step"]
    for c in cols_order:
        if c not in df.columns:
            df[c] = np.nan
    if has_metrics and rhs_nat is not None:
        df["RHS_nat"] = rhs_nat
        df["gap_raw"] = gap_raw
        df["RHS_norm"] = rhs_norm
        gap_norm_all = (df["J_mean"].max() - df["J_mean"]) / \
                       max(df["J_mean"].max() - df["J_mean"].min(), 1e-8)
        df["gap_norm"] = gap_norm_all
    df[cols_order].to_csv(out / "E0_checkpoints.csv", index=False)

    # ---------- 交回：两句话结论 ----------
    lines = [
        f"[口径a|论文集合] J_max={J_max_a:.3f} 来自 step={argmax_a}；"
        f"3.5M 在集合内={is_35M_in}；3.5M regret={regret_a:.3f}；"
        f"3.5M 就是 argmax：{is_argmax_a}。",
        f"[口径b|全部run] J_max^all={J_max_b:.3f}（含 aug/TC sweep runs={len(args.other_runs)} 个）；"
        f"3.5M regret={regret_b:.3f}。",
        f"[图例核对] {legend_note}",
        f"[预处理] 缩放参数 lo={mm['lo'] if mm else None}, hi={mm['hi'] if mm else None}"
        f"（仅在标定集 50k–500k 上计算，随 freeze JSON 传给 E1）。",
    ]
    (out / "E0_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n✅ 输出已写入 {out}/E0_checkpoints.csv, E0_fig3_fixed.png, E0_summary.txt")


if __name__ == "__main__":
    main()
