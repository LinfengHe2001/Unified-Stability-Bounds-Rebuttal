#!/usr/bin/env python3
"""
E1_frozen_beta_test.py — Rebuttal E1（最重要）：跨 run 冻结 β 的 held-out 测试
=====================================================================
回答 2q75 的 W1 与 bajJ 的 Q3：β 不重新标定时，bound 能否覆盖独立 run 上的
真实失败。TC sweep 过度压缩 runs 是天然证伪测试。

核心协议（严格按任务书，防止任何形式的"事后调参"）：
 1. 每个 checkpoint 在【同一个】5000 帧固定评估集上计算 κ/TC/LEE + J。
 2. 统一 J*：J_max^all = 所有 run 所有 checkpoint 的最高回报；
    gap = J_max^all − J；归一化分母 = J_max^all − J_min^all。
 3. 标定集：仅 baseline_main 的 50k–500k。求 β*（最小覆盖）、β_90%，
    并在该子集上拟合 5–95% min–max 缩放参数。三者一并【冻结】到
    freeze_beta.json。
 4. 测试集：其余全部 run 的全部 checkpoint。不重标定、不重算缩放。
 5. 指标：覆盖率（总体/分run/健康vs过度压缩分组）、紧度（gap>0 的
    β*RHS/gap 四分位）、区分力（失败=gap>25%·J-range，AUROC + 按 run
    block bootstrap 95% CI）、违反点明细。
 6. Leave-one-run-out：轮流用每个 baseline seed 标定，其余 run 测试。

输入：
  --runs path=name 格式（可多次）：
      logdir/baseline_main=baseline_main
      logdir/aug_seed0=aug_seed0 ...（3 aug + 3 baseline_aug）
      logdir/tc_lam0.01=tc_sweep_lam0.01 ...（全部 λ_TC）
      TC sweep runs 请用名字前缀 tc_sweep_* 以触发分组。
  --eval_set 统一评估集 npz
  --calib_run baseline_main --calib_steps 50000..500000

输出（--output_dir）：
  E1_all_checkpoints.csv  每行一个 checkpoint（含 run/seed/λ_TC/组别）
  E1_summary.csv          每 run 一行：覆盖率/紧度/违反数/组别
  E1_group_summary.csv    健康 vs 过度压缩 分组
  E1_scatter.png          RHS–gap 散点，按 run 着色，y=β*x
  freeze_beta.json        冻结的 β*/β90/minmax 参数（E0 可复用）
  E1_violations.csv       违反点明细（run,step,gap,RHS,κ,TC,LEE）
  E1_auroc.csv            AUROC + 按 run bootstrap CI
  E1_loro.csv             leave-one-run-out 覆盖率范围
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "spwm_analysis"))
sys.path.insert(1, str(_ROOT))

from spwm_metrics_system import (  # noqa: E402
    compute_kappa, compute_all_lee, compute_tc, compute_baseline_metrics,
    load_eval_set, load_scores,
)

FAIL_Q = 0.25          # 事先固定：失败阈值 = J 范围的 25%
BOOT_B = 1000          # AUROC CI 的按 run 重采样次数
SEED = 42


def parse_step(ckpt_dir: Path) -> int:
    m = re.search(r"checkpoint_(\d+)", ckpt_dir.name)
    return int(m.group(1)) if m else -1


def collect_run(run_path: Path, name: str, eval_set, device: str,
                batch: int, n_dir: int, include_base: bool) -> list[dict]:
    """审计一个 run 的全部 checkpoint_* 目录。"""
    from dreamerv3_adapter import DreamerV3Adapter
    dirs = sorted(run_path.glob("checkpoint_*"))
    rows = []
    for ck in dirs:
        step = parse_step(ck)
        if step < 0:
            continue
        print(f"[{name}] analyzing {ck.name}")
        try:
            model = DreamerV3Adapter(str(ck), device=device)
            x = eval_set["x"][:batch].to(device)
            a = eval_set["a"][:batch].to(device)
            m = {}
            m.update(compute_kappa(model, x, n_directions=n_dir))
            m.update(compute_all_lee(model, x))
            z = model.encode(x)
            m["tc"] = compute_tc(z)
            if include_base:
                base = compute_baseline_metrics(
                    model, {k: v[:batch].to(device) for k, v in eval_set.items()})
                m.update(base)
            del model
        except Exception as e:
            print(f"[ERROR] {ck}: {e}")
            continue
        m["run"] = name
        m["step"] = step
        m["ckpt"] = str(ck)
        rows.append(m)
    return rows


def auroc_by_run_bootstrap(rhs: np.ndarray, fail: np.ndarray,
                           run_ids: np.ndarray, B: int = BOOT_B,
                           seed: int = SEED) -> dict:
    """AUROC + 按 run 重采样的 95% CI（任务书要求按 run bootstrap）。"""
    from sklearn.metrics import roc_auc_score  # 延迟导入
    runs = np.unique(run_ids)
    rng = np.random.default_rng(seed)

    def _auroc(idx_runs):
        sel = np.isin(run_ids, idx_runs)
        if fail[sel].sum() == 0 or (~fail[sel].astype(bool)).sum() == 0:
            return np.nan
        return roc_auc_score(fail[sel].astype(int), rhs[sel])

    point = _auroc(runs)
    boots = []
    for _ in range(B):
        pick = rng.choice(runs, size=len(runs), replace=True)
        boots.append(_auroc(pick))
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    return {"auroc": point,
            "ci_lo": float(np.percentile(boots, 2.5)) if len(boots) else np.nan,
            "ci_hi": float(np.percentile(boots, 97.5)) if len(boots) else np.nan}


def coverage_stats(rhs: np.ndarray, gap: np.ndarray,
                   beta: float) -> dict:
    covered = beta * rhs >= gap - 1e-12
    pos = gap > 0
    tight = (beta * rhs[pos] / gap[pos]) if pos.any() else np.array([])
    return {"coverage": float(covered.mean()) if len(covered) else np.nan,
            "tight_med": float(np.median(tight)) if len(tight) else np.nan,
            "tight_q1": float(np.percentile(tight, 25)) if len(tight) else np.nan,
            "tight_q3": float(np.percentile(tight, 75)) if len(tight) else np.nan,
            "n_violations": int((~covered).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True,
                    help="path=name，可多次")
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--calib_run", default="baseline_main")
    ap.add_argument("--calib_lo", type=int, default=50000)
    ap.add_argument("--calib_hi", type=int, default=500000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=5000,
                    help="任务书：5000 帧（与论文一致）")
    ap.add_argument("--kappa_dirs", type=int, default=128)
    ap.add_argument("--output_dir", default="rebuttal/E1")
    ap.add_argument("--metrics_cache", default="rebuttal/E1_all_checkpoints.csv",
                    help="已审计过的汇总 CSV；存在则直接读，不再过模型")
    ap.add_argument("--fresh", action="store_true", help="忽略缓存强制重算")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 读取或计算指标 ----------
    cache = Path(args.metrics_cache)
    if cache.exists() and not args.fresh:
        print(f"[cache] 复用 {cache}")
        df = pd.read_csv(cache)
    else:
        eval_set = load_eval_set(args.eval_set)
        frames = []
        for spec in args.runs:
            path, name = spec.split("=", 1)
            frames += collect_run(Path(path), name, eval_set, args.device,
                                    args.batch, args.kappa_dirs,
                                    include_base=True)
        df = pd.DataFrame(frames)
        df.to_csv(cache, index=False)

    # run 元信息：seed 与 λ_TC（从名字提取，符合任务书"记录可复现信息"）
    def _seed_of(run):
        m = re.search(r"seed(\d+)", run)
        return int(m.group(1)) if m else 0

    def _lam_of(run):
        m = re.search(r"lam([\d.]+)", run) or re.search(r"tc_sweep_([\d.]+)", run)
        return float(m.group(1)) if m else 0.0

    df["seed"] = df["run"].map(_seed_of)
    df["lambda_tc"] = df["run"].map(_lam_of)
    df["group"] = np.where(df["run"].str.startswith("tc_sweep"),
                            "overcompressed_sweep", "healthy")
    # 任务书注：过度压缩 = λ_TC 大（如 ≥0.1）导致性能下降的 runs；
    # 若 sweep 中 λ 小反而更好，分组判定在 E2 用 t_drop 数据驱动复核。
    df["lee"] = df[["lee_rotation", "lee_translation", "lee_scale"]].mean(axis=1)

    # ---------- 2. 统一 J*（跨 run 全局） ----------
    df = df.sort_values(["run", "step"]).reset_index(drop=True)
    J_max_all = df.groupby("run")["J_mean"].max().max() \
        if "J_mean" in df.columns else np.nan
    if np.isnan(J_max_all):
        # J_mean 可能尚未并入：从 scores.jsonl 补
        for spec in args.runs:
            path, name = spec.split("=", 1)
            try:
                s = load_scores(Path(path) / "scores.jsonl")
                key = next((k for k in ("episode/score", "score",
                                            "episode_reward", "return")
                                if k in s.columns), None)
                if key:
                    s2 = s[["step", key]].copy()
                    s2["step"] = pd.to_numeric(s2["step"], errors="coerce")
                    s2 = s2.groupby("step", as_index=False)[key].mean()
                    s2 = s2.rename(columns={key: "J_mean", "step": "step"})
                    df.loc[df["run"] == name, "J_mean"] = (
                        df.loc[df["run"] == name, "step"]
                          .map(s2.set_index("step")["J_mean"]))
            except Exception as e:
                print(f"[WARN] scores for {name}: {e}")
        J_max_all = df["J_mean"].max()
    J_min_all = df["J_mean"].min()
    df["J_max_all"] = J_max_all
    df["gap_raw"] = J_max_all - df["J_mean"]
    df["gap_norm"] = df["gap_raw"] / max(J_max_all - J_min_all, 1e-8)
    df["rhs_nat"] = df["kappa_mean"] + df["tc"] + df["lee"]
    df["fail"] = df["gap_norm"] > FAIL_Q

    # ---------- 3. 标定（仅 baseline_main 50k–500k）并冻结 ----------
    cal = df[(df["run"] == args.calib_run) &
             (df["step"] >= args.calib_lo) & (df["step"] <= args.calib_hi)]
    assert len(cal) >= 3, f"标定集太小: {len(cal)}"
    mm = {"lo": float(np.percentile(cal["rhs_nat"], 5)),
           "hi": float(np.percentile(cal["rhs_nat"], 95))}
    df["rhs_norm"] = np.clip((df["rhs_nat"] - mm["lo"]) /
                              max(mm["hi"] - mm["lo"], 1e-8), 0, 1)
    # cal 是 df 的视图（filter 结果），取 rhs_norm/gap_norm 需按索引对齐
    cal_idx = cal.index
    ratio = (df.loc[cal_idx, "gap_norm"] /
               np.maximum(df.loc[cal_idx, "rhs_norm"], 1e-12))
    beta_star = float(ratio.max())
    beta_90 = float(np.percentile(ratio, 90))
    freeze = {"beta_star": beta_star, "beta_90": beta_90,
               "minmax_params_nat": mm,
               "calib_run": args.calib_run,
               "calib_steps": [args.calib_lo, args.calib_hi],
               "J_max_all": float(J_max_all), "J_min_all": float(J_min_all),
               "fail_q": FAIL_Q}
    (out / "freeze_beta.json").write_text(json.dumps(freeze, indent=2))
    print(f"[freeze] β*={beta_star:.4f}  β90={beta_90:.4f}  "
           f"minmax={mm}")

    # ---------- 5. 测试集指标 ----------
    df["covered"] = beta_star * df["rhs_norm"] >= df["gap_norm"] - 1e-12

    # E1_summary.csv：每 run 一行
    summ = []
    for run, g in df.groupby("run"):
        st = coverage_stats(g["rhs_norm"].values, g["gap_norm"].values,
                             beta_star)
        summ.append({"run": run, "seed": g["seed"].iloc[0],
                      "lambda_tc": g["lambda_tc"].iloc[0],
                      "group": g["group"].iloc[0],
                      "n_ckpt": len(g), **st})
    pd.DataFrame(summ).to_csv(out / "E1_summary.csv", index=False)

    # 分组：健康 vs 过度压缩
    grp = df.groupby("group").apply(
        lambda g: pd.Series(coverage_stats(g["rhs_norm"].values,
                                            g["gap_norm"].values, beta_star)))
    grp.to_csv(out / "E1_group_summary.csv")

    # 违反点明细
    vio = df[~df["covered"]][["run", "seed", "lambda_tc", "step",
                               "gap_norm", "rhs_norm",
                               "kappa_mean", "tc", "lee"]]
    vio.to_csv(out / "E1_violations.csv", index=False)

    # AUROC（区分失败 vs 健康）
    try:
        au = auroc_by_run_bootstrap(df["rhs_norm"].values,
                                     df["fail"].values.astype(float),
                                     df["run"].values)
        pd.DataFrame([au]).to_csv(out / "E1_auroc.csv", index=False)
    except ImportError:
        pd.DataFrame([{"auroc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                        "note": "sklearn 未安装"}]).to_csv(
            out / "E1_auroc.csv", index=False)

    # ---------- 6. leave-one-run-out（baseline seeds 轮流标定） ----------
    loro_rows = []
    base_seeds = sorted(df[df["group"] == "healthy"]["seed"].unique())
    for s in base_seeds:
        cal_s = df[(df["run"].str.contains("baseline", regex=False)) &
                    (df["seed"] == s) &
                    (df["step"] >= args.calib_lo) &
                    (df["step"] <= args.calib_hi)]
        if len(cal_s) < 3:
            continue
        r_s = cal_s["gap_norm"] / np.maximum(cal_s["rhs_norm"], 1e-12)
        b_s = float(r_s.max())
        test = df[df["seed"] != s] if False else df[df["run"] !=
                    df.loc[cal_s.index[0], "run"]]
        cov = float((b_s * test["rhs_norm"] >=
                      test["gap_norm"] - 1e-12).mean())
        loro_rows.append({"calib_seed": s, "beta_star": b_s,
                           "test_coverage": cov})
    pd.DataFrame(loro_rows).to_csv(out / "E1_loro.csv", index=False)

    # ---------- 散点图 ----------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for run, g in df.groupby("run"):
        c = "tab:red" if run.startswith("tc_sweep") else \
             ("tab:orange" if "aug" in run else "tab:blue")
        ax.scatter(g["rhs_norm"], g["gap_norm"], s=26, alpha=0.75, c=c,
                    label=run)
    xs = np.linspace(0, df["rhs_norm"].max() * 1.05, 50)
    ax.plot(xs, beta_star * xs, "k--", label=f"y=β*x (β*={beta_star:.2f})")
    ax.set_xlabel("RHS (normalized, frozen calibration params)")
    ax.set_ylabel("gap (normalized, unified J*)")
    ax.set_title("E1: frozen-β coverage across runs")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out / "E1_scatter.png", dpi=200)
    plt.close()

    # ---------- 交回 CSV ----------
    keep = ["run", "seed", "lambda_tc", "group", "step", "ckpt",
             "J_mean", "gap_raw", "gap_norm", "kappa_mean", "tc",
             "lee_rotation", "lee_translation", "lee_scale", "lee",
             "rhs_nat", "rhs_norm", "covered", "fail",
             "one_step_mse", "recon_mse", "reward_mae"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df[keep].to_csv(out / "E1_all_checkpoints.csv", index=False)

    print(f"""
✅ E1 完成。
   总覆盖率: {df['covered'].mean():.1%}   健康组: {grp.loc['healthy','coverage'] if 'healthy' in grp.index else 'n/a'}
   过度压缩组: {grp.loc['overcompressed_sweep','coverage'] if 'overcompressed_sweep' in grp.index else 'n/a'}
   AUROC: 见 {out}/E1_auroc.csv
   ⚠ 若过度压缩组违反 bound（覆盖率明显 <1），这是最关键信息，直接报告。
""")


if __name__ == "__main__":
    main()
