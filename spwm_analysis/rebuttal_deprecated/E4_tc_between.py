#!/usr/bin/env python3
"""
E4_tc_between.py — Rebuttal E4：分类潜变量的组间 TC 修正
=====================================================================
回答 2q75 的 Q4。DreamerV3 的 stoch 潜变量是 32 组 × 64 类的 categorical
one-hot。对 one-hot 直接算 Gaussian TC：每组分量和恒为 1 → 协方差奇异，
simplex 约束会人为抬高 TC。

任务书步骤：
 1. 写清楚 Z 的定义（本仓库 adapter 实测）：
     Z = [deter (8192), stoch (32×64 = 2048, one-hot 采样展平)]
     —— dreamerv3_adapter.encode(): z_jax = concat([feat['deter'], stoch_flat])
     —— rssm.py observe(): stoch = self._dist(logit).sample(...)
     即 stoch 是【采样的 one-hot】（非 logits/概率）。
     ridge：compute_tc 用 eps=1e-3（协方差对角加 eps）。
 2. 组间 TC：TC_between = Σ_g logdet Σ̂_gg − logdet Σ̂
     每组先去掉一个类别（消 simplex 奇异），h 单独一块。
 3. h（deter）单独报告 TC_h。
 4. 新旧 TC 并排：E4_tc_compare.csv（step, run, TC_old, TC_between, TC_h）
     并给出 E1 汇总指标在新旧 TC 下的对照。

用法：
  python E4_tc_between.py --ckpt_root logdir/baseline_main \
      --eval_set shared_eval_set.npz --output_dir rebuttal/E4
  （或 --use_e1_csv rebuttal/E1_all_checkpoints.csv 直接复用 E1 的
   已存 latent 统计——若 E1 运行时用 --save_z 保存了 Z）
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "spwm_analysis"))
sys.path.insert(1, str(_ROOT))


def tc_gauss(Z: np.ndarray, eps: float = 1e-3) -> float:
    """与 spwm_metrics_system.compute_tc 同式（torch→numpy）。"""
    Zc = Z - Z.mean(0, keepdims=True)
    N, D = Zc.shape
    var = Zc.var(0, ddof=1) + eps
    t1 = np.log(var).sum()
    # det 引理：D>N 时用 N×N 小矩阵
    ZZt = (Zc @ Zc.T) / max(N - 1, 1)
    sign, logdet_small = np.linalg.slogdet(ZZt + eps * np.eye(N))
    logdet = (D - N) * np.log(eps) + logdet_small
    return 0.5 * (t1 - logdet)


def tc_between_groups(Z_stoch: np.ndarray, n_groups: int,
                       n_classes: int, eps: float = 1e-3) -> float:
    """组间 TC：每组去掉一个类别 → 组内 simplex 约束消除。

    Z_stoch: [N, n_groups*n_classes] one-hot 展平。
    Σ̂ = blockdiag over groups（加 eps 正则后用 slogdet）。
    TC_between = Σ_g logdet Σ̂_gg − logdet Σ̂
    其中 Σ̂_gg 是第 g 组协方差块。
    """
    Zc = Z_stoch - Z_stoch.mean(0, keepdims=True)
    N, D = Zc.shape

    blocks = []
    sum_block_logdet = 0.0
    for g in range(n_groups):
        B = Zc[:, g * n_classes:(g + 1) * n_classes][:, :-1]  # 去一维
        Cg = (B.T @ B) / max(N - 1, 1) + eps * np.eye(B.shape[1])
        sign, ld = np.linalg.slogdet(Cg)
        blocks.append(Cg)
        sum_block_logdet += ld

    # 全协方差（同样去掉每组最后一维，与块定义一致）
    Zd = np.concatenate(
        [Z_stoch[:, g * n_classes:(g + 1) * n_classes][:, :-1]
         for g in range(n_groups)], axis=1)
    Zd = Zd - Zd.mean(0, keepdims=True)
    C = (Zd.T @ Zd) / max(N - 1, 1) + eps * np.eye(Zd.shape[1])
    sign, ld_full = np.linalg.slogdet(C)

    return sum_block_logdet - ld_full


def tc_of_h(Z_deter: np.ndarray, eps: float = 1e-3) -> float:
    return tc_gauss(Z_deter, eps=eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--output_dir", default="rebuttal/E4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=5000)
    ap.add_argument("--n_groups", type=int, default=32)
    ap.add_argument("--n_classes", type=int, default=64)
    ap.add_argument("--use_e1_csv", default="",
                    help="提供时并排对比 E1 的旧 tc 列")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import torch  # 延迟
    from dreamerv3_adapter import DreamerV3Adapter

    data = np.load(args.eval_set)
    x_all = torch.from_numpy(data["x"]).float()

    rows = []
    for ck in sorted(Path(args.ckpt_root).glob("checkpoint_*")):
        m = re.search(r"checkpoint_(\d+)", ck.name)
        if not m:
            continue
        step = int(m.group(1))
        print(f"[E4] {ck.name}")
        try:
            model = DreamerV3Adapter(str(ck), device=args.device)
            with torch.no_grad():
                z = model.encode(x_all[:args.batch].to(args.device))
            z_np = z.cpu().numpy().astype(np.float64)

            D_deter = model.deter_size
            deter = z_np[:, :D_deter]
            stoch = z_np[:, D_deter:]          # [N, 2048]
            assert stoch.shape[1] == args.n_groups * args.n_classes, \
                f"stoch 形状 {stoch.shape} 与 {args.n_groups}×{args.n_classes} 不符"

            tc_old_full = tc_gauss(z_np)                    # 旧口径（整个 Z）
            tc_between = tc_between_groups(
                stoch, args.n_groups, args.n_classes)       # 组间（去一维）
            tc_h = tc_of_h(deter)                            # h 单独

            rows.append({"step": step, "run": Path(args.ckpt_root).name,
                          "TC_old": tc_old_full,
                          "TC_between": tc_between,
                          "TC_h": tc_h,
                          "z_def": "deter(8192)+stoch_onehot(32x64, sampled)"})
            del model
        except Exception as e:
            print(f"[ERROR] {ck}: {e}")

    df = pd.DataFrame(rows)

    # 并排 E1 旧值
    if args.use_e1_csv and Path(args.use_e1_csv).exists():
        e1 = pd.read_csv(args.use_e1_csv)
        e1s = e1[["run", "step", "tc"]].rename(columns={"tc": "TC_old_e1"})
        df = df.merge(e1s, on=["run", "step"], how="left")

    df.to_csv(out / "E4_tc_compare.csv", index=False)

    # ---------- E1 对照（新旧 TC 下重算 rhs/覆盖——交给 E1 重跑，这里给对照底稿） ----------
    if args.use_e1_csv and Path(args.use_e1_csv).exists():
        e1 = pd.read_csv(args.use_e1_csv)
        e1 = e1.merge(df[["run", "step", "TC_between"]],
                          on=["run", "step"], how="left")
        e1["rhs_nat_newtc"] = (e1["kappa_mean"] + e1["TC_between"].fillna(e1["tc"])
                                  + e1["lee"])
        e1["rhs_norm_newtc"] = np.clip(
            (e1["rhs_nat_newtc"] - e1["rhs_nat_newtc"].min()) /
            max(e1["rhs_nat_newtc"].max() - e1["rhs_nat_newtc"].min(), 1e-8),
            0, 1)
        # 与旧 RHS 归一化相关性（诊断级对照）
        from scipy.stats import spearmanr
        rho = spearmanr(e1["rhs_norm"], e1["rhs_norm_newtc"],
                          nan_policy="omit").correlation
        pd.DataFrame([{"spearman_rhs_old_vs_new": rho}]).to_csv(
            out / "E4_e1_compare.csv", index=False)

    # ---------- 交回说明文字（直接用于 rebuttal） ----------
    note = (
        "Z 的定义（本仓库实测）：Z = [deter ∈ R^{8192}; stoch ∈ "
        "{0,1}^{32×64}]，stoch 为 categorical 采样后的 one-hot 展平"
        "（rssm.py observe() 中 self._dist(logit).sample()），"
        "非 logits/概率。旧 TC（论文）对整个 Z 直接套 Gaussian 闭式解，"
        "协方差以 eps=1e-3 岭正则。one-hot 的 simplex 约束使每组协方差"
        "奇异，人为抬高 TC；修正后的 TC_between 每组去掉一个类别再按组"
        "分块计算，h（deter）作为独立块单独报告为 TC_h。\n"
    )
    (out / "E4_z_definition.txt").write_text(note, encoding="utf-8")
    print(note)
    print(f"✅ E4 完成 → {out}/E4_tc_compare.csv")


if __name__ == "__main__":
    main()
