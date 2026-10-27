# DEPRECATED — 请勿使用

本目录中的脚本由另一轮方案（E0–E4 + agent.py TC 惩罚补丁）产生，**已被 rebuttal_v2 取代**，
不再作为 rebuttal 的复现路径。保留仅为留档。

## 为什么被取代

| 脚本                    | 主要问题                                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `E0_fact_check.py`      | J 取 ±5000 步内的训练回报；口径 (b) 用其他 run 单局回报最大值抬高 J_max^all；对 “late 500K” 的 step 是猜测而非实测 |
| `E1_frozen_beta_test.py`| 先把三通道原始值相加再 min–max 截到 [0,1]，被数值最大的 TC 主导；最健康标定点 RHS=0 而 gap>0，导致 β* 爆炸（合成数据复现出 β*≈2e11）；J 按精确 step 匹配多为 NaN；AUROC 与训练阶段混淆 |
| `E2_leadlag.py`         | 每个 seed 只有一个 baseline run，“seed 间标准差” 为 NaN，退化为时间差分中位数；TC 惩罚 run 中 TC 先报警是惩罚项自身造成却被当作证据；无统计误报率 |
| `E3_baselines.py`       | 同样对原始总和 min–max；one-step MSE 来自采样得到的潜变量（带采样噪声）                                              |
| `E4_tc_between.py`      | 在采样的 one-hot 上计算；缺少 ½ 因子；TC_old 用 5000 帧而论文用 256 帧，口径不可比                              |

对应的 `dreamerv3/agent.py` 中的 TC 惩罚补丁也已还原：
`losses['tc_pen']` 为标量，与 `assert all(x == (B, T))` 冲突；且不在 `self.scales` 中；
λ=0 时该分支不执行，故冒烟测试无法暴露该崩溃。已通过
`git checkout dreamerv3/agent.py dreamerv3/configs.yaml` 还原到 HEAD `7544ec5`。

## 用什么代替

- 测量与审计 → `spwm_analysis/audit_v2.py`（后验概率上的 κ_pair / tc_between / lee_v2 / lee_fit / alias rate / one-step KL；先跑 `--selftest`）
- 分析与统计 → `spwm_analysis/analysis_v2.py`（纯 CPU：冻结 β 的 held-out 测试、conformal 校准、重拟合 β 消融、基线对比、lead-lag）
- 事实核对 → 只需人工打开 `bound_vs_gap_with_late_note.txt` 看 step，并用 CSV 核对 Table 2 / Fig. 3–4 的显示数字
- 原口径对照与噪声底 → `audit_v2.py --paper_diag`

原始来源：`rebuttal_v2/`（已归档在仓库根目录）。
