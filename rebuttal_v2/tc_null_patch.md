# T0 补丁：给 tc_between 加零点对照

## 为什么需要

`tc_between = 0.5·(Σ_g logdet S_gg − logdet S)` 在有限样本下有系统性正偏差。
满协方差是 2016×2016（32 组 × 63 类），块只有 63×63，两项的 logdet 偏差
**不会抵消**，且偏差随后验熵下降而急剧增大。

本地合成数据实测（组间完全独立，真实 TC = 0，N=5000，ridge=1e-3）：

| 后验熵 (nats/组) | 4.06 | 3.74 | 2.88 | 2.36 | 1.81 | 1.13 |
| --- | --- | --- | --- | --- | --- | --- |
| tc_between | 0.4 | 6.9 | 52.0 | 87.5 | 122.5 | 156.3 |

论文 Fig.5 报的 TC 75–128 nats 落在熵 1.8–2.9 这一档。

κ 有 `null_kappa_mean`、LEE 有 `floor_resample_norm`，**只有 TC 没有零点**，
而它是进 RHS 的三项之一。ridge 三档敏感性不能替代零点——实测 ridge 从
1e-4 挪到 1e-2，TC 可以差近千倍。

## 补丁

在 `audit_v2.py` 的 `tc_between` 之后新增：

```python
def tc_between_null(p, ridge=1e-3, n_rep=3, seed=0):
    """Same estimator with between-group dependence destroyed.

    Permutes the frame order independently within each group. Every group's
    marginal distribution -- and therefore its entropy -- is unchanged, so the
    only thing removed is the between-group dependence. What remains is the
    finite-sample bias at that entropy.
    """
    N, G, C = p.shape
    vals = []
    for r in range(n_rep):
        rng = np.random.default_rng(seed + r)
        q = np.empty_like(p)
        for g in range(G):
            q[:, g, :] = p[rng.permutation(N), g, :]
        vals.append(tc_between(q, ridge=ridge))
    return float(np.mean(vals))
```

在 `audit_checkpoint()` 中，`row['tc_between'] = row['tc_between_r0.001']`
之后加入两行：

```python
row['tc_null'] = tc_between_null(p, ridge=1e-3)
row['tc_snr'] = row['tc_between'] / max(row['tc_null'], 1e-9)
```

## 开销

`tc_between` 每次要算一次 2016×2016 的 `Z.T @ Z` 和 `slogdet`，N=5000 时约 3–7 秒。
原本每个 checkpoint 跑 3 次（三个 ridge），加 3 次零点后变成 6 次，
单点增加约 10–20 秒，相对 A2 单点 3–5 分钟是约 7% 的增量。

如果 A2 时间紧张，可以只对 M0 的 3 个点开零点（给 `--steps` 的时候单独跑一次），
其余点沿用 M0 得到的 `tc_snr` 量级做说明。但这一条要写进保密意见。

## 预注册文件同步改两处

`channels.tc` 末尾追加：

```
; tc_null = same estimator after permuting frame order independently within
each group (3 reps, seed 0), tc_snr = tc_between / tc_null
```

M0 的判据增加一项：

- `tc_snr > 2`：TC 通道可用，可按 v2 口径进 RHS
- `tc_snr ≈ 1`：TC 在该设置下测不出东西；这也解释了论文「TC 随回报上升」为何是伪影

两种结果都要如实报告，不要预先选边。

## 校验方式

把 `tc_between_null` 作用在**本来就独立**的数据上，`tc_snr` 应该 ≈ 1.00
（本地实测：真独立时 13.2 / 13.2 = 1.00）。如果明显偏离 1，说明实现有问题。

## 另外要看的两个数

`audit_v2.py` 已经记录 `post_entropy`，但 `analysis_v2.py` 里没有任何熵的处理，
所以下面两件事需要人工对照：

1. `post_entropy` 随 step 的走向
2. `tc_between` 与 `post_entropy` 是否强负相关

若两者强负相关，说明 TC 通道的跨 checkpoint 比较被后验熵混淆；
这也正好是审稿人 2q75 W2/Q2 所观察到的「TC 随回报上升」的机制解释。
