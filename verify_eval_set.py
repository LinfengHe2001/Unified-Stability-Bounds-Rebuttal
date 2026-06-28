#!/usr/bin/env python3
"""
verify_eval_set.py  —  开跑 2 天训练之前的最后一道保险。

用途:加载 create_eval_set_dreamer.py 生成的 eval_set.npz,判断它装的是
【真实 Crafter 游戏帧】还是【静默退回的合成噪声】。

背景:create_eval_set_dreamer.py 的 --method env 有一个隐蔽陷阱——如果
crafter 包没装好,它会 silently 调用 create_synthetic_eval_set() 给你
torch.rand 噪声,且不报错。在噪声上测 LEE 是无意义的。这个脚本帮你
在烧 GPU 之前拦住这种情况。

无需 GPU / JAX / torch,纯 numpy。用法:
    python verify_eval_set.py shared_eval_set.npz
"""
import sys
import json
import numpy as np


def describe(arr, name):
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, "
          f"range=[{arr.min():.3f}, {arr.max():.3f}], mean={arr.mean():.3f}")


def is_real_frames(x):
    """
    用多个统计信号区分真实游戏帧 vs 合成噪声。
    返回 (verdict: bool, reasons: list[str]).

    真实 Crafter 帧的特征:
      - 强空间自相关(相邻像素相似,因为有大块草地/物体)
      - 颜色分布不均匀(大量绿色草地,直方图有尖峰)
      - 通道间相关(RGB 不独立)
    合成 torch.rand 噪声的特征:
      - 几乎无空间自相关(相邻像素独立)
      - 颜色均匀分布(直方图平坦)
      - 通道间无相关
    """
    reasons = []
    # x: (N, 3, 64, 64) in [0,1]; 取前 256 帧做统计足够
    n = min(256, x.shape[0])
    sample = x[:n].astype(np.float64)

    # ---- 信号 1:空间自相关(相邻像素差的方差) ----
    # 真实帧:相邻像素很像 -> 差很小;噪声:差很大
    dh = np.abs(sample[:, :, 1:, :] - sample[:, :, :-1, :]).mean()
    dw = np.abs(sample[:, :, :, 1:] - sample[:, :, :, :-1]).mean()
    neighbor_diff = (dh + dw) / 2
    # 经验阈值:真实帧邻差通常 < 0.08;均匀噪声邻差 ≈ 0.33
    sig1_real = neighbor_diff < 0.15
    reasons.append(
        f"空间自相关: 邻像素平均差={neighbor_diff:.4f} "
        f"({'真实帧特征(<0.15)' if sig1_real else '噪声特征(>=0.15)'})")

    # ---- 信号 2:颜色分布的峰度/不均匀性 ----
    # 真实帧直方图有尖峰(草地绿);噪声直方图平坦
    hist, _ = np.histogram(sample.ravel(), bins=32, range=(0, 1), density=True)
    hist_std = hist.std()  # 平坦分布 std 小,有峰 std 大
    sig2_real = hist_std > 0.5
    reasons.append(
        f"颜色分布: 直方图std={hist_std:.3f} "
        f"({'真实帧特征(>0.5,有尖峰)' if sig2_real else '噪声特征(<=0.5,平坦)'})")

    # ---- 信号 3:通道间相关 ----
    # 真实帧 RGB 相关;噪声 RGB 独立(相关≈0)
    r = sample[:, 0].ravel(); g = sample[:, 1].ravel(); b = sample[:, 2].ravel()
    rg = abs(np.corrcoef(r, g)[0, 1])
    gb = abs(np.corrcoef(g, b)[0, 1])
    chan_corr = (rg + gb) / 2
    sig3_real = chan_corr > 0.15
    reasons.append(
        f"通道相关: |corr(R,G)|,|corr(G,B)|均值={chan_corr:.3f} "
        f"({'真实帧特征(>0.15)' if sig3_real else '噪声特征(<=0.15,独立)'})")

    # 三个信号至少 2 个支持"真实",才判定为真实帧
    votes = sum([sig1_real, sig2_real, sig3_real])
    verdict = votes >= 2
    reasons.append(f"投票: {votes}/3 信号支持'真实帧'")
    return verdict, reasons


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_eval_set.py <eval_set.npz>")
        sys.exit(1)
    path = sys.argv[1]
    print("=" * 60)
    print(f"验证 eval_set: {path}")
    print("=" * 60)

    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    print(f"包含字段: {keys}\n")

    # 形状检查
    print("字段详情:")
    for k in ['x', 'a', 'x_next', 'r']:
        if k in data:
            describe(data[k], k)
    print()

    # metadata 检查(如果 create_eval_set 写了)
    if 'metadata' in data:
        try:
            meta = json.loads(str(data['metadata']))
            print(f"metadata: {meta}")
            if meta.get('method') == 'synthetic':
                print("  ⚠️ metadata 显示 method=synthetic — 这是合成数据!")
            print()
        except Exception:
            pass

    x = data['x']
    # 基本形状断言
    print("形状检查:")
    if x.shape[1:] == (3, 64, 64):
        print(f"  ✓ 观测形状 {x.shape[1:]} == (3,64,64)")
    else:
        print(f"  ✗ 观测形状 {x.shape[1:]} != (3,64,64),格式可能不对!")
    if x.shape[0] >= 1000:
        print(f"  ✓ 样本数 {x.shape[0]} 足够 (>=1000)")
    else:
        print(f"  ⚠️ 样本数 {x.shape[0]} 偏少,LEE 估计方差会大")
    print()

    # 核心:真实帧 vs 噪声
    print("真实帧 vs 合成噪声判定:")
    verdict, reasons = is_real_frames(x)
    for r in reasons:
        print(f"  - {r}")
    print()
    print("=" * 60)
    if verdict:
        print("✅ 判定: 真实 Crafter 帧。可以用于 LEE 干预实验。")
    else:
        print("❌ 判定: 大概率是合成噪声!")
        print("   原因: crafter 包可能没装好,create_eval_set 静默退回了噪声。")
        print("   处理: 1) pip install crafter")
        print("         2) 重新跑 --method env 生成 eval_set")
        print("         3) 再次运行本脚本确认,然后才开始训练")
        print("   ⚠️ 不要用噪声 eval_set 跑 2 天训练,LEE 结果会无意义!")
    print("=" * 60)


if __name__ == "__main__":
    main()