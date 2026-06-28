#!/usr/bin/env python3
"""
collect_real_crafter.py — 用 crafter 原生 API 收集真实游戏帧,绕过 embodied。

针对 vast.ai 上 embodied 循环导入问题的干净解法:
不依赖 embodied,不依赖 gym.make("Crafter-v0"),直接用 crafter.Env。

已核实的 crafter 原生 API:
  - crafter.Env(seed) -> reset() 返回 (64,64,3) uint8 numpy
  - step(int_action) 返回 (obs, reward, done, info)  # 老 gym 4元组
  - action 空间 17 个离散动作,用 np.random.randint(0,17)

输出格式与 spwm 的 LEE 测量对齐:
  - x, x_next: (N, 3, 64, 64) float32 in [0,1]  (CHW!)
  - a: (N, 17) one-hot float32
  - r: (N, 1) float32
"""
import argparse
import numpy as np


def collect(n_samples=5000, n_episodes=50, seed=42):
    import crafter
    rng = np.random.RandomState(seed)

    def to_chw01(img_hwc_uint8):
        # (64,64,3) uint8 [0,255] -> (3,64,64) float32 [0,1]
        t = img_hwc_uint8.astype(np.float32) / 255.0
        return np.transpose(t, (2, 0, 1))  # HWC -> CHW

    def to_onehot17(a_int):
        one = np.zeros(17, dtype=np.float32)
        one[int(a_int)] = 1.0
        return one

    xs, as_, xns, rs = [], [], [], []
    collected = 0
    ep = 0
    while collected < n_samples and ep < n_episodes * 10:  # 安全上限
        env = crafter.Env(seed=seed + ep)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        curr = np.asarray(obs)
        done = False
        steps_this_ep = 0
        while not done and collected < n_samples:
            a = rng.randint(0, 17)
            result = env.step(a)
            # 老 gym 4元组: (obs, reward, done, info)
            nxt, reward, done = result[0], result[1], result[2]
            nxt = np.asarray(nxt)
            xs.append(to_chw01(curr))
            as_.append(to_onehot17(a))
            xns.append(to_chw01(nxt))
            rs.append(np.array([reward], dtype=np.float32))
            curr = nxt
            collected += 1
            steps_this_ep += 1
            if steps_this_ep > 10000:  # 防止单 episode 卡死
                break
        ep += 1
        if collected % 500 == 0:
            print(f"  collected {collected}/{n_samples} frames ({ep} episodes)")

    x = np.stack(xs); a = np.stack(as_); xn = np.stack(xns); r = np.stack(rs)
    assert x.shape[1:] == (3, 64, 64), f"obs shape wrong: {x.shape}"
    assert a.shape[1] == 17, f"action dim wrong: {a.shape}"
    return {'x': x, 'a': a, 'x_next': xn, 'r': r}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_samples', type=int, default=5000)
    p.add_argument('--n_episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_path', type=str, default='shared_eval_set.npz')
    args = p.parse_args()

    print(f"Collecting {args.n_samples} REAL Crafter frames via native API...")
    es = collect(args.n_samples, args.n_episodes, args.seed)
    np.savez_compressed(args.output_path, **es)
    print(f"\n✓ Saved {es['x'].shape[0]} real frames to {args.output_path}")
    print(f"  x: {es['x'].shape} {es['x'].dtype} range=[{es['x'].min():.3f},{es['x'].max():.3f}]")
    print(f"  a: {es['a'].shape}, r: {es['r'].shape}")


if __name__ == '__main__':
    main()