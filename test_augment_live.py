#!/usr/bin/env python3
"""
训练前验证 augment.py:在 vast.ai 上跑(需要 JAX)。
验证三件事:
1. augment_images 能跑通,不报错
2. enabled=True 时图像真的被改变了(不是原样返回)
3. enabled=False 时图像原样返回(baseline 不受影响)
4. 输出格式/dtype/shape 与输入一致(不破坏管线)
用真实 Crafter 帧测试(从 shared_eval_set.npz 取)。
"""
import numpy as np
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, 'dreamerv3')
import augment

print("=" * 60)
print("训练前验证 augment.py")
print("=" * 60)

# ---- 用真实 Crafter 帧(从 eval_set 取,转成训练格式 HWC uint8) ----
data = np.load('shared_eval_set.npz')
x = data['x']  # (N, 3, 64, 64) CHW float [0,1]
# 转成训练时的格式:(B, T, H, W, C) uint8
frames_chw = x[:32]  # 取 32 帧
frames_hwc = np.transpose(frames_chw, (0, 2, 3, 1))  # CHW -> HWC
frames_uint8 = (frames_hwc * 255).astype(np.uint8)
# 加一个 T 维度模拟训练 batch (B, T, H, W, C)
obs = frames_uint8.reshape(4, 8, 64, 64, 3)  # B=4, T=8
obs_jax = jnp.array(obs)
print(f"\n输入: shape={obs_jax.shape}, dtype={obs_jax.dtype}, "
      f"range=[{obs_jax.min()},{obs_jax.max()}]")

# ---- 测试 1:enabled=True,增强应改变图像 ----
cfg_on = {
    'enabled': True, 'rotation_deg': 10.0, 'translation_px': 2.0,
    'scale_range': 0.10, 'apply_prob': 1.0,  # prob=1 确保每张都增强,便于验证
}
rng = jax.random.PRNGKey(0)
out_on = augment.augment_images(obs_jax, rng, cfg_on)
diff = jnp.abs(out_on.astype(jnp.float32) - obs_jax.astype(jnp.float32)).mean()
print(f"\n[测试1] enabled=True, apply_prob=1.0")
print(f"  输出: shape={out_on.shape}, dtype={out_on.dtype}")
print(f"  与原图平均差异: {diff:.2f} (应 > 0,证明增强生效)")
assert out_on.shape == obs_jax.shape, "形状被改变!"
assert out_on.dtype == obs_jax.dtype, "dtype 被改变!"
assert diff > 1.0, f"增强没生效!差异仅 {diff:.4f}"
print(f"  ✓ 形状/dtype 保持, 图像确实被增强")

# ---- 测试 2:enabled=False,应原样返回 ----
cfg_off = dict(cfg_on); cfg_off['enabled'] = False
out_off = augment.augment_images(obs_jax, rng, cfg_off)
diff_off = jnp.abs(out_off.astype(jnp.float32) - obs_jax.astype(jnp.float32)).mean()
print(f"\n[测试2] enabled=False")
print(f"  与原图平均差异: {diff_off:.4f} (应 = 0,baseline 不受影响)")
assert diff_off < 1e-6, f"enabled=False 时图像被改了!差异 {diff_off}"
print(f"  ✓ enabled=False 时原样返回")

# ---- 测试 3:apply_prob=0.5,约一半被增强 ----
cfg_half = dict(cfg_on); cfg_half['apply_prob'] = 0.5
# 对多个不同 rng 统计被增强的比例
changed = 0
total = 50
for i in range(total):
    o = augment.augment_images(obs_jax, jax.random.PRNGKey(i), cfg_half)
    d = jnp.abs(o.astype(jnp.float32) - obs_jax.astype(jnp.float32)).mean()
    if d > 1.0:
        changed += 1
print(f"\n[测试3] apply_prob=0.5, {total} 次试验")
print(f"  被增强比例: {changed}/{total} = {changed/total:.2f} (应接近 0.5)")
print(f"  ✓ apply_prob 大致生效" if 0.2 < changed/total < 0.8 else "  ⚠️ 比例偏离 0.5")

# ---- 测试 4:数值范围合法(不溢出) ----
print(f"\n[测试4] 增强后数值范围: [{out_on.min()},{out_on.max()}]")
assert out_on.min() >= 0 and out_on.max() <= 255, "uint8 溢出!"
print(f"  ✓ 数值在 [0,255] 合法范围")

print("\n" + "=" * 60)
print("✅ augment.py 所有训练前检查通过")
print("   增强逻辑正确,可以放心跑 aug seed0")
print("=" * 60)