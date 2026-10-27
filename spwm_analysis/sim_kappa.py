# Reproduce the repo's compute_kappa pipeline on a model whose encoder output
# does NOT depend on tiny input changes, to see what kappa_hat / sigma values
# arise purely from (a) uint8 quantisation and (b) categorical resampling.
import numpy as np
rng = np.random.default_rng(0)
B, K, eps = 256, 128, 1e-3
H = W = 64; C = 3; D = H*W*C
G, CL, DETER = 32, 64, 8192

# (a) how many uint8 pixels change when x -> x + eps*v, ||v||=1 (per sample)?
# Real frames are stored as k/255, so use quantized values (continuous x would
# wrongly suggest that the probe is erased; see sim_quant_tc.py).
x = (rng.integers(0, 256, (B, D)) / 255.0).astype(np.float32)
v = rng.standard_normal((B, D)); v /= np.linalg.norm(v, axis=1, keepdims=True)
q0 = (np.clip(x, 0, 1)*255).astype(np.uint8)
q1 = (np.clip(x+eps*v, 0, 1)*255).astype(np.uint8)
changed = (q0 != q1).sum(1)
dq = (q1.astype(int) - q0.astype(int)) / 255.0
print(f"pixels changed by eps*v after uint8 cast: {changed.mean()/D*100:.1f}%, "
      f"effective probe norm {np.linalg.norm(dq, axis=1).mean():.3f} (= {np.linalg.norm(dq, axis=1).mean()/eps:.0f} x eps)")

# (b) encoder that is blind to the perturbation: posterior probs fixed per frame,
# z = [constant deter ; one-hot sample]. Each encode() call resamples.
def run(temp):
    logits = rng.standard_normal((B, G, CL)) * temp
    p = np.exp(logits); p /= p.sum(-1, keepdims=True)
    p = 0.99*p + 0.01/CL                     # unimix 0.01 as in config
    ent = -(p*np.log(p)).sum(-1).mean()
    def sample():
        u = rng.random((B, G, 1)); c = p.cumsum(-1)
        idx = (u > c).sum(-1).clip(max=CL-1)
        oh = np.zeros((B, G, CL)); np.put_along_axis(oh, idx[..., None], 1, -1)
        return np.concatenate([np.zeros((B, DETER)), oh.reshape(B, -1)], 1)
    z0 = sample()
    s = np.stack([np.linalg.norm(sample()-z0, axis=1)/eps for _ in range(K)])
    smax, smin = s.max(0), s.min(0)
    kap = smax/(smin+1e-8)
    return ent, kap.mean(), np.median(kap), np.quantile(kap, .9), smax.mean(), smin.mean()

print("temp  entropy/group  kappa_mean  kappa_p50  kappa_p90  sigma_max  sigma_min")
for t in [2.0, 3.0, 4.0, 5.0, 6.0]:
    r = run(t)
    print(f"{t:4.1f}  {r[0]:13.2f}  {r[1]:10.2f}  {r[2]:9.2f}  {r[3]:9.2f}  {r[4]:9.0f}  {r[5]:9.0f}")
print("Paper Fig.5: kappa_mean 1.5-1.8, kappa_p90 1.6-2.2, sigma_max 5250-6500, sigma_min 3000-4500")
