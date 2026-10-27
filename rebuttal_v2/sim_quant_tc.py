"""Two checks with realistic inputs.
(1) uint8 cast on frames that are already k/255 (real Crafter frames are uint8):
    how much of the eps=1e-3 probe survives?
(2) Paper TC estimator (compute_tc, N=256, eps=1e-3) on a latent with
    32 INDEPENDENT one-hot groups (true TC = 0) plus a constant 8192-dim h."""
import numpy as np
rng = np.random.default_rng(0)
B, D = 256, 64 * 64 * 3
k = rng.integers(0, 256, (B, D))
x = (k / 255.0).astype(np.float32)                    # how frames are stored (float32 CHW)
v = rng.standard_normal((B, D)).astype(np.float32)
v /= np.linalg.norm(v, axis=1, keepdims=True)
q0 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
q1 = (np.clip(x + 1e-3 * v, 0, 1) * 255).astype(np.uint8)
dq = (q1.astype(int) - q0.astype(int)) / 255.0
print(f"(1) baseline cast already off by one level: {(q0 != k).mean()*100:.1f}% of pixels")
print(f"    pixels changed by the probe: {(q0 != q1).mean()*100:.1f}%;"
      f" effective probe norm {np.linalg.norm(dq, axis=1).mean():.3f} vs eps=1e-3"
      f" -> {np.linalg.norm(dq, axis=1).mean()/1e-3:.0f}x")

def tc_paper(Z, eps=1e-3):
    Z = Z - Z.mean(0, keepdims=True); N, Dz = Z.shape
    t1 = np.log(Z.var(0, ddof=1) + eps).sum()
    _, ls = np.linalg.slogdet(Z @ Z.T / (N - 1) + eps * np.eye(N))
    return 0.5 * (t1 - ((Dz - N) * np.log(eps) + ls))
G, C = 32, 64
for ent_temp in (2.0, 4.0, 6.0):
    logits = rng.standard_normal((G, C)) * ent_temp            # same distribution for every frame
    p = np.exp(logits); p /= p.sum(-1, keepdims=True); p = .99 * p + .01 / C
    idx = np.stack([rng.choice(C, size=256, p=p[g]) for g in range(G)], 1)
    oh = np.zeros((256, G, C)); np.put_along_axis(oh, idx[..., None], 1, -1)
    Z = np.concatenate([np.full((256, 8192), 0.3), oh.reshape(256, -1)], 1)
    ent = -(p * np.log(p)).sum(-1).mean()
    print(f"(2) groups independent (true TC=0), entropy/group {ent:.2f}: paper TC = {tc_paper(Z):.1f} nats")
print("    paper Fig. 5 TC: 75-128 nats")
