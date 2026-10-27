#!/usr/bin/env python3
"""
collect_crafter_with_info.py -- same frames as collect_real_crafter.py
(random policy, crafter.Env native API, seed 42) but ALSO saves the semantic
state needed for the d_S checks (2q75 Q3, bajJ Q2):

  pos      (N, 2)  player position
  inv      (N, K)  inventory counts (sorted item names, see 'inv_keys')
  sem_win  (N, 9, 7) local semantic window around the player (-1 outside map)

CPU only, a few minutes. Output keys x, a, x_next, r are identical in format to
collect_real_crafter.py, so audit_v2.py can use the file directly.
"""
import argparse
import json
import numpy as np


def local_window(sem, pos, w=9, h=7):
  H, W = sem.shape
  out = -np.ones((w, h), dtype=np.int16)
  x0, y0 = int(pos[0]) - w // 2, int(pos[1]) - h // 2
  for i in range(w):
    for j in range(h):
      xi, yj = x0 + i, y0 + j
      if 0 <= xi < H and 0 <= yj < W:
        out[i, j] = sem[xi, yj]
  return out


def collect(n_samples=5000, seed=42, max_episodes=500):
  import crafter
  rng = np.random.RandomState(seed)
  chw = lambda img: np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
  X, A, XN, R, POS, INV, WIN = [], [], [], [], [], [], []
  inv_keys = None
  ep = 0
  while len(X) < n_samples and ep < max_episodes:
    env = crafter.Env(seed=seed + ep)
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    # first step only to obtain the info of the current frame
    prev_obs, prev_info, done = np.asarray(obs), None, False
    steps = 0
    while not done and len(X) < n_samples and steps < 10000:
      a = rng.randint(0, 17)
      nxt, rew, done, info = env.step(a)[:4]
      nxt = np.asarray(nxt)
      if prev_info is not None:  # we know the semantic state of prev_obs
        if inv_keys is None:
          inv_keys = sorted(prev_info['inventory'].keys())
        X.append(chw(prev_obs)); XN.append(chw(nxt))
        one = np.zeros(17, np.float32); one[a] = 1.0; A.append(one)
        R.append(np.array([rew], np.float32))
        POS.append(np.asarray(prev_info['player_pos'], np.int16))
        INV.append(np.array([prev_info['inventory'][k] for k in inv_keys], np.int16))
        WIN.append(local_window(np.asarray(prev_info['semantic']), prev_info['player_pos']))
      prev_obs, prev_info = nxt, info
      steps += 1
    ep += 1
    print(f'  {len(X)}/{n_samples} frames after {ep} episodes', flush=True)
  return dict(x=np.stack(X), a=np.stack(A), x_next=np.stack(XN), r=np.stack(R),
              pos=np.stack(POS), inv=np.stack(INV), sem_win=np.stack(WIN),
              inv_keys=np.array(json.dumps(inv_keys)))


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--n_samples', type=int, default=5000)
  p.add_argument('--seed', type=int, default=42)
  p.add_argument('--output_path', default='shared_eval_set_info.npz')
  a = p.parse_args()
  d = collect(a.n_samples, a.seed)
  np.savez_compressed(a.output_path, **d)
  print('saved', a.output_path, {k: v.shape for k, v in d.items()})
