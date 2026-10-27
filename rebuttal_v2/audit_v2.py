#!/usr/bin/env python3
"""
audit_v2.py -- corrected checkpoint audit for the rebuttal.

Why this exists (see AUDIT_FINDINGS.md):
  * The paper pipeline encodes with a SAMPLED one-hot latent (rssm._observe calls
    .sample()) and a fresh RNG on every encode() call. Inputs are cast to uint8
    by truncation, so on real (k/255) frames the eps=1e-3 probe becomes a
    one-level step on ~50% of pixels (~300x eps): kappa is not a local Jacobian
    estimate, and kappa/LEE are dominated by categorical resampling noise.
  * The paper's Gaussian TC on [h; one-hot] is dominated by the sum-to-one
    constraint and the ridge (independent groups give 86-892 nats; see
    sim_quant_tc.py).
  * The deterministic part h of Z is constant across inputs for a single observe
    step from the initial carry.
This script measures the channels on the deterministic posterior probabilities
and also reports the noise-floor diagnostics that show the problem.

Place this file in spwm_analysis/ (next to dreamerv3_adapter.py) and run on the
GPU machine. It was written against the repo code but could NOT be executed in
the environment where it was written (no JAX). Run the --selftest first.

Usage
  python spwm_analysis/audit_v2.py --selftest --ckpt logdir/<run>/checkpoint_00500000 \
      --eval_set shared_eval_set_info.npz
  python spwm_analysis/audit_v2.py --run_dir logdir/<run> --run_name main --seed 0 \
      --group main --eval_set shared_eval_set_info.npz --out results_v2/main.csv
Options
  --steps 50000,100000,...   restrict to these checkpoint steps
  --mask_k K                 clamp the first K categorical groups (T4 experiment)
  --paper_diag               also run the paper's sampled kappa/LEE/TC + noise floors (slower)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(1, str(_HERE.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ninjax as nj  # noqa: E402
import torch  # noqa: E402  (only for the paper's torchvision transforms)

from dreamerv3_adapter import DreamerV3Adapter  # noqa: E402
from spwm_metrics_system import (  # noqa: E402
    affine_rotate, translate, isotropic_scale, compute_tc,
)

CHUNK = 500


# ----------------------------------------------------------------------------
# Deterministic posterior
# ----------------------------------------------------------------------------
class ProbAdapter(DreamerV3Adapter):
  """DreamerV3Adapter + deterministic access to posterior logits/probabilities."""

  def __init__(self, ckpt, device='cuda', mask_k=0, jit=True):
    super().__init__(ckpt, device=device)
    self.mask_k = int(mask_k)
    dyn_cfg = self.config.get('agent', self.config)['dyn']['rssm']
    self.unimix = float(dyn_cfg.get('unimix', 0.01))
    # observe -> return logits (deterministic) and deter
    self._post = nj.pure(lambda c, tok, act, reset: self.dyn.observe(
        c, tok, act, reset, training=False, single=True))
    self._prior_fn = nj.pure(lambda deter: self.dyn._prior(deter))
    # The paper adapter calls these un-jitted (slow). jit once per chunk shape.
    if jit:
      self._enc_apply = jax.jit(self._enc_apply)
      self._dyn_observe = jax.jit(self._dyn_observe)
      self._post = jax.jit(self._post)
      self._prior_fn = jax.jit(self._prior_fn)

  # ---- helpers -------------------------------------------------------------
  @staticmethod
  def _to_uint8(x_chw01: np.ndarray) -> np.ndarray:
    x = np.transpose(np.clip(x_chw01, 0.0, 1.0), (0, 2, 3, 1))
    return (x * 255).astype(np.uint8)  # identical cast to the paper adapter

  def _tokens(self, img_u8):
    B = img_u8.shape[0]
    obs = {'image': jnp.asarray(img_u8)}
    for k in self.obs_space:
      if k in ('is_first', 'is_last', 'is_terminal'):
        obs[k] = jnp.zeros((B,), bool)
      elif k == 'reward':
        obs[k] = jnp.zeros((B,), jnp.float32)
    reset = jnp.zeros((B,), bool)
    self.state, (_, _, tokens) = self._enc_apply(
        self.state, self.enc.initial(B), obs, reset, seed=self._next_rng())
    return tokens, reset

  def _probs_from_logit(self, logit):
    logit = np.asarray(jnp.asarray(logit, jnp.float32))
    logit = logit - logit.max(-1, keepdims=True)
    p = np.exp(logit)
    p /= p.sum(-1, keepdims=True)
    p = (1.0 - self.unimix) * p + self.unimix / p.shape[-1]
    return p  # [B, G, C]

  def _apply_mask(self, p):
    if self.mask_k <= 0:
      return p
    p = p.copy()
    p[:, :self.mask_k, :] = 0.0
    p[:, :self.mask_k, 0] = 1.0
    return p

  # ---- public API ----------------------------------------------------------
  def posterior(self, x_chw01: np.ndarray, with_deter=False):
    """Deterministic posterior probabilities [B, G, C] (and deter [B, D])."""
    out_p, out_d = [], []
    for s in range(0, len(x_chw01), CHUNK):
      img = self._to_uint8(x_chw01[s:s + CHUNK])
      B = img.shape[0]
      tokens, reset = self._tokens(img)
      act = {'action': jnp.zeros((B,), jnp.int32)}
      self.state, (_, _, feat) = self._post(
          self.state, self.dyn.initial(B), tokens, act, reset,
          seed=self._next_rng())
      out_p.append(self._apply_mask(self._probs_from_logit(feat['logit'])))
      if with_deter:
        out_d.append(np.asarray(jnp.asarray(feat['deter'], jnp.float32)))
    p = np.concatenate(out_p, 0)
    return (p, np.concatenate(out_d, 0)) if with_deter else p

  def mode_latent(self, x_chw01):
    """[deter ; one-hot argmax] as a torch tensor, for decode()/reward()."""
    p, d = self.posterior(x_chw01, with_deter=True)
    oh = np.zeros_like(p)
    np.put_along_axis(oh, p.argmax(-1)[..., None], 1.0, -1)
    z = np.concatenate([d, oh.reshape(len(oh), -1)], 1).astype(np.float32)
    return torch.from_numpy(z).to(self.device), p

  def one_step_kl(self, x, a_onehot, x_next):
    """KL(post(x_{t+1}) || prior(h_{t+1})) with a deterministic (mode) carry."""
    kls = []
    for s in range(0, len(x), CHUNK):
      xs, xn = x[s:s + CHUNK], x_next[s:s + CHUNK]
      a = a_onehot[s:s + CHUNK].argmax(-1).astype(np.int32)
      B = len(xs)
      tok0, reset = self._tokens(self._to_uint8(xs))
      tok1, _ = self._tokens(self._to_uint8(xn))
      act0 = {'action': jnp.zeros((B,), jnp.int32)}
      self.state, (c1, _, f0) = self._post(
          self.state, self.dyn.initial(B), tok0, act0, reset, seed=self._next_rng())
      G, C = f0['logit'].shape[-2:]
      mode = jax.nn.one_hot(jnp.argmax(f0['logit'], -1), C, dtype=c1['stoch'].dtype)
      c1 = dict(deter=c1['deter'], stoch=mode)
      act1 = {'action': jnp.asarray(a)}
      self.state, (_, _, f1) = self._post(
          self.state, c1, tok1, act1, reset, seed=self._next_rng())
      self.state, prior_logit = self._prior_fn(
          self.state, f1['deter'], seed=self._next_rng())
      q = self._probs_from_logit(f1['logit'])
      r = self._probs_from_logit(prior_logit)
      kls.append((q * (np.log(q) - np.log(r))).sum((-1, -2)))
    return float(np.concatenate(kls).mean())

  def sampled_z(self, x_chw01):
    """The paper's Z: [deter ; sampled one-hot], fresh RNG per call."""
    return self.encode(torch.from_numpy(x_chw01).float()).cpu().numpy()


# ----------------------------------------------------------------------------
# Channels
# ----------------------------------------------------------------------------
def kappa_pair(p_flat, x_flat, n_pairs=20000, seed=0, d_sem=None):
  rng = np.random.default_rng(seed)
  N = len(p_flat)
  i = rng.integers(0, N, n_pairs)
  j = rng.integers(0, N, n_pairs)
  keep = i != j
  i, j = i[keep], j[keep]
  dz = np.linalg.norm(p_flat[i] - p_flat[j], axis=1)
  dx = np.linalg.norm(x_flat[i] - x_flat[j], axis=1)
  ok = dx > 0
  r = dz[ok] / dx[ok]
  out = {'kappa_pair': float(np.quantile(r, .9) / max(np.quantile(r, .1), 1e-12)),
         'kappa_pair_q99q01': float(np.quantile(r, .99) / max(np.quantile(r, .01), 1e-12)),
         'lip_pix_q90': float(np.quantile(r, .9))}
  if d_sem is not None:
    ds = d_sem(i, j)
    ok = ds > 0
    rs = dz[ok] / ds[ok]
    out['kappa_sem'] = float(np.quantile(rs, .9) / max(np.quantile(rs, .1), 1e-12))
    # pairs that are semantically identical but far apart in latent space
    same = ds == 0
    out['frac_sem_equal_pairs'] = float(same.mean())
    out['dz_when_sem_equal'] = float(dz[same].mean()) if same.any() else float('nan')
  return out


def tc_between(p, ridge=1e-3):
  """0.5*(sum_g logdet S_gg - logdet S); one class dropped per group (simplex)."""
  N, G, C = p.shape
  Z = p[:, :, :-1].reshape(N, G * (C - 1)).astype(np.float64)
  Z = Z - Z.mean(0, keepdims=True)
  S = Z.T @ Z / max(N - 1, 1) + ridge * np.eye(Z.shape[1])
  _, ld_full = np.linalg.slogdet(S)
  ld_blocks = 0.0
  k = C - 1
  for g in range(G):
    _, ld = np.linalg.slogdet(S[g * k:(g + 1) * k, g * k:(g + 1) * k])
    ld_blocks += ld
  return float(0.5 * (ld_blocks - ld_full))


def tc_gauss_np(Z, eps=1e-3):
  """Same formula as spwm_metrics_system.compute_tc (paper), numpy."""
  Z = Z - Z.mean(0, keepdims=True)
  N, D = Z.shape
  t1 = np.log(Z.var(0, ddof=1) + eps).sum()
  _, ls = np.linalg.slogdet(Z @ Z.T / max(N - 1, 1) + eps * np.eye(N))
  return float(0.5 * (t1 - ((D - N) * np.log(eps) + ls)))


def lee_v2(model, x, p_x):
  x_t = torch.from_numpy(x).float()
  specs = {
      'rotation': (lambda s: affine_rotate(x_t, s), 5.0, 10.0),
      'translation1': (lambda s: translate(x_t, int(s), 0), 1.0, 2.0),
      'translation2': (lambda s: translate(x_t, int(s), 0), 2.0, 4.0),  # paper code
      'scale': (lambda s: isotropic_scale(x_t, 1 + s / 100.0), 5.0, 0.10),
  }
  out = {}
  flat = lambda p: p.reshape(len(p), -1)
  for name, (fn, step, denom) in specs.items():
    pp = flat(model.posterior(fn(+step).numpy()))
    pm = flat(model.posterior(fn(-step).numpy()))
    out[f'lee_{name}'] = float(np.linalg.norm(pp - pm, axis=1).mean() / denom)
  # identity control: must be ~0 for a deterministic latent
  out['lee_identity'] = float(np.linalg.norm(
      flat(model.posterior(x)) - flat(p_x), axis=1).mean())
  out['lee_v2'] = float(np.mean([out['lee_rotation'], out['lee_translation1'],
                                 out['lee_scale']]))
  return out


def lee_fit(model, x, p_x, n_pc=128, ridge=1e-2, seed=0):
  """E6: fit linear rho_g on PCA(p) with half the frames, residual on the other half."""
  rng = np.random.default_rng(seed)
  idx = rng.permutation(len(x))
  a, b = idx[: len(x) // 2], idx[len(x) // 2:]
  P = p_x.reshape(len(p_x), -1)
  mu = P[a].mean(0)
  U, S, Vt = np.linalg.svd(P[a] - mu, full_matrices=False)
  V = Vt[:n_pc].T
  proj = lambda Q: (Q - mu) @ V
  x_t = torch.from_numpy(x).float()
  out = {}
  for name, fn in {'rotation': lambda: affine_rotate(x_t, 5.0),
                   'translation1': lambda: translate(x_t, 1, 0),
                   'scale': lambda: isotropic_scale(x_t, 1.05)}.items():
    PT = model.posterior(fn().numpy()).reshape(len(x), -1)
    X0, X1 = proj(P), proj(PT)
    A = np.linalg.solve(X0[a].T @ X0[a] + ridge * np.eye(n_pc), X0[a].T @ X1[a])
    resid = np.linalg.norm(X1[b] - X0[b] @ A, axis=1).mean()
    move = np.linalg.norm(X1[b] - X0[b], axis=1).mean()
    out[f'leefit_{name}'] = float(resid)
    out[f'leefit_ratio_{name}'] = float(resid / max(move, 1e-12))
  out['lee_fit'] = float(np.mean([out['leefit_rotation'], out['leefit_translation1'],
                                  out['leefit_scale']]))
  return out


def alias_rate(p_flat, sem_id, reward, k=10):
  """Fraction of latent k-NN with a different semantic state / next reward."""
  N = len(p_flat)
  sq = (p_flat ** 2).sum(1)
  diff_sem, diff_rew = [], []
  for s in range(0, N, 500):
    D = sq[s:s + 500, None] + sq[None] - 2 * p_flat[s:s + 500] @ p_flat.T
    for r in range(D.shape[0]):
      D[r, s + r] = np.inf
    nn = np.argpartition(D, k, axis=1)[:, :k]
    rows = np.arange(s, min(s + 500, N))[:, None]
    if sem_id is not None:
      diff_sem.append((sem_id[nn] != sem_id[rows]).mean(1))
    diff_rew.append((np.abs(reward[nn] - reward[rows]) > 1e-6).mean(1))
  out = {'alias_reward': float(np.concatenate(diff_rew).mean())}
  if sem_id is not None:
    out['alias_sem'] = float(np.concatenate(diff_sem).mean())
  return out


def paper_diagnostics(model, x256):
  """Paper kappa/LEE/TC on the sampled Z, plus the resampling noise floors."""
  t = torch.from_numpy(x256).float()
  z0 = model.sampled_z(x256)
  # kappa exactly as compute_kappa, but with v = 0: input identical, only RNG differs
  s_null = np.stack([np.linalg.norm(model.sampled_z(x256) - z0, axis=1) / 1e-3
                     for _ in range(128)])
  k_null = s_null.max(0) / (s_null.min(0) + 1e-8)
  floor = np.linalg.norm(model.sampled_z(x256) - model.sampled_z(x256), axis=1).mean()
  from spwm_metrics_system import compute_kappa, compute_all_lee
  out = {f'paper_{k}': v for k, v in compute_kappa(model, t, n_directions=128).items()}
  out.update({f'paper_{k}': v for k, v in compute_all_lee(model, t).items()})
  out.update({
      'null_kappa_mean': float(k_null.mean()),
      'null_sigma_max_mean': float(s_null.max(0).mean()),
      'null_sigma_min_mean': float(s_null.min(0).mean()),
      # same quantity in the units of each paper LEE (divide by its denominator)
      'floor_resample_norm': float(floor),
      'floor_as_lee_rotation': float(floor / 10.0),
      'floor_as_lee_translation': float(floor / 4.0),
      'floor_as_lee_scale': float(floor / 0.10),
      'paper_tc_n256': float(compute_tc(torch.from_numpy(z0).float())),
  })
  return out


# ----------------------------------------------------------------------------
def audit_checkpoint(ckpt, data, args, info=None):
  t0 = time.time()
  model = ProbAdapter(str(ckpt), device=args.device, mask_k=args.mask_k,
                      jit=not args.no_jit)
  x, a, xn, r = data['x'], data['a'], data['x_next'], data['r'].reshape(-1)
  p, deter = model.posterior(x, with_deter=True)
  pf = p.reshape(len(p), -1)
  row = {'deter_std_max': float(deter.std(0).max()),
         'post_entropy': float(-(p * np.log(p)).sum(-1).mean()),
         'post_maxprob': float(p.max(-1).mean())}
  d_sem = info['d_sem'] if info else None
  row.update(kappa_pair(pf, x.reshape(len(x), -1), d_sem=d_sem))
  for rid in (1e-4, 1e-3, 1e-2):
    row[f'tc_between_r{rid:g}'] = tc_between(p, ridge=rid)
  row['tc_between'] = row['tc_between_r0.001']
  row.update(lee_v2(model, x, p))
  if args.lee_fit:
    row.update(lee_fit(model, x, p))
  row.update(alias_rate(pf, info['sem_id'] if info else None, r))
  # baselines (deterministic)
  zm, _ = model.mode_latent(x)
  rec = model.decode(zm).cpu().numpy()
  row['recon_mse'] = float(((rec - x) ** 2).mean())
  # the reward head predicts the reward received on arriving at a state -> use x_next
  zn, _ = model.mode_latent(xn)
  rp = model.reward(zn, torch.from_numpy(a)).cpu().numpy().reshape(-1)
  row['reward_mae'] = float(np.abs(rp - r).mean())
  row['one_step_kl'] = model.one_step_kl(x, a, xn)
  if args.paper_diag:
    row.update(paper_diagnostics(model, x[:256]))
  row['audit_seconds'] = time.time() - t0
  del model
  return row


def selftest(args, data):
  m = ProbAdapter(args.ckpt, device=args.device, jit=not args.no_jit)
  x = data['x'][:64]
  p1, d1 = m.posterior(x, with_deter=True)
  p2 = m.posterior(x)
  print('posterior shape', p1.shape, 'sums to 1:', np.allclose(p1.sum(-1), 1, atol=1e-4))
  print('deterministic (max |p1-p2|):', float(np.abs(p1 - p2).max()), '(expect 0)')
  print('deter std across inputs (max over dims):', float(d1.std(0).max()),
        '(expect ~0 -> h is constant; excluded from Z)')
  z1, z2 = m.sampled_z(x), m.sampled_z(x)
  print('paper sampled Z: mean ||z1-z2|| for identical input:',
        float(np.linalg.norm(z1 - z2, axis=1).mean()), '(>0 means resampling noise)')
  q = (np.clip(x, 0, 1) * 255).astype(np.uint8)
  v = np.random.default_rng(0).standard_normal(x.shape)
  v /= np.linalg.norm(v.reshape(len(v), -1), axis=1)[:, None, None, None]
  q2 = (np.clip(x + 1e-3 * v, 0, 1) * 255).astype(np.uint8)
  print('uint8 pixels changed by the paper eps=1e-3 probe: %.1f%% (expect ~50%% on real frames)' % (100 * (q != q2).mean()))
  print('one_step_kl on 64 frames:', m.one_step_kl(x, data['a'][:64], data['x_next'][:64]))
  t0 = time.time(); m.posterior(data['x'][:CHUNK]); m.posterior(data['x'][:CHUNK])
  print('seconds per 500-frame posterior pass (after compile): %.2f' % ((time.time() - t0) / 2))
  print('SELFTEST DONE')


def load_info(path, n):
  """Semantic distance from collect_crafter_with_info.py outputs (optional)."""
  d = np.load(path)
  if 'pos' not in d:
    return None
  pos, inv, win = d['pos'][:n].astype(float), d['inv'][:n].astype(float), d['sem_win'][:n]
  sem_id = np.array([hash((tuple(pp), tuple(ii), w.tobytes()))
                     for pp, ii, w in zip(pos, inv, win)])

  def d_sem(i, j):
    # fixed weights (preregistered): position L1 / 10 + inventory L1 / 5
    # + fraction of differing cells in the 9x7 local semantic window
    dp = np.abs(pos[i] - pos[j]).sum(1)
    di = np.abs(inv[i] - inv[j]).sum(1)
    dw = (win[i] != win[j]).reshape(len(i), -1).mean(1)
    return dp / 10.0 + di / 5.0 + dw
  return {'d_sem': d_sem, 'sem_id': sem_id}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--selftest', action='store_true')
  ap.add_argument('--ckpt', default='')
  ap.add_argument('--run_dir', default='')
  ap.add_argument('--run_name', default='')
  ap.add_argument('--seed', type=int, default=-1)
  ap.add_argument('--group', default='main')
  ap.add_argument('--lambda_tc', type=float, default=0.0)
  ap.add_argument('--aug', type=int, default=0)
  ap.add_argument('--steps', default='')
  ap.add_argument('--mask_k', type=int, default=0)
  ap.add_argument('--eval_set', required=True)
  ap.add_argument('--paper_diag', action='store_true')
  ap.add_argument('--lee_fit', action='store_true')
  ap.add_argument('--device', default='cuda')
  ap.add_argument('--no_jit', action='store_true', help='use if jax.jit fails on your ninjax version')
  ap.add_argument('--out', default='results_v2/audit.csv')
  args = ap.parse_args()

  d = np.load(args.eval_set)
  data = {k: d[k] for k in ('x', 'a', 'x_next', 'r')}
  if args.selftest:
    return selftest(args, data)
  info = load_info(args.eval_set, len(data['x']))

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  want = {int(s) for s in args.steps.split(',') if s.strip()}
  rows = []
  for ck in sorted(Path(args.run_dir).glob('checkpoint_*')):
    m = re.search(r'checkpoint_(\d+)', ck.name)
    if not m:
      continue
    step = int(m.group(1))
    if want and step not in want:
      continue
    print(f'[{args.run_name}] step {step}', flush=True)
    row = audit_checkpoint(ck, data, args, info)
    row.update(run=args.run_name, seed=args.seed, group=args.group,
               lambda_tc=args.lambda_tc, aug=args.aug, mask_k=args.mask_k,
               step=step, ckpt=str(ck))
    rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)  # write after each checkpoint
  print('wrote', out)


if __name__ == '__main__':
  main()
