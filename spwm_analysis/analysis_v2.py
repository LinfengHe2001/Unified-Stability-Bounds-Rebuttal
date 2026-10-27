#!/usr/bin/env python3
"""
analysis_v2.py -- frozen-beta held-out tests for the rebuttal (CPU only).

Replaces E1/E2/E3 from spwm_analysis/rebuttal/, which have these defects:
  * E1 min-max scales the SUMMED raw RHS (TC dominates) and clips at 0, so the
    lowest calibration checkpoint gets RHS = 0 and beta* blows up (vacuous bound);
    J is matched on exact steps (mostly NaN); AUROC is confounded with training stage.
  * E2 takes the "seed SD" from a single baseline run (NaN) and falls back to a
    temporal statistic; a TC alarm in a TC-penalty run is counted as evidence.
Everything here follows PREREGISTRATION.json.

Inputs
  --audit   one or more CSVs from audit_v2.py (one row per checkpoint)
  --runs    run_name=path/to/logdir  (to read scores.jsonl), repeat
  --prereg  PREREGISTRATION.json
Outputs (--out): summary.json, per_run.csv, checkpoints.csv, ablation.csv,
  baselines.csv, leadlag.csv (if TC-sweep runs), scatter.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ['kappa_pair', 'tc_between', 'lee_v2']
BASELINES = ['one_step_kl', 'recon_mse', 'reward_mae']


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def returns_for_run(scores_path: Path, steps, window: int):
  recs = [json.loads(l) for l in open(scores_path) if l.strip()]
  s = pd.DataFrame(recs)
  key = next(k for k in ('episode/score', 'score', 'episode_reward', 'return')
             if k in s.columns)
  s = s[['step', key]].dropna()
  s['step'] = pd.to_numeric(s['step'], errors='coerce')
  out = []
  for t in steps:
    v = s.loc[(s['step'] - t).abs() <= window, key].astype(float).values
    out.append({'step': t, 'J_mean': v.mean() if len(v) else np.nan,
                'J_sem': v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                'n_eps': len(v)})
  return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------
# Natural zero of each channel: the value that means "no defect".
# kappa (a ratio of quantiles) is >= 1, so its zero is 1; TC, LEE, and the
# baseline losses are >= 0.
NATURAL_ZERO = {'kappa_pair': 1.0, 'kappa_sem': 1.0, 'paper_kappa_mean': 1.0}


def fit_scaler(cal: pd.DataFrame, cols, method='natural'):
  """method='natural' (preregistered): (v - zero_c) / median_cal(v - zero_c).
  method='minmax' (paper, sensitivity only): (v - q05_cal) / (q95_cal - q05_cal).
  Min-max puts the healthiest calibration checkpoint at RHS = 0, so any
  positive gap there makes beta* infinite; natural-zero scaling avoids this."""
  sc = {}
  for c in cols:
    if method == 'minmax':
      sc[c] = ('minmax', float(np.quantile(cal[c], .05)), float(np.quantile(cal[c], .95)))
    else:
      z = NATURAL_ZERO.get(c, 0.0)
      sc[c] = ('natural', z, float(np.median(cal[c] - z)))
  return sc


def apply_scaler(df, scaler, lower_clip=0.0, upper_clip=None):
  out = {}
  for c, (kind, a, b) in scaler.items():
    if kind == 'minmax':
      v = (df[c].values - a) / max(b - a, 1e-12)
    else:
      v = (df[c].values - a) / max(abs(b), 1e-12)
    v = np.maximum(v, lower_clip) if lower_clip is not None else v
    v = np.minimum(v, upper_clip) if upper_clip is not None else v
    out[c] = v
  return pd.DataFrame(out, index=df.index)


def beta_from(gap, rhs):
  ok = rhs > 0
  if (~ok & (gap > 0)).any():
    return np.inf, np.inf  # a calibration point with zero RHS and positive gap
  r = gap[ok] / rhs[ok]
  return float(r.max()), float(np.quantile(r, .9))


def coverage_block(gap, rhs, beta, tight_min_gap=0.05):
  if not np.isfinite(beta):
    return {'n': int(len(gap)), 'coverage': np.nan, 'n_violations': -1,
            'tight_median': np.nan, 'tight_q25': np.nan, 'tight_q75': np.nan,
            'note': 'beta infinite: a calibration point has RHS=0 and gap>0'}
  cov = beta * rhs >= gap - 1e-12
  pos = gap > tight_min_gap
  t = beta * rhs[pos] / gap[pos] if pos.any() else np.array([])
  return {'n': int(len(gap)), 'coverage': float(cov.mean()) if len(gap) else np.nan,
          'n_violations': int((~cov).sum()),
          'tight_median': float(np.median(t)) if len(t) else np.nan,
          'tight_q25': float(np.quantile(t, .25)) if len(t) else np.nan,
          'tight_q75': float(np.quantile(t, .75)) if len(t) else np.nan}


def auroc(score, label):
  """Mann-Whitney AUROC; label 1 = failure; higher score should mean failure."""
  pos, neg = score[label == 1], score[label == 0]
  if len(pos) == 0 or len(neg) == 0:
    return np.nan
  allv = np.concatenate([pos, neg])
  ranks = pd.Series(allv).rank().values
  return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) /
               (len(pos) * len(neg)))


def auroc_stratified(score, label, strata):
  """Pairs compared only within the same stratum (step): removes training-stage
  confounding. Returns pair-weighted AUROC."""
  num, den = 0.0, 0.0
  for s in np.unique(strata):
    m = strata == s
    p, n = score[m & (label == 1)], score[m & (label == 0)]
    if len(p) and len(n):
      diff = p[:, None] - n[None, :]
      num += (diff > 0).sum() + 0.5 * (diff == 0).sum()
      den += diff.size
  return float(num / den) if den else np.nan


def run_bootstrap(df, fn, B=2000, seed=0):
  runs = df['run'].unique()
  rng = np.random.default_rng(seed)
  vals = []
  for _ in range(B):
    pick = rng.choice(runs, len(runs), replace=True)
    sub = pd.concat([df[df['run'] == r] for r in pick], ignore_index=True)
    v = fn(sub)
    if np.isfinite(v):
      vals.append(v)
  if not vals:
    return np.nan, np.nan
  return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def conformal_beta(scores, alpha):
  """Split-conformal over exchangeable runs. Valid only if alpha >= 1/(n+1)."""
  n = len(scores)
  k = int(np.ceil((n + 1) * (1 - alpha)))
  if k > n:
    return np.inf
  return float(np.sort(scores)[k - 1])


# ---------------------------------------------------------------------------
def build_table(audit_csvs, run_paths, prereg):
  df = pd.concat([pd.read_csv(p) for p in audit_csvs], ignore_index=True)
  w = prereg['return']['window_steps']
  parts = []
  for run, g in df.groupby('run'):
    if run not in run_paths:
      raise SystemExit(f'no --runs path for {run}')
    J = returns_for_run(Path(run_paths[run]) / 'scores.jsonl', g['step'].unique(), w)
    parts.append(g.merge(J, on='step', how='left'))
  df = pd.concat(parts, ignore_index=True)
  missing = df['J_mean'].isna().sum()
  if missing:
    print(f'[WARN] {missing} checkpoints have no episodes within +-{w} steps')
  return df


def analyse(df, prereg, out: Path, channels=CHANNELS):
  out.mkdir(parents=True, exist_ok=True)
  cal_run = prereg['calibration']['run']
  lo, hi = prereg['calibration']['steps']
  df = df.dropna(subset=['J_mean']).copy()
  Jstar, Jmin = df['J_mean'].max(), df['J_mean'].min()
  df['gap'] = (Jstar - df['J_mean']) / max(Jstar - Jmin, 1e-12)
  is_cal = (df['run'] == cal_run) & df['step'].between(lo, hi)
  cal = df[is_cal]
  if len(cal) < 5:
    raise SystemExit(f'calibration set too small ({len(cal)})')

  method = prereg['scaling'].get('method_key', 'natural')

  def rhs_for(cols, ref=cal, frame=df):
    sc = fit_scaler(ref, cols, method)
    return apply_scaler(frame, sc, prereg['scaling']['lower_clip'],
                        prereg['scaling']['upper_clip']).sum(1).values, sc

  df['rhs'], scaler = rhs_for(channels)
  b_star, b90 = beta_from(df.loc[is_cal, 'gap'].values, df.loc[is_cal, 'rhs'].values)

  # test-set definitions
  sets = {
      'T1_same_run_late': (df['run'] == cal_run) & (df['step'] > hi),
      'T2_cross_run': df['group'].isin(['aug_base', 'aug']),
      'T3_tc_sweep': df['group'].eq('tc_sweep'),
      'T4_latent_mask': df['group'].eq('t4'),
  }
  sets['ALL_TEST'] = ~is_cal
  summ = {'J_star': float(Jstar), 'J_min': float(Jmin), 'beta_star': b_star,
          'beta_90': b90, 'n_calibration': int(is_cal.sum()), 'scaler': scaler,
          'channels': channels, 'test': {}}
  for name, m in sets.items():
    if m.sum() == 0:
      summ['test'][name] = 'unavailable'
      continue
    sub = df[m]
    blk = {'beta_star': coverage_block(sub['gap'].values, sub['rhs'].values, b_star),
           'beta_90': coverage_block(sub['gap'].values, sub['rhs'].values, b90)}
    fail = (sub['gap'] > prereg_fail_q(prereg)).astype(int).values
    blk['auroc_pooled'] = auroc(sub['rhs'].values, fail)
    blk['auroc_pooled_ci'] = run_bootstrap(
        sub, lambda s: auroc(s['rhs'].values,
                             (s['gap'] > prereg_fail_q(prereg)).astype(int).values))
    blk['auroc_step_matched'] = auroc_stratified(sub['rhs'].values, fail, sub['step'].values)
    blk['spearman_rhs_gap'] = float(pd.Series(sub['rhs'].values).corr(
        pd.Series(sub['gap'].values), method='spearman'))
    summ['test'][name] = blk

  # per-run table
  rows = []
  for run, g in df.groupby('run'):
    r = coverage_block(g['gap'].values, g['rhs'].values, b_star)
    r.update(run=run, group=g['group'].iloc[0], calibration=bool(run == cal_run),
             J_last=float(g.sort_values('step')['J_mean'].iloc[-1]))
    rows.append(r)
  pd.DataFrame(rows).to_csv(out / 'per_run.csv', index=False)

  # conformal across runs (healthy runs only, excluding calibration run)
  healthy = df[df['group'].isin(['aug_base', 'main']) & (df['run'] != cal_run)]
  if healthy['run'].nunique() >= 3:
    sc_h = []
    for run, g in healthy.groupby('run'):
      ok = g['rhs'] > 0
      sc_h.append(float((g.loc[ok, 'gap'] / g.loc[ok, 'rhs']).max()))
    n = len(sc_h)
    summ['conformal'] = {'n_runs': n, 'alpha_min_valid': 1 / (n + 1),
                         'beta_at_alpha_min': conformal_beta(np.array(sc_h), 1 / (n + 1))}

  # ablation with REFIT beta
  abl = []
  for drop in [None] + channels:
    cols = [c for c in channels if c != drop]
    r_all, _ = rhs_for(cols)
    bs, _ = beta_from(df.loc[is_cal, 'gap'].values, r_all[is_cal.values])
    test = ~is_cal.values
    cb = coverage_block(df['gap'].values[test], r_all[test], bs)
    fail = (df['gap'].values[test] > prereg_fail_q(prereg)).astype(int)
    abl.append({'rhs': '+'.join(cols), 'dropped': drop or '-', 'beta_star_refit': bs,
                **cb, 'auroc_step_matched': auroc_stratified(
                    r_all[test], fail, df['step'].values[test])})
  pd.DataFrame(abl).to_csv(out / 'ablation.csv', index=False)

  # baselines under the identical protocol
  base_rows = []
  for spec in [channels] + [[b] for b in BASELINES if b in df.columns] + \
              ([channels + ['one_step_kl']] if 'one_step_kl' in df.columns else []):
    r_all, _ = rhs_for(spec)
    bs, _ = beta_from(df.loc[is_cal, 'gap'].values, r_all[is_cal.values])
    test = ~is_cal.values
    cb = coverage_block(df['gap'].values[test], r_all[test], bs)
    fail = (df['gap'].values[test] > prereg_fail_q(prereg)).astype(int)
    base_rows.append({'rhs': '+'.join(spec), 'beta_star': bs, **cb,
                      'auroc_pooled': auroc(r_all[test], fail),
                      'auroc_step_matched': auroc_stratified(
                          r_all[test], fail, df['step'].values[test])})
  pd.DataFrame(base_rows).to_csv(out / 'baselines.csv', index=False)

  # LEE intervention, recomputed (per seed, steps >= 250k)
  if {'aug', 'aug_base'} <= set(df['group']) and 'lee_v2' in df.columns:
    late = df[df['step'] >= 250000]
    per_seed = []
    for sd in sorted(late['seed'].unique()):
      a_ = late[(late['group'] == 'aug') & (late['seed'] == sd)]
      b_ = late[(late['group'] == 'aug_base') & (late['seed'] == sd)]
      if len(a_) and len(b_):
        per_seed.append({'seed': int(sd),
                         'lee_change': float(a_['lee_v2'].mean() / b_['lee_v2'].mean() - 1),
                         'return_change': float(a_['J_mean'].mean() / b_['J_mean'].mean() - 1)})
    if per_seed:
      ch = np.array([r['lee_change'] for r in per_seed])
      rng = np.random.default_rng(0)
      boots = [rng.choice(ch, len(ch)).mean() for _ in range(2000)]
      summ['lee_intervention'] = {'per_seed': per_seed, 'mean_lee_change': float(ch.mean()),
                                  'ci': [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                                  'paper_value': -0.227}
  # pixel kappa vs semantic kappa across checkpoints
  if 'kappa_sem' in df.columns and df['kappa_sem'].notna().sum() >= 5:
    summ['kappa_pix_vs_sem_spearman'] = float(df[['kappa_pair', 'kappa_sem']].corr(method='spearman').iloc[0, 1])
    summ['kappa_pix_vs_sem_ci'] = run_bootstrap(
        df.dropna(subset=['kappa_sem']),
        lambda s: s[['kappa_pair', 'kappa_sem']].corr(method='spearman').iloc[0, 1])

  df.to_csv(out / 'checkpoints.csv', index=False)
  summ['prereg_sha256'] = hashlib.sha256(
      json.dumps(prereg, sort_keys=True).encode()).hexdigest()[:16]
  (out / 'summary.json').write_text(json.dumps(summ, indent=2, default=str))
  plot(df, is_cal, b_star, out)
  return df, summ


def prereg_fail_q(prereg):
  txt = prereg['failure_labels']['absolute']
  return float(txt.split('>')[1].strip())


# ---------------------------------------------------------------------------
# Lead-lag (only meaningful if TC-sweep runs exist)
# ---------------------------------------------------------------------------
def lead_lag(df, cols, out: Path):
  base = df[df['group'].isin(['aug_base', 'main'])]
  if base['run'].nunique() < 3 or not df['group'].eq('tc_sweep').any():
    return None

  def ref(exclude=None):
    b = base if exclude is None else base[base['run'] != exclude]
    return b.groupby('step')[['J_mean'] + cols].agg(['mean', 'std'])

  def first_alarm(g, R, c, k=2.0):
    for _, row in g.sort_values('step').iterrows():
      s = row['step']
      if s not in R.index or not np.isfinite(R.loc[s, (c, 'std')]):
        continue
      mu, sd = R.loc[s, (c, 'mean')], R.loc[s, (c, 'std')]
      if row[c] > mu + k * sd:
        return s, 'up'
      if row[c] < mu - k * sd:
        return s, 'down'
    return None, ''

  def first_drop(g, R):
    g = g.sort_values('step')
    below = [(s in R.index) and (j < R.loc[s, ('J_mean', 'mean')] - R.loc[s, ('J_mean', 'std')])
             for s, j in zip(g['step'], g['J_mean'])]
    for i in range(len(below) - 2):
      if below[i] and below[i + 1] and below[i + 2]:
        return g['step'].iloc[i]
    return None

  rows = []
  R = ref()
  for run, g in df[df['group'].eq('tc_sweep')].groupby('run'):
    td = first_drop(g, R)
    for c in cols:
      ta, d = first_alarm(g, R, c)
      rows.append({'run': run, 'lambda_tc': g['lambda_tc'].iloc[0], 'channel': c,
                   't_drop': td, 't_alarm': ta, 'direction': d,
                   'lead': (td - ta) if (td is not None and ta is not None) else None,
                   'counts_as_evidence': c not in ('tc_between',)})
  # false alarms on held-out healthy seeds
  for run, g in base.groupby('run'):
    Rx = ref(exclude=run)
    for c in cols:
      ta, d = first_alarm(g, Rx, c)
      rows.append({'run': run, 'lambda_tc': 0.0, 'channel': c, 't_drop': None,
                   't_alarm': ta, 'direction': d, 'lead': None,
                   'counts_as_evidence': False, 'false_alarm_check': True})
  res = pd.DataFrame(rows)
  res.to_csv(out / 'leadlag.csv', index=False)
  return res


def plot(df, is_cal, beta, out):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(figsize=(6.4, 5))
  styles = {'main': ('#1f4e79', 'o'), 'aug_base': ('#3a7d44', 's'), 'aug': ('#c7792a', '^'),
            'tc_sweep': ('#b03a2e', 'D'), 't4': ('#6c3483', 'v')}
  for grp, g in df[~is_cal].groupby('group'):
    c, m = styles.get(grp, ('gray', 'x'))
    ax.scatter(g['rhs'], g['gap'], s=22, c=c, marker=m, alpha=.8, label=f'test: {grp}')
  ax.scatter(df.loc[is_cal, 'rhs'], df.loc[is_cal, 'gap'], s=34, facecolors='none',
             edgecolors='k', label='calibration')
  xs = np.linspace(0, max(df['rhs'].max(), 1e-6) * 1.05, 50)
  if np.isfinite(beta):
    ax.plot(xs, beta * xs, 'k--', lw=1, label=f'frozen β*={beta:.3g}')
  ax.set_xlabel('RHS (per-channel scaling fitted on calibration only)')
  ax.set_ylabel('normalized gap (J* over all runs)')
  ax.legend(fontsize=7, frameon=False)
  fig.tight_layout()
  fig.savefig(out / 'scatter.png', dpi=180)
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--audit', nargs='+', required=True)
  ap.add_argument('--runs', action='append', required=True, help='name=logdir')
  ap.add_argument('--prereg', default='PREREGISTRATION.json')
  ap.add_argument('--out', default='results_v2/analysis')
  ap.add_argument('--minmax_sensitivity', action='store_true',
                  help='also run with the paper min-max scaling (sensitivity only)')
  ap.add_argument('--paper_channels', action='store_true',
                  help='also run with the paper channels (paper_kappa_mean, paper_tc_n256, paper LEE mean)')
  a = ap.parse_args()
  prereg = json.loads(Path(a.prereg).read_text())
  paths = dict(s.split('=', 1) for s in a.runs)
  df = build_table(a.audit, paths, prereg)
  out = Path(a.out)
  df2, summ = analyse(df, prereg, out)
  lead_lag(df2, CHANNELS + [b for b in BASELINES if b in df2.columns], out)
  if a.minmax_sensitivity:
    pr2 = json.loads(json.dumps(prereg)); pr2['scaling']['method_key'] = 'minmax'
    analyse(df, pr2, out / 'minmax_sensitivity')
  if a.paper_channels:
    df['paper_lee'] = df[['paper_lee_rotation', 'paper_lee_translation',
                          'paper_lee_scale']].mean(1)
    analyse(df, prereg, out / 'paper_channels',
            channels=['paper_kappa_mean', 'paper_tc_n256', 'paper_lee'])
  print(json.dumps(summ, indent=2, default=str)[:3000])


if __name__ == '__main__':
  main()
