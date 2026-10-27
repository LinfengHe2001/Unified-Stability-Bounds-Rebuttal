"""Synthetic end-to-end test for analysis_v2.py (no GPU, no checkpoints).
Builds fake audit CSVs + scores.jsonl with known structure, runs the analysis,
and checks the outputs are finite and behave as designed."""
import json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import analysis_v2 as A

def make(root: Path, seed=0):
  rng = np.random.default_rng(seed)
  sched = [50000*i for i in range(1, 11)] + [650000, 800000, 950000, 1100000, 1250000,
           1400000, 1550000, 1700000, 1850000, 2000000, 2500000, 3000000, 3500000]
  runs = [('main', 'main', 0, 0.0, sched)]
  runs += [(f'aug_base_s{s}', 'aug_base', s, 0.0, sched[:10]) for s in range(3)]
  runs += [(f'aug_s{s}', 'aug', s, 0.0, sched[:10]) for s in range(3)]
  runs += [(f'tc_lam{l}', 'tc_sweep', 0, l, sched[:10]) for l in (0.01, 0.1, 1.0)]
  rows, paths = [], {}
  for name, grp, sd, lam, steps in runs:
    d = root / name; d.mkdir(parents=True, exist_ok=True); paths[name] = str(d)
    eps = []
    for t in range(0, max(steps) + 30000, 7000):
      prog = 1 - np.exp(-t / 300000)
      J = 2 + 10 * prog - (1.5 if grp == 'aug' else 0)
      if lam >= 1.0 and t > 250000: J -= 4 * min(1, (t - 250000) / 150000)
      eps.append({'step': t, 'episode/score': J + rng.normal(0, 1.0)})
    (d / 'scores.jsonl').write_text('\n'.join(json.dumps(e) for e in eps))
    for t in steps:
      prog = 1 - np.exp(-t / 300000)
      bad = (lam >= 1.0 and t > 200000)
      rows.append(dict(run=name, group=grp, seed=sd, lambda_tc=lam, aug=int(grp=='aug'),
        mask_k=0, step=t,
        kappa_pair=3 - 1.2*prog + (0.8 if bad else 0) + rng.normal(0, .05),
        tc_between=40 - 15*prog - (10*lam) + rng.normal(0, 1),
        lee_v2=0.5 - 0.2*prog - (0.1 if grp=='aug' else 0) + rng.normal(0, .01),
        one_step_kl=5 - 3*prog + (1.5 if bad else 0) + rng.normal(0, .1),
        recon_mse=.02 - .01*prog + rng.normal(0, .001),
        reward_mae=.1 - .05*prog + rng.normal(0, .005),
        kappa_sem=4 - 1.5*prog + rng.normal(0, .1)))
  pd.DataFrame(rows).to_csv(root / 'audit.csv', index=False)
  return paths

root = Path(sys.argv[1])
paths = make(root)
pr = json.loads(Path('PREREGISTRATION.json').read_text()); pr['calibration']['run'] = 'main'
df = A.build_table([root / 'audit.csv'], paths, pr)
df2, s = A.analyse(df, pr, root / 'out')
ll = A.lead_lag(df2, A.CHANNELS + A.BASELINES, root / 'out')
print('beta*', round(s['beta_star'], 4), 'beta90', round(s['beta_90'], 4))
for k, v in s['test'].items():
  if isinstance(v, dict):
    b = v['beta_star']
    print(f"{k:18s} n={b['n']:3d} cov={b['coverage']:.2f} viol={b['n_violations']:2d} "
          f"tight_med={b['tight_median']:.2f} auroc={v['auroc_pooled']:.2f} "
          f"auroc_step={v['auroc_step_matched']:.2f}")
  else:
    print(k, v)
print('conformal', s.get('conformal'))
print('lee_intervention', s.get('lee_intervention'))
print('kappa pix vs sem', s.get('kappa_pix_vs_sem_spearman'), s.get('kappa_pix_vs_sem_ci'))
print(pd.read_csv(root/'out'/'ablation.csv')[['rhs','beta_star_refit','coverage','auroc_step_matched']].to_string(index=False))
print(pd.read_csv(root/'out'/'baselines.csv')[['rhs','beta_star','coverage','auroc_step_matched']].to_string(index=False))
print(ll[ll.get('false_alarm_check', pd.Series(False, index=ll.index)).fillna(False) == False][['run','channel','t_drop','t_alarm','direction','lead','counts_as_evidence']].head(12).to_string(index=False))
assert np.isfinite(s['beta_star'])

# ---- reproduce the E1 defect: minmax of SUMMED raw RHS with clip -> beta* blow-up
cal = df2[(df2.run=='main') & df2.step.between(50000, 500000)].copy()
raw = cal.kappa_pair + cal.tc_between + cal.lee_v2
lo, hi = np.percentile(raw, 5), np.percentile(raw, 95)
rn = np.clip((raw - lo) / (hi - lo), 0, 1)
print('E1-style: calibration RHS==0 points:', int((rn == 0).sum()),
      ' E1 beta* =', float((cal.gap / np.maximum(rn, 1e-12)).max()))
