"""汇总 baseline vs +Aug 的 metrics，画 LEE-vs-step 和 Return-vs-step 两张图。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONDS = ['baseline', 'aug']
SEEDS = [0, 1, 2]
SCORE_KEYS = ['episode_reward', 'return', 'score', 'eval/return',
              'train/return', 'eval/score', 'episode/score']


def load_scores(jsonl):
    recs = [json.loads(l) for l in open(jsonl)]
    df = pd.DataFrame(recs)
    key = next((k for k in SCORE_KEYS if k in df.columns), None)
    return df, key


def collect():
    rows = []
    for cond in CONDS:
        for seed in SEEDS:
            mpath = Path(f'results/{cond}_seed{seed}/metrics.csv')
            spath = Path(f'logdir/{cond}_seed{seed}/scores.jsonl')
            if not mpath.exists():
                print(f'[skip] {mpath} missing'); continue
            m = pd.read_csv(mpath)
            s, ykey = load_scores(spath)
            s = s.sort_values('step')
            # nearest-step return for each checkpoint
            for _, r in m.iterrows():
                step = r['step']
                j = (s['step'] - step).abs().idxmin()
                rows.append({
                    'cond': cond, 'seed': seed, 'step': step,
                    'lee_rot': r.get('lee_rotation', np.nan),
                    'kappa': r.get('kappa_mean', np.nan),
                    'tc': r.get('tc', np.nan),
                    'return': s.loc[j, ykey],
                })
    return pd.DataFrame(rows)


def plot(df):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7, 2.8))
    colors = {'baseline': 'C0', 'aug': 'C1'}
    labels = {'baseline': 'Baseline', 'aug': '+Aug (targeted)'}
    for cond in CONDS:
        g = df[df.cond == cond].groupby('step')
        # Panel A: Rotation LEE
        mA, sA = g['lee_rot'].mean(), g['lee_rot'].std()
        axA.plot(mA.index, mA.values, color=colors[cond], label=labels[cond])
        axA.fill_between(mA.index, mA - sA, mA + sA, color=colors[cond], alpha=0.2)
        # Panel B: Return
        mB, sB = g['return'].mean(), g['return'].std()
        axB.plot(mB.index, mB.values, color=colors[cond], label=labels[cond])
        axB.fill_between(mB.index, mB - sB, mB + sB, color=colors[cond], alpha=0.2)
    axA.set_xlabel('Training step'); axA.set_ylabel('Rotation LEE')
    axA.set_title('(A) Symmetry channel'); axA.legend(fontsize=8)
    axB.set_xlabel('Training step'); axB.set_ylabel('Episode return')
    axB.set_title('(B) Downstream performance'); axB.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('lee_intervention.pdf')
    print('Saved lee_intervention.pdf')
    # Also dump the channel-specificity check
    print('\n=== Channel specificity (mean over late checkpoints) ===')
    late = df[df.step >= df.step.max() * 0.7]
    print(late.groupby('cond')[['lee_rot', 'kappa', 'tc', 'return']].mean())


if __name__ == '__main__':
    df = collect()
    df.to_csv('results/intervention_curves.csv', index=False)
    plot(df)