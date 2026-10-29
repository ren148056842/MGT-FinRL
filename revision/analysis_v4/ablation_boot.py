"""Fixed-PPO ablation: paired block-bootstrap SR differences against A0 and against Vanilla (F1)."""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from backtest import metrics
V3, V4, AL, OUT = sys.argv[1:5]
def ret(sec, v, s):
    if v == 'Vanilla': p = '%s/E1_%s_Vanilla_ppo_s%d_F1' % (V3, sec, s)
    elif v == 'A0': p = '%s/E1_%s_A0_ppo_s%d_F1' % (V4, sec, s)
    elif v.startswith('I'): p = '%s/E2_%s_%s_ppo_s%d_F1' % (AL, sec, v, s)
    else: p = '%s/E2_%s_%s_ppo_s%d_F1' % (V4, sec, v, s)
    return pd.read_csv(p + '/trajectory.csv').daily_return.iloc[1:].values
def boot(A, B, block=10, Bn=2000, seed=1729):
    rng = np.random.default_rng(seed); n = A.shape[0]
    est = lambda ix: np.mean([metrics(A[ix, s])['SR'] - metrics(B[ix, s])['SR'] for s in range(3)])
    p = est(np.arange(n)); S = []
    for _ in range(Bn):
        st = rng.integers(0, n, size=int(np.ceil(n / block))); ix = ((st[:, None] + np.arange(block)) % n).ravel()[:n]; S.append(est(ix))
    lo, hi = np.percentile(S, [2.5, 97.5]); return p, lo, hi
rows = []
for sec in ['bank', 'healthcare']:
    M = {v: np.stack([ret(sec, v, s) for s in [1, 2, 3]], 1) for v in ['Vanilla', 'A0', 'A1', 'A2', 'A6', 'A7', 'A8', 'A9', 'I1', 'I2', 'I3']}
    for v in M:
        for ref in ['Vanilla', 'A0']:
            if v == ref: continue
            p, lo, hi = boot(M[v], M[ref]); rows.append(dict(sector=sec, variant=v, reference=ref, dSR=p, lo=lo, hi=hi, significant=(lo > 0) or (hi < 0)))
d = pd.DataFrame(rows); d.to_csv(OUT, index=False); print(d.round(3).to_string())
