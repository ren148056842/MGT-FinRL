"""RL and non-RL portfolio analysis for v4 (MGT agents from v4; Vanilla agents reused from v3).

python rl_eval.py <v3_eval_dir> <v4_eval_dir> <v4_nonrl_dir> <alpha_eval_dir or -> <panel_dir> <out_dir>
"""
import sys, pathlib, json
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtest import metrics

V3, V4, NONRL, ALPHA, PANEL, OUT = sys.argv[1:7]
OUT = pathlib.Path(OUT); OUT.mkdir(parents=True, exist_ok=True)
ALGS = ['a2c', 'ddpg', 'ppo', 'sac', 'td3']; FEES = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6']


def traj(path):
    t = pd.read_csv(pathlib.Path(path) / 'trajectory.csv')
    return t


def run_dir(sec, v, a, s, fee):
    if v == 'Vanilla':
        return pathlib.Path(V3) / ('E1_%s_Vanilla_%s_s%d_%s' % (sec, a, s, fee))
    if v == 'A0':
        return pathlib.Path(V4) / ('E1_%s_A0_%s_s%d_%s' % (sec, a, s, fee))
    if v.startswith('I'):
        return pathlib.Path(ALPHA) / ('E2_%s_%s_ppo_s%d_%s' % (sec, v, s, fee))
    return pathlib.Path(V4) / ('E2_%s_%s_ppo_s%d_%s' % (sec, v, s, fee))


def stats(t):
    r = t.daily_return.iloc[1:].to_numpy(float); m = metrics(r)
    m['turnover'] = t.gross_turnover.iloc[1:].mean(); m['cost'] = t.cost.sum(); m['final_nav'] = t.nav.iloc[-1]
    return m, r


rows = []
for sec in ['bank', 'healthcare']:
    for a in ALGS:
        for v in ['Vanilla', 'A0']:
            for s in [1, 2, 3]:
                for fee in FEES:
                    d = run_dir(sec, v, a, s, fee)
                    if (d / 'trajectory.csv').exists():
                        m, _ = stats(traj(d)); rows.append(dict(sector=sec, variant=v, algorithm=a, seed=s, fee=fee, **m))
    for v in ['A1', 'A2', 'A6', 'A7', 'A8', 'A9', 'I1', 'I2', 'I3']:
        for s in [1, 2, 3]:
            for fee in ['F0', 'F1']:
                d = run_dir(sec, v, 'ppo', s, fee)
                if (d / 'trajectory.csv').exists():
                    m, _ = stats(traj(d)); rows.append(dict(sector=sec, variant=v, algorithm='ppo', seed=s, fee=fee, **m))
R = pd.DataFrame(rows); R.to_csv(OUT / 'rl_runs.csv', index=False)
g = R.groupby(['sector', 'variant', 'algorithm', 'fee'])
S = g[['AR', 'SR', 'MDD', 'Calmar', 'Vol', 'turnover', 'cost']].agg(['mean', 'std']); S.columns = ['%s_%s' % c for c in S.columns]
S['n'] = g.size(); S = S.reset_index(); S.to_csv(OUT / 'rl_mean_sd.csv', index=False)

# cost deltas relative to F0 (same fixed policy)
P = R.pivot_table(index=['sector', 'variant', 'algorithm', 'seed'], columns='fee', values=['AR', 'SR', 'MDD'])
cd = []
for fee in FEES[1:]:
    if ('AR', fee) not in P: continue
    x = pd.DataFrame({'dAR_pp': 100 * (P[('AR', fee)] - P[('AR', 'F0')]), 'dSR': P[('SR', fee)] - P[('SR', 'F0')], 'dMDD_pp': 100 * (P[('MDD', fee)] - P[('MDD', 'F0')])}).dropna()
    x['fee'] = fee; cd.append(x.reset_index())
C = pd.concat(cd); C.to_csv(OUT / 'cost_deltas.csv', index=False)
CG = C[C.variant.isin(['Vanilla', 'A0'])].groupby(['sector', 'variant', 'algorithm', 'fee'])[['dAR_pp', 'dSR', 'dMDD_pp']].mean().reset_index()
CG.to_csv(OUT / 'cost_deltas_group.csv', index=False)
print('Cost F1 vs F0 over groups: median dAR %.2f (range %.2f..%.2f), dSR %.3f (%.3f..%.3f), dMDD %.2f (max %.2f)' % tuple(
    [CG[CG.fee == 'F1'][c].median() for c in ['dAR_pp']] + [CG[CG.fee == 'F1'].dAR_pp.min(), CG[CG.fee == 'F1'].dAR_pp.max()] +
    [CG[CG.fee == 'F1'].dSR.median(), CG[CG.fee == 'F1'].dSR.min(), CG[CG.fee == 'F1'].dSR.max(), CG[CG.fee == 'F1'].dMDD_pp.median(), CG[CG.fee == 'F1'].dMDD_pp.max()]))
print('Cost F4 vs F0: median dAR %.2f (range %.2f..%.2f), dSR %.3f (%.3f..%.3f), dMDD %.2f (max %.2f)' % tuple(
    [CG[CG.fee == 'F4'].dAR_pp.median(), CG[CG.fee == 'F4'].dAR_pp.min(), CG[CG.fee == 'F4'].dAR_pp.max(), CG[CG.fee == 'F4'].dSR.median(), CG[CG.fee == 'F4'].dSR.min(), CG[CG.fee == 'F4'].dSR.max(), CG[CG.fee == 'F4'].dMDD_pp.median(), CG[CG.fee == 'F4'].dMDD_pp.max()]))


# paired circular block bootstrap (same resampled dates for both strategies and all seeds)
def boot(A, B, block=10, Bn=2000, seed=1729):
    rng = np.random.default_rng(seed); n = A.shape[0]; keys = ['AR', 'SR', 'MDD']
    def est(ix):
        return np.mean([[metrics(A[ix, s])[k] - metrics(B[ix, s])[k] for k in keys] for s in range(A.shape[1])], axis=0)
    point = est(np.arange(n)); Sx = []
    for _ in range(Bn):
        st = rng.integers(0, n, size=int(np.ceil(n / block))); ix = ((st[:, None] + np.arange(block)) % n).ravel()[:n]; Sx.append(est(ix))
    Sx = np.array(Sx); lo, hi = np.percentile(Sx, [2.5, 97.5], axis=0)
    return point, lo, hi


def returns(sec, v, a, fee='F1'):
    cols = []
    for s in [1, 2, 3]:
        t = traj(run_dir(sec, v, a, s, fee)); cols.append(t.set_index('date').daily_return.iloc[1:])
    return pd.concat(cols, axis=1)


brow = []
for sec in ['bank', 'healthcare']:
    for a in ALGS:
        try:
            A = returns(sec, 'A0', a); B = returns(sec, 'Vanilla', a)
        except FileNotFoundError:
            continue
        assert (A.index == B.index).all()
        wins = sum(metrics(A.values[:, s])['SR'] > metrics(B.values[:, s])['SR'] for s in range(3))
        for block in [10, 5, 20]:
            p, lo, hi = boot(A.values, B.values, block=block)
            for k, m in enumerate(['AR', 'SR', 'MDD']):
                brow.append(dict(sector=sec, algorithm=a, block=block, metric=m, effect=p[k], ci_low=lo[k], ci_high=hi[k], seeds_A0_higher_SR=wins))
BS = pd.DataFrame(brow); BS.to_csv(OUT / 'paired_bootstrap_F1.csv', index=False)

# sub-periods around the 24 September 2024 policy announcement
sub = []
for sec in ['bank', 'healthcare']:
    for v in ['Vanilla', 'A0']:
        for a in ALGS:
            try:
                X = returns(sec, v, a)
            except FileNotFoundError:
                continue
            X.index = pd.to_datetime(X.index)
            for lab, mask in [('Jan-Sep 23', X.index < '2024-09-24'), ('Sep 24-Dec', X.index >= '2024-09-24')]:
                sub.append(dict(sector=sec, variant=v, algorithm=a, period=lab, SR=np.mean([metrics(X[mask].values[:, s])['SR'] for s in range(3)]),
                                cumret=np.mean([np.prod(1 + X[mask].values[:, s]) - 1 for s in range(3)])))
pd.DataFrame(sub).to_csv(OUT / 'subperiods.csv', index=False)

# tracking versus the equal-weight daily rebalanced portfolio (same costs)
ew = {}
for f in pathlib.Path(NONRL).glob('Baseline_*_equal_daily_F1'):
    sec = f.name.split('_')[1]; t = traj(f); ew[sec] = t.set_index('date').daily_return.iloc[1:]
tr = []
for sec in ew:
    for v in ['Vanilla', 'A0']:
        for a in ALGS:
            for s in [1, 2, 3]:
                d = run_dir(sec, v, a, s, 'F1')
                if not (d / 'trajectory.csv').exists(): continue
                r = traj(d).set_index('date').daily_return.iloc[1:]
                tr.append(dict(sector=sec, variant=v, algorithm=a, seed=s, corr_ew=np.corrcoef(r, ew[sec].loc[r.index])[0, 1], TE=np.std(r - ew[sec].loc[r.index], ddof=1) * np.sqrt(252)))
pd.DataFrame(tr).to_csv(OUT / 'tracking_vs_ew.csv', index=False)

# non-RL baselines and forecast-ranking rules
nr = []
for f in sorted(pathlib.Path(NONRL).glob('Baseline_*')):
    if not (f / 'trajectory.csv').exists(): continue
    m, _ = stats(traj(f)); parts = f.name.split('_'); sec = parts[1]; fee = parts[-1]; strat = '_'.join(parts[2:-1])
    nr.append(dict(sector=sec, strategy=strat, fee=fee, **m))
pd.DataFrame(nr).to_csv(OUT / 'nonrl.csv', index=False)
print('done', len(R), 'rl rows', len(nr), 'nonrl rows')
