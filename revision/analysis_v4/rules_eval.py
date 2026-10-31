"""Forecast-ranking rules versus equal-weight daily rebalancing (2024): paired block bootstrap."""
import sys, pathlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from backtest import metrics
NONRL, FCM, OUT = map(pathlib.Path, sys.argv[1:4])
def r(name):
    t = pd.read_csv(NONRL / name / 'trajectory.csv'); return t.set_index('date').daily_return.iloc[1:]
def boot(a, b, block=10, B=2000, seed=1729):
    rng = np.random.default_rng(seed); n = len(a)
    def est(ix):
        ma, mb = metrics(a[ix]), metrics(b[ix]); return [ma['AR'] - mb['AR'], ma['SR'] - mb['SR']]
    p = est(np.arange(n)); S = []
    for _ in range(B):
        st = rng.integers(0, n, size=int(np.ceil(n / block))); ix = ((st[:, None] + np.arange(block)) % n).ravel()[:n]; S.append(est(ix))
    lo, hi = np.percentile(np.array(S), [2.5, 97.5], axis=0); return p, lo, hi
fm = pd.read_csv(FCM / 'forecast_metrics_all.csv'); fm = fm[(fm.kind != 'seed')]
rows = []
for sec in ['bank', 'healthcare']:
    for fee in ['F0', 'F1']:
        ew = r('Baseline_%s_equal_daily_%s' % (sec, fee))
        for f in sorted(NONRL.glob('Baseline_%s_Forecast_*_%s' % (sec, fee))):
            name = f.name[len('Baseline_%s_Forecast_' % sec):-len('_' + fee)]
            x = r(f.name); assert (x.index == ew.index).all()
            p, lo, hi = boot(x.values, ew.values)
            m = metrics(x.values); t = pd.read_csv(f / 'trajectory.csv')
            ic = fm[(fm.sector == sec) & (fm.model == name) & (fm.test_year.astype(str) == '2024')]
            rows.append(dict(sector=sec, fee=fee, model=name, AR=m['AR'], SR=m['SR'], MDD=m['MDD'], turnover=t.gross_turnover.iloc[1:].mean(),
                             dAR=p[0], dAR_lo=lo[0], dAR_hi=hi[0], dSR=p[1], dSR_lo=lo[1], dSR_hi=hi[1],
                             IC2_2024=ic.IC_next.iloc[0] if len(ic) else np.nan))
d = pd.DataFrame(rows); d.to_csv(OUT / 'forecast_rules_vs_ew.csv', index=False)
for sec in ['bank', 'healthcare']:
    z = d[(d.sector == sec) & (d.fee == 'F0')]
    print(sec, 'Spearman(IC2_2024, SR_F0) =', round(spearmanr(z.IC2_2024, z.SR)[0], 3), 'n', len(z))
pd.set_option('display.width', 250); print(d.round(3).to_string())
