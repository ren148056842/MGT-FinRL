"""Plan B3, post-hoc addendum Y (PROTOCOL_B3.md): descriptive analyses requested by the reviewers.

python b3x_refs.py all --panel-dir D:/paper/2/xiu/data2025/panels --out results_x

1. Allocation baselines with the registered ledger and costs (2024 and 2025): long-only minimum variance
   (252-day sample covariance of the RL state), rebalanced monthly and daily, and inverse volatility, monthly.
2. Regimes within 2025: days split at the median of the trailing 20-day volatility of the daily
   equal-weight portfolio; Sharpe ratios of the 25 MGAT and 25 naive policies (F1) on each half and the
   paired difference with a circular block bootstrap (block 10, 2000 resamples) of the days of that half.
3. Market structure by sector and year (2022-2025): mean pairwise correlation of daily log returns, share
   of the first principal component of the correlation matrix, mean cross-sectional standard deviation of
   daily returns, and the volatility of the equal-weight portfolio.
Outputs: rl/references_x/, <out>/allocation_baselines.csv, regimes_2025.csv, market_structure.csv
"""
import sys, json, pathlib, argparse, itertools
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'revision_v4' / 'forecaster'))
import numpy as np
import pandas as pd
import b3_rl as R
from v3_data import RealPortfolioEnv
from v2_data import export_trace
from backtest import FEES

ALGOS, SEEDS = R.ALGOS, R.SEEDS


def metrics(r):
    r = np.asarray(r, float); curve = np.r_[1., np.cumprod(1 + r)]; sd = np.std(r, ddof=1)
    mdd = np.max(1 - curve / np.maximum.accumulate(curve)); ar = curve[-1] ** (252 / len(r)) - 1
    return dict(AR=ar, SR=np.sqrt(252) * r.mean() / sd if sd > 0 else np.nan, MDD=mdd, Vol=np.sqrt(252) * sd,
                Calmar=ar / mdd if mdd > 0 else np.nan)


def sr(R_):
    sd = R_.std(axis=0, ddof=1)
    return np.where(sd > 0, np.sqrt(252) * R_.mean(axis=0) / np.where(sd > 0, sd, 1), np.nan)


def block_index(n, block, rng):
    st = rng.integers(0, n, size=int(np.ceil(n / block)))
    return ((st[:, None] + np.arange(block)) % n).ravel()[:n]


def min_var(S):
    """Long-only minimum-variance weights: projected gradient on the simplex (exact enough for 15 assets)."""
    n = S.shape[0]; S = 0.5 * (S + S.T) + 1e-10 * np.eye(n)
    try:
        from scipy.optimize import minimize
        res = minimize(lambda w: w @ S @ w, np.ones(n) / n, jac=lambda w: 2 * S @ w, method='SLSQP',
                       bounds=[(0, 1)] * n, constraints=[dict(type='eq', fun=lambda w: w.sum() - 1)],
                       options=dict(ftol=1e-14, maxiter=500))
        w = np.clip(res.x, 0, None)
        return w / w.sum()
    except Exception:
        w = np.ones(n) / n; L = 2 * np.linalg.eigvalsh(S).max()
        for _ in range(5000):
            v = w - (2 * S @ w) / L
            u = np.sort(v)[::-1]; css = np.cumsum(u); k = np.nonzero(u * np.arange(1, n + 1) > css - 1)[0][-1]
            w = np.maximum(v - (css[k] - 1) / (k + 1), 0)
        return w


def allocation(out):
    rows = []
    for sec in R.SECTORS:
        for period, part in [('2024', 'validation'), ('2025', 'test')]:
            data = R.build(sec, 'Naive', need_test=True)[0][part]; N = len(data.tickers)
            for name in ['min_var_monthly', 'min_var_daily', 'inv_vol_monthly']:
                for fee in FEES:
                    d = R.RUN / 'references_x' / ('%s_%s_%s_%s' % (sec, name, period, fee))
                    env = RealPortfolioEnv(data, FEES[fee]); env.reset(); done = False
                    while not done:
                        i = env.i; S = np.asarray(data.states[i][:N], float)
                        new_month = i == 0 or data.dates[i].month != data.dates[i - 1].month
                        if name == 'min_var_daily' or new_month:
                            w = min_var(S) if name.startswith('min_var') else (1 / np.sqrt(np.diag(S))) / (1 / np.sqrt(np.diag(S))).sum()
                        else:
                            w = None
                        _, _, done, _, _ = env.advance(w)
                    info = dict(run_id=d.name, sector=sec, strategy=name, period=period, fee=fee)
                    export_trace(env, d, info)
                    t = pd.read_csv(d / 'trajectory.csv')
                    rows.append(dict(info, **metrics(t.daily_return.iloc[1:].to_numpy()), turnover=float(t.gross_turnover.iloc[1:].mean()),
                                     max_weight=float(np.nan)))
            R.log('allocation baselines', sector=sec, period=period)
    pd.DataFrame(rows).to_csv(out / 'allocation_baselines.csv', index=False)


def returns(path):
    t = pd.read_csv(path / 'trajectory.csv')
    return t.set_index('date').daily_return.iloc[1:]


def regimes(out):
    rows = []
    for sec in R.SECTORS:
        ew = returns(R.RUN / 'references' / ('%s_equal_daily_2025_F1' % sec))
        vol = ew.shift(1).rolling(20, min_periods=5).std()
        vol.iloc[0] = vol.dropna().iloc[0] if vol.isna().iloc[0] else vol.iloc[0]
        vol = vol.bfill(); high = (vol > vol.median()).to_numpy()
        pan = {}
        for src in ['MGAT_T', 'Naive']:
            cols = [returns(R.RUN / 'results' / ('B3_%s_%s_%s_s%d_F1' % (sec, src, a, s))) for a in ALGOS for s in SEEDS]
            pan[src] = pd.concat(cols, axis=1).loc[ew.index].to_numpy(float)
        for reg, m in [('high volatility', high), ('low volatility', ~high), ('all days', np.ones(len(high), bool))]:
            G, B = pan['MGAT_T'][m], pan['Naive'][m]; rng = np.random.default_rng(1729); n = len(G)
            d = float(np.nanmean(sr(G) - sr(B))); bs = np.empty(2000)
            for b in range(2000):
                ix = block_index(n, 10, rng); bs[b] = np.nanmean(sr(G[ix]) - sr(B[ix]))
            e = ew.to_numpy()[m]
            rows.append(dict(sector=sec, regime=reg, days=int(n), ew_SR=float(np.sqrt(252) * e.mean() / e.std(ddof=1)),
                             ew_ann_vol=float(np.sqrt(252) * e.std(ddof=1)), ew_ann_return=float(252 * e.mean()),
                             MGAT_SR=float(np.nanmean(sr(G))), Naive_SR=float(np.nanmean(sr(B))), dSR=d,
                             ci90_low=float(np.percentile(bs, 5)), ci90_high=float(np.percentile(bs, 95)),
                             p_one_sided=float(np.mean((bs - d) >= d)),
                             first=str(ew.index[m][0]), last=str(ew.index[m][-1])))
    pd.DataFrame(rows).to_csv(out / 'regimes_2025.csv', index=False)


def structure(out, panel_dir):
    import fc_data as D
    for s, f in R.PANELS.items():
        D.SECTORS[s]['panel'] = str((pathlib.Path(panel_dir) / f).resolve())
    rows = []
    for sec in R.SECTORS:
        p = D.load_panel(sec); C = p['C']; ok = p['traded']
        r = np.log(C[1:] / C[:-1]); r[~(ok[1:] & ok[:-1])] = np.nan
        dates = p['dates'][1:]
        for y in [2022, 2023, 2024, 2025]:
            x = r[dates.year == y]; df = pd.DataFrame(x)
            c = df.corr(min_periods=100).to_numpy(); iu = np.triu_indices_from(c, 1)
            ev = np.linalg.eigvalsh(np.nan_to_num(c, nan=0.0) + np.diag(1 - np.diag(np.nan_to_num(c, nan=0.0))))
            ew = np.nanmean(np.exp(x) - 1, axis=1)
            rows.append(dict(sector=sec, year=y, days=int(len(x)), mean_pairwise_corr=float(np.nanmean(c[iu])),
                             pc1_share=float(ev.max() / ev.sum()), xs_sd=float(np.nanmean(np.nanstd(x, axis=1, ddof=1))),
                             ew_ann_vol=float(np.sqrt(252) * np.nanstd(ew, ddof=1)), ew_return=float(np.prod(1 + ew) - 1)))
    pd.DataFrame(rows).to_csv(out / 'market_structure.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['all', 'allocation', 'regimes', 'structure'])
    ap.add_argument('--panel-dir', required=True)
    ap.add_argument('--out', default='results_x')
    a = ap.parse_args()
    out = HERE / a.out; out.mkdir(parents=True, exist_ok=True)
    R.use_extended(a.panel_dir)
    if a.command in ('all', 'structure'):
        structure(out, a.panel_dir)
    if a.command in ('all', 'regimes'):
        regimes(out)
    if a.command in ('all', 'allocation'):
        allocation(out)
    print('REFS DONE', flush=True)


if __name__ == '__main__':
    main()
