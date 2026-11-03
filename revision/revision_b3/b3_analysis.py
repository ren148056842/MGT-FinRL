"""Plan B3 analysis (PROTOCOL_B3.md, section 5): forecast accuracy, H1-H3, descriptive tables.

python b3_analysis.py <panel_dir> <out_dir>
Reads revision_b3/rl/{results,validation,references}, revision_b2/forecasts and revision_b3/forecasts.
Writes to <out_dir>: forecast_metrics.csv, forecast_ic_diffs.csv, rl_runs.csv, rl_summary.csv, rl_tests.csv,
rl_by_algorithm.csv, validation_2024.csv, references.csv, ablation.csv, h3_tests.csv, summary.md
"""
import sys, json, math, pathlib, itertools
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'revision_v4' / 'forecaster'))
import fc_data as D

PANEL_DIR, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
RL = HERE / 'rl'
B2F, B3F = ROOT / 'revision_b2' / 'forecasts', ROOT / 'revision_b3' / 'forecasts'
SECTORS = ['bank', 'healthcare']
ENSEMBLE = {'MGAT_T': 'B2_GAT_Transformer_R', 'A1_R': 'A1_Transformer_only_R', 'GCN_R': 'B1_GCN_Transformer_R',
            'A10_R': 'A10_No_cross_stock_R', 'MGT_R': 'A0_Full_MGT_R'}
FC = ['MGAT_T', 'A1_R', 'GCN_R', 'A10_R', 'MGT_R', 'Ridge_R']
SOURCES = ['Naive', 'Ridge_R', 'A1_R', 'MGAT_T']
ABLATION = ['GCN_R', 'A10_R', 'MGT_R']
ALGOS = ['ppo', 'a2c', 'sac', 'td3', 'ddpg']
SEEDS = [1, 2, 3, 4, 5]
FEES = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6']
PANELS = {'bank': 'bank_panel_20251231.csv', 'healthcare': 'healthcare_panel_20251231.csv'}
try:
    from scipy.stats import t as tdist, norm
    def p_one(tstat, df):
        return float(1 - tdist.cdf(tstat, df))
except Exception:
    def p_one(tstat, df):
        return float(0.5 * math.erfc(tstat / math.sqrt(2)))


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); out = np.empty(m); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); out[i] = run
    return out


def block_index(n, block, rng):
    st = rng.integers(0, n, size=int(np.ceil(n / block)))
    return ((st[:, None] + np.arange(block)) % n).ravel()[:n]


def metrics(r):
    r = np.asarray(r, float); curve = np.r_[1., np.cumprod(1 + r)]; sd = np.std(r, ddof=1)
    mdd = np.max(1 - curve / np.maximum.accumulate(curve)); ar = curve[-1] ** (252 / len(r)) - 1
    return dict(AR=ar, SR=np.sqrt(252) * r.mean() / sd if sd > 0 else np.nan, MDD=mdd, Vol=np.sqrt(252) * sd,
                Calmar=ar / mdd if mdd > 0 else np.nan)


def sr(R):                                    # columnwise annualised Sharpe ratio
    sd = R.std(axis=0, ddof=1)
    return np.where(sd > 0, np.sqrt(252) * R.mean(axis=0) / np.where(sd > 0, sd, 1), np.nan)


# ---------------------------------------------------------------- forecasts
for s, f in PANELS.items():
    D.SECTORS[s]['panel'] = str((PANEL_DIR / f).resolve())


def load_fc(sec, name, fold):
    if name == 'Ridge_R':
        f = B3F / 'runs' / sec / str(fold) / 'Ridge_R' / 'pred.npz'
    else:
        f = (B2F if fold in (2022, 2023) else B3F) / 'ensembles' / ('%s_%d_%s.npz' % (sec, fold, ENSEMBLE[name]))
    return np.load(f, allow_pickle=False) if f.exists() else None


def spearman_rows(P, A, V):
    out = np.full(P.shape[0], np.nan)
    for t in range(P.shape[0]):
        v = V[t] & np.isfinite(P[t]) & np.isfinite(A[t])
        if v.sum() >= 5:
            a = pd.Series(P[t][v]).rank().to_numpy(); b = pd.Series(A[t][v]).rank().to_numpy()
            if a.std() > 0 and b.std() > 0:
                out[t] = np.corrcoef(a, b)[0, 1]
    return out


fm, icd, daily_ic = [], [], {}
for sec in SECTORS:
    p = D.load_panel(sec); Y, M = D.targets(p); dates = [str(d.date()) for d in p['dates']]; row = {d: i for i, d in enumerate(dates)}
    tick_p = ['PAB' if (sec == 'bank' and t == 'PAD') else t for t in p['tickers']]
    for fold in [2022, 2023, 2024, 2025]:
        for name in FC + ['Naive']:
            if name == 'Naive':
                z = load_fc(sec, 'MGAT_T', fold)
                if z is None:
                    continue
                idx = np.array([row[str(d)] for d in z['test_dates']])
                F = D.naive_forecast(p, idx)
            else:
                z = load_fc(sec, name, fold)
                if z is None:
                    continue
                tk = ['PAB' if (sec == 'bank' and str(t) == 'PAD') else str(t) for t in z['tickers']]
                if tk != tick_p:
                    raise ValueError('ticker order differs')
                idx = np.array([row[str(d)] for d in z['test_dates']]); F = z['test']
            Yt, Mt = Y[idx], M[idx]
            q, qm = Yt[:, :, 5] - Yt[:, :, 2], Mt[:, :, 5] & Mt[:, :, 2]
            qh = F[:, :, 5] - F[:, :, 2]
            ic2 = spearman_rows(qh, q, qm) if name != 'Naive' else np.full(len(idx), np.nan)
            ic1 = spearman_rows(F[:, :, 2], Yt[:, :, 2], Mt[:, :, 2]) if name not in ('Naive', 'Ridge_R') else np.full(len(idx), np.nan)
            daily_ic[(sec, fold, name)] = pd.Series(ic2, index=[dates[i] for i in idx])
            r = dict(sector=sec, test_year=fold, model=name, days=len(idx))
            for k, ch in [('h1', 0), ('l1', 1), ('c1', 2)]:
                if name == 'Ridge_R':
                    r['RMSE_' + k] = np.nan; continue
                e = (F[:, :, ch] - Yt[:, :, ch])[Mt[:, :, ch]]
                r['RMSE_' + k] = float(np.sqrt(np.mean(e ** 2)))
            for k, v in [('IC1', ic1), ('IC2', ic2)]:
                v = v[np.isfinite(v)]
                r[k] = float(v.mean()) if len(v) else np.nan
                r[k + '_t'] = float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))) if len(v) > 2 else np.nan
            fm.append(r)
fm = pd.DataFrame(fm)
naive = fm[fm.model == 'Naive'].set_index(['sector', 'test_year'])
for k in ['h1', 'l1', 'c1']:
    fm['RMSE_%s_rel_naive' % k] = [r['RMSE_' + k] / naive.loc[(r.sector, r.test_year), 'RMSE_' + k] - 1 for _, r in fm.iterrows()]
fm.to_csv(OUT / 'forecast_metrics.csv', index=False)

h3 = []
rng = np.random.default_rng(1729)
for sec in SECTORS:
    for year in [2024, 2025]:
        a = daily_ic.get((sec, year, 'MGAT_T'))
        if a is None:
            continue
        v = a.dropna(); t = v.mean() / v.std(ddof=1) * np.sqrt(len(v))
        if year == 2025:
            h3.append(dict(sector=sec, year=year, IC2=v.mean(), t=t, days=len(v), p_one_sided=p_one(t, len(v) - 1)))
        for other in ['A1_R', 'GCN_R', 'A10_R', 'MGT_R', 'Ridge_R']:
            b = daily_ic.get((sec, year, other))
            if b is None:
                continue
            d = (a - b).dropna().to_numpy()
            bs = [d[block_index(len(d), 10, rng)].mean() for _ in range(2000)]
            icd.append(dict(sector=sec, year=year, comparison='MGAT_T - ' + other, dIC2=d.mean(),
                            ci95_low=np.percentile(bs, 2.5), ci95_high=np.percentile(bs, 97.5), days=len(d)))
h3 = pd.DataFrame(h3)
if len(h3):
    h3['p_holm'] = holm(h3.p_one_sided)
h3.to_csv(OUT / 'h3_tests.csv', index=False)
pd.DataFrame(icd).to_csv(OUT / 'forecast_ic_diffs.csv', index=False)

# ---------------------------------------------------------------- RL
def returns(path):
    f = path / 'trajectory.csv'
    if not f.exists():
        return None
    t = pd.read_csv(f)
    return t.set_index('date').daily_return.iloc[1:], t


R, rows = {}, []
for sec, src, algo, seed, fee in itertools.product(SECTORS, SOURCES + ABLATION, ALGOS, SEEDS, FEES):
    x = returns(RL / 'results' / ('B3_%s_%s_%s_s%d_%s' % (sec, src, algo, seed, fee)))
    if x is None:
        continue
    r, t = x; R[(sec, src, algo, seed, fee)] = r
    rows.append(dict(sector=sec, source=src, algorithm=algo, seed=seed, fee=fee, **metrics(r.to_numpy()),
                     turnover=float(t.gross_turnover.iloc[1:].mean())))
runs = pd.DataFrame(rows); runs.to_csv(OUT / 'rl_runs.csv', index=False)
if len(runs):
    summ = runs.groupby(['sector', 'source', 'algorithm', 'fee'])[['AR', 'SR', 'MDD', 'Calmar', 'Vol', 'turnover']].agg(['mean', 'std']).reset_index()
    summ.columns = ['_'.join(c).strip('_') for c in summ.columns]; summ.to_csv(OUT / 'rl_summary.csv', index=False)
    pooled = runs.groupby(['sector', 'source', 'fee'])[['AR', 'SR', 'MDD', 'Calmar', 'Vol', 'turnover']].agg(['mean', 'std']).reset_index()
    pooled.columns = ['_'.join(c).strip('_') for c in pooled.columns]; pooled.to_csv(OUT / 'rl_pooled.csv', index=False)


def panel(sec, src, fee='F1', algos=ALGOS):
    keys = [(sec, src, a, s, fee) for a in algos for s in SEEDS]
    if not all(k in R for k in keys):
        return None
    return pd.concat([R[k] for k in keys], axis=1).to_numpy(float)


def boot(A, B, block, n_boot=2000, seed=1729):
    rng = np.random.default_rng(seed); n = A.shape[0]
    point = np.nanmean(sr(A) - sr(B)); s = np.empty(n_boot)
    for b in range(n_boot):
        ix = block_index(n, block, rng); s[b] = np.nanmean(sr(A[ix]) - sr(B[ix]))
    return point, np.percentile(s, 5), np.percentile(s, 95), float(np.mean((s - point) >= point))


tests = []
for sec in SECTORS:
    G = panel(sec, 'MGAT_T')
    if G is None:
        continue
    for ref, hyp in [('Naive', 'H1'), ('A1_R', 'H2a'), ('Ridge_R', 'H2b')]:
        Bm = panel(sec, ref)
        if Bm is None:
            continue
        for block in [5, 10, 20]:
            d, lo, hi, pv = boot(G, Bm, block)
            tests.append(dict(sector=sec, hypothesis=hyp, comparison='MGAT_T - ' + ref, fee='F1', block=block,
                              dSR=d, ci90_low=lo, ci90_high=hi, p_one_sided=pv))
T = pd.DataFrame(tests)
if len(T):
    T['p_holm'] = np.nan
    main = T[T.block == 10]
    h1 = main[main.hypothesis == 'H1']
    T.loc[h1.index, 'p_holm_H1_across_sectors'] = holm(h1.p_one_sided)
    for sec in SECTORS:
        g = main[main.sector == sec]
        T.loc[g.index, 'p_holm_within_sector'] = holm(g.p_one_sided)
    T = T.drop(columns='p_holm')
T.to_csv(OUT / 'rl_tests.csv', index=False)

byalg = []
for sec, algo in itertools.product(SECTORS, ALGOS):
    G = panel(sec, 'MGAT_T', algos=[algo])
    for ref in ['Naive', 'A1_R', 'Ridge_R']:
        Bm = panel(sec, ref, algos=[algo])
        if G is None or Bm is None:
            continue
        d, lo, hi, pv = boot(G, Bm, 10)
        byalg.append(dict(sector=sec, algorithm=algo, comparison='MGAT_T - ' + ref, fee='F1', dSR=d, ci90_low=lo, ci90_high=hi, p_one_sided=pv))
for fee in FEES:
    for sec in SECTORS:
        G, Bm = panel(sec, 'MGAT_T', fee), panel(sec, 'Naive', fee)
        if G is None or Bm is None:
            continue
        d, lo, hi, pv = boot(G, Bm, 10)
        byalg.append(dict(sector=sec, algorithm='all', comparison='MGAT_T - Naive', fee=fee, dSR=d, ci90_low=lo, ci90_high=hi, p_one_sided=pv))
pd.DataFrame(byalg).to_csv(OUT / 'rl_by_algorithm.csv', index=False)

abl = []
for sec in SECTORS:
    G = panel(sec, 'MGAT_T', algos=['ppo'])
    for ref in ['Naive'] + ABLATION + ['A1_R', 'Ridge_R']:
        Bm = panel(sec, ref, algos=['ppo'])
        if G is None or Bm is None:
            continue
        d, lo, hi, pv = boot(G, Bm, 10)
        abl.append(dict(sector=sec, comparison='MGAT_T - ' + ref, algorithm='ppo', fee='F1', dSR=d, ci90_low=lo, ci90_high=hi))
pd.DataFrame(abl).to_csv(OUT / 'ablation.csv', index=False)

val = []
for f in sorted((RL / 'validation').glob('B3_*_F1')):
    m = json.loads((f / 'metrics.json').read_text(encoding='utf-8'))
    val.append({k: m.get(k) for k in ['sector', 'source', 'algorithm', 'seed', 'AR', 'SR', 'MDD']})
val = pd.DataFrame(val); val.to_csv(OUT / 'validation_2024.csv', index=False)

ref = []
for f in sorted((RL / 'references').glob('*')):
    x = returns(f)
    if x is None:
        continue
    m = json.loads((f / 'metrics.json').read_text(encoding='utf-8'))
    ref.append(dict(sector=m['sector'], strategy=m['strategy'], period=m['period'], fee=m['fee'], **metrics(x[0].to_numpy()),
                    turnover=float(x[1].gross_turnover.iloc[1:].mean())))
ref = pd.DataFrame(ref); ref.to_csv(OUT / 'references.csv', index=False)

# ---------------------------------------------------------------- summary
def md(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


L = ['# Plan B3 results', '']
L += ['## Forecasts (test years)', '', fm[fm.test_year.isin([2024, 2025])].round(4).pipe(md), '']
L += ['## H3: 2025 IC2 of MGAT-T', '', h3.round(4).pipe(md) if len(h3) else 'n/a', '']
if len(runs):
    L += ['## RL, 2025, F1, pooled over agents and seeds', '', pooled[pooled.fee == 'F1'].round(4).pipe(md), '']
L += ['## H1/H2 tests (F1)', '', T.round(4).pipe(md) if len(T) else 'n/a', '']
L += ['## References', '', ref[ref.fee.isin(['F0', 'F1'])].round(4).pipe(md) if len(ref) else 'n/a', '']
(OUT / 'summary.md').write_text('\n'.join(L), encoding='utf-8')
print('\n'.join(L))
