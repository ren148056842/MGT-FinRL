"""Out-of-sample evaluation of the v4 forecasts (test years 2022, 2023, 2024).

Usage: python fc_eval.py <xiu_root> <forecasts_dir> <out_dir>
Reads per-seed predictions (runs/<sector>/<fold>/<model>/seed<k>/pred.npz), ensembles, Ridge and Naive,
recomputes the targets from the trade panels with fc_data, and writes forecast_metrics_*.csv.
"""
import sys, pathlib, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'revision_v4' / 'forecaster'))
import fc_data as D

ROOT, FC, OUT = map(pathlib.Path, sys.argv[1:4])
D.ROOT = ROOT
OUT.mkdir(parents=True, exist_ok=True)


def nw_var(d, lag=5):
    d = d - d.mean(); n = len(d); v = (d @ d) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return v


def dm(e_model, e_base, valid):
    """Diebold-Mariano on the daily cross-sectional mean squared-error differential (HAC, 5 lags)."""
    m = np.where(valid, e_model ** 2, np.nan); b = np.where(valid, e_base ** 2, np.nan)
    d = np.nanmean(m, 1) - np.nanmean(b, 1); d = d[np.isfinite(d)]
    stat = d.mean() / np.sqrt(nw_var(d) / len(d))
    return stat, 2 * (1 - norm.cdf(abs(stat)))


def ic_series(P, A, valid, min_n=5):
    out = []
    for t in range(P.shape[0]):
        v = valid[t] & np.isfinite(P[t]) & np.isfinite(A[t])
        if v.sum() >= min_n and np.std(P[t][v]) > 0 and np.std(A[t][v]) > 0:
            out.append(spearmanr(P[t][v], A[t][v])[0])
    out = np.array(out)
    return out.mean(), out.mean() / out.std(ddof=1) * np.sqrt(len(out)), len(out)


def metrics(F, Y, M, NV):
    """F, Y: [n, N, 6] log ratios (h1,l1,c1,h2,l2,c2); M: masks; NV: naive forecast."""
    r = {}
    v1 = M[:, :, 2]
    e = F[:, :, 2] - Y[:, :, 2]; e0 = -Y[:, :, 2]            # zero-return benchmark (= previous close)
    r['close_rmse_pct'] = 100 * np.sqrt(np.mean(e[v1] ** 2))
    r['close_mape_pct'] = 100 * np.mean(np.abs(np.exp(e[v1]) - 1))
    r['R2_os_pct'] = 100 * (1 - np.sum(e[v1] ** 2) / np.sum(e0[v1] ** 2))
    r['DM_vs_prevclose'], r['DM_p'] = dm(e, e0, v1)
    nz = v1 & (Y[:, :, 2] != 0) & (F[:, :, 2] != 0)
    r['dir_acc_pct'] = 100 * np.mean(np.sign(F[:, :, 2][nz]) == np.sign(Y[:, :, 2][nz])) if nz.any() else np.nan
    r['IC'], r['IC_t'], _ = ic_series(F[:, :, 2], Y[:, :, 2], v1) if np.nanstd(F[:, :, 2]) > 0 else (np.nan, np.nan, 0)
    v2 = M[:, :, 2] & M[:, :, 5]
    pd2, ad2 = F[:, :, 5] - F[:, :, 2], Y[:, :, 5] - Y[:, :, 2]  # log return from t+1 to t+2 (traded by the next-day rule)
    r['IC_next'], r['IC_next_t'], _ = ic_series(pd2, ad2, v2) if np.nanstd(pd2) > 0 else (np.nan, np.nan, 0)
    nz2 = v2 & (ad2 != 0) & (pd2 != 0)
    r['dir_next_pct'] = 100 * np.mean(np.sign(pd2[nz2]) == np.sign(ad2[nz2])) if nz2.any() else np.nan
    for c, k in [('high', 0), ('low', 1)]:
        vv = M[:, :, k]
        r[c + '_rmse_pct'] = 100 * np.sqrt(np.mean((F[:, :, k] - Y[:, :, k])[vv] ** 2))
        r[c + '_rmse_ratio_naive'] = np.sqrt(np.mean((F[:, :, k] - Y[:, :, k])[vv] ** 2)) / np.sqrt(np.mean((NV[:, :, k] - Y[:, :, k])[vv] ** 2))
        r[c + '_DM_vs_naive'], r[c + '_DM_p'] = dm(F[:, :, k] - Y[:, :, k], NV[:, :, k] - Y[:, :, k], vv)
    vr = M[:, :, 0] & M[:, :, 1]
    pr, ar = F[:, :, 0] - F[:, :, 1], Y[:, :, 0] - Y[:, :, 1]
    r['range_rmse_ratio_naive'] = np.sqrt(np.mean((pr - ar)[vr] ** 2)) / np.sqrt(np.mean(((NV[:, :, 0] - NV[:, :, 1]) - ar)[vr] ** 2))
    r['n_days'] = F.shape[0]
    return r


rows = []
POOL = {}   # (sector, model, seed) -> list of (F, Y, M, NV)
for sec in ['bank', 'healthcare']:
    p = D.load_panel(sec); Y, M = D.targets(p)
    for fold in [2022, 2023, 2024]:
        s = D.split_indices(p['dates'], fold); idx = s['test']
        dates = [str(x.date()) for x in p['dates'][idx]]
        Yt, Mt = Y[idx], M[idx]
        NV = D.naive_forecast(p, idx)
        base = FC / 'runs' / sec / str(fold)
        if not base.exists():
            continue
        for model_dir in sorted(base.iterdir()):
            name = model_dir.name
            files = sorted(model_dir.glob('seed*/pred.npz')) or ([model_dir / 'pred.npz'] if (model_dir / 'pred.npz').exists() else [])
            for f in files:
                z = np.load(f)
                if list(z['test_dates']) != dates:
                    raise ValueError('date mismatch ' + str(f))
                r = metrics(z['test'], Yt, Mt, NV)
                seed = f.parent.name if f.parent.name.startswith('seed') else 'single'
                POOL.setdefault((sec, name, seed), []).append((fold, z['test'], Yt, Mt, NV))
                meta = json.loads((f.parent / 'meta.json').read_text()) if (f.parent / 'meta.json').exists() else {}
                rows.append(dict(sector=sec, test_year=fold, model=name, seed=seed, kind='seed' if seed != 'single' else 'single',
                                 epochs=meta.get('epochs_run'), best_epoch=meta.get('best_epoch'), train_seconds=meta.get('train_seconds'),
                                 infer_seconds=meta.get('test_inference_seconds'), params=meta.get('params'), **r))
            ens = FC / 'ensembles' / ('%s_%d_%s.npz' % (sec, fold, name))
            if ens.exists():
                z = np.load(ens)
                POOL.setdefault((sec, name, 'ensemble'), []).append((fold, z['test'], Yt, Mt, NV))
                rows.append(dict(sector=sec, test_year=fold, model=name, seed='ensemble', kind='ensemble', **metrics(z['test'], Yt, Mt, NV)))
for (sec, name, seed), parts in POOL.items():
    if sorted(x[0] for x in parts) != [2022, 2023, 2024]:
        continue
    cat = [np.concatenate([x[i] for x in parts]) for i in range(1, 5)]
    kind = 'ensemble' if seed == 'ensemble' else ('single' if seed == 'single' else 'seed')
    rows.append(dict(sector=sec, test_year='pooled', model=name, seed=seed, kind=kind, **metrics(*cat)))
d = pd.DataFrame(rows)
d.to_csv(OUT / 'forecast_metrics_all.csv', index=False)
num = [c for c in d.columns if c not in ['sector', 'test_year', 'model', 'seed', 'kind']]
seeds = d[d.kind == 'seed'].groupby(['sector', 'test_year', 'model'])[num].agg(['mean', 'std'])
seeds.columns = ['%s_%s' % c for c in seeds.columns]
seeds.reset_index().to_csv(OUT / 'forecast_metrics_seed_mean_sd.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(d[d.kind != 'seed'][['sector', 'test_year', 'model', 'seed', 'close_rmse_pct', 'R2_os_pct', 'DM_p', 'dir_acc_pct', 'IC', 'IC_t', 'IC_next', 'IC_next_t', 'high_rmse_ratio_naive', 'low_rmse_ratio_naive', 'range_rmse_ratio_naive']].round(3).to_string())
