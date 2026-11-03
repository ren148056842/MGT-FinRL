"""Plan B2 gate (PROTOCOL.md): rank IC of the tradable-horizon return in the development years.

python gate_eval.py <xiu_root> <b2_forecasts_dir> <v4_forecasts_dir> <out_dir>
Uses only the test years of folds 2022 and 2023.  Writes gate_metrics.csv, gate_decision.json.
"""
import sys, json, pathlib
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'revision_v4' / 'forecaster'))
import fc_data as D

ROOT, B2, V4, OUT = map(pathlib.Path, sys.argv[1:5])
D.ROOT = ROOT
OUT.mkdir(parents=True, exist_ok=True)
FOLDS = [2022, 2023]
B2_MODELS = ['A0_Full_MGT_R', 'A10_No_cross_stock_R', 'A1_Transformer_only_R', 'A2_MGNN_only_R', 'B1_GCN_Transformer_R', 'B2_GAT_Transformer_R', 'Ridge_R']
V4_MODELS = ['A0_Full_MGT', 'A10_No_cross_stock', 'A1_Transformer_only', 'Ridge']        # reference (v4 objective)
GATE_RIVALS = ['A10_No_cross_stock_R', 'A1_Transformer_only_R', 'Ridge_R']


def rank(x):
    return pd.Series(x).rank(method='average').to_numpy()


def daily_ic(P, A, valid, min_n=5):
    out = np.full(P.shape[0], np.nan)
    for t in range(P.shape[0]):
        v = valid[t] & np.isfinite(P[t]) & np.isfinite(A[t])
        if v.sum() >= min_n and np.std(P[t][v]) > 0 and np.std(A[t][v]) > 0:
            out[t] = np.corrcoef(rank(P[t][v]), rank(A[t][v]))[0, 1]
    return out


def load_pred(sector, fold, model):
    if model in ['Ridge_R']:
        f = B2 / 'runs' / sector / str(fold) / model / 'pred.npz'
    elif model.endswith('_R'):
        f = B2 / 'ensembles' / ('%s_%d_%s.npz' % (sector, fold, model))
    elif model == 'Ridge':
        f = V4 / 'runs' / sector / str(fold) / 'Ridge' / 'pred.npz'
    else:
        f = V4 / 'ensembles' / ('%s_%d_%s.npz' % (sector, fold, model))
    return np.load(f) if f.exists() else None


rows, series = [], {}
for sec in ['bank', 'healthcare']:
    p = D.load_panel(sec); Y, M = D.targets(p)
    for model in B2_MODELS + V4_MODELS:
        ics = []
        for fold in FOLDS:
            z = load_pred(sec, fold, model)
            if z is None:
                ics = None; break
            s = D.split_indices(p['dates'], fold)
            if list(z['test_dates']) != [str(x.date()) for x in p['dates'][s['test']]]:
                raise ValueError('dates do not match: %s %s %d' % (sec, model, fold))
            F, Yt, Mt = z['test'], Y[s['test']], M[s['test']]
            v2 = Mt[:, :, 2] & Mt[:, :, 5]
            ic = daily_ic(F[:, :, 5] - F[:, :, 2], Yt[:, :, 5] - Yt[:, :, 2], v2)
            ics.append(ic)
            ok = ic[np.isfinite(ic)]
            rows.append(dict(sector=sec, model=model, period=str(fold), IC2=ok.mean(), t=ok.mean() / ok.std(ddof=1) * np.sqrt(len(ok)), days=len(ok)))
        if ics is None:
            continue
        ic = np.concatenate(ics); series[(sec, model)] = ic
        ok = ic[np.isfinite(ic)]
        rows.append(dict(sector=sec, model=model, period='2022-2023', IC2=ok.mean(), t=ok.mean() / ok.std(ddof=1) * np.sqrt(len(ok)), days=len(ok)))
res = pd.DataFrame(rows)
res.to_csv(OUT / 'gate_metrics.csv', index=False)

decision = {}
for sec in ['bank', 'healthcare']:
    r = res[(res.sector == sec) & (res.period == '2022-2023')].set_index('model')
    if 'A0_Full_MGT_R' not in r.index or any(m not in r.index for m in GATE_RIVALS):
        decision[sec] = dict(status='incomplete'); continue
    mgt = r.loc['A0_Full_MGT_R']
    g1 = bool(mgt.IC2 > 0 and mgt.t >= 2.0)
    g2 = {m: bool(mgt.IC2 >= r.loc[m].IC2) for m in GATE_RIVALS}
    diffs = {}
    for m in GATE_RIVALS:
        d = series[(sec, 'A0_Full_MGT_R')] - series[(sec, m)]; d = d[np.isfinite(d)]
        diffs[m] = dict(mean=float(d.mean()), t=float(d.mean() / d.std(ddof=1) * np.sqrt(len(d))))
    decision[sec] = dict(IC2=float(mgt.IC2), t=float(mgt.t), G1=g1, G2=g2, passed=bool(g1 and all(g2.values())),
                         paired_difference_vs_rivals=diffs)
(OUT / 'gate_decision.json').write_text(json.dumps(decision, indent=1), encoding='utf-8')
print(res.round(4).to_string(index=False))
print(json.dumps(decision, indent=1))
