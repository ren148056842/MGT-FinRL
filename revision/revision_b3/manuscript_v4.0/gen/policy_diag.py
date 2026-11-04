"""Descriptive diagnostics of the 2025 policies (post hoc; not part of the registered tests)."""
import numpy as np, pandas as pd, glob, os
U = '/mnt/user-data/uploads/paper--2--xiu'
rows = []
def spear(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return np.corrcoef(ra, rb)[0, 1] if ra.std() > 0 and rb.std() > 0 else np.nan
for sec in ['bank', 'healthcare']:
    z = np.load(f'{U}/revision_b3/forecasts/ensembles/{sec}_2025_B2_GAT_Transformer_R.npz')
    tk = ['PAB' if (sec == 'bank' and str(t) == 'PAD') else str(t) for t in z['tickers']]
    qh = pd.DataFrame(z['test'][:, :, 5] - z['test'][:, :, 2], index=[str(d) for d in z['test_dates']], columns=tk)
    p = pd.read_csv(f'{U}/data2025/panels/{sec}_panel_20251231.csv', dtype={'tic': str}); p['tic'] = p.tic.str.strip().replace({'PAD': 'PAB'} if sec == 'bank' else {})
    p['date'] = pd.to_datetime(p.date); C = p.pivot(index='date', columns='tic', values='close').sort_index()
    q = np.log(C.shift(-2) / C.shift(-1)); q.index = [str(d.date()) for d in q.index]
    dates = list(C.index.strftime('%Y-%m-%d')); prev = {dates[i]: dates[i - 1] for i in range(1, len(dates))}
    for src in ['MGAT_T', 'Naive']:
        for algo in ['ppo', 'a2c', 'sac', 'td3', 'ddpg']:
            for s in range(1, 6):
                f = f'{U}/revision_b3/rl/results/B3_{sec}_{src}_{algo}_s{s}_F1/positions.csv'
                d = pd.read_csv(f)
                W = d.pivot(index='date', columns='tic', values='target_weight').sort_index()
                W = W[W.sum(1) > 0.5]
                N = W.shape[1]
                act = (W - 1 / N).abs().sum(1).mean() / 2
                load, own = [], []
                for dt, w in W.iterrows():
                    t = prev.get(dt)
                    if t is None or t not in qh.index: continue
                    load.append(spear(w[qh.columns].to_numpy(), qh.loc[t].to_numpy()))
                    if t in q.index and q.loc[t].notna().all():
                        own.append(spear(w[q.columns].to_numpy(), q.loc[t].to_numpy()))
                rows.append(dict(sector=sec, source=src, algorithm=algo, seed=s, active_share=act, max_weight=W.max(1).mean(),
                                 forecast_loading=np.nanmean(load), weight_ic=np.nanmean(own), days=len(load)))
D = pd.DataFrame(rows); D.to_csv('/home/claude/v40/policy_diag.csv', index=False)
pd.set_option('display.width', 200)
print(D.groupby(['sector', 'source', 'algorithm'])[['active_share', 'max_weight', 'forecast_loading', 'weight_ic']].mean().round(3))
print(D.groupby(['sector', 'source'])[['active_share', 'max_weight', 'forecast_loading', 'weight_ic']].mean().round(3))
