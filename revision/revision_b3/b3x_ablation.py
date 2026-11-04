"""Plan B3, post-hoc addendum (PROTOCOL_B3.md, "Addendum X"): relation-layer and complete-graph ablation
of the MGAT forecaster, and a descriptive analysis of its graph-attention coefficients.

Run after the registered 2025 evaluation; nothing here changes the registered forecaster, agents or tests.

python b3x_ablation.py smoke     --panel-dir D:/paper/2/xiu/data2025/panels
python b3x_ablation.py train     --folds 2022,2023 --sectors bank --threads 4 --panel-dir ...
python b3x_ablation.py attention --panel-dir ...          # MGAT fold 2025, seeds 1-3 (registered weights)
python b3x_ablation.py evaluate  --panel-dir ... --out results_x

Variants (identical to B2_GAT_Transformer except for the admissible neighbours of graph attention):
  X1_GAT_no_KD   relations QA + SH         X2_GAT_no_QA   relations KD + SH
  X3_GAT_no_SH   relations KD + QA         X4_GAT_complete  all ordered pairs (unrestricted inter-stock attention)
Training code, loss, seeds 1-3, folds and panels are those of the registered runs (fc_run_b3.py).
Outputs: revision_b3/forecasts_x/{runs,ensembles}, revision_b3/<out>/*.csv
"""
import sys, json, time, argparse, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'revision_b2'))
sys.path.insert(0, str(ROOT / 'revision_v4' / 'forecaster'))
import numpy as np
import pandas as pd
import fc_data as D
import fc_run as V4
import fc_run_b2 as B2
import fc_run_b3 as B3                         # sets fold 2025 and the extended-panel loader

OUTX = HERE / 'forecasts_x'
B2.OUT = OUTX
B2.TAG = 'b3x'
VARIANTS = {'X1_GAT_no_KD': [0, 1, 1], 'X2_GAT_no_QA': [1, 0, 1], 'X3_GAT_no_SH': [1, 1, 0], 'X4_GAT_complete': [1, 1, 1],
            'X5_GAT_1layer': [1, 1, 1], 'X6_GAT_2layers': [1, 1, 1], 'X7_Transformer_2blocks': [1, 1, 1]}
EXTRA = {'X5_GAT_1layer': dict(units=[64]), 'X6_GAT_2layers': dict(units=[32, 64]), 'X7_Transformer_2blocks': dict(blocks=2)}
FIRST = ['X1_GAT_no_KD', 'X2_GAT_no_QA', 'X3_GAT_no_SH', 'X4_GAT_complete']
ORIG_PANEL = {s: D.SECTORS[s]['panel'] for s in D.SECTORS}


def register():
    import tensorflow as tf
    import fc_models as FM
    for k, w in VARIANTS.items():
        FM.CONFIGS[k] = dict(graph='gat', w=list(w), learn_alpha=False, temporal='transformer', complete=(k == 'X4_GAT_complete'),
                             **EXTRA.get(k, {}))
    base = FM.Forecaster
    if getattr(base, '_b3x', False):
        return

    class ForecasterX(base):
        _b3x = True

        def __init__(self, N, F, P, graphs, cfg, **kw):
            old = FM.UNITS_GNN
            if cfg.get('units'):
                FM.UNITS_GNN = list(cfg['units'])              # depth variants (read only in Forecaster.__init__)
            try:
                super().__init__(N, F, P, graphs, cfg, **kw)
            finally:
                FM.UNITS_GNN = old
            if cfg.get('complete'):
                for layer in self.gnn:
                    layer.mask = tf.constant(np.ones((N, N), bool))
            self.n_blocks = int(cfg.get('blocks', 1))
            if self.n_blocks == 2:                             # a second, identical encoder block
                nh, dff, dr = kw.get('num_heads', 4), kw.get('dff', 128), kw.get('dropout', 0.1)
                self.attn2 = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=max(1, self.D // nh))
                self.n3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                self.n4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                self.ffn2 = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(self.D)])
                self.drop2 = tf.keras.layers.Dropout(dr)

        def call(self, x, training=False):
            if self.n_blocks == 1:
                return super().call(x, training=training)
            if self.graph:
                x = tf.concat([x, self.graph_branch(x)], axis=-1)
            B, P, N = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
            s = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [B * N, P, self.D]) + self.pos
            a = self.attn(s, s, s, training=training)
            s = self.n1(s + self.drop(a, training=training))
            s = self.n2(s + self.drop(self.ffn(s), training=training))
            a = self.attn2(s, s, s, training=training)
            s = self.n3(s + self.drop2(a, training=training))
            s = self.n4(s + self.drop2(self.ffn2(s), training=training))
            return self.out(tf.reshape(s[:, -1], [B, N, -1]))
    FM.Forecaster = ForecasterX


def set_panel(fold, panel_dir):
    for s in D.SECTORS:
        D.SECTORS[s]['panel'] = ORIG_PANEL[s]
    if fold == 2025:
        B3.use_extended(panel_dir)


def neighbourhoods(graphs, w, complete=False):
    import fc_models as FM
    N = graphs[0].shape[0]
    if complete:
        return np.ones((N, N), bool)
    return (FM.fused_static(graphs, w) > 0) | np.eye(N, dtype=bool)


# ------------------------------------------------------------------ training
def train(folds, sectors, threads, panel_dir, smoke=False, names=None):
    register()
    names = list(names or FIRST)
    B2.TAG = 'b3x_%s_%s_%s' % ('_'.join(sectors), '_'.join(map(str, folds)), names[0][:2])
    V4.set_threads(threads or None)
    total = len(folds) * len(sectors) * len(names) * len(B2.SEEDS); done = 0
    for fold in folds:
        set_panel(fold, panel_dir)
        for sec in sectors:
            fd = V4.Fold(sec, fold)
            for k in names:
                m = neighbourhoods(fd.graphs, VARIANTS[k], k == 'X4_GAT_complete')
                print(B2.now(), sec, fold, k, 'admissible ordered pairs (excl. self) %.3f' % ((m.sum() - m.shape[0]) / (m.shape[0] * (m.shape[0] - 1))), flush=True)
            if smoke:
                for k in names:
                    meta = B2.train_nn(fd, k, 1, epochs=1, keep=False)
                    print('smoke ok', sec, fold, k, meta['params'], flush=True)
                continue
            for k in names:
                for seed in B2.SEEDS:
                    meta = B2.train_nn(fd, k, seed)
                    done += 1
                    print('%s done %d/%d %s %d %s_R s%d epochs=%s best=%s %.1fs' % (B2.now(), done, total, sec, fold, k, seed,
                          meta.get('epochs_run'), meta.get('best_epoch'), meta.get('train_seconds', 0)), flush=True)
                B2.ensemble(sec, fold, k + '_R')
    if not smoke:
        (OUTX / ('DONE_%s_%s_%s' % ('_'.join(sectors), '_'.join(map(str, folds)), names[0][:2]))).write_text(B2.now(), encoding='utf-8')


# ------------------------------------------------------------------ attention analysis
def attention(panel_dir, out, smoke=False):
    """Graph-attention coefficients of the registered MGAT (fold 2025, seeds 1-3) on the last day of each 2025 window."""
    import tensorflow as tf
    import fc_models as FM
    set_panel(2025, panel_dir)
    rows, firm_rows = [], []
    for sec in ['bank', 'healthcare']:
        fd = V4.Fold(sec, 2025)
        KD, QA, SH = [np.asarray(g, float) for g in fd.graphs]
        N = fd.N; off = ~np.eye(N, dtype=bool)
        adm = (FM.fused_static(fd.graphs, [1, 1, 1]) > 0) & off
        rel = {'SH': (SH > 0) & off, 'QA': (QA > 0) & off, 'KD': (KD > 0) & off}
        rel['KD only'] = rel['KD'] & ~rel['SH'] & ~rel['QA']
        x = fd.x['test'] if not smoke else fd.x['test'][:8]
        att_all = []
        for seed in [1, 2, 3]:
            d = B3.OUT / 'runs' / sec / '2025' / 'B2_GAT_Transformer_R' / ('seed%d' % seed)
            model = FM.Forecaster(N, fd.F, D.WINDOW, fd.graphs, FM.CONFIGS['B2_GAT_Transformer'])
            _ = model(tf.zeros((1, D.WINDOW, N, fd.F)), training=False)
            model.load_weights(str(d / 'weights.h5'))
            pred = model(tf.constant(np.asarray(x[:4], np.float32)), training=False).numpy()
            ref = np.load(d / 'pred.npz')['test'][:4]
            chk = float(np.nanmax(np.abs(fd.sc.y_inv(pred) - ref)))
            h = tf.constant(np.asarray(x, np.float32)); per_layer = []
            for layer in model.gnn:
                z = tf.einsum('bpnf,fu->bpnu', h, layer.W)
                f1 = tf.einsum('bpnu,uk->bpnk', z, layer.a1); f2 = tf.einsum('bpnu,uk->bpkn', z, layer.a2)
                e = tf.nn.leaky_relu(f1 + f2, 0.2)
                e = tf.where(layer.mask, e, tf.fill(tf.shape(e), -1e9))
                att = tf.nn.softmax(e, axis=-1)
                per_layer.append(att[:, -1].numpy())                      # [days, N, N], last day of each window
                h = tf.nn.elu(tf.matmul(att, z) + layer.b)
            att_all.append(np.stack(per_layer))                              # [layers, days, N, N]
            print(sec, 'seed', seed, 'reload check max abs diff %.2e' % chk, flush=True)
            tf.keras.backend.clear_session()
        A = np.stack(att_all)                                               # [seeds, layers, days, N, N]
        deg = adm.sum(1)
        for li in range(A.shape[1]):
            a = A[:, li]                                                    # [seeds, days, N, N]
            mean_a = a.mean(axis=(0, 1))                                    # [N, N]
            r = dict(sector=sec, layer=li + 1, self_weight=float(np.mean(np.diagonal(a, axis1=2, axis2=3))),
                     uniform_self=float(np.mean(1.0 / (deg + 1))))
            nb = mean_a[adm].mean()
            for k, msk in rel.items():
                msk = msk & adm
                r['ratio_' + k] = float(mean_a[msk].mean() / nb) if msk.any() else np.nan
                r['pairs_' + k] = int(msk.sum())
            p = np.where(adm[None, None] | np.eye(N, dtype=bool)[None, None], a, np.nan)
            ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=-1)     # [seeds, days, N]
            r['entropy_rel_uniform'] = float(np.mean(ent / np.log(deg + 1)[None, None]))
            sd_t = a.std(axis=1).mean(axis=0)                               # day-to-day sd, averaged over seeds
            r['cv_over_days_median'] = float(np.median(sd_t[adm] / mean_a[adm]))
            r['spearman_att_kd'] = float(np.corrcoef(pd.Series(mean_a[adm]).rank(), pd.Series(KD[adm]).rank())[0, 1])
            r['seed_agreement'] = float(np.mean([np.corrcoef(a[i].mean(0)[adm], a[j].mean(0)[adm])[0, 1]
                                                 for i in range(3) for j in range(i + 1, 3)]))
            rows.append(r)
            if li == 0:
                recv = mean_a.copy(); np.fill_diagonal(recv, 0)
                for j, t in enumerate(fd.p['tickers']):
                    firm_rows.append(dict(sector=sec, ticker=t, attention_received=float(recv[:, j].sum()),
                                          in_degree=int(adm[:, j].sum()), self_weight=float(np.mean(a[:, :, j, j]))))
        np.savez_compressed(out / ('attention_%s.npz' % sec), mean=A.mean(axis=(0, 2)), tickers=np.array(fd.p['tickers']))
    pd.DataFrame(rows).to_csv(out / 'attention_summary.csv', index=False)
    pd.DataFrame(firm_rows).to_csv(out / 'attention_firms.csv', index=False)


# ------------------------------------------------------------------ evaluation
def spearman_rows(P, A, V):
    out = np.full(P.shape[0], np.nan)
    for t in range(P.shape[0]):
        v = V[t] & np.isfinite(P[t]) & np.isfinite(A[t])
        if v.sum() >= 5:
            a = pd.Series(P[t][v]).rank().to_numpy(); b = pd.Series(A[t][v]).rank().to_numpy()
            if a.std() > 0 and b.std() > 0:
                out[t] = np.corrcoef(a, b)[0, 1]
    return out


def block_index(n, block, rng):
    st = rng.integers(0, n, size=int(np.ceil(n / block)))
    return ((st[:, None] + np.arange(block)) % n).ravel()[:n]


def evaluate(panel_dir, out):
    B2F, B3F = ROOT / 'revision_b2' / 'forecasts', HERE / 'forecasts'
    for s, f in B3.PANELS.items():
        D.SECTORS[s]['panel'] = str((pathlib.Path(panel_dir) / f).resolve())
    rows, diffs, meta_rows = [], [], []
    rng = np.random.default_rng(1729)
    for sec in ['bank', 'healthcare']:
        p = D.load_panel(sec); Y, M = D.targets(p); dates = [str(d.date()) for d in p['dates']]; row = {d: i for i, d in enumerate(dates)}
        for fold in [2022, 2023, 2024, 2025]:
            ref = (B2F if fold in (2022, 2023) else B3F) / 'ensembles' / ('%s_%d_B2_GAT_Transformer_R.npz' % (sec, fold))
            files = {'MGAT_T': ref}
            reg = {'A1_R': 'A1_Transformer_only_R', 'GCN_R': 'B1_GCN_Transformer_R', 'A10_R': 'A10_No_cross_stock_R', 'MGT_R': 'A0_Full_MGT_R'}
            files.update({k: (B2F if fold in (2022, 2023) else B3F) / 'ensembles' / ('%s_%d_%s.npz' % (sec, fold, v)) for k, v in reg.items()})
            files['Ridge_R'] = B3F / 'runs' / sec / str(fold) / 'Ridge_R' / 'pred.npz'
            files.update({k: OUTX / 'ensembles' / ('%s_%d_%s_R.npz' % (sec, fold, k)) for k in VARIANTS})
            z0 = np.load(ref); idx = np.array([row[str(d)] for d in z0['test_dates']]); Yt, Mt = Y[idx], M[idx]
            Fn = D.naive_forecast(p, idx)
            q, qm = Yt[:, :, 5] - Yt[:, :, 2], Mt[:, :, 5] & Mt[:, :, 2]
            ic = {}
            for name, f in files.items():
                if not f.exists():
                    continue
                z = np.load(f); assert list(z['test_dates']) == list(z0['test_dates'])
                F = z['test']; ic[name] = spearman_rows(F[:, :, 5] - F[:, :, 2], q, qm)
                r = dict(sector=sec, test_year=fold, model=name, days=len(idx))
                for k, ch in [('h1', 0), ('l1', 1), ('c1', 2)]:
                    if name == 'Ridge_R':
                        r['RMSE_%s_rel_naive' % k] = np.nan; continue
                    e = (F[:, :, ch] - Yt[:, :, ch])[Mt[:, :, ch]]; en = (Fn[:, :, ch] - Yt[:, :, ch])[Mt[:, :, ch]]
                    r['RMSE_%s_rel_naive' % k] = float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(en ** 2)) - 1)
                # direction measures at the tradable horizon: sign of q, and sign of q relative to the day's cross-sectional mean
                qh = F[:, :, 5] - F[:, :, 2]; ok = qm & np.isfinite(qh)
                qhd = qh - np.nanmean(np.where(ok, qh, np.nan), 1, keepdims=True); qd_ = q - np.nanmean(np.where(ok, q, np.nan), 1, keepdims=True)
                hit_abs = np.where(ok & (q != 0) & (qh != 0), (np.sign(qh) == np.sign(q)).astype(float), np.nan)
                hit_rel = np.where(ok & (qd_ != 0) & (qhd != 0), (np.sign(qhd) == np.sign(qd_)).astype(float), np.nan)
                for nm, hv in [('DA_q', hit_abs), ('DA_rel', hit_rel)]:
                    dly = np.nanmean(hv, 1); dly = dly[np.isfinite(dly)]
                    r[nm] = float(np.nanmean(hv)) if np.isfinite(hv).any() else np.nan
                    r[nm + '_t'] = float((dly.mean() - 0.5) / dly.std(ddof=1) * np.sqrt(len(dly))) if len(dly) > 2 else np.nan
                v = ic[name][np.isfinite(ic[name])]
                r['IC2'] = float(v.mean()); r['IC2_t'] = float(v.mean() / v.std(ddof=1) * np.sqrt(len(v)))
                rows.append(r)
            for name in list(VARIANTS) + ['A1_R', 'GCN_R', 'A10_R', 'MGT_R', 'Ridge_R']:
                if name not in ic:
                    continue
                d = ic['MGAT_T'] - ic[name]; d = d[np.isfinite(d)]
                bs = [d[block_index(len(d), 10, rng)].mean() for _ in range(2000)]
                diffs.append(dict(sector=sec, test_year=fold, comparison='MGAT_T - ' + name, diff=float(d.mean()),
                                  ci95_low=float(np.percentile(bs, 2.5)), ci95_high=float(np.percentile(bs, 97.5))))
        for k in ['B2_GAT_Transformer'] + list(VARIANTS):
            base = (B3F if k == 'B2_GAT_Transformer' else OUTX) / 'runs' / sec / '2025' / (k + '_R')
            for s in [1, 2, 3]:
                f = base / ('seed%d' % s) / 'meta.json'
                if f.exists():
                    m = json.loads(f.read_text(encoding='utf-8'))
                    meta_rows.append(dict(sector=sec, model=k, seed=s, params=m['params'], epochs=m['epochs_run'],
                                          best_epoch=m['best_epoch'], seconds=m['train_seconds']))
    pd.DataFrame(rows).to_csv(out / 'ablation_forecast.csv', index=False)
    pd.DataFrame(diffs).to_csv(out / 'ablation_ic_diffs.csv', index=False)
    pd.DataFrame(meta_rows).to_csv(out / 'ablation_meta_2025.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['smoke', 'smoke2', 'train', 'attention', 'evaluate'])
    ap.add_argument('--folds', default='2022,2023,2024,2025')
    ap.add_argument('--sectors', default='bank,healthcare')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--panel-dir', required=True)
    ap.add_argument('--out', default='results_x')
    ap.add_argument('--variants', default='')
    a = ap.parse_args()
    out = HERE / a.out; out.mkdir(parents=True, exist_ok=True)
    folds = [int(x) for x in a.folds.split(',')]; sectors = a.sectors.split(',')
    if a.command == 'smoke2':
        train([2022], ['bank'], a.threads, a.panel_dir, smoke=True, names=list(EXTRA))
        print('SMOKE2 OK', flush=True)
    elif a.command == 'smoke':
        train([2022, 2025], ['healthcare'], a.threads, a.panel_dir, smoke=True)
        (out / 'smoke').mkdir(exist_ok=True)
        attention(a.panel_dir, out / 'smoke', smoke=True)
        print('SMOKE OK', flush=True)
    elif a.command == 'train':
        train(folds, sectors, a.threads, a.panel_dir, names=a.variants.split(',') if a.variants else None)
    elif a.command == 'attention':
        attention(a.panel_dir, out)
    else:
        evaluate(a.panel_dir, out)


if __name__ == '__main__':
    main()
