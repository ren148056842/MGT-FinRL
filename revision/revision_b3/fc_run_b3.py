"""Plan B3: walk-forward forecasters for folds 2024 and 2025 and the two-target Ridge_R (PROTOCOL_B3.md).

python fc_run_b3.py ridge --folds 2022,2023,2024 --sectors bank,healthcare
python fc_run_b3.py train --folds 2024 --sectors bank --threads 9
python fc_run_b3.py train --folds 2025 --sectors bank --threads 9 --panel-dir D:/paper/2/xiu/data2025/panels
python fc_run_b3.py ridge --folds 2025 --panel-dir D:/paper/2/xiu/data2025/panels

Training code, loss (MSE + ranking term, lambda = 1), architectures and seeds are those of Plan B2
(revision_b2/fc_run_b2.py), imported unchanged; only the output folder, the folds and the Ridge_R
targets differ.  Fold 2025 (training t+2 <= 2023-12-31, validation 2024, test 2025) needs the extended
panels written by b3_ingest.py; earlier folds use the original panels (identical rows through 2024).
Outputs: revision_b3/forecasts/runs/<sector>/<fold>/<model>/..., ensembles/<sector>_<fold>_<model>.npz
"""
import sys, time, argparse, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'revision_b2'))
sys.path.insert(0, str(ROOT / 'revision_v4' / 'forecaster'))
import numpy as np
import fc_data as D
import fc_run as V4
import fc_run_b2 as B2

OUT = ROOT / 'revision_b3' / 'forecasts'
B2.OUT = OUT                                   # every B2 writer resolves OUT at call time
MODELS = ['B2_GAT_Transformer', 'A1_Transformer_only', 'B1_GCN_Transformer', 'A10_No_cross_stock', 'A0_Full_MGT']
D.FOLDS[2025] = dict(train_end='2023-12-31', val=2024, test=2025)
PANELS = {'bank': 'bank_panel_20251231.csv', 'healthcare': 'healthcare_panel_20251231.csv'}


def use_extended(panel_dir):
    """Point the loader at the extended panels (graphs still come from the sector folders)."""
    for sec, f in PANELS.items():
        p = pathlib.Path(panel_dir).resolve() / f
        if not p.exists():
            raise FileNotFoundError(p)
        D.SECTORS[sec]['panel'] = str(p)        # absolute path: Path(root) / folder / abs -> abs


def ridge_b3(fd):
    """Ridge on the flattened 20-day window: q = c2 - c1 (penalty by validation rank IC of q) and the
    log range h1 - l1 (penalty by validation MSE).  Stored as six channels: c2 = q_hat, c1 = 0,
    h1 = range_hat, l1 = 0, others NaN."""
    base = OUT / 'runs' / fd.sector / str(fd.fold) / 'Ridge_R'
    if (base / 'pred.npz').exists():
        return
    t0 = time.perf_counter()
    Yraw = {k: fd.sc.y_inv(fd.y[k]) for k in ['train', 'val', 'test']}

    def flat(part, kind):
        x = fd.x[part]; n, P, N, F = x.shape
        if kind == 'q':
            y = Yraw[part][:, :, 5] - Yraw[part][:, :, 2]; m = fd.m[part][:, :, 2] & fd.m[part][:, :, 5]
        else:
            y = Yraw[part][:, :, 0] - Yraw[part][:, :, 1]; m = fd.m[part][:, :, 0] & fd.m[part][:, :, 1]
        return x.transpose(0, 2, 1, 3).reshape(n * N, P * F), y.reshape(n * N), m.reshape(n * N)

    nv, ns = len(fd.x['val']), len(fd.x['test'])
    fits = {}
    for kind in ['q', 'range']:
        Xt, yt, mt = flat('train', kind); Xt, yt = Xt[mt], yt[mt]
        xm, ym = Xt.mean(0), yt.mean()
        A = (Xt - xm).T @ (Xt - xm); b = (Xt - xm).T @ (yt - ym)
        Xv, yv, mv = flat('val', kind); best = None
        for a in V4.RIDGE_ALPHAS:
            w = np.linalg.solve(A + a * np.eye(A.shape[0]), b)
            pv = (Xv - xm) @ w + ym
            if kind == 'q':
                score = B2.spearman_ic(pv.reshape(nv, fd.N), yv.reshape(nv, fd.N), mv.reshape(nv, fd.N))
            else:
                score = -float(np.mean((pv[mv] - yv[mv]) ** 2))
            if best is None or (np.isfinite(score) and score > best[1]):
                best = (a, score, w)
        fits[kind] = (xm, ym) + best

    def six(part, n):
        Xp, _, _ = flat(part, 'q')
        F = np.full((n, fd.N, 6), np.nan)
        for kind, ch in [('q', 5), ('range', 0)]:
            xm, ym, a, s, w = fits[kind]
            F[:, :, ch] = ((Xp - xm) @ w + ym).reshape(n, fd.N)
        F[:, :, 2] = 0.0; F[:, :, 1] = 0.0
        return F
    base.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base / 'pred.npz', test=six('test', ns), val=six('val', nv), test_dates=fd.dates('test'),
                        val_dates=fd.dates('val'), tickers=np.array(fd.p['tickers']))
    V4.save_json(base / 'meta.json', dict(model='Ridge_R (B3)', targets=['q = log(C_t+2/C_t+1)', 'log(H_t+1/L_t+1)'],
                                          alpha_q=fits['q'][2], val_rank_ic_q=fits['q'][3],
                                          alpha_range=fits['range'][2], val_mse_range=-fits['range'][3],
                                          alphas=V4.RIDGE_ALPHAS, panel_sha256=fd.inputs_sha, seconds=time.perf_counter() - t0))
    B2.log_row(kind='ridge', sector=fd.sector, fold=fd.fold, model='Ridge_R', alpha=fits['q'][2], best_val=fits['q'][3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['train', 'ridge'])
    ap.add_argument('--folds', default='2024')
    ap.add_argument('--sectors', default='bank,healthcare')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--panel-dir', default='')
    a = ap.parse_args()
    sectors = a.sectors.split(','); folds = [int(x) for x in a.folds.split(',')]
    if 2025 in folds and not a.panel_dir:
        raise SystemExit('Fold 2025 needs --panel-dir (extended panels from b3_ingest.py)')
    if a.panel_dir:
        use_extended(a.panel_dir)
    B2.TAG = 'b3_%s_%s_%s' % (a.command, a.sectors.replace(',', '_'), a.folds.replace(',', '_'))
    if a.command == 'ridge':
        for fold in folds:
            for sec in sectors:
                ridge_b3(V4.Fold(sec, fold)); print(B2.now(), 'ridge done', sec, fold, flush=True)
        return
    V4.set_threads(a.threads or None)
    total = len(folds) * len(sectors) * len(MODELS) * len(B2.SEEDS); done = 0
    for fold in folds:
        for sec in sectors:
            fd = V4.Fold(sec, fold); ridge_b3(fd)
            for name in MODELS:
                for seed in B2.SEEDS:
                    B2.status(sector=sec, fold=fold, model=name + '_R', seed=seed, done=done, total=total)
                    m = B2.train_nn(fd, name, seed)
                    done += 1
                    print('%s done %d/%d %s %d %s_R s%d epochs=%s val=%.4f %.1fs' % (B2.now(), done, total, sec, fold, name, seed,
                          m.get('epochs_run'), m.get('best_val_loss', np.nan), m.get('train_seconds', 0)), flush=True)
                B2.ensemble(sec, fold, name + '_R')
    B2.status(finished=True, done=done, total=total, folds=folds, sectors=sectors)
    (OUT / ('DONE_%s' % B2.TAG)).write_text(B2.now(), encoding='utf-8')


if __name__ == '__main__':
    main()
