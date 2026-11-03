"""Plan B2, stage 1: ranking-aware walk-forward forecasters (see PROTOCOL.md).

python fc_run_b2.py smoke  --sectors bank            # 2 epochs, fold 2022, nothing kept
python fc_run_b2.py train  --folds 2022,2023 --sectors bank --threads 9
python fc_run_b2.py train  --folds 2024 ...           # only after the gate (PROTOCOL.md)

Identical to the v4 forecaster (revision_v4/forecaster) except for the loss:
    L = masked MSE (six standardised outputs) + LAMBDA * (1 - mean daily cross-sectional Pearson
        correlation between q_hat = c2_hat - c1_hat and q = c2 - c1, in log-return units).
Outputs: revision_b2/forecasts/runs/<sector>/<fold>/<model>_R/seed<k>/..., ensembles/<sector>_<fold>_<model>_R.npz
"""
import os, sys, json, time, argparse, pathlib, platform
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'revision_v4' / 'forecaster'))
import numpy as np
import pandas as pd
import fc_data as D
import fc_run as V4                                    # v4 runner: Fold, set_threads, sha, save_json, constants

OUT = ROOT / 'revision_b2' / 'forecasts'
LAMBDA = 1.0
MIN_STOCKS = 5
MODELS = ['A0_Full_MGT', 'A10_No_cross_stock', 'A1_Transformer_only', 'A2_MGNN_only', 'B1_GCN_Transformer', 'B2_GAT_Transformer']
SEEDS = V4.SEEDS
EPOCHS, PATIENCE, BATCH, LR = V4.EPOCHS, V4.PATIENCE, V4.BATCH, V4.LR
TAG = 'b2'


def now():
    return V4.now()


def log_row(**kw):
    f = OUT / ('run_log_%s.csv' % TAG); f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(time=now(), **kw)]).to_csv(f, mode='a', header=not f.exists(), index=False)


def make_losses(ymu, ysd):
    import tensorflow as tf
    from fc_models import masked_mse
    mu = tf.constant(np.asarray(ymu, np.float32)); sd = tf.constant(np.asarray(ysd, np.float32))

    def rank_term(y, yhat, m):
        q_hat = (yhat[..., 5] * sd[5] + mu[5]) - (yhat[..., 2] * sd[2] + mu[2])
        q = (y[..., 5] * sd[5] + mu[5]) - (y[..., 2] * sd[2] + mu[2])
        v = tf.cast(tf.logical_and(m[..., 2], m[..., 5]), tf.float32)          # [B, N]
        n = tf.reduce_sum(v, axis=1)                                            # [B]
        nn = tf.maximum(n, 1.0)
        dp = (q_hat - tf.reduce_sum(v * q_hat, 1, keepdims=True) / nn[:, None]) * v
        da = (q - tf.reduce_sum(v * q, 1, keepdims=True) / nn[:, None]) * v
        cov = tf.reduce_sum(dp * da, 1)
        corr = cov / tf.sqrt(tf.reduce_sum(dp * dp, 1) * tf.reduce_sum(da * da, 1) + 1e-12)
        ok = tf.cast(n >= MIN_STOCKS, tf.float32)
        mean_corr = tf.reduce_sum(ok * corr) / tf.maximum(tf.reduce_sum(ok), 1.0)
        return 1.0 - mean_corr

    def total(y, yhat, m):
        return masked_mse(y, yhat, m) + LAMBDA * rank_term(y, yhat, m)
    return total, rank_term


def train_nn(fd, name, seed, epochs=EPOCHS, keep=True):
    import tensorflow as tf
    from fc_models import Forecaster, CONFIGS
    rname = name + '_R'
    out = OUT / 'runs' / fd.sector / str(fd.fold) / rname / ('seed%d' % seed)
    if keep and (out / 'meta.json').exists():
        return json.loads((out / 'meta.json').read_text(encoding='utf-8'))
    tf.keras.utils.set_random_seed(seed)
    model = Forecaster(fd.N, fd.F, D.WINDOW, fd.graphs, CONFIGS[name])
    _ = model(tf.zeros((1, D.WINDOW, fd.N, fd.F)), training=False)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    loss_fn, rank_fn = make_losses(fd.sc.ymu, fd.sc.ysd)
    Xtr, Ytr, Mtr = tf.constant(fd.x['train']), tf.constant(fd.y['train']), tf.constant(fd.m['train'])

    @tf.function(input_signature=[tf.TensorSpec([None], tf.int32)])
    def step(ix):
        xb, yb, mb = tf.gather(Xtr, ix), tf.gather(Ytr, ix), tf.gather(Mtr, ix)
        with tf.GradientTape() as tape:
            loss = loss_fn(yb, model(xb, training=True), mb)
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients([(a, b) for a, b in zip(g, model.trainable_variables) if a is not None])
        return loss

    @tf.function(input_signature=[tf.TensorSpec([None, D.WINDOW, fd.N, fd.F], tf.float32)])
    def infer(x):
        return model(x, training=False)

    def predict(part):
        x = fd.x[part]
        return np.concatenate([infer(tf.constant(x[i:i + 256])).numpy() for i in range(0, len(x), 256)])

    def val_scores():
        yv, mv, pv = tf.constant(fd.y['val']), tf.constant(fd.m['val']), tf.constant(predict('val'))
        return float(loss_fn(yv, pv, mv)), float(1.0 - rank_fn(yv, pv, mv))

    rng = np.random.RandomState(seed)
    n = len(fd.x['train']); hist = []; best = np.inf; best_w = None; best_ep = 0; wait = 0
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n).astype(np.int32); losses = []
        for i in range(0, n, BATCH):
            losses.append(float(step(tf.constant(perm[i:i + BATCH]))))
        vl, vic = val_scores(); hist.append(dict(epoch=ep, loss=float(np.mean(losses)), val_loss=vl, val_pearson_q=vic))
        if np.isfinite(vl) and vl < best - 1e-7:
            best, best_w, best_ep, wait = vl, [w.copy() for w in model.get_weights()], ep, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    train_s = time.perf_counter() - t0
    if best_w is None:
        raise RuntimeError('Validation loss never finite for %s seed %d' % (rname, seed))
    model.set_weights(best_w)
    zt = predict('test'); zv = predict('val')
    meta = dict(sector=fd.sector, fold=fd.fold, model=rname, base_model=name, seed=seed, objective='mse+rank', lam=LAMBDA,
                epochs_run=len(hist), best_epoch=best_ep, best_val_loss=best, train_seconds=train_s,
                n_train=int(len(fd.x['train'])), n_val=int(len(fd.x['val'])), n_test=int(len(fd.x['test'])),
                params=int(np.sum([np.prod(v.shape) for v in model.trainable_variables])),
                window=D.WINDOW, batch=BATCH, lr=LR, patience=PATIENCE, max_epochs=epochs,
                panel_sha256=fd.inputs_sha, scaler=fd.sc.state(), finished=now(),
                tensorflow=tf.__version__, python=platform.python_version(), cpu_count=os.cpu_count())
    if hasattr(model, 'alpha'):
        meta['alpha_final'] = model.alpha.numpy().tolist()
    if not keep:
        tf.keras.backend.clear_session()
        return meta
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'pred.npz', test=fd.sc.y_inv(zt).astype(np.float64), val=fd.sc.y_inv(zv).astype(np.float64),
                        test_dates=fd.dates('test'), val_dates=fd.dates('val'), tickers=np.array(fd.p['tickers']))
    pd.DataFrame(hist).to_csv(out / 'history.csv', index=False)
    model.save_weights(str(out / 'weights.h5'))
    meta['pred_sha256'] = V4.sha(out / 'pred.npz')
    V4.save_json(out / 'meta.json', meta)
    log_row(kind='nn', sector=fd.sector, fold=fd.fold, model=rname, seed=seed, epochs=len(hist), best_epoch=best_ep,
            best_val=best, train_s=round(train_s, 2), params=meta['params'])
    tf.keras.backend.clear_session()
    return meta


def _rank(x):
    return pd.Series(x).rank(method='average').to_numpy()


def spearman_ic(P, A, valid):
    out = []
    for t in range(P.shape[0]):
        v = valid[t] & np.isfinite(P[t]) & np.isfinite(A[t])
        if v.sum() >= MIN_STOCKS and np.std(P[t][v]) > 0 and np.std(A[t][v]) > 0:
            out.append(np.corrcoef(_rank(P[t][v]), _rank(A[t][v]))[0, 1])
    return float(np.mean(out)) if out else np.nan


def ridge_r(fd):
    """Ridge regression of q = c2 - c1 on the own 20-day window; penalty by validation rank IC of q."""
    base = OUT / 'runs' / fd.sector / str(fd.fold) / 'Ridge_R'
    if (base / 'pred.npz').exists():
        return
    t0 = time.perf_counter()
    Yraw = {k: fd.sc.y_inv(fd.y[k]) for k in ['train', 'val', 'test']}

    def flat(part):
        x = fd.x[part]; n, P, N, F = x.shape
        q = (Yraw[part][:, :, 5] - Yraw[part][:, :, 2]).reshape(n * N)
        m = (fd.m[part][:, :, 2] & fd.m[part][:, :, 5]).reshape(n * N)
        return x.transpose(0, 2, 1, 3).reshape(n * N, P * F), q, m
    Xt, qt, mt = flat('train'); Xt, qt = Xt[mt], qt[mt]
    xm, qm = Xt.mean(0), qt.mean()
    A = (Xt - xm).T @ (Xt - xm); b = (Xt - xm).T @ (qt - qm)
    Xv, qv, mv = flat('val'); nv = len(fd.x['val'])
    best = None
    for a in V4.RIDGE_ALPHAS:
        w = np.linalg.solve(A + a * np.eye(A.shape[0]), b)
        pv = ((Xv - xm) @ w + qm).reshape(nv, fd.N)
        ic = spearman_ic(pv, qv.reshape(nv, fd.N), mv.reshape(nv, fd.N))
        if best is None or (np.isfinite(ic) and ic > best[1]):
            best = (a, ic, w)
    a, ic, w = best
    Xs, _, _ = flat('test'); ns = len(fd.x['test'])

    def six(Xp, n):                                   # store q_hat in the c2 channel, c1 = 0 (others NaN)
        q = ((Xp - xm) @ w + qm).reshape(n, fd.N)
        F = np.full((n, fd.N, 6), np.nan); F[:, :, 2] = 0.0; F[:, :, 5] = q
        return F
    base.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base / 'pred.npz', test=six(Xs, ns), val=six(Xv, nv), test_dates=fd.dates('test'),
                        val_dates=fd.dates('val'), tickers=np.array(fd.p['tickers']))
    V4.save_json(base / 'meta.json', dict(model='Ridge_R', target='q = log(C_t+2 / C_t+1)', alpha=a, val_rank_ic=ic,
                                          alphas=V4.RIDGE_ALPHAS, seconds=time.perf_counter() - t0))
    log_row(kind='ridge', sector=fd.sector, fold=fd.fold, model='Ridge_R', alpha=a, best_val=ic)


def ensemble(sector, fold, rname):
    runs = [OUT / 'runs' / sector / str(fold) / rname / ('seed%d' % s) / 'pred.npz' for s in SEEDS]
    if not all(f.exists() for f in runs):
        return None
    z = [np.load(f) for f in runs]
    f = OUT / 'ensembles' / ('%s_%d_%s.npz' % (sector, fold, rname)); f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, test=np.mean([x['test'] for x in z], 0), val=np.mean([x['val'] for x in z], 0),
                        test_dates=z[0]['test_dates'], val_dates=z[0]['val_dates'], tickers=z[0]['tickers'],
                        seeds=np.array(SEEDS), sources=np.array([V4.sha(r) for r in runs]))
    return f


def status(**kw):
    V4.save_json(OUT / ('status_%s.json' % TAG), dict(time=now(), **kw))


def main():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['train', 'smoke'])
    ap.add_argument('--folds', default='2022,2023')
    ap.add_argument('--sectors', default='bank,healthcare')
    ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args()
    sectors = a.sectors.split(','); folds = [int(x) for x in a.folds.split(',')]
    TAG = 'b2_%s_%s' % (a.sectors.replace(',', '_'), a.folds.replace(',', '_'))
    V4.set_threads(a.threads or None)
    if a.command == 'smoke':
        for sec in sectors:
            fd = V4.Fold(sec, 2022)
            for name in MODELS:
                m = train_nn(fd, name, 1, epochs=2, keep=False)
                print('SMOKE', sec, name, 'val', round(m['best_val_loss'], 4), 'sec', round(m['train_seconds'], 1), flush=True)
        return
    total = len(folds) * len(sectors) * len(MODELS) * len(SEEDS); done = 0
    for fold in folds:
        for sec in sectors:
            fd = V4.Fold(sec, fold); ridge_r(fd)
            for name in MODELS:
                for seed in SEEDS:
                    status(sector=sec, fold=fold, model=name + '_R', seed=seed, done=done, total=total)
                    m = train_nn(fd, name, seed)
                    done += 1
                    print('%s done %d/%d %s %d %s_R s%d epochs=%s val=%.4f %.1fs' % (now(), done, total, sec, fold, name, seed,
                          m.get('epochs_run'), m.get('best_val_loss', np.nan), m.get('train_seconds', 0)), flush=True)
                ensemble(sec, fold, name + '_R')
    status(finished=True, done=done, total=total, folds=folds, sectors=sectors)
    (OUT / ('DONE_%s' % TAG)).write_text(now(), encoding='utf-8')


if __name__ == '__main__':
    main()
