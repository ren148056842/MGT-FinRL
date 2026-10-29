"""v4 walk-forward forecaster runner.

python fc_run.py train  --stage rl     # fold 2022, models used by the RL experiments (writes merges)
python fc_run.py train  --stage all    # every fold, model, and seed (skips finished runs)
python fc_run.py merges                # (re)write the 2022 merges from the fold-2022 ensembles
python fc_run.py smoke                 # 2-epoch run of every model on fold 2022 (no outputs kept)

Outputs (D:/paper/2/xiu/revision_v4/forecasts):
  runs/<sector>/<fold>/<model>/seed<k>/{pred.npz, history.csv, meta.json, weights.h5}
  runs/<sector>/<fold>/{Ridge,Naive}/pred.npz
  ensembles/<sector>_<fold>_<model>.npz      (mean of seeds 1-3, log-ratio space)
  merges/<sector>_<model>_2022.csv           (2022 H/L/C replaced by one-step-ahead forecasts)
  status.json, run_log.csv
"""
import os, sys, json, time, argparse, hashlib, pathlib, datetime, platform
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import fc_data as D

OUT = D.ROOT / 'revision_v4' / 'forecasts'
SEEDS = [1, 2, 3]
EPOCHS, PATIENCE, BATCH, LR = 100, 10, 32, 1e-3
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def sha(f):
    h = hashlib.sha256()
    with open(f, 'rb') as s:
        for b in iter(lambda: s.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save_json(f, obj):
    f = pathlib.Path(f); f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=1, default=str), encoding='utf-8'); tmp.replace(f)


TAG = 'all'


def log_row(**kw):
    f = OUT / ('run_log_%s.csv' % TAG); f.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([dict(time=now(), **kw)])
    row.to_csv(f, mode='a', header=not f.exists(), index=False)


class Fold:
    """Arrays of one sector and fold, standardised with training statistics only."""
    def __init__(self, sector, fold):
        self.sector, self.fold = sector, fold
        p = D.load_panel(sector)
        self.p = p
        X = D.features(p); Y, M = D.targets(p)
        s = D.split_indices(p['dates'], fold)
        sc = D.Scaler(X, Y, M, s['train'])
        Z, Zy = sc.x(X), sc.y(Y)
        self.idx, self.sc = s, sc
        self.x = {k: D.windows(Z, v) for k, v in s.items()}
        self.y = {k: Zy[v] for k, v in s.items()}
        self.m = {k: M[v] for k, v in s.items()}
        self.graphs = p['graphs']
        self.N, self.F = Z.shape[1], Z.shape[2]
        self.inputs_sha = sha(p['panel_file'])

    def dates(self, part):
        return np.array([str(x.date()) for x in self.p['dates'][self.idx[part]]])


def set_threads(n=None):
    import tensorflow as tf
    n = n or max(1, (os.cpu_count() or 2) - 1)
    tf.config.threading.set_intra_op_parallelism_threads(n)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()


def train_nn(fd, name, seed, epochs=EPOCHS, keep=True):
    import tensorflow as tf
    from fc_models import Forecaster, CONFIGS, masked_mse
    out = OUT / 'runs' / fd.sector / str(fd.fold) / name / ('seed%d' % seed)
    if keep and (out / 'meta.json').exists():
        return json.loads((out / 'meta.json').read_text(encoding='utf-8'))
    tf.keras.utils.set_random_seed(seed)
    model = Forecaster(fd.N, fd.F, D.WINDOW, fd.graphs, CONFIGS[name])
    _ = model(tf.zeros((1, D.WINDOW, fd.N, fd.F)), training=False)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    Xtr, Ytr, Mtr = (tf.constant(fd.x['train']), tf.constant(fd.y['train']), tf.constant(fd.m['train']))

    @tf.function(input_signature=[tf.TensorSpec([None], tf.int32)])
    def step(ix):
        xb, yb, mb = tf.gather(Xtr, ix), tf.gather(Ytr, ix), tf.gather(Mtr, ix)
        with tf.GradientTape() as tape:
            loss = masked_mse(yb, model(xb, training=True), mb)
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients([(a, b) for a, b in zip(g, model.trainable_variables) if a is not None])
        return loss

    @tf.function(input_signature=[tf.TensorSpec([None, D.WINDOW, fd.N, fd.F], tf.float32)])
    def infer(x):
        return model(x, training=False)

    def predict(part):
        x = fd.x[part]
        return np.concatenate([infer(tf.constant(x[i:i + 256])).numpy() for i in range(0, len(x), 256)])

    def val_loss():
        return float(masked_mse(tf.constant(fd.y['val']), tf.constant(predict('val')), tf.constant(fd.m['val'])))

    rng = np.random.RandomState(seed)
    n = len(fd.x['train']); hist = []; best = np.inf; best_w = None; best_ep = 0; wait = 0
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n).astype(np.int32); losses = []
        for i in range(0, n, BATCH):
            losses.append(float(step(tf.constant(perm[i:i + BATCH]))))
        vl = val_loss(); hist.append(dict(epoch=ep, loss=float(np.mean(losses)), val_loss=vl))
        if vl < best - 1e-7:
            best, best_w, best_ep, wait = vl, [w.copy() for w in model.get_weights()], ep, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    train_s = time.perf_counter() - t0
    if best_w is None:
        raise RuntimeError('Validation loss never finite for %s seed %d' % (name, seed))
    model.set_weights(best_w)
    t1 = time.perf_counter(); zt = predict('test'); infer_s = time.perf_counter() - t1
    zv = predict('val')
    meta = dict(sector=fd.sector, fold=fd.fold, model=name, seed=seed, epochs_run=len(hist), best_epoch=best_ep,
                best_val_loss=best, train_seconds=train_s, intra_threads=tf.config.threading.get_intra_op_parallelism_threads(), test_inference_seconds=infer_s,
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
    meta['pred_sha256'] = sha(out / 'pred.npz')
    save_json(out / 'meta.json', meta)
    log_row(kind='nn', sector=fd.sector, fold=fd.fold, model=name, seed=seed, epochs=len(hist), best_epoch=best_ep,
            best_val=best, train_s=round(train_s, 2), infer_s=round(infer_s, 3), params=meta['params'])
    tf.keras.backend.clear_session()
    return meta


def ridge_and_naive(fd):
    base = OUT / 'runs' / fd.sector / str(fd.fold)
    if not (base / 'Naive' / 'pred.npz').exists():
        (base / 'Naive').mkdir(parents=True, exist_ok=True)
        t1 = time.perf_counter()
        nt, nv = D.naive_forecast(fd.p, fd.idx['test']), D.naive_forecast(fd.p, fd.idx['val'])
        np.savez_compressed(base / 'Naive' / 'pred.npz', test=nt, val=nv, test_dates=fd.dates('test'),
                            val_dates=fd.dates('val'), tickers=np.array(fd.p['tickers']))
        save_json(base / 'Naive' / 'meta.json', dict(model='Naive', rule='H,L,C of t+k forecast by H_t, L_t, C_t', seconds=time.perf_counter() - t1))
    if (base / 'Ridge' / 'pred.npz').exists():
        return
    def flat(part):
        x = fd.x[part]; n, P, N, F = x.shape
        return x.transpose(0, 2, 1, 3).reshape(n * N, P * F), fd.y[part].reshape(n * N, 6), fd.m[part].reshape(n * N, 6)
    t0 = time.perf_counter()
    Xt, Yt, Mt = flat('train'); ok = Mt.all(1); Xt, Yt = Xt[ok], Yt[ok]
    xm, ym = Xt.mean(0), Yt.mean(0)
    A = (Xt - xm).T @ (Xt - xm); B = (Xt - xm).T @ (Yt - ym)
    Xv, Yv, Mv = flat('val'); best = None
    for a in RIDGE_ALPHAS:
        W = np.linalg.solve(A + a * np.eye(A.shape[0]), B)
        pv = (Xv - xm) @ W + ym
        mse = float((Mv * (pv - Yv) ** 2).sum() / Mv.sum())
        if best is None or mse < best[1]:
            best = (a, mse, W)
    a, mse, W = best
    Xs, _, _ = flat('test')
    def pred(Xp, part):
        n = len(fd.x[part]); return fd.sc.y_inv(((Xp - xm) @ W + ym).reshape(n, fd.N, 6))
    (base / 'Ridge').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base / 'Ridge' / 'pred.npz', test=pred(Xs, 'test'), val=pred(Xv, 'val'), test_dates=fd.dates('test'),
                        val_dates=fd.dates('val'), tickers=np.array(fd.p['tickers']))
    save_json(base / 'Ridge' / 'meta.json', dict(model='Ridge', alpha=a, val_mse=mse, alphas=RIDGE_ALPHAS, seconds=time.perf_counter() - t0,
                                                 pooled_over_stocks=True, inputs='own 20-day window of the 8 features'))
    log_row(kind='ridge', sector=fd.sector, fold=fd.fold, model='Ridge', alpha=a, best_val=mse)


def ensemble(sector, fold, name):
    runs = [OUT / 'runs' / sector / str(fold) / name / ('seed%d' % s) / 'pred.npz' for s in SEEDS]
    if not all(f.exists() for f in runs):
        return None
    z = [np.load(f) for f in runs]
    f = OUT / 'ensembles' / ('%s_%d_%s.npz' % (sector, fold, name)); f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, test=np.mean([x['test'] for x in z], 0), val=np.mean([x['val'] for x in z], 0),
                        test_dates=z[0]['test_dates'], val_dates=z[0]['val_dates'], tickers=z[0]['tickers'],
                        seeds=np.array(SEEDS), sources=np.array([sha(r) for r in runs]))
    return f


def write_merge(sector, name):
    """2022 H/L/C replaced by the one-step-ahead forecast made at the previous trading day."""
    src = OUT / 'ensembles' / ('%s_2022_%s.npz' % (sector, name)) if name not in ['Ridge', 'Naive'] else OUT / 'runs' / sector / '2022' / name / 'pred.npz'
    if not src.exists():
        return None
    p = D.load_panel(sector); z = np.load(src)
    s = D.split_indices(p['dates'], 2022)
    if list(z['test_dates']) != [str(x.date()) for x in p['dates'][s['test']]]:
        raise ValueError('Forecast origins do not match')
    P = D.to_prices(p, s['test'], z['test'])               # [n, N, 6]
    tgt = s['test'] + 1                                     # forecast target rows (all of 2022)
    frame = p['frame'].copy()
    n = len(p['tickers'])
    H, L, C = p['H'].copy(), p['L'].copy(), p['C'].copy()
    keep = ~p['traded'][tgt]                                # suspended on the target day: observed price kept
    H[tgt] = np.where(keep, H[tgt], P[:, :, 0]); L[tgt] = np.where(keep, L[tgt], P[:, :, 1]); C[tgt] = np.where(keep, C[tgt], P[:, :, 2])
    frame['high'], frame['low'], frame['close'] = H.reshape(-1), L.reshape(-1), C.reshape(-1)
    frame['date'] = frame.date.dt.strftime('%Y-%m-%d')
    out = OUT / 'merges' / ('%s_%s_2022.csv' % (sector, name)); out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    changed = int((~keep).sum())
    save_json(out.with_suffix('.json'), dict(sector=sector, model=name, source=src.relative_to(OUT).as_posix(), source_sha256=sha(src),
                                             replaced_stock_days=changed, kept_suspended_stock_days=int(keep.sum()),
                                             first_target=str(p['dates'][tgt[0]].date()), last_target=str(p['dates'][tgt[-1]].date()),
                                             sha256=sha(out)))
    return out


def status(**kw):
    save_json(OUT / ('status_%s.json' % TAG), dict(time=now(), **kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['train', 'merges', 'smoke'])
    ap.add_argument('--stage', default='all', choices=['rl', 'all'])
    ap.add_argument('--sectors', default='bank,healthcare')
    ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args()
    global TAG
    TAG = a.sectors.replace(',', '_')
    from fc_models import CONFIGS, RL_VARIANTS
    sectors = a.sectors.split(',')
    if a.command == 'merges':
        for sec in sectors:
            for name in list(CONFIGS) + ['Ridge', 'Naive']:
                print('merge', sec, name, write_merge(sec, name), flush=True)
        return
    set_threads(a.threads or None)
    if a.command == 'smoke':
        for sec in sectors:
            fd = Fold(sec, 2022)
            for name in CONFIGS:
                m = train_nn(fd, name, 1, epochs=2, keep=False)
                print('SMOKE', sec, name, 'params', m['params'], 'val', round(m['best_val_loss'], 4), 'sec', round(m['train_seconds'], 1), flush=True)
        return
    plan = []
    folds = [2022] if a.stage == 'rl' else [2022, 2023, 2024]
    for fold in folds:
        for sec in sectors:
            names = RL_VARIANTS if a.stage == 'rl' else (RL_VARIANTS + [n for n in CONFIGS if n not in RL_VARIANTS])
            plan.append((sec, fold, names))
    total = sum(len(n) * len(SEEDS) for _, _, n in plan); done = 0
    for sec, fold, names in plan:
        fd = Fold(sec, fold); ridge_and_naive(fd)
        for name in names:
            for seed in SEEDS:
                status(stage=a.stage, sector=sec, fold=fold, model=name, seed=seed, done=done, total=total)
                m = train_nn(fd, name, seed)
                done += 1
                print('%s done %d/%d %s %d %s s%d epochs=%s val=%.4f %.1fs' % (now(), done, total, sec, fold, name, seed,
                      m.get('epochs_run'), m.get('best_val_loss', np.nan), m.get('train_seconds', 0)), flush=True)
            if not (OUT / 'ensembles' / ('%s_%d_%s.npz' % (sec, fold, name))).exists():
                ensemble(sec, fold, name)
            if fold == 2022 and not (OUT / 'merges' / ('%s_%s_2022.csv' % (sec, name))).exists():
                write_merge(sec, name)
        if fold == 2022:
            for name in ['Ridge', 'Naive']:
                if not (OUT / 'merges' / ('%s_%s_2022.csv' % (sec, name))).exists():
                    write_merge(sec, name)
    status(stage=a.stage, finished=True, done=done, total=total)
    if a.stage == 'rl':
        (OUT / ('RL_MERGES_READY_%s' % TAG)).write_text(now(), encoding='utf-8')
    else:
        (OUT / ('ALL_FORECASTS_DONE_%s' % TAG)).write_text(now(), encoding='utf-8')


if __name__ == '__main__':
    main()
