"""Plan B2, stage 2: RL agents with forecasts in the state (revision_b2/PROTOCOL.md).

python b2_rl.py prepare                                  # training/validation episodes (2022, 2023)
python b2_rl.py train --sector bank --algorithm ppo      # 4 forecast sources x seeds 1-10
python b2_rl.py evaluate                                 # 2024, once, after the fold-2024 forecasts exist

State = v4 state ((N+4) x N: 252-day covariance and MACD/RSI/CCI/DX from observed prices) plus two rows:
cross-sectional z-score of q_hat (predicted log return from the close of t+1 to the close of t+2) and the
predicted log range h1_hat - l1_hat (standardised with 2022 statistics). Forecasts are made at the close of
the decision day by the forecasters of the fold whose test year contains the next trading day (out of sample).
Accounting, rewards and execution are those of v4 (observed prices, next-day execution, F0-F6 costs).
"""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
import sys, json, math, time, shutil, pathlib, argparse, traceback, datetime, hashlib
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from v4_audit import PROJECT
from v4_data import clean_history, sequence_states, episode
from v3_data import RealPortfolioEnv
from v2_data import export_trace
from v2_engine import algorithms, stats_from_env
from backtest import FEES, validate_splits

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE / 'runs' / 'b2_forecast_state'
B2F = PROJECT / 'revision_b2' / 'forecasts'
V4F = PROJECT / 'revision_v4' / 'forecasts'
P = json.loads((HERE / 'runs' / 'v4_forecast_train_only' / 'protocol_v4.json').read_text(encoding='utf-8'))
SPLITS = P['splits']
SOURCES = ['Naive', 'Ridge_R', 'A10_R', 'MGT_R']
ENSEMBLE = {'MGT_R': 'A0_Full_MGT_R', 'A10_R': 'A10_No_cross_stock_R'}
ALGOS = ['ppo', 'sac']
SEEDS = list(range(1, 11))
FOLDS = [2022, 2023, 2024]


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
    t = f.with_suffix(f.suffix + '.tmp'); t.write_text(json.dumps(obj, indent=1, default=str), encoding='utf-8'); t.replace(f)


def log(msg, **kw):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'records.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps(dict(time=now(), msg=msg, **kw), default=str) + '\n')
    print(now(), msg, kw if kw else '', flush=True)


def canon(t, sec):
    return 'PAB' if (sec == 'bank' and t == 'PAD') else t


def forecast_rows(sec, src, tickers, raw):
    """origin date -> (q_hat, range_hat) in environment ticker order, from the out-of-sample test sets of the folds."""
    rows = {}
    if src == 'Naive':
        h = raw.pivot(index='date', columns='tic', values='high').reindex(columns=tickers)
        l = raw.pivot(index='date', columns='tic', values='low').reindex(columns=tickers)
        rg = np.log(h.to_numpy(float) / l.to_numpy(float))
        for d, r in zip(h.index, rg):
            rows[str(d.date())] = (np.zeros(len(tickers)), r)
        return rows, ['observed log(H_t/L_t)']
    used = []
    for fold in FOLDS:
        if src == 'Ridge_R':
            fq = B2F / 'runs' / sec / str(fold) / 'Ridge_R' / 'pred.npz'
            fr = V4F / 'runs' / sec / str(fold) / 'Ridge' / 'pred.npz'
        else:
            fq = fr = B2F / 'ensembles' / ('%s_%d_%s.npz' % (sec, fold, ENSEMBLE[src]))
        if not (fq.exists() and fr.exists()):
            continue
        zq, zr = np.load(fq, allow_pickle=False), np.load(fr, allow_pickle=False)
        if list(zq['test_dates']) != list(zr['test_dates']):
            raise ValueError('Forecast dates differ between return and range sources')
        tick = [canon(str(t), sec) for t in zq['tickers']]
        order = [tick.index(t) for t in tickers]
        q = (zq['test'][:, :, 5] - zq['test'][:, :, 2])[:, order]
        r = (zr['test'][:, :, 0] - zr['test'][:, :, 1])[:, order]
        for k, d in enumerate(zq['test_dates']):
            if str(d) in rows:
                raise ValueError('Origin %s appears in two folds' % d)
            rows[str(d)] = (q[k], r[k])
        used += [str(fq.relative_to(PROJECT)), str(fr.relative_to(PROJECT))]
    return rows, used


def augment(dates, states, rows, train_mask):
    T, _, N = states.shape
    Q = np.full((T, N), np.nan); R = np.full((T, N), np.nan); have = np.zeros(T, bool)
    for i, d in enumerate(dates):
        k = str(d.date())
        if k in rows:
            Q[i], R[i] = rows[k]; have[i] = True
    zq = np.zeros((T, N))
    sd = np.nanstd(Q, axis=1)
    ok = have & (sd > 0)
    zq[ok] = (Q[ok] - np.nanmean(Q[ok], axis=1, keepdims=True)) / sd[ok][:, None]
    mu, s = np.nanmean(R[train_mask & have]), np.nanstd(R[train_mask & have])
    zr = np.where(np.isfinite(R), (R - mu) / s, 0.0)
    zq = np.nan_to_num(zq); zr = np.nan_to_num(zr)
    return np.concatenate([states, zq[:, None, :], zr[:, None, :]], axis=1).astype(np.float32), have


def build(sec, src, need_test=False):
    raw = clean_history(sec)
    dates, states, tickers = sequence_states(raw)
    rows, used = forecast_rows(sec, src, tickers, raw)
    train_mask = (dates >= pd.Timestamp(SPLITS['train'][0])) & (dates <= pd.Timestamp(SPLITS['train'][1]))
    aug, have = augment(dates, states, rows, train_mask)
    end = SPLITS['test'][1] if need_test else SPLITS['validation'][1]
    span = (dates >= pd.Timestamp(SPLITS['train'][0])) & (dates <= pd.Timestamp(end))
    missing = [str(d.date()) for d in dates[span & ~have]]
    if len(missing) > 1 or (missing and missing[0] != str(dates[span][-1].date())):
        raise ValueError('%s %s: forecasts missing for %s' % (sec, src, missing[:5]))
    parts = {k: episode(raw, dates, aug, tickers, *SPLITS[k]) for k in (['train', 'validation', 'test'] if need_test else ['train', 'validation'])}
    return parts, dict(sector=sec, source=src, forecast_files=used, state_shape=list(aug.shape[1:]), tickers=list(tickers),
                       missing_last_day=missing)


def train_one(sec, src, algo, seed):
    task = 'B2_%s_%s_%s_s%d' % (sec, src, algo, seed)
    out = RUN / 'models' / task
    if (out / 'done.json').exists():
        return
    parts, meta = build(sec, src)
    tr, va = parts['train'], parts['validation']
    validate_splits(tr.dates, va.dates, pd.DatetimeIndex([pd.Timestamp(SPLITS['test'][0])]))
    cls = algorithms()[algo]
    env = RealPortfolioEnv(tr, FEES['F1'])
    params = P['model_params'][algo].copy(); params.update(seed=int(seed), device='cpu', verbose=0)
    model = cls('MlpPolicy', env, **params)
    nsteps = int(getattr(model, 'n_steps', 1)); target = int(P['steps'][algo])
    target = int(math.ceil(target / nsteps) * nsteps) if algo in ['a2c', 'ppo'] else target
    cps = sorted(set(int(math.ceil(target * f / nsteps) * nsteps) if algo in ['a2c', 'ppo'] else int(math.ceil(target * f)) for f in [.2, .4, .6, .8, 1.]))
    out.mkdir(parents=True, exist_ok=True); best, best_path, vals = -np.inf, None, []; t0 = time.perf_counter()
    for step in cps:
        model.learn(total_timesteps=step - model.num_timesteps, reset_num_timesteps=False, progress_bar=False)
        ck = out / ('checkpoint_%06d.zip' % model.num_timesteps); model.save(str(ck))
        venv = RealPortfolioEnv(va, FEES['F1']); obs, _ = venv.reset(); done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True); obs, _, done, _, _ = venv.step(a)
        vm = stats_from_env(venv); score = vm['SR'] if np.isfinite(vm['SR']) else -np.inf
        vals.append(dict(steps=int(model.num_timesteps), validation_SR=vm['SR'], validation_AR=vm['AR'], validation_MDD=vm['MDD'], checkpoint=ck.name))
        if best_path is None or score > best:
            best, best_path = score, ck
    shutil.copy2(str(best_path), str(out / 'best_policy.zip'))
    pd.DataFrame(vals).to_csv(out / 'validation.csv', index=False)
    save_json(out / 'done.json', dict(task=task, sector=sec, source=src, algorithm=algo, seed=seed, selected=best_path.name,
                                      best_validation_SR=best, train_seconds=time.perf_counter() - t0, budget=target,
                                      policy_sha256=sha(out / 'best_policy.zip'), finished=now(), **meta))
    log('trained', task=task, seconds=round(time.perf_counter() - t0, 1), val_SR=round(float(best), 4))


def train(sector, algorithm, sources=None):
    for src in (sources or SOURCES):
        for seed in SEEDS:
            try:
                train_one(sector, src, algorithm, seed)
            except Exception as e:
                d = RUN / 'models' / ('B2_%s_%s_%s_s%d' % (sector, src, algorithm, seed)); d.mkdir(parents=True, exist_ok=True)
                (d / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8'); log('failed', sector=sector, source=src, algorithm=algorithm, seed=seed, error=repr(e))


def evaluate(sectors):
    for sec in sectors:
        for src in SOURCES:
            parts, meta = build(sec, src, need_test=True)
            te = parts['test']
            for algo in ALGOS:
                model_cls = algorithms()[algo]
                for seed in SEEDS:
                    task = 'B2_%s_%s_%s_s%d' % (sec, src, algo, seed)
                    m = RUN / 'models' / task
                    if not (m / 'done.json').exists():
                        log('missing model', task=task); continue
                    model = model_cls.load(str(m / 'best_policy.zip'), device='cpu')
                    for fee in FEES:
                        out = RUN / 'results' / (task + '_' + fee)
                        if (out / 'metrics.json').exists():
                            continue
                        env = RealPortfolioEnv(te, FEES[fee]); obs, _ = env.reset(); done = False
                        while not done:
                            a, _ = model.predict(obs, deterministic=True); obs, _, done, _, _ = env.step(a)
                        info = dict(run_id=out.name, sector=sec, source=src, algorithm=algo, seed=seed, fee=fee)
                        export_trace(env, out, info)
                        save_json(out / 'metrics.json', {**stats_from_env(env), **info})
                    log('evaluated', task=task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'train', 'evaluate'])
    ap.add_argument('--sector', choices=['bank', 'healthcare'])
    ap.add_argument('--sectors', default='bank,healthcare')
    ap.add_argument('--algorithm', choices=ALGOS)
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if a.command == 'prepare':
        rows = []
        for sec in a.sectors.split(','):
            for src in SOURCES:
                parts, meta = build(sec, src)
                rows.append(dict(meta, train_days=len(parts['train'].dates), validation_days=len(parts['validation'].dates)))
        save_json(RUN / 'prepared.json', rows); log('prepared', n=len(rows))
    elif a.command == 'train':
        train(a.sector, a.algorithm)
    else:
        evaluate(a.sectors.split(','))


if __name__ == '__main__':
    main()
