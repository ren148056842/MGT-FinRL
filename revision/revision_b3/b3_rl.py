"""Plan B3: RL agents with forecasts in the state; training 2022-2023, selection 2024, evaluation 2025.

python b3_rl.py prepare                                   # build and check the 2022-2024 episodes of every source
python b3_rl.py worker --id 1                             # claims and trains tasks until none is left
python b3_rl.py unclaim                                   # release claims of unfinished tasks (after a crash)
python b3_rl.py status
python b3_rl.py evaluate   --panel-dir D:/paper/2/xiu/data2025/panels    # 2025, once (PROTOCOL_B3.md)
python b3_rl.py references --panel-dir D:/paper/2/xiu/data2025/panels    # equal weight, buy-and-hold, top-3 rules

State = v4 state ((N+4) x N: 252-day covariance and MACD/RSI/CCI/DX from observed prices) plus two rows:
the cross-sectional z-score of q_hat (predicted log return from the close of t+1 to the close of t+2) and
the predicted log range h1_hat - l1_hat (standardised with 2022-2023 statistics).  Forecasts are made at
the close of the decision day by the forecasters of the fold whose test year contains the next trading
day.  Accounting, rewards and execution are those of v4 (observed unadjusted prices, next-day execution).
"""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
import sys, json, math, time, shutil, pathlib, argparse, traceback, datetime, hashlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'revision_backtest'))
import numpy as np
import pandas as pd
import v4_data
from v4_audit import PROJECT
from v4_data import sequence_states, episode
from v3_data import RealPortfolioEnv
from v2_data import export_trace
from backtest import FEES, validate_splits

RUN = HERE / 'rl'
B2F = PROJECT / 'revision_b2' / 'forecasts'
B3F = PROJECT / 'revision_b3' / 'forecasts'
P = json.loads((PROJECT / 'revision_backtest' / 'runs' / 'v4_forecast_train_only' / 'protocol_v4.json').read_text(encoding='utf-8'))
SPLITS = dict(train=['2022-01-01', '2023-12-31'], validation=['2024-01-01', '2024-12-31'], test=['2025-01-01', '2025-12-31'])
ENSEMBLE = {'MGAT_T': 'B2_GAT_Transformer_R', 'A1_R': 'A1_Transformer_only_R', 'GCN_R': 'B1_GCN_Transformer_R',
            'A10_R': 'A10_No_cross_stock_R', 'MGT_R': 'A0_Full_MGT_R'}
SOURCES = ['Naive', 'Ridge_R', 'A1_R', 'MGAT_T']
ABLATION = ['GCN_R', 'A10_R', 'MGT_R']
ALGOS = ['ppo', 'a2c', 'sac', 'td3', 'ddpg']
SEEDS = [1, 2, 3, 4, 5]
SECTORS = ['bank', 'healthcare']
FOLDS = [2022, 2023, 2024, 2025]
COST = {'ddpg': 5, 'sac': 4, 'td3': 3, 'ppo': 1, 'a2c': 1}          # relative run time, for scheduling only
PANELS = {'bank': 'bank_panel_20251231.csv', 'healthcare': 'healthcare_panel_20251231.csv'}


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
    t = f.with_suffix(f.suffix + '.tmp%d' % os.getpid()); t.write_text(json.dumps(obj, indent=1, default=str), encoding='utf-8'); t.replace(f)


def log(msg, **kw):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'records.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps(dict(time=now(), pid=os.getpid(), msg=msg, **kw), default=str) + '\n')
    print(now(), msg, kw if kw else '', flush=True)


def use_extended(panel_dir):
    paths = {s: pathlib.Path(panel_dir).resolve() / f for s, f in PANELS.items()}
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(p)
    v4_data.find_market = lambda sec: paths[sec]          # clean_history resolves find_market at call time


def canon(t, sec):
    return 'PAB' if (sec == 'bank' and t == 'PAD') else t


def forecast_file(sec, src, fold):
    if src == 'Ridge_R':
        return B3F / 'runs' / sec / str(fold) / 'Ridge_R' / 'pred.npz'
    base = B2F if fold in (2022, 2023) else B3F
    return base / 'ensembles' / ('%s_%d_%s.npz' % (sec, fold, ENSEMBLE[src]))


def forecast_rows(sec, src, tickers, raw):
    """origin date -> (q_hat, range_hat) in environment ticker order (out-of-sample test sets of the folds)."""
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
        f = forecast_file(sec, src, fold)
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        tick = [canon(str(t), sec) for t in z['tickers']]
        order = [tick.index(t) for t in tickers]
        q = (z['test'][:, :, 5] - z['test'][:, :, 2])[:, order]
        r = (z['test'][:, :, 0] - z['test'][:, :, 1])[:, order]
        for k, d in enumerate(z['test_dates']):
            if str(d) in rows:
                raise ValueError('Origin %s appears in two folds' % d)
            rows[str(d)] = (q[k], r[k])
        used.append('%s sha256=%s' % (f.relative_to(PROJECT), sha(f)))
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
    return np.concatenate([states, zq[:, None, :], zr[:, None, :]], axis=1).astype(np.float32), have, Q


_CACHE = {}


def build(sec, src, need_test=False):
    key = (sec, src, need_test)
    if key in _CACHE:
        return _CACHE[key]
    raw = v4_data.clean_history(sec)
    dates, states, tickers = sequence_states(raw)
    rows, used = forecast_rows(sec, src, tickers, raw)
    train_mask = (dates >= pd.Timestamp(SPLITS['train'][0])) & (dates <= pd.Timestamp(SPLITS['train'][1]))
    aug, have, Q = augment(dates, states, rows, train_mask)
    end = SPLITS['test'][1] if need_test else SPLITS['validation'][1]
    span = (dates >= pd.Timestamp(SPLITS['train'][0])) & (dates <= pd.Timestamp(end))
    missing = [str(d.date()) for d in dates[span & ~have]]
    if len(missing) > 1 or (missing and missing[0] != str(dates[span][-1].date())):
        raise ValueError('%s %s: forecasts missing for %s' % (sec, src, missing[:5]))
    names = ['train', 'validation', 'test'] if need_test else ['train', 'validation']
    parts = {k: episode(raw, dates, aug, tickers, *SPLITS[k]) for k in names}
    if need_test and parts['test'].dates[-1] < pd.Timestamp('2025-12-01'):
        raise ValueError('Test episode ends at %s' % parts['test'].dates[-1].date())
    meta = dict(sector=sec, source=src, forecast_files=used, state_shape=list(aug.shape[1:]), tickers=list(tickers),
                missing_last_day=missing, panel=str(v4_data.find_market(sec)), panel_sha256=sha(v4_data.find_market(sec)))
    qd = {str(d.date()): Q[i] for i, d in enumerate(dates) if have[i]}
    _CACHE[key] = (parts, meta, qd)
    return _CACHE[key]


def task_name(sec, src, algo, seed):
    return 'B3_%s_%s_%s_s%d' % (sec, src, algo, seed)


def all_tasks():
    t = [(sec, src, algo, seed) for sec in SECTORS for src in SOURCES for algo in ALGOS for seed in SEEDS]
    t += [(sec, src, 'ppo', seed) for sec in SECTORS for src in ABLATION for seed in SEEDS]
    return sorted(t, key=lambda x: (-COST[x[2]], x[3], x[0], x[1]))


def rollout(model, data, fee):
    env = RealPortfolioEnv(data, FEES[fee]); obs, _ = env.reset(); done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True); obs, _, done, _, _ = env.step(a)
    return env


def train_one(sec, src, algo, seed):
    import torch
    torch.set_num_threads(1)
    from v2_engine import algorithms, stats_from_env
    task = task_name(sec, src, algo, seed)
    out = RUN / 'models' / task
    if (out / 'done.json').exists():
        return
    parts, meta, _ = build(sec, src)
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
        vm = stats_from_env(rollout(model, va, 'F1')); score = vm['SR'] if np.isfinite(vm['SR']) else -np.inf
        vals.append(dict(steps=int(model.num_timesteps), validation_SR=vm['SR'], validation_AR=vm['AR'], validation_MDD=vm['MDD'], checkpoint=ck.name))
        if best_path is None or score > best:
            best, best_path = score, ck
    shutil.copy2(str(best_path), str(out / 'best_policy.zip'))
    pd.DataFrame(vals).to_csv(out / 'validation.csv', index=False)
    best_model = cls.load(str(out / 'best_policy.zip'), device='cpu')
    venv = rollout(best_model, va, 'F1')
    info = dict(run_id=task + '_F1', sector=sec, source=src, algorithm=algo, seed=seed, fee='F1', period='2024')
    export_trace(venv, RUN / 'validation' / (task + '_F1'), info)
    save_json(RUN / 'validation' / (task + '_F1') / 'metrics.json', {**stats_from_env(venv), **info})
    save_json(out / 'done.json', dict(task=task, sector=sec, source=src, algorithm=algo, seed=seed, selected=best_path.name,
                                      best_validation_SR=best, train_seconds=time.perf_counter() - t0, budget=target,
                                      policy_sha256=sha(out / 'best_policy.zip'), finished=now(), **meta))
    log('trained', task=task, seconds=round(time.perf_counter() - t0, 1), val_SR=round(float(best), 4))


def smoke():
    """Short PPO, SAC and DDPG runs on one source (outputs under rl/smoke, not used anywhere)."""
    import torch
    torch.set_num_threads(1)
    from v2_engine import algorithms, stats_from_env
    parts, meta, _ = build('bank', 'MGAT_T')
    for algo, steps in [('ppo', 2048), ('sac', 300), ('ddpg', 300)]:
        t0 = time.perf_counter()
        params = P['model_params'][algo].copy(); params.update(seed=1, device='cpu', verbose=0)
        if algo == 'ppo':
            params['n_steps'] = 512
        model = algorithms()[algo]('MlpPolicy', RealPortfolioEnv(parts['train'], FEES['F1']), **params)
        model.learn(total_timesteps=steps, progress_bar=False)
        ck = RUN / 'smoke' / ('%s.zip' % algo); ck.parent.mkdir(parents=True, exist_ok=True); model.save(str(ck))
        m = algorithms()[algo].load(str(ck), device='cpu')
        vm = stats_from_env(rollout(m, parts['validation'], 'F1'))
        log('smoke', algorithm=algo, steps=steps, seconds=round(time.perf_counter() - t0, 1), val_SR=vm['SR'],
            state_shape=meta['state_shape'])


def claim(task):
    d = RUN / 'claims'; d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(d / (task + '.claim')), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, ('%d %s' % (os.getpid(), now())).encode()); os.close(fd)
    return True


def worker(wid):
    n = 0
    for sec, src, algo, seed in all_tasks():
        task = task_name(sec, src, algo, seed)
        if (RUN / 'models' / task / 'done.json').exists() or not claim(task):
            continue
        log('start', task=task, worker=wid)
        try:
            train_one(sec, src, algo, seed); n += 1
        except Exception as e:
            d = RUN / 'models' / task; d.mkdir(parents=True, exist_ok=True)
            (d / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
            log('failed', task=task, error=repr(e))
    log('worker finished', worker=wid, trained=n)


def unclaim():
    k = 0
    for c in (RUN / 'claims').glob('*.claim'):
        if not (RUN / 'models' / c.stem / 'done.json').exists():
            c.unlink(); k += 1
    log('released claims', n=k)


def status():
    tasks = all_tasks(); done = [t for t in tasks if (RUN / 'models' / task_name(*t) / 'done.json').exists()]
    failed = [task_name(*t) for t in tasks if (RUN / 'models' / task_name(*t) / 'failure.log').exists() and t not in done]
    s = dict(time=now(), total=len(tasks), done=len(done), failed=failed)
    save_json(RUN / 'status.json', s); print(json.dumps(s, indent=1))


def evaluate():
    from v2_engine import algorithms, stats_from_env
    tasks = all_tasks()
    for sec in SECTORS:
        for src in SOURCES + ABLATION:
            mine = [t for t in tasks if t[0] == sec and t[1] == src]
            parts, meta, _ = build(sec, src, need_test=True)
            te = parts['test']
            save_json(RUN / 'evaluation_inputs' / ('%s_%s.json' % (sec, src)), dict(meta, test_first=str(te.dates[0].date()),
                      test_last=str(te.dates[-1].date()), test_days=len(te.dates)))
            for _, _, algo, seed in mine:
                task = task_name(sec, src, algo, seed); m = RUN / 'models' / task
                if not (m / 'done.json').exists():
                    log('missing model', task=task); continue
                model = algorithms()[algo].load(str(m / 'best_policy.zip'), device='cpu')
                for fee in FEES:
                    out = RUN / 'results' / (task + '_' + fee)
                    if (out / 'metrics.json').exists():
                        continue
                    env = rollout(model, te, fee)
                    info = dict(run_id=out.name, sector=sec, source=src, algorithm=algo, seed=seed, fee=fee, period='2025')
                    export_trace(env, out, info)
                    save_json(out / 'metrics.json', {**stats_from_env(env), **info})
                log('evaluated', task=task)


def references():
    """Equal-weight daily, equal-weight buy-and-hold and top-3 forecast-ranking rules, 2024 and 2025, all costs."""
    from v2_engine import stats_from_env
    rule_sources = ['MGAT_T', 'A1_R', 'GCN_R', 'A10_R', 'MGT_R', 'Ridge_R']
    for sec in SECTORS:
        for period, part in [('2024', 'validation'), ('2025', 'test')]:
            data = build(sec, 'Naive', need_test=True)[0][part]
            strategies = {'equal_daily': None, 'buy_hold': None}
            for src in rule_sources:
                strategies['Forecast_' + src] = build(sec, src, need_test=True)[2]
            for name, qd in strategies.items():
                for fee in FEES:
                    out = RUN / 'references' / ('%s_%s_%s_%s' % (sec, name, period, fee))
                    if (out / 'metrics.json').exists():
                        continue
                    env = RealPortfolioEnv(data, FEES[fee]); env.reset(); done = False
                    while not done:
                        N = len(data.tickers)
                        if name == 'equal_daily' or (name == 'buy_hold' and env.i == 0):
                            w = np.ones(N) / N
                        elif name == 'buy_hold':
                            w = None
                        else:
                            d = str(data.dates[env.i].date())
                            if d not in qd:
                                raise ValueError('No forecast made at %s (%s %s)' % (d, sec, name))
                            ix = np.argsort(-qd[d], kind='mergesort')[:3]; w = np.zeros(N); w[ix] = 1 / 3
                        _, _, done, _, _ = env.advance(w)
                    info = dict(run_id=out.name, sector=sec, strategy=name, period=period, fee=fee)
                    export_trace(env, out, info)
                    save_json(out / 'metrics.json', {**stats_from_env(env), **info})
            log('references', sector=sec, period=period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'smoke', 'worker', 'unclaim', 'status', 'evaluate', 'references'])
    ap.add_argument('--id', type=int, default=0)
    ap.add_argument('--panel-dir', default='')
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if a.command in ('evaluate', 'references'):
        if not a.panel_dir:
            raise SystemExit('2025 evaluation needs --panel-dir (extended panels from b3_ingest.py)')
        use_extended(a.panel_dir)
    if a.command == 'prepare':
        rows = []
        for sec in SECTORS:
            for src in SOURCES + ABLATION:
                parts, meta, _ = build(sec, src)
                rows.append(dict(meta, train_days=len(parts['train'].dates), validation_days=len(parts['validation'].dates),
                                 train_first=str(parts['train'].dates[0].date()), validation_last=str(parts['validation'].dates[-1].date())))
        save_json(RUN / 'prepared.json', rows); log('prepared', n=len(rows), tasks=len(all_tasks()))
    elif a.command == 'smoke':
        smoke()
    elif a.command == 'worker':
        worker(a.id)
    elif a.command == 'unclaim':
        unclaim()
    elif a.command == 'status':
        status()
    elif a.command == 'evaluate':
        evaluate()
    else:
        references()


if __name__ == '__main__':
    main()
