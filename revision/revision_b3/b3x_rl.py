"""Plan B3, post-hoc addendum Y (PROTOCOL_B3.md): sensitivity of the PPO comparison to the learning rate and
the discount factor.  Descriptive; the registered agents, selection and tests are unchanged.

python b3x_rl.py worker  --id 1 --of 8 --panel-dir D:/paper/2/xiu/data2025/panels
python b3x_rl.py analyze --panel-dir ... --out results_x

PPO agents with the MGAT and naive forecast sources, seeds 1-5, both sectors, trained exactly as the registered
PPO agents (b3_rl.train_one: training 2022-2023, cost setting F1, checkpoints at 20-100% of the budget,
selection by the 2024 Sharpe ratio), with one setting changed: learning rate 3e-4 or 3e-3 (registered 1e-3),
or discount factor 0.9 or 0.999 (registered 0.99).  The selected policies are evaluated on 2025 under F1.
Outputs: rl_x/{models,validation,results}, <out>/rl_sensitivity.csv
"""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
import sys, json, math, time, shutil, pathlib, argparse
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import b3_rl as R
from v3_data import RealPortfolioEnv
from v2_data import export_trace
from backtest import FEES, validate_splits

RUNX = HERE / 'rl_x'
VARIANTS = {'lr3e-4': dict(learning_rate=3e-4), 'lr3e-3': dict(learning_rate=3e-3), 'gamma0.9': dict(gamma=0.9), 'gamma0.999': dict(gamma=0.999)}
SOURCES = ['MGAT_T', 'Naive']


def tasks():
    return [(sec, src, var, seed) for var in VARIANTS for sec in R.SECTORS for src in SOURCES for seed in R.SEEDS]


def name(sec, src, var, seed):
    return 'X_%s_%s_ppo_%s_s%d' % (sec, src, var, seed)


def train_eval(sec, src, var, seed):
    import torch
    torch.set_num_threads(1)
    from v2_engine import algorithms, stats_from_env
    task = name(sec, src, var, seed); out = RUNX / 'models' / task; res = RUNX / 'results' / (task + '_F1')
    if (res / 'metrics.json').exists():
        return
    parts, meta, _ = R.build(sec, src, need_test=True)
    tr, va, te = parts['train'], parts['validation'], parts['test']
    validate_splits(tr.dates, va.dates, te.dates)
    cls = algorithms()['ppo']
    env = RealPortfolioEnv(tr, FEES['F1'])
    params = R.P['model_params']['ppo'].copy(); params.update(VARIANTS[var]); params.update(seed=int(seed), device='cpu', verbose=0)
    model = cls('MlpPolicy', env, **params)
    nsteps = int(model.n_steps); target = int(math.ceil(int(R.P['steps']['ppo']) / nsteps) * nsteps)
    cps = sorted(set(int(math.ceil(target * f / nsteps) * nsteps) for f in [.2, .4, .6, .8, 1.]))
    out.mkdir(parents=True, exist_ok=True); best, best_path, vals = -np.inf, None, []; t0 = time.perf_counter()
    for step in cps:
        model.learn(total_timesteps=step - model.num_timesteps, reset_num_timesteps=False, progress_bar=False)
        ck = out / ('checkpoint_%06d.zip' % model.num_timesteps); model.save(str(ck))
        vm = stats_from_env(R.rollout(model, va, 'F1')); score = vm['SR'] if np.isfinite(vm['SR']) else -np.inf
        vals.append(dict(steps=int(model.num_timesteps), validation_SR=vm['SR'], checkpoint=ck.name))
        if best_path is None or score > best:
            best, best_path = score, ck
    shutil.copy2(str(best_path), str(out / 'best_policy.zip'))
    pd.DataFrame(vals).to_csv(out / 'validation.csv', index=False)
    bm = cls.load(str(out / 'best_policy.zip'), device='cpu')
    env = R.rollout(bm, te, 'F1')
    info = dict(run_id=task + '_F1', sector=sec, source=src, algorithm='ppo', variant=var, seed=seed, fee='F1', period='2025',
                params={k: v for k, v in params.items() if k != 'device'}, selected=best_path.name, best_validation_SR=best,
                train_seconds=time.perf_counter() - t0)
    export_trace(env, res, {k: v for k, v in info.items() if k != 'params'})
    R.save_json(res / 'metrics.json', {**stats_from_env(env), **info})
    R.log('x trained+evaluated', task=task, seconds=round(time.perf_counter() - t0, 1))


def worker(i, of):
    mine = [t for k, t in enumerate(tasks()) if k % of == i - 1]
    for t in mine:
        train_eval(*t)


def sr(R_):
    sd = R_.std(axis=0, ddof=1)
    return np.where(sd > 0, np.sqrt(252) * R_.mean(axis=0) / np.where(sd > 0, sd, 1), np.nan)


def block_index(n, block, rng):
    st = rng.integers(0, n, size=int(np.ceil(n / block)))
    return ((st[:, None] + np.arange(block)) % n).ravel()[:n]


def analyze(out):
    def ret(d):
        t = pd.read_csv(d / 'trajectory.csv'); return t.set_index('date').daily_return.iloc[1:]
    rows = []
    for sec in R.SECTORS:
        for var in ['registered'] + list(VARIANTS):
            pan = {}
            for src in SOURCES:
                if var == 'registered':
                    ds = [R.RUN / 'results' / ('B3_%s_%s_ppo_s%d_F1' % (sec, src, s)) for s in R.SEEDS]
                else:
                    ds = [RUNX / 'results' / (name(sec, src, var, s) + '_F1') for s in R.SEEDS]
                if not all((d / 'trajectory.csv').exists() for d in ds):
                    break
                pan[src] = pd.concat([ret(d) for d in ds], axis=1).to_numpy(float)
            if len(pan) < 2:
                continue
            G, B = pan['MGAT_T'], pan['Naive']; rng = np.random.default_rng(1729); n = len(G)
            d = float(np.nanmean(sr(G) - sr(B))); bs = np.empty(2000)
            for b in range(2000):
                ix = block_index(n, 10, rng); bs[b] = np.nanmean(sr(G[ix]) - sr(B[ix]))
            rows.append(dict(sector=sec, setting=var, MGAT_SR=float(np.mean(sr(G))), Naive_SR=float(np.mean(sr(B))),
                             MGAT_SR_sd=float(np.std(sr(G), ddof=1)), Naive_SR_sd=float(np.std(sr(B), ddof=1)), dSR=d,
                             ci90_low=float(np.percentile(bs, 5)), ci90_high=float(np.percentile(bs, 95)),
                             p_one_sided=float(np.mean((bs - d) >= d)), pairs_MGAT_higher=int((sr(G) > sr(B)).sum())))
    pd.DataFrame(rows).to_csv(out / 'rl_sensitivity.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['worker', 'analyze'])
    ap.add_argument('--id', type=int, default=1)
    ap.add_argument('--of', type=int, default=1)
    ap.add_argument('--panel-dir', required=True)
    ap.add_argument('--out', default='results_x')
    a = ap.parse_args()
    R.use_extended(a.panel_dir)
    out = HERE / a.out; out.mkdir(parents=True, exist_ok=True)
    if a.command == 'worker':
        worker(a.id, a.of)
    else:
        analyze(out)
    print('DONE', a.command, flush=True)


if __name__ == '__main__':
    main()
