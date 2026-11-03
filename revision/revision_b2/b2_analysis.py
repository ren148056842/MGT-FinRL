"""Plan B2 analysis of the 2024 evaluation (PROTOCOL.md hypotheses H1 and H2).

python b2_analysis.py <b2_results_dir> <out_dir> [sectors]
Reads results/B2_<sector>_<source>_<algo>_s<seed>_<fee>/trajectory.csv and writes:
  b2_runs.csv           metrics of every run
  b2_summary.csv        mean and sd over seeds by sector, source, algorithm and fee
  b2_tests.csv          paired block-bootstrap tests (H1, H2; block lengths 5, 10, 20), Holm-adjusted
"""
import sys, pathlib, itertools
import numpy as np, pandas as pd

RES, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
SECTORS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['bank', 'healthcare']
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = ['Naive', 'Ridge_R', 'A10_R', 'MGT_R']
ALGOS = ['ppo', 'sac']; SEEDS = list(range(1, 11)); FEES = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6']


def metrics(r):
    r = np.asarray(r, float); curve = np.r_[1., np.cumprod(1 + r)]
    sd = np.std(r, ddof=1)
    return dict(AR=curve[-1] ** (252 / len(r)) - 1, SR=np.sqrt(252) * r.mean() / sd if sd > 0 else np.nan,
                MDD=np.max(1 - curve / np.maximum.accumulate(curve)), Vol=np.sqrt(252) * sd)


def returns(sec, src, algo, seed, fee):
    f = RES / ('B2_%s_%s_%s_s%d_%s' % (sec, src, algo, seed, fee)) / 'trajectory.csv'
    if not f.exists():
        return None
    t = pd.read_csv(f)
    return t.set_index('date').daily_return.iloc[1:]


rows = []; R = {}
for sec, src, algo, seed, fee in itertools.product(SECTORS, SOURCES, ALGOS, SEEDS, FEES):
    r = returns(sec, src, algo, seed, fee)
    if r is None:
        continue
    R[(sec, src, algo, seed, fee)] = r
    rows.append(dict(sector=sec, source=src, algorithm=algo, seed=seed, fee=fee, **metrics(r.to_numpy())))
runs = pd.DataFrame(rows); runs.to_csv(OUT / 'b2_runs.csv', index=False)
summ = runs.groupby(['sector', 'source', 'algorithm', 'fee'])[['AR', 'SR', 'MDD', 'Vol']].agg(['mean', 'std']).reset_index()
summ.columns = ['_'.join(c).strip('_') for c in summ.columns]; summ.to_csv(OUT / 'b2_summary.csv', index=False)


def panel(sec, src, fee='F1'):
    cols = [R[(sec, src, a, s, fee)] for a in ALGOS for s in SEEDS if (sec, src, a, s, fee) in R]
    if len(cols) != len(ALGOS) * len(SEEDS):
        return None
    m = pd.concat(cols, axis=1)
    return m.to_numpy(float)


def mean_sr_diff(A, B, ix):
    return np.mean([metrics(A[ix, j])['SR'] - metrics(B[ix, j])['SR'] for j in range(A.shape[1])])


def boot(A, B, block, n_boot=2000, seed=1729):
    rng = np.random.default_rng(seed); n = A.shape[0]
    point = mean_sr_diff(A, B, np.arange(n)); s = []
    for _ in range(n_boot):
        st = rng.integers(0, n, size=int(np.ceil(n / block))); ix = ((st[:, None] + np.arange(block)) % n).ravel()[:n]
        s.append(mean_sr_diff(A, B, ix))
    s = np.array(s)
    # one-sided p-value for "difference > 0": share of centred bootstrap draws at least as large as the estimate
    p = float(np.mean((s - point) >= point))
    lo, hi = np.percentile(s, [5, 95])
    return point, lo, hi, p


tests = []
for sec in SECTORS:
    base = {src: panel(sec, src) for src in SOURCES}
    if base['MGT_R'] is None:
        continue
    for ref, hyp in [('Naive', 'H1'), ('Ridge_R', 'H2'), ('A10_R', 'H2')]:
        if base[ref] is None:
            continue
        for block in [5, 10, 20]:
            d, lo, hi, p = boot(base['MGT_R'], base[ref], block)
            tests.append(dict(sector=sec, hypothesis=hyp, comparison='MGT_R - ' + ref, block=block, dSR=d, ci90_low=lo, ci90_high=hi, p_one_sided=p))
T = pd.DataFrame(tests)
if len(T):
    T['p_holm'] = np.nan
    main = T[T.block == 10]
    # Holm across sectors for H1; across the three comparisons within each sector for H1+H2
    for (key, grp) in [('H1', main[main.hypothesis == 'H1']), *[(s, main[main.sector == s]) for s in SECTORS]]:
        if not len(grp):
            continue
        order = grp.p_one_sided.sort_values().index; m = len(order); running = 0
        for k, ix in enumerate(order):
            running = max(running, min(1.0, (m - k) * T.loc[ix, 'p_one_sided']))
            col = 'p_holm_H1_across_sectors' if key == 'H1' else 'p_holm_within_sector'
            T.loc[ix, col] = running
    T.to_csv(OUT / 'b2_tests.csv', index=False)
print(summ[summ.fee == 'F1'].round(3).to_string(index=False))
print(T.round(4).to_string(index=False) if len(T) else 'no tests')
