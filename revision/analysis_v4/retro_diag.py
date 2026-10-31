"""Retrospective selection diagnostic (Appendix E, Table 14).

For each sector and fixed-PPO training input, compares the mean 2024 Sharpe ratio over the three
paired seeds (the statistic used in Table 7) with the Sharpe ratio of the seed that is best in 2024.
The best-seed selection uses the evaluation year and is reported only as a diagnostic.

Usage: python retro_diag.py <results/rl> <results>
"""
import sys, pathlib
import pandas as pd

R = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
d = pd.read_csv(R / 'rl_runs.csv')
d = d[(d.fee == 'F1') & (d.algorithm == 'ppo')]
ORDER = [('Vanilla', 'Vanilla (observed prices)'), ('A0', 'A0 Full MGT'), ('A1', 'A1 Transformer only'), ('A2', 'A2 MGNN only'),
         ('A6', 'A6 SH+QA (no KD)'), ('A7', 'A7 SH+KD (no QA)'), ('A8', 'A8 QA+KD (no SH)'), ('A9', 'A9 Fixed fusion'),
         ('I1', r'I1 $\alpha_r(0)=1$'), ('I2', r'I2 $\alpha_r(0)=0.1/\rho_r$'), ('I3', r'I3 $\alpha_r(0)=10/\rho_r$')]
SEC = {'bank': 'Banking', 'healthcare': 'Healthcare'}
rows = []
for sec in ['bank', 'healthcare']:
    x = d[d.sector == sec]
    assert set(x.groupby('variant').size()) == {3}, 'expected three seeds per input'
    g = x.groupby('variant').SR
    t = pd.DataFrame({'mean': g.mean(), 'best': g.max(), 'best_seed': x.loc[g.idxmax(), ['variant', 'seed']].set_index('variant').seed})
    t = t.loc[[k for k, _ in ORDER]]
    t['rank_mean'] = t['mean'].rank(ascending=False, method='min').astype(int)
    t['rank_best'] = t['best'].rank(ascending=False, method='min').astype(int)
    t['gain'] = t['best'] - t['mean']
    t.insert(0, 'sector', sec)
    rows.append(t.reset_index())
res = pd.concat(rows, ignore_index=True)
(R / 'retro_best_seed.csv').write_text(res.to_csv(index=False), encoding='utf-8')

def n(v, nd=3):
    s = ('%.' + str(nd) + 'f') % (round(float(v), nd) + 0.0)
    return s.replace('-', '$-$')

L = [r'\begin{table}[!h]\centering\small\color{blue}',
     r'\caption{Retrospective selection diagnostic for the fixed-PPO comparison (F1). Mean: mean 2024 SR over the three paired seeds, as in Table~\ref{tab:ablation_ppo}. Best seed: SR of the seed with the highest 2024 SR. Ranks are among the eleven training inputs of each sector (1 = highest). The best-seed selection uses the evaluation year and is not out-of-sample evidence.}',
     r'\label{tab:retro}', r'\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{lrrrrrrrr}\toprule',
     r' & \multicolumn{4}{c}{Banking} & \multicolumn{4}{c}{Healthcare} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}',
     r'Training input & Mean & Best seed & \multicolumn{2}{c}{Rank} & Mean & Best seed & \multicolumn{2}{c}{Rank} \\',
     r' & & & mean & best & & & mean & best \\ \midrule']
b = res[res.sector == 'bank'].set_index('variant'); h = res[res.sector == 'healthcare'].set_index('variant')
for k, lab in ORDER:
    L.append('%s & %s & %s & %d & %d & %s & %s & %d & %d \\\\' % (lab, n(b.loc[k, 'mean']), n(b.loc[k, 'best']), b.loc[k, 'rank_mean'], b.loc[k, 'rank_best'],
                                                               n(h.loc[k, 'mean']), n(h.loc[k, 'best']), h.loc[k, 'rank_mean'], h.loc[k, 'rank_best']))
L += [r'\bottomrule', r'\end{tabular}\end{table}']
(OUT / 'tab_retro.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')
for sec, t in res.groupby('sector'):
    print(sec, 'gain %.3f-%.3f' % (t.gain.min(), t.gain.max()), 'top best:', t.loc[t.rank_best.idxmin(), 'variant'], 'top mean:', t.loc[t.rank_mean.idxmin(), 'variant'],
          'A0 ranks', int(t[t.variant == 'A0'].rank_mean.iloc[0]), int(t[t.variant == 'A0'].rank_best.iloc[0]))
