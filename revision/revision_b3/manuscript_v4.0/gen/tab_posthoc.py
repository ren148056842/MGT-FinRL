"""Post-hoc tables (addenda X and Y) for v4.0 from revision_b3/results_x."""
import sys, pandas as pd, numpy as np
R, OUT = sys.argv[1], sys.argv[2]
SEC = [('bank', 'Banking'), ('healthcare', 'Healthcare')]
def m(x): return '--' if not np.isfinite(x) else ('$-$' if x < 0 else '') + ('%.2f' % abs(x))
def m3(x): return '--' if not np.isfinite(x) else ('$-$' if x <= -0.0005 else '') + ('%.3f' % abs(x))
def pc(x): return '--' if not np.isfinite(x) else ('$-$' if x < 0 else '') + ('%.1f' % abs(100 * x))
def ci(lo, hi): return '[%s, %s]' % (m3(lo), m3(hi))
RS = r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%'

# ---------------- forecaster ablation (addenda X and Y)
fa = pd.read_csv(R + '/ablation_forecast.csv')
def _pairs():
    G = {'bank': '/mnt/user-data/uploads/paper--2--xiu/bank sectordata/', 'healthcare': '/mnt/user-data/uploads/paper--2--xiu/haelthcaresectordata/'}
    F = {'bank': ['bankvolume_kendall.npy', 'bank_qa_graph.npy', 'bankshareholding_relation_adjacent_matrix_tongyi.npy'],
         'healthcare': ['techvolume_kendall.npy', 'tech_qa_graph.npy', 'techstockholder.npy']}
    out = {}
    for sec in G:
        K, Q, S = [(np.load(G[sec] + f, allow_pickle=True).astype(float) > 0) & ~np.eye(15, dtype=bool) for f in F[sec]]
        pct = lambda m: round(100 * m.sum() / 210)
        full = pct(K | Q | S)
        out[sec] = dict(MGAT_T=full, X1_GAT_no_KD=pct(Q | S), X2_GAT_no_QA=pct(K | S), X3_GAT_no_SH=pct(K | Q), X4_GAT_complete=100,
                        X5_GAT_1layer=full, X6_GAT_2layers=full, X7_Transformer_2blocks=full)
    return out
PAIRS = _pairs()
NAMES = [('MGAT_T', 'MGAT (registered runs)'), ('X1_GAT_no_KD', 'without Kendall layer'), ('X2_GAT_no_QA', 'without Q\\&A layer'),
         ('X3_GAT_no_SH', 'without shareholder layer'), ('X4_GAT_complete', 'all pairs admissible'), ('X5_GAT_1layer', 'one GAT layer (64)'),
         ('X6_GAT_2layers', 'two GAT layers (32, 64)'), ('X7_Transformer_2blocks', 'two Transformer blocks')]
NAMES = [x for x in NAMES if x[0] in set(fa.model)]
L = [r'\begin{table*}[!t]\centering\small\color{blue}',
     r'\caption{Post-hoc forecaster ablation (not part of the registered tests). Each variant differs from MGAT only in the stated respect and is trained in the same four folds with seeds 1--3. IC$_2$: mean daily rank correlation of the predicted and realised tradable return (Eq.~\eqref{eq:ic2}), with its $t$-statistic for 2025; 2022--23: mean of the two development years. High, Low: change in the RMSE of the next-day log high and low in 2025 relative to the previous-day values (\%). Pairs: admissible ordered stock pairs (\%).}',
     r'\label{tab:xablation}', r'\setlength{\tabcolsep}{3pt}', RS + '\n' + r'\begin{tabular}{lrrrrrrrrrrrr}\toprule',
     r' & \multicolumn{6}{c}{Banking} & \multicolumn{6}{c}{Healthcare} \\ \cmidrule(lr){2-7}\cmidrule(lr){8-13}',
     r'Forecaster & Pairs & 2022--23 & 2024 & 2025 ($t$) & High & Low & Pairs & 2022--23 & 2024 & 2025 ($t$) & High & Low \\ \midrule']
for k, n in NAMES:
    cells = []
    for sec, _ in SEC:
        cells.append('%d' % PAIRS[sec][k])
        g = fa[(fa.sector == sec) & (fa.model == k)].set_index('test_year')
        dev = g.loc[[y for y in (2022, 2023) if y in g.index], 'IC2'].mean() if len(g) else np.nan
        v24 = g.IC2.get(2024, np.nan); v25 = g.IC2.get(2025, np.nan); t25 = g.IC2_t.get(2025, np.nan)
        cells += [m3(dev), m3(v24), '%s (%s)' % (m3(v25), ('$-$' if t25 < 0 else '') + '%.1f' % abs(t25)) if np.isfinite(t25) else '--',
                  pc(g.RMSE_h1_rel_naive.get(2025, np.nan)), pc(g.RMSE_l1_rel_naive.get(2025, np.nan))]
    L.append(n + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule\end{tabular}}', r'\end{table*}']
open(OUT + '/tab_xablation.tex', 'w').write('\n'.join(L))

# ---------------- attention
at = pd.read_csv(R + '/attention_summary.csv')
L = [r'\begin{table}[!t]\centering\small\color{blue}',
     r'\caption{Post-hoc analysis of the graph-attention coefficients of MGAT (fold 2025, three seeds, last day of every 2025 input window). Self: mean weight on the stock itself (uniform attention in parentheses). SH, QA, KD only: mean attention on admissible pairs linked by the shareholder graph, by the Q\&A graph, or only by the Kendall graph, relative to the mean over all admissible pairs (1 = no preference). Entropy: relative to uniform attention. CV: median day-to-day coefficient of variation of the coefficients. Seeds: mean correlation of the average coefficients between seeds.}',
     r'\label{tab:attention}', r'\setlength{\tabcolsep}{3pt}',
     r'\begin{tabular}{llrrrrrrr}\toprule', r'Sector & Layer & Self & SH & QA & KD only & Entropy & CV & Seeds \\ \midrule']
for sec, sn in SEC:
    g = at[at.sector == sec].sort_values('layer'); first = True
    for _, r in g.iterrows():
        L.append((sn if first else '') + ' & %d & %.3f (%.3f) & %s & %s & %s & %s & %s & %s \\\\' % (r.layer, r.self_weight, r.uniform_self, m(r.ratio_SH), m(r['ratio_QA']), m(r['ratio_KD only']), m(r.entropy_rel_uniform), m(r.cv_over_days_median), m(r.seed_agreement)))
        first = False
    if sec == 'bank': L.append(r'\midrule')
L += [r'\bottomrule\end{tabular}', r'\end{table}']
open(OUT + '/tab_attention.tex', 'w').write('\n'.join(L))

# ---------------- portfolio: allocation baselines, regimes, PPO sensitivity
al = pd.read_csv(R + '/allocation_baselines.csv'); ref = pd.read_csv(R + '/references_b3.csv')
rg = pd.read_csv(R + '/regimes_2025.csv'); import os; ps = pd.read_csv(R + '/rl_sensitivity.csv') if os.path.exists(R + '/rl_sensitivity.csv') else pd.DataFrame(columns=['sector', 'setting']); runs = pd.read_csv(R + '/rl_runs_b3.csv')
L = [r'\begin{table*}[!t]\centering\small\color{blue}',
     r'\caption{Post-hoc portfolio analyses in 2025 (not part of the registered tests). Panel A: allocation baselines with the same ledger and costs (F0 without costs; F1 baseline costs); minimum variance uses the 252-day covariance of the RL state. Panel B: Sharpe ratios on the days with above- and below-median trailing 20-day volatility of daily equal weighting, mean over the 25 MGAT and 25 naive policies (F1), and the paired difference with its 90\% block-bootstrap interval. Panel C: PPO agents (seeds 1--5, F1) with one hyperparameter changed; registered: learning rate $10^{-3}$, $\gamma=0.99$.}',
     r'\label{tab:posthoc_port}', r'\setlength{\tabcolsep}{3pt}', RS + '\n' + r'\begin{tabular}{lrrrrrrrrrr}\toprule',
     r'\multicolumn{11}{l}{\textit{Panel A: allocation baselines}} \\',
     r' & \multicolumn{5}{c}{Banking} & \multicolumn{5}{c}{Healthcare} \\ \cmidrule(lr){2-6}\cmidrule(lr){7-11}',
     r'Strategy & SR (F0) & AR (\%) & SR & MDD (\%) & Turn. (\%) & SR (F0) & AR (\%) & SR & MDD (\%) & Turn. (\%) \\ \midrule']
rows = [('equal_daily', 'Equal weight, daily', 'ref'), ('min_var_monthly', 'Minimum variance, monthly', 'al'), ('min_var_daily', 'Minimum variance, daily', 'al'),
        ('inv_vol_monthly', 'Inverse volatility, monthly', 'al'), ('MGAT_T', 'MGAT agents (mean)', 'rl'), ('Naive', 'Naive agents (mean)', 'rl')]
for k, n, kind in rows:
    cells = []
    for sec, _ in SEC:
        def get(fee):
            if kind == 'ref':
                x = ref[(ref.sector == sec) & (ref.strategy == k) & (ref.period == 2025) & (ref.fee == fee)].iloc[0]; return x.AR, x.SR, x.MDD, x.turnover
            if kind == 'al':
                x = al[(al.sector == sec) & (al.strategy == k) & (al.period == 2025) & (al.fee == fee)].iloc[0]; return x.AR, x.SR, x.MDD, x.turnover
            x = runs[(runs.sector == sec) & (runs.source == k) & (runs.fee == fee)]; return x.AR.mean(), x.SR.mean(), x.MDD.mean(), x.turnover.mean()
        a0 = get('F0'); a1 = get('F1')
        cells += [m(a0[1]), pc(a1[0]), m(a1[1]), pc(a1[2]), pc(a1[3])]
    L.append(n + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\midrule', r'\multicolumn{11}{l}{\textit{Panel B: volatility regimes within 2025}} \\',
      r' & \multicolumn{5}{c}{Banking} & \multicolumn{5}{c}{Healthcare} \\ \cmidrule(lr){2-6}\cmidrule(lr){7-11}',
      r'Regime & EW & MGAT & Naive & $\Delta$SR & 90\% interval & EW & MGAT & Naive & $\Delta$SR & 90\% interval \\ \midrule']
for reg, n in [('high volatility', 'High volatility'), ('low volatility', 'Low volatility')]:
    cells = []
    for sec, _ in SEC:
        x = rg[(rg.sector == sec) & (rg.regime == reg)].iloc[0]
        cells += [m(x.ew_SR), m(x.MGAT_SR), m(x.Naive_SR), m3(x.dSR), ci(x.ci90_low, x.ci90_high)]
    L.append(n + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\midrule', r'\multicolumn{11}{l}{\textit{Panel C: PPO hyperparameter sensitivity}} \\',
      r' & \multicolumn{5}{c}{Banking} & \multicolumn{5}{c}{Healthcare} \\ \cmidrule(lr){2-6}\cmidrule(lr){7-11}',
      r'Setting & MGAT & Naive & $\Delta$SR & 90\% interval & Higher & MGAT & Naive & $\Delta$SR & 90\% interval & Higher \\ \midrule']
for k, n in [('registered', 'Registered'), ('lr3e-4', 'Learning rate $3\\times10^{-4}$'), ('lr3e-3', 'Learning rate $3\\times10^{-3}$'), ('gamma0.9', '$\\gamma=0.9$'), ('gamma0.999', '$\\gamma=0.999$')]:
    cells = []
    for sec, _ in SEC:
        x = ps[(ps.sector == sec) & (ps.setting == k)]
        if not len(x):
            cells += ['--'] * 5; continue
        x = x.iloc[0]
        cells += [m(x.MGAT_SR), m(x.Naive_SR), m3(x.dSR), ci(x.ci90_low, x.ci90_high), '%d/5' % x.pairs_MGAT_higher]
    L.append(n + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule\end{tabular}}', r'\end{table*}']
open(OUT + '/tab_posthoc_port.tex', 'w').write('\n'.join(L))
print('posthoc tables written')
