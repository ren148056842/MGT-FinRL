"""RL tables for v4.0 from the B3 results directory."""
import sys, pandas as pd, numpy as np
R, OUT = sys.argv[1], sys.argv[2]
runs = pd.read_csv(R + '/rl_runs.csv'); tests = pd.read_csv(R + '/rl_tests.csv'); byalg = pd.read_csv(R + '/rl_by_algorithm.csv')
abl = pd.read_csv(R + '/ablation.csv'); ref = pd.read_csv(R + '/references.csv')
SRC = [('Naive', 'Naive'), ('Ridge_R', 'Ridge'), ('A1_R', 'Graph-free'), ('MGAT_T', 'MGAT')]
SEC = [('bank', 'Banking'), ('healthcare', 'Healthcare')]
ALG = [('ppo', 'PPO'), ('a2c', 'A2C'), ('sac', 'SAC'), ('td3', 'TD3'), ('ddpg', 'DDPG')]
def m(x): return ('$-$' if x < 0 else '') + ('%.2f' % abs(x))
def m3(x): return ('$-$' if x < 0 else '') + ('%.3f' % abs(x))
def pc(x): return ('$-$' if x < 0 else '') + ('%.1f' % abs(100 * x))
def ms(a, s, f=m): return f(a) + r'\,$\pm$\,' + (f(s).replace('$-$', ''))
def p3(p): return '$<$0.001' if p < 0.001 else '%.3f' % p
# ---------------- main table: 2025, F1, pooled over 5 agents x 5 seeds
L = [r'\begin{table*}[!t]\centering\small\color{blue}',
     r'\caption{Portfolio performance in the evaluation year 2025 under cost setting F1 (3~bp commission, 2~bp slippage, stamp duty). RL rows: mean\,$\pm$\,standard deviation over the 25 policies of each forecast source (five agents $\times$ five seeds), each trained on 2022--2023 and selected on 2024. AR: annualised return; SR: annualised Sharpe ratio; MDD: maximum drawdown; turnover: mean daily gross turnover. Reference strategies are deterministic.}',
     r'\label{tab:main_f1}', r'\setlength{\tabcolsep}{4pt}', r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%' + '\n' + r'\begin{tabular}{llrrrrr}\toprule',
     r'Sector & Strategy & AR (\%) & SR & MDD (\%) & Calmar & Turnover (\%/day) \\ \midrule']
for sec, sn in SEC:
    first = True
    for k, n in SRC:
        g = runs[(runs.sector == sec) & (runs.source == k) & (runs.fee == 'F1') & runs.algorithm.isin([a for a, _ in ALG])]
        L.append((sn if first else '') + ' & ' + n + ' agents & ' + ' & '.join([ms(g.AR.mean(), g.AR.std(), pc), ms(g.SR.mean(), g.SR.std()), ms(g.MDD.mean(), g.MDD.std(), pc), ms(g.Calmar.mean(), g.Calmar.std()), pc(g.turnover.mean())]) + r' \\')
        first = False
    for k, n in [('equal_daily', 'Equal weight, daily'), ('buy_hold', 'Equal weight, buy-and-hold'), ('Forecast_MGAT_T', 'Top-3 MGAT rule')]:
        r = ref[(ref.sector == sec) & (ref.strategy == k) & (ref.period == 2025) & (ref.fee == 'F1')]
        if len(r):
            r = r.iloc[0]
            L.append(' & ' + n + ' & ' + ' & '.join([pc(r.AR), m(r.SR), pc(r.MDD), m(r.Calmar), pc(r.turnover)]) + r' \\')
    if sec == 'bank': L.append(r'\midrule')
L += [r'\bottomrule\end{tabular}}', r'\end{table*}']
open(OUT + '/tab_main_f1.tex', 'w').write('\n'.join(L))
# ---------------- tests table
L = [r'\begin{table*}[!t]\centering\small\color{blue}',
     r'\caption{Pre-registered tests in 2025 (F1). $\Delta$SR: mean paired difference in annualised Sharpe ratio over the 25 agent--seed pairs; 90\% interval and one-sided $p$-value from a paired circular block bootstrap of daily returns (block length 10; 2{,}000 resamples); $p_{5}$ and $p_{20}$: block lengths 5 and 20. Holm: H1 across sectors; H1--H2b within each sector.}',
     r'\label{tab:paired_boot}', r'\setlength{\tabcolsep}{3pt}', r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%' + '\n' + r'\begin{tabular}{llrrrrrr}\toprule',
     r'Sector & Hypothesis & $\Delta$SR & 90\% interval & $p$ & $p_{5}$ & $p_{20}$ & Holm $p$ \\ \midrule']
for sec, sn in SEC:
    first = True
    for hyp, lab in [('H1', 'H1: MGAT $-$ Naive'), ('H2a', 'H2a: MGAT $-$ Graph-free'), ('H2b', 'H2b: MGAT $-$ Ridge')]:
        g = tests[(tests.sector == sec) & (tests.hypothesis == hyp)]
        if not len(g): continue
        r10 = g[g.block == 10].iloc[0]; p5 = g[g.block == 5].p_one_sided.iloc[0]; p20 = g[g.block == 20].p_one_sided.iloc[0]
        holm = r10.p_holm_H1_across_sectors if hyp == 'H1' else r10.p_holm_within_sector
        hs = p3(holm) + (r'\,/\,' + p3(r10.p_holm_within_sector) if hyp == 'H1' else '')
        L.append((sn if first else '') + ' & ' + lab + ' & ' + m3(r10.dSR) + ' & [' + m3(r10.ci90_low) + ', ' + m3(r10.ci90_high) + '] & ' + p3(r10.p_one_sided) + ' & ' + p3(p5) + ' & ' + p3(p20) + ' & ' + hs + r' \\')
        first = False
    if sec == 'bank': L.append(r'\midrule')
L += [r'\bottomrule\end{tabular}}', r'\end{table*}']
open(OUT + '/tab_tests.tex', 'w').write('\n'.join(L))
# ---------------- by algorithm
L = [r'\begin{table*}[!t]\centering\small\color{blue}',
     r'\caption{Sharpe ratio in 2025 (F1) by agent: mean over five seeds for the naive and MGAT sources, and the paired difference with its 90\% block-bootstrap interval (block length 10).}',
     r'\label{tab:byagent}', r'\setlength{\tabcolsep}{4pt}', r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%' + '\n' + r'\begin{tabular}{lrrrrrrrr}\toprule',
     r' & \multicolumn{4}{c}{Banking} & \multicolumn{4}{c}{Healthcare} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}',
     r'Agent & Naive & MGAT & $\Delta$SR & 90\% interval & Naive & MGAT & $\Delta$SR & 90\% interval \\ \midrule']
for a, an in ALG:
    cells = []
    for sec, _ in SEC:
        nv = runs[(runs.sector == sec) & (runs.source == 'Naive') & (runs.algorithm == a) & (runs.fee == 'F1')].SR.mean()
        mg = runs[(runs.sector == sec) & (runs.source == 'MGAT_T') & (runs.algorithm == a) & (runs.fee == 'F1')].SR.mean()
        b = byalg[(byalg.sector == sec) & (byalg.algorithm == a) & (byalg.comparison == 'MGAT_T - Naive') & (byalg.fee == 'F1')].iloc[0]
        cells += [m(nv), m(mg), m3(b.dSR), '[' + m3(b.ci90_low) + ', ' + m3(b.ci90_high) + ']']
    L.append(an + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule\end{tabular}}', r'\end{table*}']
open(OUT + '/tab_byagent.tex', 'w').write('\n'.join(L))
# ---------------- costs
L = [r'\begin{table}[!t]\centering\small\color{blue}',
     r'\caption{Cost sensitivity in 2025: mean Sharpe ratio of the 25 MGAT and naive policies and their paired difference (90\% block-bootstrap interval) under each cost setting. Commission/slippage in bp: F0 0/0 without stamp duty, F1 3/2, F2 5/2, F3 10/2, F4 20/2, F5 3/0, F6 3/5.}',
     r'\label{tab:cost}', r'\setlength{\tabcolsep}{3pt}', r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%' + '\n' + r'\begin{tabular}{lrrrrrr}\toprule',
     r' & \multicolumn{3}{c}{Banking} & \multicolumn{3}{c}{Healthcare} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}',
     r'Cost & MGAT & Naive & $\Delta$SR [90\%] & MGAT & Naive & $\Delta$SR [90\%] \\ \midrule']
for fee in ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6']:
    cells = []
    for sec, _ in SEC:
        g = runs[(runs.sector == sec) & (runs.fee == fee) & runs.algorithm.isin([a for a, _ in ALG])]
        b = byalg[(byalg.sector == sec) & (byalg.algorithm == 'all') & (byalg.fee == fee)].iloc[0]
        cells += [m(g[g.source == 'MGAT_T'].SR.mean()), m(g[g.source == 'Naive'].SR.mean()), m3(b.dSR) + ' [' + m3(b.ci90_low) + ', ' + m3(b.ci90_high) + ']']
    L.append(fee + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule\end{tabular}}', r'\end{table}']
open(OUT + '/tab_cost.tex', 'w').write('\n'.join(L))
# ---------------- ablation (PPO)
NAMES = {'Naive': 'Naive', 'GCN_R': 'GCN', 'A10_R': 'No cross-stock', 'MGT_R': 'MGNN', 'A1_R': 'Graph-free', 'Ridge_R': 'Ridge'}
L = [r'\begin{table}[!t]\centering\small\color{blue}',
     r'\caption{Forecast-source ablation with PPO (seeds 1--5), 2025, F1: mean Sharpe ratio of each source and the paired difference MGAT minus source (90\% block-bootstrap interval).}',
     r'\label{tab:ablation_ppo}', r'\setlength{\tabcolsep}{3pt}', r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%' + '\n' + r'\begin{tabular}{lrrrr}\toprule',
     r' & \multicolumn{2}{c}{Banking} & \multicolumn{2}{c}{Healthcare} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}',
     r'Source & SR & MGAT $-$ source & SR & MGAT $-$ source \\ \midrule']
cells = []
for sec, _ in SEC:
    cells.append(m(runs[(runs.sector == sec) & (runs.source == 'MGAT_T') & (runs.algorithm == 'ppo') & (runs.fee == 'F1')].SR.mean())); cells.append('--')
L.append('MGAT & ' + ' & '.join(cells) + r' \\')
for k in ['GCN_R', 'A10_R', 'MGT_R', 'A1_R', 'Ridge_R', 'Naive']:
    cells = []
    for sec, _ in SEC:
        sr = runs[(runs.sector == sec) & (runs.source == k) & (runs.algorithm == 'ppo') & (runs.fee == 'F1')].SR.mean()
        b = abl[(abl.sector == sec) & (abl.comparison == 'MGAT_T - ' + k)].iloc[0]
        cells += [m(sr), m3(b.dSR) + ' [' + m3(b.ci90_low) + ', ' + m3(b.ci90_high) + ']']
    L.append(NAMES[k] + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule\end{tabular}}', r'\end{table}']
open(OUT + '/tab_ablation.tex', 'w').write('\n'.join(L))
# ---------------- rules
L = [r'\begin{table*}[!t]\centering\small\color{blue}',
     r'\caption{Forecast-ranking rules in 2025: equal weights on the three stocks with the highest predicted return from the close of $t+1$ to the close of $t+2$, executed at the close of $t+1$, compared with daily equal weighting.}',
     r'\label{tab:forecast_rules}', r'\setlength{\tabcolsep}{4pt}', r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%' + '\n' + r'\begin{tabular}{llrrrrrr}\toprule',
     r' & & \multicolumn{3}{c}{F0 (no costs)} & \multicolumn{3}{c}{F1} \\ \cmidrule(lr){3-5}\cmidrule(lr){6-8}',
     r'Sector & Strategy & AR (\%) & SR & MDD (\%) & AR (\%) & SR & MDD (\%) \\ \midrule']
RULES = [('equal_daily', 'Equal weight, daily'), ('Forecast_MGAT_T', 'MGAT'), ('Forecast_GCN_R', 'GCN'), ('Forecast_A10_R', 'No cross-stock'), ('Forecast_MGT_R', 'MGNN'), ('Forecast_A1_R', 'Graph-free'), ('Forecast_Ridge_R', 'Ridge')]
for sec, sn in SEC:
    first = True
    for k, n in RULES:
        cells = []
        for fee in ['F0', 'F1']:
            r = ref[(ref.sector == sec) & (ref.strategy == k) & (ref.period == 2025) & (ref.fee == fee)].iloc[0]
            cells += [pc(r.AR), m(r.SR), pc(r.MDD)]
        L.append((sn if first else '') + ' & ' + n + ' & ' + ' & '.join(cells) + r' \\'); first = False
    if sec == 'bank': L.append(r'\midrule')
L += [r'\bottomrule\end{tabular}}', r'\end{table*}']
open(OUT + '/tab_rules.tex', 'w').write('\n'.join(L))
print('ok')
# single-column tables in sn-jnl cannot hold \resizebox: strip it from these
for nm in ['tab_cost', 'tab_ablation']:
    f = OUT + '/' + nm + '.tex'; s = open(f).read()
    s = s.replace('\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n', '').replace('\\end{tabular}}', '\\end{tabular}')
    open(f, 'w').write(s)
