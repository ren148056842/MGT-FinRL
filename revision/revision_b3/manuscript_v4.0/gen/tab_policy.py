import pandas as pd
D = pd.read_csv('/home/claude/v40/policy_diag.csv')
ALG = [('ppo', 'PPO'), ('a2c', 'A2C'), ('sac', 'SAC'), ('td3', 'TD3'), ('ddpg', 'DDPG')]
def f(x, d=2): return ('$-$' if x < 0 else '') + ('%.*f' % (d, abs(x)))
L = [r'\begin{table*}[!t]\centering\small\color{blue}',
     r'\caption{Behaviour of the 2025 policies (F1; mean over five seeds; descriptive, not part of the registered tests). Active share: half the $L_1$ distance between the daily target weights and equal weights (0 for equal weighting). Loading: mean daily Spearman correlation between the weights executed at the close of $t+1$ and the MGAT forecast $\hat q_t$; for the naive agents, which do not observe $\hat q_t$, it serves as a reference. Weight IC: mean daily Spearman correlation between the executed weights and the realised return $q_t$.}',
     r'\label{tab:policy}', r'\setlength{\tabcolsep}{3.5pt}', r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%' + '\n' + r'\begin{tabular}{lrrrrrrrrrrrr}\toprule',
     r' & \multicolumn{6}{c}{Banking} & \multicolumn{6}{c}{Healthcare} \\ \cmidrule(lr){2-7}\cmidrule(lr){8-13}',
     r' & \multicolumn{2}{c}{Active share} & \multicolumn{2}{c}{Loading} & \multicolumn{2}{c}{Weight IC} & \multicolumn{2}{c}{Active share} & \multicolumn{2}{c}{Loading} & \multicolumn{2}{c}{Weight IC} \\',
     r'Agent & Naive & MGAT & Naive & MGAT & Naive & MGAT & Naive & MGAT & Naive & MGAT & Naive & MGAT \\ \midrule']
for a, an in ALG + [('all', 'All')]:
    cells = []
    for sec in ['bank', 'healthcare']:
        for col, d in [('active_share', 2), ('forecast_loading', 3), ('weight_ic', 3)]:
            for src in ['Naive', 'MGAT_T']:
                g = D[(D.sector == sec) & (D.source == src) & ((D.algorithm == a) if a != 'all' else True)]
                cells.append(f(g[col].mean(), d))
    if a == 'all': L.append(r'\midrule')
    L.append(an + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule\end{tabular}}', r'\end{table*}']
open('/home/claude/v40/tables/tab_policy.tex', 'w').write('\n'.join(L)); print('\n'.join(L[-9:]))
