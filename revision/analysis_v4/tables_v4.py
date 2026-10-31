import sys, pathlib
import pandas as pd, numpy as np
R = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
d = pd.read_csv(R / 'rl_mean_sd.csv'); nr = pd.read_csv(R / 'nonrl.csv'); bs = pd.read_csv(R / 'paired_bootstrap_F1.csv'); bs = bs[bs.block == 10]
cg = pd.read_csv(R / 'cost_deltas.csv'); fr = pd.read_csv(R / 'forecast_rules_vs_ew.csv')
ALG = {'a2c': 'A2C', 'ddpg': 'DDPG', 'ppo': 'PPO', 'sac': 'SAC', 'td3': 'TD3'}; SEC = {'bank': 'Banking', 'healthcare': 'Healthcare'}
def n(x, nd=2):
    x = round(float(x), nd) + 0.0; s = ('%.' + str(nd) + 'f') % x
    if s.startswith('-') and float(s) == 0: s = s[1:]
    return s.replace('-', '$-$')
def pm(m, s, sc=100, nd=2): return '%s $\\pm$ %s' % (n(m * sc, nd), n(s * sc, nd).replace('$-$', ''))
def base(sec, strat, fee='F1'): return nr[(nr.sector == sec) & (nr.strategy == strat) & (nr.fee == fee)].iloc[0]
# ---------- main F1 table ----------
L = [r'\begin{table*}[!t]\centering\small\color{blue}',
     r'\caption{2024 evaluation under the baseline cost setting (F1: 3 bp commission, 2 bp slippage, statutory sell-side stamp duty). RL agents: mean $\pm$ s.d. over three paired seeds. AR, MDD, and turnover are in \%. Vanilla and MGT agents use identical, fully observed 2023 validation and 2024 test inputs; they differ only in the 2022 training observations, which for the MGT agents are built from one-step-ahead MGT forecasts. The ridge rules are fitted on the 2022 training observations; the forecast rule holds the three stocks with the highest MGT forecast of the next-day-but-one return (Section~\ref{subsec:comparison}).}',
     r'\label{tab:main_f1}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{llrrrrr}\toprule',
     r'Sector & Strategy & AR (\%) & SR & MDD (\%) & Calmar & Turnover (\%/day) \\ \midrule']
for sec in ['bank', 'healthcare']:
    first = True
    for a in ['a2c', 'ddpg', 'ppo', 'sac', 'td3']:
        for v, lab in [('Vanilla', 'Vanilla-'), ('A0', 'MGT-')]:
            r = d[(d.sector == sec) & (d.variant == v) & (d.algorithm == a) & (d.fee == 'F1')].iloc[0]
            L.append('%s & %s%s & %s & %s & %s & %s & %s \\\\' % (SEC[sec] if first else '', lab, ALG[a], pm(r.AR_mean, r.AR_std), pm(r.SR_mean, r.SR_std, 1, 3), pm(r.MDD_mean, r.MDD_std), n(r.Calmar_mean), n(r.turnover_mean * 100)))
            first = False
    for st, lab in [('equal_buy_hold', 'Equal-weight buy-and-hold'), ('equal_daily', 'Equal-weight daily rebalancing'), ('Ridge_Vanilla', 'Ridge top-3 (vanilla training inputs)'),
                    ('Ridge_A0', 'Ridge top-3 (MGT training inputs)'), ('Forecast_A0_Full_MGT', 'MGT forecast top-3')]:
        r = base(sec, st)
        L.append(' & %s & %s & %s & %s & %s & %s \\\\' % (lab, n(r.AR * 100), n(r.SR, 3), n(r.MDD * 100), n(r.Calmar), n(r.turnover * 100)))
    L.append(r'\midrule' if sec == 'bank' else r'\bottomrule')
L.append(r'\end{tabular}}\end{table*}')
(OUT / 'tab_main_f1.tex').write_text('\n'.join(L), encoding='utf-8')
# ---------- bootstrap ----------
L = [r'\begin{table*}[!t]\centering\small\color{blue}',
     r'\caption{Paired differences (MGT $-$ Vanilla) under F1. Point estimates average the per-seed differences over three paired seeds; 95\% intervals use a circular block bootstrap (block length 10 trading days, 2{,}000 resamples) that resamples the same dates for both strategies and all seeds. $\Delta$AR and $\Delta$MDD are in percentage points; a negative $\Delta$MDD indicates a smaller drawdown for MGT. The last column counts the seeds for which the MGT agent has the higher SR.}',
     r'\label{tab:paired_boot}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{llrrrc}\toprule', r'Sector & Agent & $\Delta$AR (pp) & $\Delta$SR & $\Delta$MDD (pp) & Seeds \\ \midrule']
for sec in ['bank', 'healthcare']:
    first = True
    for a in ['a2c', 'ddpg', 'ppo', 'sac', 'td3']:
        cells = []
        for m, sc, nd in [('AR', 100, 2), ('SR', 1, 3), ('MDD', 100, 2)]:
            r = bs[(bs.sector == sec) & (bs.algorithm == a) & (bs.metric == m)].iloc[0]
            cells.append('%s [%s, %s]' % (n(r.effect * sc, nd), n(r.ci_low * sc, nd), n(r.ci_high * sc, nd)))
        L.append('%s & %s & %s & %d/3 \\\\' % (SEC[sec] if first else '', ALG[a], ' & '.join(cells), int(r.seeds_A0_higher_SR))); first = False
    L.append(r'\midrule' if sec == 'bank' else r'\bottomrule')
L.append(r'\end{tabular}}\end{table*}')
(OUT / 'tab_paired_boot.tex').write_text('\n'.join(L), encoding='utf-8')
# ---------- cost table ----------
FEE = {'F0': '0/0/0', 'F1': '3/2/5', 'F2': '5/2/5', 'F3': '10/2/5', 'F4': '20/2/5', 'F5': '3/0/5', 'F6': '3/5/5'}
cm = cg.groupby(['sector', 'variant', 'algorithm', 'fee'])[['dAR_pp', 'dSR', 'dMDD_pp']].mean().reset_index()
L = [r'\begin{table*}[!t]\centering\small\color{blue}',
     r'\caption{Cost sensitivity of the fixed policies (mean over three seeds). Cost settings are commission/slippage/sell-side stamp duty in basis points; stamp duty follows the statutory rate (5 bp in 2024). $\Delta$ columns are changes relative to the frictionless setting F0 (AR and MDD in percentage points). Policies are not retrained for each cost level.}',
     r'\label{tab:cost}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{lllrrrrrr}\toprule',
     r'Sector/agent & Variant & Setting & AR (\%) & SR & MDD (\%) & $\Delta$AR & $\Delta$SR & $\Delta$MDD \\ \midrule']
for sec, a in [('bank', 'a2c'), ('healthcare', 'sac'), ('healthcare', 'ppo')]:
    for v, lab in [('Vanilla', 'Vanilla'), ('A0', 'MGT')]:
        for f in ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6']:
            r = d[(d.sector == sec) & (d.variant == v) & (d.algorithm == a) & (d.fee == f)].iloc[0]
            if f == 'F0': dd = dict(dAR_pp=0., dSR=0., dMDD_pp=0.)
            else: dd = cm[(cm.sector == sec) & (cm.variant == v) & (cm.algorithm == a) & (cm.fee == f)].iloc[0]
            head = '%s %s' % (SEC[sec], ALG[a]) if (v == 'Vanilla' and f == 'F0') else ''
            L.append('%s & %s & %s (%s) & %s & %s & %s & %s & %s & %s \\\\' % (head, lab if f == 'F0' else '', f, FEE[f], n(r.AR_mean * 100), n(r.SR_mean, 3), n(r.MDD_mean * 100), n(dd['dAR_pp']), n(dd['dSR'], 3), n(dd['dMDD_pp'])))
        L.append(r'\addlinespace')
    L.append(r'\midrule')
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}}\end{table*}')
(OUT / 'tab_cost.tex').write_text('\n'.join(L), encoding='utf-8')
# ---------- ablation ----------
LAB = {'Vanilla': 'Vanilla (observed prices)', 'A0': 'A0 Full MGT', 'A1': 'A1 Transformer only', 'A2': 'A2 MGNN only', 'A6': 'A6 SH+QA (no KD)', 'A7': 'A7 SH+KD (no QA)',
       'A8': 'A8 QA+KD (no SH)', 'A9': 'A9 Fixed fusion', 'I1': r'I1 $\alpha_r(0)=1$', 'I2': r'I2 $\alpha_r(0)=0.1/\rho_r$', 'I3': r'I3 $\alpha_r(0)=10/\rho_r$'}
L = [r'\begin{table}[!t]\centering\small\color{blue}',
     r'\caption{Representation ablations at a fixed, prespecified policy (PPO) with fixed hyperparameters, three paired seeds, and the baseline cost setting F1. Each variant changes only the 2022 training observations, which are built from the one-step-ahead forecasts of the corresponding forecaster; validation and evaluation inputs are identical across rows. I1--I3 change only the initialisation of the relation scalars. Values are mean $\pm$ s.d. over seeds; AR and MDD in \%.}',
     r'\label{tab:ablation_ppo}', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{llrrr}\toprule', r'Sector & Training input & AR (\%) & SR & MDD (\%) \\ \midrule']
for sec in ['bank', 'healthcare']:
    first = True
    for v in ['Vanilla', 'A0', 'A1', 'A2', 'A6', 'A7', 'A8', 'A9', 'I1', 'I2', 'I3']:
        r = d[(d.sector == sec) & (d.variant == v) & (d.algorithm == 'ppo') & (d.fee == 'F1')].iloc[0]
        L.append('%s & %s & %s & %s & %s \\\\' % (SEC[sec] if first else '', LAB[v], pm(r.AR_mean, r.AR_std), pm(r.SR_mean, r.SR_std, 1, 3), pm(r.MDD_mean, r.MDD_std))); first = False
    L.append(r'\midrule' if sec == 'bank' else r'\bottomrule')
L.append(r'\end{tabular}\end{table}')
(OUT / 'tab_ablation.tex').write_text('\n'.join(L), encoding='utf-8')
# ---------- forecast rules ----------
NM = {'Ridge': 'Ridge regression', 'A0_Full_MGT': 'MGT (A0)', 'A1_Transformer_only': 'A1 Transformer only', 'A2_MGNN_only': 'A2 MGNN only', 'A3_SH_only': 'A3 SH only',
      'A4_QA_only': 'A4 QA only', 'A5_KD_only': 'A5 KD only', 'A6_SH_QA': 'A6 SH+QA', 'A7_SH_KD': 'A7 SH+KD', 'A8_QA_KD': 'A8 QA+KD', 'A9_Fixed_fusion': 'A9 Fixed fusion',
      'A10_No_cross_stock': 'A10 No cross-stock', 'B1_GCN_Transformer': 'Multi-GCN-Transformer', 'B2_GAT_Transformer': 'Multi-GAT-Transformer', 'B3_GNN_GRU': 'Multi-GNN-GRU',
      'B4_GCN_GRU': 'Multi-GCN-GRU', 'B5_GAT_GRU': 'Multi-GAT-GRU'}
ORDER = ['Ridge', 'A0_Full_MGT', 'A1_Transformer_only', 'A2_MGNN_only', 'A3_SH_only', 'A4_QA_only', 'A5_KD_only', 'A6_SH_QA', 'A7_SH_KD', 'A8_QA_KD', 'A9_Fixed_fusion',
         'A10_No_cross_stock', 'B1_GCN_Transformer', 'B2_GAT_Transformer', 'B3_GNN_GRU', 'B4_GCN_GRU', 'B5_GAT_GRU']
L = [r'\begin{table*}[!t]\centering\small\color{blue}',
     r'\caption{Forecast-ranking rules in 2024. Each rule holds, in equal weights, the three stocks with the highest forecast of the return from the close of $t+1$ to the close of $t+2$ (forecasts of the 2024 fold, averaged over three seeds), with the same execution and ledger as all other strategies. $\Delta$SR is the difference from the equal-weight daily-rebalanced portfolio under the same costs (SR 1.711 and 1.702 in banking, $-0.286$ and $-0.295$ in healthcare, at F0 and F1), with a 95\% circular block-bootstrap interval (block length 10). AR in \%; turnover in \% of the account per day.}',
     r'\label{tab:forecast_rules}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{llrrrrrrr}\toprule',
     r' & & \multicolumn{3}{c}{Frictionless (F0)} & \multicolumn{3}{c}{Baseline costs (F1)} & \\ \cmidrule(lr){3-5}\cmidrule(lr){6-8}',
     r'Sector & Forecaster & AR & SR & $\Delta$SR [95\% CI] & AR & SR & $\Delta$SR [95\% CI] & Turnover \\ \midrule']
for sec in ['bank', 'healthcare']:
    first = True
    for m in ORDER:
        a = fr[(fr.sector == sec) & (fr.model == m) & (fr.fee == 'F0')].iloc[0]; b = fr[(fr.sector == sec) & (fr.model == m) & (fr.fee == 'F1')].iloc[0]
        L.append('%s & %s & %s & %s & %s [%s, %s] & %s & %s & %s [%s, %s] & %s \\\\' % (SEC[sec] if first else '', NM[m], n(a.AR * 100, 1), n(a.SR, 2), n(a.dSR, 2), n(a.dSR_lo, 2), n(a.dSR_hi, 2),
                                                                                    n(b.AR * 100, 1), n(b.SR, 2), n(b.dSR, 2), n(b.dSR_lo, 2), n(b.dSR_hi, 2), n(a.turnover * 100, 0))); first = False
    L.append(r'\midrule' if sec == 'bank' else r'\bottomrule')
L.append(r'\end{tabular}}\end{table*}')
(OUT / 'tab_forecast_rules.tex').write_text('\n'.join(L), encoding='utf-8')
print('RL tables written')
