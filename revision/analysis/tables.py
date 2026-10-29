import pandas as pd, numpy as np
d=pd.read_csv('main_mean_sd.csv'); b=pd.read_csv('baseline_comparison.csv'); bs=pd.read_csv('paired_bootstrap_all_F1.csv')
ALG={'a2c':'A2C','ddpg':'DDPG','ppo':'PPO','sac':'SAC','td3':'TD3'}
SEC={'bank':'Banking','healthcare':'Healthcare'}
def pm(m,s,scale=100,nd=2): return f"{m*scale:.{nd}f} $\\pm$ {s*scale:.{nd}f}"
out=[]
# Table: main F1
L=[]
L.append(r"\begin{table*}[!t]\centering\small")
L.append(r"\caption{2024 evaluation under the baseline cost setting (F1: 3 bp commission, 2 bp slippage, statutory sell-side stamp duty). Values are mean $\pm$ s.d. over three paired RL seeds. AR, MDD and turnover are in \%. Vanilla and MGT agents use identical, fully observed 2023 validation and 2024 test inputs; they differ only in the 2022 training observations.}")
L.append(r"\label{tab:main_f1}")
L.append(r"\begin{tabular}{llrrrrr}\toprule")
L.append(r"Sector & Strategy & AR (\%) & SR & MDD (\%) & Calmar & Daily turnover (\%) \\ \midrule")
for sec in ['bank','healthcare']:
    first=True
    for a in ['a2c','ddpg','ppo','sac','td3']:
        for v,lab in [('Vanilla','Vanilla-'),('A0','MGT-')]:
            r=d[(d.sector==sec)&(d.variant==v)&(d.algorithm==a)&(d.fee=='F1')].iloc[0]
            L.append(f"{SEC[sec] if first else ''} & {lab}{ALG[a]} & {pm(r.AR_mean,r.AR_std)} & {pm(r.SR_mean,r.SR_std,1,3)} & {pm(r.MDD_mean,r.MDD_std)} & {r.Calmar_mean:.2f} & {r.mean_gross_turnover_mean*100:.2f} \\\\")
            first=False
    for st,lab in [('equal_buy_hold','Equal-weight buy-and-hold'),('equal_daily','Equal-weight daily rebalancing'),('Ridge_Vanilla','Ridge top-3 (vanilla inputs)'),('Ridge_A0','Ridge top-3 (MGT inputs)')]:
        r=b[(b.sector==sec)&(b.variant==st)&(b.fee=='F1')].iloc[0]
        L.append(f" & {lab} & {r.AR*100:.2f} & {r.SR:.3f} & {r.MDD*100:.2f} & {r.Calmar:.2f} & {r.mean_gross_turnover*100:.2f} \\\\")
    L.append(r"\midrule" if sec=='bank' else r"\bottomrule")
L.append(r"\end{tabular}\end{table*}")
open('tab_main_f1.tex','w').write('\n'.join(L))
# Bootstrap table
L=[r"\begin{table}[!t]\centering\small",
r"\caption{Paired differences (MGT $-$ Vanilla) under F1. Point estimates average the per-seed differences over three paired seeds; 95\% intervals use a circular block bootstrap (block length 10 trading days, 2{,}000 resamples) that resamples the same dates for both strategies and all seeds. $\Delta$AR and $\Delta$MDD are in percentage points; a negative $\Delta$MDD indicates a smaller drawdown for MGT.}",
r"\label{tab:paired_boot}",r"\begin{tabular}{llrrr}\toprule",r"Sector & Agent & $\Delta$AR (pp) & $\Delta$SR & $\Delta$MDD (pp) \\ \midrule"]
for sec in ['bank','healthcare']:
    first=True
    for a in ['a2c','ddpg','ppo','sac','td3']:
        cells=[]
        for m,sc,nd in [('AR',100,2),('SR',1,3),('MDD',100,2)]:
            r=bs[(bs.sector==sec)&(bs.algorithm==a)&(bs.metric==m)].iloc[0]
            cells.append(f"{r.effect*sc:.{nd}f} [{r.ci_low*sc:.{nd}f}, {r.ci_high*sc:.{nd}f}]")
        L.append(f"{SEC[sec] if first else ''} & {ALG[a]} & "+' & '.join(cells)+r" \\"); first=False
    L.append(r"\midrule" if sec=='bank' else r"\bottomrule")
L+= [r"\end{tabular}\end{table}"]
open('tab_paired_boot.tex','w').write('\n'.join(L))
# Cost sensitivity
FEE={'F0':'0/0/0','F1':'3/2/5','F2':'5/2/5','F3':'10/2/5','F4':'20/2/5','F5':'3/0/5','F6':'3/5/5'}
L=[r"\begin{table*}[!t]\centering\small",
r"\caption{Cost sensitivity of the fixed policies (mean over three seeds). Cost settings are commission/slippage/sell-side stamp duty in basis points; stamp duty follows the statutory rate (5 bp in 2024). $\Delta$ columns are changes relative to the frictionless setting F0 (AR and MDD in percentage points). Policies are not retrained for each cost level.}",
r"\label{tab:cost}",r"\begin{tabular}{lllrrrrrr}\toprule",
r"Sector/agent & Variant & Setting & AR (\%) & SR & MDD (\%) & $\Delta$AR & $\Delta$SR & $\Delta$MDD \\ \midrule"]
for sec,a in [('bank','a2c'),('healthcare','sac'),('healthcare','ppo')]:
    for v,lab in [('Vanilla','Vanilla'),('A0','MGT')]:
        for f in ['F0','F1','F2','F3','F4','F5','F6']:
            r=d[(d.sector==sec)&(d.variant==v)&(d.algorithm==a)&(d.fee==f)].iloc[0]
            head=f"{SEC[sec]} {ALG[a]}" if (v=='Vanilla' and f=='F0') else ''
            vl=lab if f=='F0' else ''
            L.append(f"{head} & {vl} & {f} ({FEE[f]}) & {r.AR_mean*100:.2f} & {r.SR_mean:.3f} & {r.MDD_mean*100:.2f} & {r.delta_AR_vs_F0_pp_mean:.2f} & {r.delta_SR_vs_F0_mean:.3f} & {r.delta_MDD_vs_F0_pp_mean:.2f} \\\\")
        L.append(r"\addlinespace")
    L.append(r"\midrule")
L[-1]=r"\bottomrule"
L+=[r"\end{tabular}\end{table*}"]
open('tab_cost.tex','w').write('\n'.join(L))
# Ablation fixed PPO
L=[r"\begin{table}[!t]\centering\small",
r"\caption{Representation ablations at a fixed, prespecified policy (PPO) and fixed hyperparameters, three paired seeds, under F1. Values are mean $\pm$ s.d.; AR and MDD in \%.}",
r"\label{tab:ablation_ppo}",r"\begin{tabular}{llrrr}\toprule",r"Sector & Training input & AR (\%) & SR & MDD (\%) \\ \midrule"]
for sec in ['bank','healthcare']:
    first=True
    for v in ['Vanilla','A0','A1','A2','A6','A7','A8','A9']:
        r=d[(d.sector==sec)&(d.variant==v)&(d.algorithm=='ppo')&(d.fee=='F1')].iloc[0]
        L.append(f"{SEC[sec] if first else ''} & {v} & {pm(r.AR_mean,r.AR_std)} & {pm(r.SR_mean,r.SR_std,1,3)} & {pm(r.MDD_mean,r.MDD_std)} \\\\"); first=False
    L.append(r"\midrule" if sec=='bank' else r"\bottomrule")
L+=[r"\end{tabular}\end{table}"]
open('tab_ablation.tex','w').write('\n'.join(L))
print(open('tab_main_f1.tex').read()[:3000])
