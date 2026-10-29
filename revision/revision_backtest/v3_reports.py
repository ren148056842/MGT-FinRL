"""Scope-aware real-training reports; no model or trajectory mutation."""
import pathlib,json,zipfile
import numpy as np
import pandas as pd
from v3_audit import ROOT,RUN,SCOPE,sha,load_json,save_json,save_csv,integrity,canonical
from v3_engine import records,context
from v2_reports import collect_metrics,cost_differences,write_table
from backtest import paired_bootstrap


def summary(df):
 keys=['sector','variant','algorithm','fee'];rows=[]
 if df.empty:return pd.DataFrame(columns=keys+['AR_mean','AR_std','SR_mean','SR_std','MDD_mean','MDD_std','accounting_scope'])
 for key,g in df.groupby(keys):
  row=dict(zip(keys,key));row.update(n=len(g),accounting_scope=SCOPE,economic_return_validated=False,protocol_id='v3_train_only_unadjusted_price_cost',run_ids=';'.join(g.run_id),input_hash=';'.join(g.input_hash.unique()),code_hash=';'.join(g.code_hash.unique()),checkpoint_hashes=';'.join(g.checkpoint_sha256.unique()))
  for col in ['AR','SR','SR_rf02','MDD','Vol','Calmar','cumulative_cost','mean_gross_turnover','delta_AR_vs_F0_pp','delta_SR_vs_F0','delta_MDD_vs_F0_pp','delta_AR_vs_F1_pp','delta_SR_vs_F1','delta_MDD_vs_F1_pp']:
   if col in g:row[col+'_mean']=g[col].mean();row[col+'_std']=g[col].std(ddof=1) if len(g)>1 else np.nan
  rows.append(row)
 return pd.DataFrame(rows)


def bootstrap(df):
 rows=[];status=[];cache=RUN/'analysis/bootstrap_cache';cache.mkdir(exist_ok=True,parents=True)
 for sector,algo in [('bank','a2c'),('healthcare','sac'),('healthcare','ppo')]:
  for fee in ['F1','F0']:
   aa=[];bb=[];hashes=[];dates=None;ok=True
   for seed in [1,2,3]:
    pair=[]
    for variant in ['A0','Vanilla']:
     z=df[(df.sector==sector)&(df.algorithm==algo)&(df.seed==seed)&(df.variant==variant)&(df.fee==fee)]
     if len(z)!=1:ok=False;break
     path=pathlib.Path(z.iloc[0]['directory'])/'trajectory.csv';t=pd.read_csv(path);ds=t.date.iloc[1:].tolist()
     if dates is None:dates=ds
     if dates!=ds:raise ValueError('Unaligned paired dates')
     pair.append(t.daily_return.iloc[1:].to_numpy(float));hashes.append(sha(path))
    if not ok:break
    aa.append(pair[0]);bb.append(pair[1])
   if not ok:status.append(dict(sector=sector,algorithm=algo,fee=fee,status='pending_or_blocked',reason='Three paired seed trajectories needed'));continue
   for block in [10,5,20]:
    key=canonical(dict(paths=hashes,block=block,B=2000,seed=1729));f=cache/(key+'.csv')
    if f.exists():result=pd.read_csv(f)
    else:result=paired_bootstrap(np.stack(aa,axis=1),np.stack(bb,axis=1),block,2000,1729);save_csv(f,result)
    for r in result.to_dict('records'):rows.append(dict(**r,sector=sector,algorithm=algo,fee=fee,comparison='A0 - Vanilla',accounting_scope=SCOPE,economic_return_validated=False,protocol_id='v3_train_only_unadjusted_price_cost',trajectory_hashes=';'.join(hashes),interpretation='Conditional on frozen offline augmentation and trained policies; same dates across seeds; no p-values; no total-return claim'))
   status.append(dict(sector=sector,algorithm=algo,fee=fee,status='computation_completed'))
 save_csv(RUN/'results/bootstrap_status.csv',status)
 return pd.DataFrame(rows,columns=None if rows else ['sector','algorithm','fee','metric','effect','ci_low','ci_high','block','B','accounting_scope'])


def plots(df):
 if df.empty:return
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=RUN/'figures';out.mkdir(exist_ok=True)
 fig,axes=plt.subplots(2,3,figsize=(15,7),sharex='col')
 for j,(sec,algo) in enumerate([('bank','a2c'),('healthcare','sac'),('healthcare','ppo')]):
  g=df[(df.sector==sec)&(df.algorithm==algo)&(df.fee=='F1')]
  for variant in ['Vanilla','A0']:
   paths=[];dates=None
   for _,r in g[g.variant==variant].iterrows():
    t=pd.read_csv(pathlib.Path(r.directory)/'trajectory.csv');dates=pd.to_datetime(t.date);paths.append(t.nav.to_numpy()/1e6)
   if paths:
    a=np.stack(paths);mean=a.mean(axis=0);axes[0,j].plot(dates,mean,label=variant);axes[0,j].fill_between(dates,a.min(axis=0),a.max(axis=0),alpha=.15);axes[1,j].plot(dates,100*(1-mean/np.maximum.accumulate(mean)),label=variant)
  axes[0,j].set_title(sec+' '+algo.upper());axes[0,j].legend();axes[0,j].set_ylabel('Mean normalized price-account NAV');axes[1,j].set_ylabel('Mean-path drawdown (%)')
  for ax in axes[:,j]:ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25)
 fig.suptitle('Train-only augmentation; raw2024 evaluation — unadjusted price cost scope, no total return');fig.tight_layout();fig.savefig(out/'priority_nav_drawdown.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for (sec,algo,var),g in df[(df.phase==1)&df.variant.isin(['A0','Vanilla'])].groupby(['sector','algorithm','variant']):
  g=g.groupby('fee')[['AR','SR','MDD']].mean().reindex(list(FEES_ORDER))
  for ax,col in zip(axes,['AR','SR','MDD']):ax.plot(g.index,g[col]*(100 if col!='SR' else 1),marker='o',label=sec+' '+algo+' '+var);ax.set_title(col+(' (%)' if col!='SR' else ''));ax.grid(alpha=.2)
 axes[0].legend(fontsize=5);fig.suptitle('Fixed-policy cost sensitivity — unadjusted accounting scope');fig.tight_layout();fig.savefig(out/'priority_cost_sensitivity.png',dpi=150);plt.close(fig)
FEES_ORDER=['F0','F1','F2','F3','F4','F5','F6']


def report(make_plots=True):
 context();RUN.joinpath('results').mkdir(exist_ok=True);df=collect_metrics(RUN/'results/evaluations');baseline=collect_metrics(RUN/'results/nonrl')
 cost=cost_differences(df);write_table(RUN/'results/summary_main.csv',df[df.variant.isin(['Vanilla','A0'])] if not df.empty else df);write_table(RUN/'results/cost_sensitivity.csv',cost);write_table(RUN/'results/main_mean_sd.csv',summary(cost));write_table(RUN/'results/summary_ablation_fixed_ppo.csv',df[(df.algorithm=='ppo')&df.variant.isin(['A0','A1','A2','A6','A7','A8','A9'])] if not df.empty else df);write_table(RUN/'results/ablation_mean_sd.csv',summary(df[(df.algorithm=='ppo')&df.fee.isin(['F0','F1'])] if not df.empty else df));write_table(RUN/'results/baseline_comparison.csv',baseline);write_table(RUN/'results/nonrl_cost_sensitivity.csv',cost_differences(baseline));write_table(RUN/'results/paired_bootstrap.csv',bootstrap(df))
 last={}
 for r in records():last[r['task_id']]=r
 save_csv(RUN/'run_status.csv',list(last.values()));counts={}
 for k in ['actual_real_data_train','actual_real_data_eval','real_api_short_run','nonrl_fit','nonrl_eval','regression_tests']:
  counts[k]=sum(r['status']=='completed' and r.get('kind')==k for r in last.values())
 counts.update(failed=sum(r['status']=='failed' for r in last.values()),blocked=sum(r['status']=='blocked' for r in last.values()),economic_return_validated=False,accounting_scope=SCOPE)
 save_json(RUN/'completion_counts.json',counts)
 coverage=[]
 for item,kind in [('R1.2','cost empirical scope-limited'),('R2.7','cost empirical scope-limited'),('R4/R5 costs','cost empirical scope-limited'),('R1.3/R2.6','independent validation and fixed PPO'),('R1.4/R2.5','paired RL seeds and block intervals'),('R5.4','Ridge same-training-input controls')]:coverage.append(dict(reviewer_item=item,status='computation_completed_scope_limited' if counts['actual_real_data_train'] else 'running',response=kind,fully_resolved=False,accounting_scope=SCOPE,remaining='No full economic return or complete reality-constraint claim; unknown frozen enhancer provenance disclosed'))
 for item in ['fusion_initialization','cross-stock_attention','full_predictor_seed_retraining','graph_definition_and_cutoffs','reward_grid','unseen_stocks']:coverage.append(dict(reviewer_item=item,status='not_run_or_unresolved',fully_resolved=False,remaining='Outside authorized retraining scope'))
 save_csv(RUN/'REVIEWER_STATUS.csv',coverage)
 lines=['# v3 实际执行摘要','',json.dumps(counts,ensure_ascii=False,indent=2),'','MGT训练观测仅2022增强；2023验证与2024测试及其预热历史对所有策略完全相同且全真实。真实账本在训练/验证/测试始终使用raw。','', 'Tushare不复权来自author_confirmed；公司行动/现金分红尚未纳入。结果是unadjusted_price_cost_sensitivity，cost_analysis_scope_limited，非total_return，不声称完整经济收益或全面实盘约束。2024为已见固定历史评估期。','', '## 重点比较F1实测（各seed均保留）']
 if not df.empty:
  for (sec,algo,var),g in cost[(cost.phase==1)&(cost.fee=='F1')].groupby(['sector','algorithm','variant']):lines.append('- %s %s %s：n=%d；AR均值 %.4f；SR %.4f；MDD %.4f；相对F0 delta_AR %.4f个百分点、delta_SR %.4f、delta_MDD %.4f个百分点。'%(sec,algo,var,len(g),g.AR.mean(),g.SR.mean(),g.MDD.mean(),g.delta_AR_vs_F0_pp.mean(),g.delta_SR_vs_F0.mean(),g.delta_MDD_vs_F0_pp.mean()))
 lines+=['','## 解释边界','不同表示消融是训练增强输入的比较，不是测试时实时图删除干预；三RL种子条件于固定merge，不是完整预测流水线重训。配对bootstrap共用日期块，逐seed差再平均，无p值。非正预测不裁剪，不用于账本；具体非有限状态只暂停对应变体。','', 'R1.2已有计算也须附上述会计限制，不能把全部现实约束勾为fully_resolved。未知预测器/图来源不足不被误称为已证实评估污染；也不宣称无泄漏。R1.1作者已处理。','', '论文对应实现表述：MGT-derived observations are used during policy training in 2022. Validation and testing use observed market data for all policies. The comparison therefore evaluates the effect of forecast-augmented training on subsequent portfolio performance.']
 (RUN/'SUMMARY_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 if make_plots:plots(df)
 print('V3 actual counts',counts,flush=True);return counts


def package(name='results_to_send_v3.zip'):
 integrity();target=ROOT/name
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for f in RUN.rglob('*'):
   if not f.is_file() or f.suffix in ['.npz','.zip','.joblib','.pyc'] or 'prepared' in f.parts or 'bootstrap_cache' in f.parts:continue
   z.write(f,'runs/v3_train_only/'+f.relative_to(RUN).as_posix())
  for n in list(load_json(RUN/'protocol_v3.json')['code'])+['run_all_v3.ps1']:
   if (ROOT/n).exists():z.write(ROOT/n,n)
 print('Saved',target,flush=True)
