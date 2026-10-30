"""Read-only result aggregation. Never trains, modifies checkpoints or rewrites trajectories."""
import json,pathlib,zipfile
import numpy as np
import pandas as pd
from v2_audit import ROOT,RUN,PROJECT,save_json,save_csv,load_json,sha,canonical,integrity,code_manifest
from v2_engine import records,current_context,cache_valid,assert_locked_context
from backtest import paired_bootstrap,metrics

METRIC_COLS=['run_id','protocol_id','sector','variant','algorithm','seed','fee','paper_eligible','diagnostic_only','price_basis','input_hash','code_hash','checkpoint_sha256','AR','SR','SR_rf02','MDD','Vol','Calmar','final_nav','cumulative_cost','mean_gross_turnover','annualized_gross_turnover','first_date','last_date','n']

def collect_metrics(base):
 rows=[]
 for f in pathlib.Path(base).rglob('metrics.json'):
  receipt=f.parent/'receipt.json'
  if not receipt.exists():continue
  j=load_json(receipt)
  if j.get('status')!='completed':continue
  if not j.get('artifacts') or not all((f.parent/name).is_file() and sha(f.parent/name)==h for name,h in j['artifacts'].items()):raise RuntimeError('Artifact checksum mismatch: '+str(f.parent))
  m=load_json(f);m['directory']=str(f.parent);rows.append(m)
 return pd.DataFrame(rows) if rows else pd.DataFrame(columns=METRIC_COLS+['directory'])

def cost_differences(df):
 if df.empty:return pd.DataFrame(columns=METRIC_COLS+['delta_AR_vs_F0_pp','delta_SR_vs_F0','delta_MDD_vs_F0_pp','delta_AR_vs_F1_pp','delta_SR_vs_F1','delta_MDD_vs_F1_pp'])
 rows=[]
 for keys,g in df.groupby(['sector','variant','algorithm','seed','first_date','last_date']):
  if g.checkpoint_sha256.nunique()!=1:raise ValueError('Fee comparisons require one fixed checkpoint')
  base=g.set_index('fee')
  if base.index.has_duplicates:raise ValueError('Duplicate fee outputs; do not silently choose')
  for _,r in g.iterrows():
   row=r.to_dict()
   for anchor in ['F0','F1']:
    for m in ['AR','SR','MDD']:
     row['delta_'+m+'_vs_'+anchor+('_pp' if m!='SR' else '')]=float((r[m]-base.loc[anchor,m])*(100 if m!='SR' else 1)) if anchor in base.index else np.nan
   rows.append(row)
 return pd.DataFrame(rows)

def summarize(df):
 if df.empty:return pd.DataFrame(columns=['protocol_id','sector','variant','algorithm','fee','seeds','paper_eligible','diagnostic_only','AR_mean','AR_std','SR_mean','SR_std','MDD_mean','MDD_std'])
 cols=[c for c in ['AR','SR','SR_rf02','MDD','Vol','Calmar','final_nav','cumulative_cost','mean_gross_turnover','annualized_gross_turnover','delta_AR_vs_F0_pp','delta_SR_vs_F0','delta_MDD_vs_F0_pp','delta_AR_vs_F1_pp','delta_SR_vs_F1','delta_MDD_vs_F1_pp'] if c in df]
 rows=[]
 for key,g in df.groupby(['protocol_id','sector','variant','algorithm','fee','paper_eligible','diagnostic_only','price_basis']):
  row=dict(zip(['protocol_id','sector','variant','algorithm','fee','paper_eligible','diagnostic_only','price_basis'],key));row.update(seeds=len(g),run_ids=';'.join(g.run_id),input_hash=';'.join(sorted(g.input_hash.unique())),code_hash=';'.join(sorted(g.code_hash.unique())),checkpoint_hashes=';'.join(sorted(g.checkpoint_sha256.unique())))
  for c in cols:row[c+'_mean']=float(g[c].mean());row[c+'_std']=float(g[c].std(ddof=1)) if len(g)>1 else np.nan
  rows.append(row)
 return pd.DataFrame(rows)

def write_table(f,df):
 f=pathlib.Path(f);f.parent.mkdir(exist_ok=True,parents=True);df.to_csv(f,index=False,encoding='utf-8-sig')
 view=df.drop(columns=[c for c in ['directory','context'] if c in df])
 if len(view.columns)>12:
  desired=[c for c in ['sector','variant','algorithm','seed','fee','paper_eligible','diagnostic_only','AR','SR','MDD','AR_mean','SR_mean','MDD_mean','cumulative_cost','delta_AR_vs_F0_pp','delta_SR_vs_F0','delta_MDD_vs_F0_pp'] if c in view];view=view[desired]
 if view.empty:tex='% No eligible completed results. See corresponding status files.\n'
 else:tex=view.to_latex(index=False,na_rep='NA',float_format='%.6f')
 f.with_suffix('.tex').write_text(tex,encoding='utf-8')

def paired_results(formal,p):
 rows=[];status=[]
 for sector,algo in p['phase1']:
  for fee in ['F1','F0']:
   series_a=[];series_b=[];dates=None;ok=True
   for seed in p['seeds']:
    pair=[]
    for variant in ['A0','Vanilla']:
     x=formal[(formal.sector==sector)&(formal.algorithm==algo)&(formal.seed==seed)&(formal.variant==variant)&(formal.fee==fee)]
     if len(x)!=1:ok=False;break
     data=pd.read_csv(pathlib.Path(x.iloc[0]['directory'])/'trajectory.csv');date=pd.DatetimeIndex(pd.to_datetime(data.date.iloc[1:]));ret=data.daily_return.iloc[1:].to_numpy(float)
     if dates is None:dates=date
     if not dates.equals(date):raise ValueError('Paired actual dates differ')
     pair.append(ret)
    if not ok:break
    series_a.append(pair[0]);series_b.append(pair[1])
   if not ok:status.append(dict(sector=sector,algorithm=algo,fee=fee,status='blocked',reason='Requires completed eligible A0 and Vanilla trajectories for each paired seed; no diagnostic-to-formal substitution'));continue
   a=np.stack(series_a,axis=1);b=np.stack(series_b,axis=1)
   for block in p['bootstrap']['blocks']:
    x=paired_bootstrap(a,b,block,p['bootstrap']['B'],p['bootstrap']['seed'])
    for r in x.to_dict('records'):rows.append(dict(**r,sector=sector,algorithm=algo,fee=fee,comparison='A0 - Vanilla',protocol_id=p['protocol_id'],paper_eligible=True,input_hash=p['input_manifest_sha256'],code_hash=canonical(p['code']),conditional='fixed forecasts/strategies/pool/period; shared date blocks across seeds; no p-values'))
   status.append(dict(sector=sector,algorithm=algo,fee=fee,status='completed',reason='Matched date/seed/algorithm/context'))
 save_csv(RUN/'formal/bootstrap_status.csv',status)
 return pd.DataFrame(rows,columns=None if rows else ['sector','algorithm','fee','metric','effect','ci_low','ci_high','block','B','protocol_id','paper_eligible'])

def plot_diagnostics(diag):
 if diag.empty:return []
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=RUN/'diagnostics/figures';out.mkdir(exist_ok=True,parents=True);paths=[]
 fig,axes=plt.subplots(2,2,figsize=(12,7),sharex='col')
 for col,sector in enumerate(['bank','healthcare']):
  sub=diag[(diag.sector==sector)&diag.fee.isin(['F0','F1'])]
  for _,r in sub.iterrows():
   z=pd.read_csv(pathlib.Path(r.directory)/'trajectory.csv');dates=pd.to_datetime(z.date);nav=z.nav.to_numpy();label=r['variant']+' '+r['fee'];axes[0,col].plot(dates,nav/1e6,label=label);axes[1,col].plot(dates,100*(1-nav/np.maximum.accumulate(nav)),label=label)
  axes[0,col].set_title(sector+' — unverified price basis');axes[0,col].set_ylabel('Price-sequence NAV / initial capital');axes[1,col].set_ylabel('Price-sequence drawdown (%)');axes[0,col].legend(fontsize=7)
  for ax in axes[:,col]:ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25)
 fig.suptitle('NUMERICAL DIAGNOSTICS ONLY — not economic investment returns');fig.tight_layout();f=out/'diagnostic_nav_drawdown.png';fig.savefig(f,dpi=160);plt.close(fig);paths.append(f)
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for (sector,variant),g in diag.groupby(['sector','variant']):
  g=g.set_index('fee').reindex(['F0','F1','F2','F3','F4','F5','F6'])
  for ax,m in zip(axes,['AR','SR','MDD']):ax.plot(g.index,g[m]*(100 if m!='SR' else 1),marker='o',label=sector+' '+variant);ax.set_title('Sequence '+m+(' (%)' if m!='SR' else ''));ax.grid(alpha=.2)
 axes[0].legend(fontsize=6);fig.suptitle('Cost-scenario accounting diagnostics — price basis unverified');fig.tight_layout();f=out/'diagnostic_cost_sensitivity.png';fig.savefig(f,dpi=160);plt.close(fig);paths.append(f)
 return paths

def reviewer_status(counts):
 raw=pd.read_csv(ROOT/'archive/v1/reviewer_coverage.csv');rows=[]
 for r in raw.to_dict('records'):
  i=r['reviewer_item'];s='partially_addressed';reason='Implementation and protocol corrected; empirical comparisons require eligible data';fully=False
  if i=='R1.1':s='not_run';reason='Author handled; excluded from this continuation'
  elif i in ['R1.6','R1.8','R2.11'] or any(t in i for t in ['深度','奖励网格','未见股票','大规模']):s='not_run';reason='Explicitly outside scope; no new prediction/model experiments'
  elif any(t in i for t in ['新文献','创新','数学','引用','Code availability','数据选择']):s='not_run';reason='Full manuscript/reviewer source absent; checklist only'
  elif i in ['R2.2','R5.1','R2.1','R2.3','R2.4','R4时序','R5时序']:s='blocked';reason='Frozen forecast/scaler/graph provenance unresolved; raw medical prices do not cure this'
  elif i=='R2.8':s='blocked';reason='Dates alone do not certify forecast scale/strict held-out source; no prediction retraining'
  elif i in ['R1.2','R2.7','R4成本与约束','R5.4']:
   s='empirical_completed' if counts.get('paired_priority_complete') else 'partially_addressed';reason='Real learning/evaluation APIs and cost ledger implemented; '+str(counts.get('diagnostic_evaluations',0))+' B-class baseline diagnostics do not complete R1.2 main evidence'
  rows.append(dict(reviewer_item=i,status=s,fully_resolved=fully,scope=r['scope'],evidence='runs/v2/README.md; dependency_status.csv; execution_records.jsonl',remaining=reason))
 save_csv(RUN/'REVIEWER_STATUS.csv',rows)


def report():
 p=assert_locked_context();ctx=current_context();plan=pd.read_csv(RUN/'task_plan.csv',keep_default_na=False);events=records();last={}
 for event in events:last[event['task_id']]=event
 statuses=[]
 for t in plan.to_dict('records'):
  ev=last.get(t['task_id'],{});statuses.append({**t,**ev,'status':ev.get('status','not_run'),'paper_eligible':ev.get('paper_eligible',False),'reason':ev.get('reason','' if ev.get('status')=='completed' else 'No execution receipt yet')})
 # Verification execution is counted independently of scientific task plan.
 for task,ev in last.items():
  if task not in set(plan.task_id):statuses.append(ev)
 save_csv(RUN/'run_status.csv',statuses)
 formal=collect_metrics(RUN/'formal/evaluations');diag=collect_metrics(RUN/'diagnostics/baselines')
 if not formal.empty and (not formal.paper_eligible.all() or formal.diagnostic_only.any()):raise RuntimeError('Noneligible metric attempted to enter formal tables')
 if not diag.empty and (diag.paper_eligible.any() or not diag.diagnostic_only.all()):raise RuntimeError('Diagnostic classification mismatch')
 formal_cost=cost_differences(formal);diag_cost=cost_differences(diag)
 write_table(RUN/'formal/summary_main.csv',formal[formal.variant.isin(['A0','Vanilla'])] if not formal.empty else formal)
 write_table(RUN/'formal/cost_sensitivity.csv',formal_cost)
 write_table(RUN/'formal/summary_main_mean_sd.csv',summarize(formal_cost))
 write_table(RUN/'formal/summary_ablation_fixed_ppo.csv',formal[(formal.algorithm=='ppo')&formal.variant.isin(['A0']+p['ablation'])] if not formal.empty else formal)
 write_table(RUN/'formal/baseline_comparison.csv',formal[formal.algorithm=='none'] if not formal.empty else formal)
 boot=paired_results(formal,p)
 write_table(RUN/'formal/paired_bootstrap.csv',boot)
 write_table(RUN/'diagnostics/baseline_comparison.csv',diag)
 write_table(RUN/'diagnostics/cost_sensitivity.csv',diag_cost)
 write_table(RUN/'diagnostics/cost_summary_mean_sd.csv',summarize(diag_cost))
 descriptions=[]
 for _,r in pd.concat([formal,diag],ignore_index=True).iterrows():
  z=pd.read_csv(pathlib.Path(r.directory)/'trajectory.csv');z['date']=pd.to_datetime(z.date)
  for label,mask in [('H1',z.date.dt.month<=6),('H2',z.date.dt.month>6)]:
   rets=z.loc[mask,'daily_return'].dropna().to_numpy(float)
   if len(rets):descriptions.append(dict(run_id=r.run_id,protocol_id=r.protocol_id,sector=r.sector,variant=r['variant'],fee=r.fee,period=label,paper_eligible=r.paper_eligible,diagnostic_only=r.diagnostic_only,observations=len(rets),cumulative_sequence_change=float(np.prod(1+rets)-1),mean_daily_return=float(rets.mean()),annualized_volatility=float(np.std(rets,ddof=1)*np.sqrt(252)),interpretation='Contiguous trajectory slice only; no retraining/walk-forward/causal market explanation'))
 save_csv(RUN/'descriptive_slices.csv',descriptions)
 stats={}
 for row in statuses:
  key=(row.get('kind',''),row['status']);stats[key]=stats.get(key,0)+1
 counts=dict(formal_trainings=stats.get(('train','completed'),0),formal_evaluations=stats.get(('formal_evaluation','completed'),0)+stats.get(('formal_baseline','completed'),0),diagnostic_evaluations=stats.get(('price_sequence_diagnostic','completed'),0),real_api_short_runs=stats.get(('real_api_short_run','completed'),0),synthetic_tests_this_phase=0,regression_suites=stats.get(('regression_tests','completed'),0),blocked=sum(v for (k,s),v in stats.items() if s=='blocked'),failed=sum(v for (k,s),v in stats.items() if s=='failed'),paired_priority_complete=False)
 priority=plan[(plan['kind'].isin(['train','evaluate']))&(plan.phase==1)]
 counts['paired_priority_complete']=bool(len(priority)>0 and all(last.get(t,{}).get('status')=='completed' and last[t].get('paper_eligible',False) for t in priority.task_id))
 save_json(RUN/'completion_counts.json',counts);reviewer_status(counts)
 parts={'main':formal[formal.variant.isin(['A0','Vanilla'])], 'cost_sensitivity':formal_cost, 'fixed_ppo':formal[(formal.algorithm=='ppo')&formal.variant.isin(['A0']+p['ablation'])], 'baseline':formal[formal.algorithm=='none'], 'bootstrap':boot}
 save_csv(RUN/'formal/result_status.csv',[dict(table=n,status='completed' if not table.empty else 'blocked',reason='No eligible completed rows' if table.empty else 'Completed rows only; remaining tasks explicit',paper_eligible=not table.empty) for n,table in parts.items()])
 save_csv(RUN/'prediction_metrics_status.csv',[dict(sector=s,status='blocked',reason='Independent raw now present; forecast mapping/scale and temporal provenance still not fully certified. No refit or inference used.') for s in p['market']])
 paths=plot_diagnostics(diag)
 integrity()
 changes=pd.read_csv(RUN/'input_delta.csv');delta_counts=changes.change.value_counts().to_dict();profiles=pd.read_csv(RUN/'data_profiles.csv')
 txt=['# v2 执行结果（实际记录）','', '医疗实际文件：`'+p['market']['healthcare']['raw']+'`。医疗 raw 缺失已解除，全部医疗 merge 均按 date/tic 与此独立真实表一对一对齐。','', '正式训练：%d；正式评估：%d；真实价格数值诊断：%d；真实API短跑：%d。'%(counts['formal_trainings'],counts['formal_evaluations'],counts['diagnostic_evaluations'],counts['real_api_short_runs']),'本轮只执行新增回归，不重复五算法合成训练；v1合成训练次数不并入v2。','', '## 当前结论','', '两行业行情均结构有效，但项目导出代码、表头、已有记录仍不能认证复权/分红/拆股口径。Vanilla不依赖预测，单独因为价格计价口径未证实而暂停；MGT另受2023/2024覆盖、日期/尺度及预处理时序约束。无open不是本轮阻塞项：采用已定义的次日收盘延迟执行。','', '数值诊断仅衡量给定价格序列的费用与账本敏感性；paper_eligible=false、diagnostic_only=true、price_basis=unverified，不能作为经济投资收益或R1.2主要实证。没有用allow_unverified放行RL。','', '## 费用结果（数学价格序列诊断）','']
 if not diag_cost.empty:
  for (sec,v),g in diag_cost.groupby(['sector','variant']):
   a=g[g.fee=='F1'].iloc[0];txt.append('- %s / %s：F1累计现金费用 %.2f；相对F0的年化序列指标变化 %.4f 个百分点、SR变化 %.4f、MDD变化 %.4f 个百分点。'%(sec,v,a.cumulative_cost,a.delta_AR_vs_F0_pp,a.delta_SR_vs_F0,a.delta_MDD_vs_F0_pp))
 txt+=['','费用全部为预设情景（逐侧c/s+假定A股历史卖出税），非券商报价。税种应用属于B类情景假设，不声称银行别名已经证券主表认证。每档从全现金重新闭环；未强制清仓；不预设指标单调恶化。','', '## 原始输入与剩余缺项','', 'v2研究输入增量：'+str(delta_counts)+'。说明文件另列instruction_delta.csv；旧指令删除不当作行情篡改。既有原始研究输入是否修改/删除见input_delta.csv。input_integrity.csv验证本轮运行期间原始输入不变。','', '最小补件：两份行情各自对应的导出/复权与分红拆股来源（或可认证总回报指数口径）；银行证券别名映射；A0及各消融2023/2024日期化预测、冻结scaler/生成记录与图信息截止，且需能支持2022训练所需的因果特征。2022预测不能平移，已证实预处理泄漏不能靠lag修复。','', 'R1.2/R2.7/R4/R5费用要求：程序实现与B类诊断已完成，MGT正式成本实证尚未完成；R1.3验证选择规则已实现，R1.4/R2.5多seed/配对统计缺正式轨迹。R1.1由作者处理，本轮未重复。其余见REVIEWER_STATUS.csv。','', '## 文件位置','', '- formal/：只接受已完成且paper_eligible=true的真实任务；本轮若为空则仅列字段，不填零或假数。','- diagnostics/：真实价格序列基准、费用表、逐日账本/持仓与明确标记的图。','- synthetic/：新增回归测试日志，未启动合成策略训练。','- models/：正式训练和checkpoint所在位置（如无合格输入则不存在权重）。','- execution_records.jsonl：追加式实际执行事件；report只读模型/轨迹，不运行训练。']
 (RUN/'SUMMARY_ZH.md').write_text('\n'.join(txt)+'\n',encoding='utf-8')
 print('Reported actual completed counts:',counts,flush=True)
 return counts


def package():
 target=ROOT/'results_to_send_v2.zip';selected=[]
 for f in RUN.rglob('*'):
  if not f.is_file() or f.suffix in ['.zip','.npz','.pyc'] or 'prepared' in f.relative_to(RUN).parts or f.name.endswith('.tmp'):continue
  if 'models' in f.relative_to(RUN).parts and f.suffix not in ['.json','.csv','.log','.txt']:continue
  selected.append((f,'runs/v2/'+f.relative_to(RUN).as_posix()))
 for n in ['run_revision.py','backtest.py','v2_audit.py','v2_data.py','v2_engine.py','v2_reports.py','test_v2.py','test_backtest.py','run_all_v2.ps1','README.md','build_delivery.py','supplemental_audit.py','V2_CHANGES.md']:
  f=ROOT/n
  if f.exists():selected.append((f,n))
 for n in ['protocol.json','input_hashes.csv','notebook_evidence.csv','graph_review.md','ARCHIVE_COMPLETE.json']:
  f=ROOT/'archive/v1'/n
  if f.exists():selected.append((f,'archive/v1/'+n))
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for f,name in selected:z.write(f,name)
 with zipfile.ZipFile(target) as z:
  if z.testzip() is not None:raise RuntimeError('ZIP integrity failed')
 save_json(RUN/'delivery_receipt.json',dict(archive=target.name,sha256=sha(target),entries=len(selected),excludes=['commercial raw tables','prepared price arrays','model weights','environment directories','full notebook outputs'],note='Models/checkpoints retained locally; metadata and validation/training logs included'))
 print('Packaged',target,flush=True)
