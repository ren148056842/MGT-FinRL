"""v4 read-only summaries (detailed analysis is done from the exported evaluations)."""
import pathlib,json,zipfile
import numpy as np
import pandas as pd
from v4_audit import ROOT,RUN,V3RUN,SCOPE,sha,load_json,save_json,save_csv,integrity
from v4_engine import records,context
from v2_reports import collect_metrics,cost_differences,write_table


def summary(df):
 keys=['sector','variant','algorithm','fee'];rows=[]
 for key,g in df.groupby(keys):
  row=dict(zip(keys,key));row.update(n=len(g))
  for col in ['AR','SR','MDD','Calmar','cumulative_cost','mean_gross_turnover']:
   if col in g:row[col+'_mean']=g[col].mean();row[col+'_std']=g[col].std(ddof=1) if len(g)>1 else np.nan
  rows.append(row)
 return pd.DataFrame(rows)


def report():
 context();RUN.joinpath('results').mkdir(exist_ok=True)
 v4=collect_metrics(RUN/'results/evaluations');v3=collect_metrics(V3RUN/'results/evaluations')
 v3=v3[v3.variant=='Vanilla'] if not v3.empty else v3
 rl=pd.concat([v3.assign(source='v3_vanilla'),v4.assign(source='v4')],ignore_index=True) if not v4.empty else v3.assign(source='v3_vanilla')
 base=collect_metrics(RUN/'results/nonrl')
 write_table(RUN/'results/rl_all_runs.csv',rl);write_table(RUN/'results/nonrl_all.csv',base)
 if not rl.empty:write_table(RUN/'results/rl_mean_sd.csv',summary(rl))
 if not base.empty:write_table(RUN/'results/nonrl_summary.csv',base[['sector','variant','fee','AR','SR','MDD','Calmar','cumulative_cost','mean_gross_turnover']])
 last={}
 for r in records():last[r['task_id']]=r
 save_csv(RUN/'run_status.csv',list(last.values()));counts={}
 for k in ['actual_real_data_train','actual_real_data_eval','real_api_short_run','nonrl_fit','nonrl_eval','regression_tests']:
  counts[k]=sum(r['status']=='completed' and r.get('kind')==k for r in last.values())
 counts.update(failed=sum(r['status']=='failed' for r in last.values()),blocked=sum(r['status']=='blocked' for r in last.values()),v3_vanilla_evaluations=int(len(v3)))
 save_json(RUN/'completion_counts.json',counts);print('V4 counts',counts,flush=True);return counts


def package(name=None):
 target=ROOT/(name or 'results_to_send_%s.zip'%RUN.name)
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for f in RUN.rglob('*'):
   if not f.is_file() or f.suffix in ['.npz','.zip','.joblib','.pyc'] or 'prepared' in f.parts or f.name=='positions.csv':continue
   z.write(f,'runs/'+RUN.name+'/'+f.relative_to(RUN).as_posix())
 print('Saved',target,flush=True)
