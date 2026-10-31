"""Read-only verification of completed artifacts; no training or checkpoint mutation."""
import sys,pathlib,json,hashlib
import numpy as np
import pandas as pd
ROOT=pathlib.Path(r'C:\Users\perfe\Desktop\2\xiu\revision_backtest')
sys.path.insert(0,str(ROOT))
from v3_audit import RUN,save_json,save_csv,sha
from v3_data import load_prepared
from backtest import FEES,metrics

def verify():
 rows=[];fail=[];data={}
 for parent in sorted((RUN/'models').glob('*')):
  if not (parent/'receipt.json').exists():continue
  try:
   receipt=json.loads((parent/'receipt.json').read_text(encoding='utf-8'))
   for name,h in receipt['artifacts'].items():assert sha(parent/name)==h, name+' hash mismatch'
   meta=json.loads((parent/'checkpoint_metadata.json').read_text(encoding='utf-8'));v=pd.read_csv(parent/'validation.csv')
   score=v.validation_SR.replace([np.inf,-np.inf],np.nan).fillna(-np.inf)
   expected=v.loc[score.idxmax()]
   assert expected.checkpoint==meta['selected_checkpoint'],'selection not earliest maximum validation Sharpe'
   assert sha(parent/'best_policy.zip')==expected.checkpoint_sha256
   assert len(v)==5,'expected five prespecified checkpoints'
   rows.append(dict(task_id=parent.name,kind='training',status='passed',steps=meta['num_timesteps']))
  except Exception as e:fail.append(dict(task_id=parent.name,error=repr(e)))
 for directory in sorted((RUN/'results/evaluations').glob('*')):
  if not (directory/'receipt.json').exists():continue
  try:
   receipt=json.loads((directory/'receipt.json').read_text(encoding='utf-8'))
   for name,h in receipt['artifacts'].items():assert sha(directory/name)==h,name+' hash mismatch'
   t=pd.read_csv(directory/'trajectory.csv');pos=pd.read_csv(directory/'positions.csv');m=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
   sec=m['sector']
   if sec not in data:data[sec]=load_prepared(sec,'Vanilla').subset('2024-01-01','2024-12-31')
   d=data[sec];dates=pd.to_datetime(t.date)
   assert np.array_equal(dates.values,d.dates.values),'raw date alignment'
   q=pos.pivot(index='date',columns='tic',values='quantity').reindex(index=t.date,columns=d.tickers).to_numpy()
   np.testing.assert_allclose(t.nav,t.cash+(q*d.prices).sum(axis=1),rtol=1e-11,atol=1e-5)
   assert t.cash.min()>=-1e-6 and q.min()>=-1e-10
   np.testing.assert_allclose(t.cost,t.commission+t.slippage+t.tax,atol=1e-8)
   np.testing.assert_allclose(t.pre_nav-t.cost,t.nav,rtol=1e-11,atol=1e-5)
   np.testing.assert_allclose(t.nav.iloc[1:].to_numpy()/t.nav.iloc[:-1].to_numpy()-1,t.daily_return.iloc[1:],atol=1e-12)
   np.testing.assert_allclose(t.reward.iloc[1:],.1*t.daily_return.iloc[1:],atol=1e-12)
   np.testing.assert_allclose(t.gross_turnover,(t.buy+t.sell)/t.pre_nav,atol=1e-12)
   assert t.decision_date.iloc[1:].tolist()==t.date.iloc[:-1].tolist(),'one-close delay'
   fee=FEES[m['fee']]
   np.testing.assert_allclose(t.commission,fee.commission*(t.buy+t.sell),atol=1e-8)
   np.testing.assert_allclose(t.slippage,fee.slippage*(t.buy+t.sell),atol=1e-8)
   assert np.isclose(t.nav.iloc[0],1e6) and np.isclose(t.cash.iloc[0],1e6)
   calc=metrics(t.daily_return.iloc[1:]);calc['SR_rf02']=metrics(t.daily_return.iloc[1:],.02)['SR']
   for key in ['AR','SR','SR_rf02','MDD','Vol','Calmar']:np.testing.assert_allclose(calc[key],m[key],rtol=1e-9,atol=1e-11)
   ckpt=RUN/'models'/m['parent']/'best_policy.zip';assert sha(ckpt)==m['checkpoint_sha256']
   rows.append(dict(task_id=directory.name,kind='evaluation',status='passed',steps=len(t)-1))
  except Exception as e:fail.append(dict(task_id=directory.name,error=repr(e)))
 out=RUN/'verification';out.mkdir(exist_ok=True)
 save_csv(out/'completed_artifact_checks.csv',rows);save_csv(out/'completed_artifact_failures.csv',fail,columns=['task_id','error'])
 result=dict(completed_artifacts_checked=len(rows),failures=fail,source_sha256=sha(__file__),scope='Completed receipts only; active model excluded; no training or source modifications')
 save_json(out/'completed_artifact_checks.json',result);print(json.dumps(result,ensure_ascii=False))
 return not fail
if __name__=='__main__':sys.exit(0 if verify() else 1)
