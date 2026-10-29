"""v4 SB3 training (MGT variants only; Vanilla reused from v3), fee evaluation, non-RL controls incl. forecast-ranking rules."""
import math,pathlib,shutil,time,traceback,json,gc,datetime
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback
from v4_audit import ROOT,PROJECT,RUN,V3RUN,FORECASTS,SCOPE,VARIANTS,E1_VARIANTS,RULES,sha,canonical,save_json,save_csv,load_json,code_manifest,integrity,canonicalize
from v4_data import RealPortfolioEnv,load_prepared,export_trace,load_episode
from v2_engine import algorithms,stats_from_env,cache_valid,finish_receipt,receipt
from backtest import FEES,validate_splits,metrics

class Progress(BaseCallback):
 def __init__(self,task):super().__init__();self.task=task
 def _on_step(self):
  if self.num_timesteps%2000==0:save_json(RUN/'progress.json',dict(task=self.task,num_timesteps=self.num_timesteps,status='training',time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  if self.num_timesteps%5000==0:print('PROGRESS',self.task,self.num_timesteps,flush=True)
  return True

def records():
 f=RUN/'execution_records.jsonl'
 return [json.loads(x) for x in f.read_text(encoding='utf-8').splitlines() if x.strip()] if f.exists() else []

def record(task,status,**kwargs):
 row=dict(task_id=task,status=status,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accounting_scope=SCOPE,**kwargs)
 with (RUN/'execution_records.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(row,default=str,ensure_ascii=False)+'\n');out.flush()
 return row

def context():
 p=load_json(RUN/'protocol_v4.json')
 if p['code']!=code_manifest():raise RuntimeError('Locked v4 code changed')
 return dict(protocol_sha256=sha(RUN/'protocol_v4.json'),input_sha256=sha(RUN/'input_manifest.csv'),code_sha256=canonical(p['code']),environment_sha256=sha(RUN/'environment.json'))

def plan():
 p=load_json(RUN/'protocol_v4.json');rows=[];priority={tuple(v) for v in p['phase1']}
 for sec in ['bank','healthcare']:
  for variant in p['e1_variants']:
   for algo in p['algorithms']:
    for seed in p['seeds']:
     rows.append(dict(task_id='E1_%s_%s_%s_s%d'%(sec,variant,algo,seed),sector=sec,variant=variant,algorithm=algo,seed=seed,steps=p['steps'][algo],phase=1 if (sec,algo) in priority else 2,fees=list(FEES)))
  for variant in p['ablation']:
   for seed in p['seeds']:rows.append(dict(task_id='E2_%s_%s_ppo_s%d'%(sec,variant,seed),sector=sec,variant=variant,algorithm='ppo',seed=seed,steps=p['steps']['ppo'],phase=3,fees=['F0','F1']))
 rows.sort(key=lambda x:({'ppo':0,'a2c':1,'td3':2,'sac':3,'ddpg':4}[x['algorithm']],x['phase'] if x['phase']==3 else 1,x['sector'],x['seed'],x['variant']))
 save_json(RUN/'task_plan.json',rows);return rows

def ready(task):
 f=RUN/'preparation_status.csv';d=pd.read_csv(f);x=d[(d.sector==task['sector'])&(d.variant==task['variant'])]
 return bool(len(x)==1 and x.iloc[0]['status']=='ready'),str(x.iloc[0]['reason']) if len(x)==1 else 'No prepared phase objects'

def rollout_model(model,data,fee='F1'):
 env=RealPortfolioEnv(data,FEES[fee]);obs,_=env.reset();done=False
 while not done:
  action,_=model.predict(obs,deterministic=True);obs,_,done,_,_=env.step(action)
 return env

def train_model(task,p,directory,short=False):
 """Actual SB3 .learn and checkpoint saving; no use of test data for checkpoint selection."""
 import torch
 torch.set_num_threads(1);cls=algorithms()[task['algorithm']];data=load_prepared(task['sector'],task['variant']);split=p['splits']
 train=data.subset(*split['train']);val=data.subset(*split['validation']);test_dates=data.dates[(data.dates>=pd.Timestamp(split['test'][0]))&(data.dates<=pd.Timestamp(split['test'][1]))]
 validate_splits(train.dates,val.dates,test_dates)
 env=RealPortfolioEnv(train,FEES['F1']);params=p['model_params'][task['algorithm']].copy();params.update(seed=int(task['seed']),device='cpu',verbose=0)
 model=cls('MlpPolicy',env,**params);nsteps=int(getattr(model,'n_steps',1));target=256 if short else int(task['steps']);actual_target=int(math.ceil(target/nsteps)*nsteps) if task['algorithm'] in ['a2c','ppo'] else target
 checkpoints=sorted(set(int(math.ceil(actual_target*frac/nsteps)*nsteps) if task['algorithm'] in ['a2c','ppo'] else int(math.ceil(actual_target*frac)) for frac in ([1.] if short else [.2,.4,.6,.8,1.])))
 directory=pathlib.Path(directory);directory.mkdir(parents=True,exist_ok=True);validations=[];best_score=-np.inf;best_path=None;start=time.perf_counter()
 with (directory/'training.log').open('w',encoding='utf-8') as log:
  for step in checkpoints:
   before=model.num_timesteps;model.learn(total_timesteps=step-before,reset_num_timesteps=False,progress_bar=False,callback=Progress(task['task_id']))
   ckpt=directory/('checkpoint_%06d.zip'%model.num_timesteps);model.save(str(ckpt))
   vstart=time.perf_counter();ev=rollout_model(model,val,'F1');vm=stats_from_env(ev);seconds=time.perf_counter()-vstart
   row=dict(actual_steps=int(model.num_timesteps),requested_threshold=step,validation_SR=vm['SR'],validation_AR=vm['AR'],validation_MDD=vm['MDD'],validation_seconds=seconds,checkpoint=ckpt.name,checkpoint_sha256=sha(ckpt));validations.append(row)
   score=vm['SR'] if np.isfinite(vm['SR']) else -np.inf
   if best_path is None or score>best_score:best_score=score;best_path=ckpt
   log.write(str(row)+'\n');log.flush();print('  learned',task['task_id'],model.num_timesteps,'validation SR',score,flush=True)
  shutil.copy2(str(best_path),str(directory/'best_policy.zip'));model.save(str(directory/'final_policy.zip'))
 meta=dict(task=task,formal=not short,paper_eligible=False,accounting_scope=SCOPE,result_status='cost_analysis_scope_limited',economic_return_validated=False,num_timesteps=int(model.num_timesteps),requested_steps=target,rounded_budget=actual_target,rollout_steps=nsteps,train_seconds=time.perf_counter()-start,validation_selection='net Sharpe F1, fixed thresholds; earliest tie',selected_checkpoint=best_path.name,checkpoint_sha256=sha(directory/'best_policy.zip'),algorithm_parameters=model._get_constructor_parameters() if hasattr(model,'_get_constructor_parameters') else params,actual_gamma=model.gamma,policy=str(model.policy),observation_shape=list(env.observation_space.shape),action_mapping='float64 stable softmax [-1,1] outputs to long-only simplex',reward='net daily return*.1',episode_transitions=len(train.dates)-1,train_dates=[str(train.dates[0].date()),str(train.dates[-1].date())],validation_dates=[str(val.dates[0].date()),str(val.dates[-1].date())],test_dates_not_evaluated=True)
 save_json(directory/'checkpoint_metadata.json',meta);save_csv(directory/'validation.csv',validations)
 return meta


def evaluate_task(task):
 ctx=context();directory=RUN/'models'/task['task_id']
 if not cache_valid(directory,ctx):return
 checkpoint=directory/'best_policy.zip';ctx={**ctx,'checkpoint_sha256':sha(checkpoint)};model=None
 for fee in task['fees']:
  out=RUN/'results/evaluations'/(task['task_id']+'_'+fee)
  if cache_valid(out,ctx):continue
  if receipt(out).exists():raise RuntimeError('Protected completed evaluation hash mismatch '+str(out))
  try:
   start=time.perf_counter()
   if model is None:model=algorithms()[task['algorithm']].load(str(checkpoint),device='cpu')
   data=load_prepared(task['sector'],task['variant']).subset('2024-01-01','2024-12-31');env=rollout_model(model,data,fee)
   meta=dict(run_id=out.name,protocol_id=RUN.name+'_unadjusted_price_cost',parent=task['task_id'],sector=task['sector'],variant=task['variant'],algorithm=task['algorithm'],seed=task['seed'],fee=fee,phase=task['phase'],accounting_scope=SCOPE,result_status='cost_analysis_scope_limited',economic_return_validated=False,computation_completed=True,diagnostic_only=False,paper_eligible=False,price_basis='author_confirmed_Tushare_unadjusted',input_hash=ctx['input_sha256'],code_hash=ctx['code_sha256'],checkpoint_sha256=ctx['checkpoint_sha256'])
   frame=export_trace(env,out,meta);frame['reward']=frame.daily_return*.1;save_csv(out/'trajectory.csv',frame)
   m={**stats_from_env(env),**meta,'seconds':time.perf_counter()-start};save_json(out/'metrics.json',m);finish_receipt(out,ctx,dict(metrics=m));record(out.name,'completed',kind='actual_real_data_eval',phase=task['phase'],sector=task['sector'],variant=task['variant'],algorithm=task['algorithm'],seed=task['seed'],fee=fee,seconds=m['seconds'],transitions=len(frame)-1)
  except Exception as e:
   out.mkdir(parents=True,exist_ok=True);(out/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');record(out.name,'failed',kind='actual_real_data_eval',reason=repr(e))
 del model;gc.collect()

def train(phase='all',sector=None,algorithm=None,seed=None):
 integrity();p=load_json(RUN/'protocol_v4.json');ctx=context();tasks=plan()
 for t in tasks:
  if phase!='all' and int(t['phase'])!=int(phase):continue
  if sector and t['sector']!=sector or algorithm and t['algorithm']!=algorithm or seed and t['seed']!=seed:continue
  ok,reason=ready(t)
  if not ok:record(t['task_id'],'blocked',kind='actual_real_data_train',phase=t['phase'],reason=reason);continue
  out=RUN/'models'/t['task_id']
  if cache_valid(out,ctx):evaluate_task(t);continue
  if receipt(out).exists():record(t['task_id'],'failed',kind='actual_real_data_train',reason='Existing success protected; incompatible hash');continue
  last={r['task_id']:r for r in records()}
  if last.get(t['task_id'],{}).get('status')=='failed':continue
  completed={r['task_id'] for r in records() if r['status']=='completed' and r.get('kind')=='actual_real_data_train'}
  if len(completed)>=96:raise RuntimeError('96 real training cap reached')
  smoke=RUN/'short_run'
  if not receipt(smoke).exists():
   meta=train_model(t,p,smoke,short=True);finish_receipt(smoke,ctx,meta);record('real_api_short_run','completed',kind='real_api_short_run',actual_steps=meta['num_timesteps'],seconds=meta['train_seconds'])
  record(t['task_id'],'running',kind='actual_real_data_train',phase=t['phase'],sector=t['sector'],variant=t['variant'],algorithm=t['algorithm'],seed=t['seed']);print('TRAIN',t['task_id'],flush=True)
  try:
   meta=train_model(t,p,out);finish_receipt(out,ctx,meta);record(t['task_id'],'completed',kind='actual_real_data_train',phase=t['phase'],sector=t['sector'],variant=t['variant'],algorithm=t['algorithm'],seed=t['seed'],actual_steps=meta['num_timesteps'],seconds=meta['train_seconds'],checkpoint_sha256=meta['checkpoint_sha256']);evaluate_task(t)
  except Exception as e:
   out.mkdir(parents=True,exist_ok=True);(out/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');record(t['task_id'],'failed',kind='actual_real_data_train',phase=t['phase'],sector=t['sector'],variant=t['variant'],reason=repr(e));print('FAILED',t['task_id'],repr(e),flush=True)
  gc.collect()

def evaluate(phase='all'):
 for t in plan():
  if phase=='all' or str(t['phase'])==str(phase):evaluate_task(t)

def nonrl():
 import joblib
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import Ridge
 ctx=context()
 for sec in ['bank','healthcare']:
  for variant in ['Vanilla']+E1_VARIANTS:
   task='Ridge_%s_%s'%(sec,variant);folder=RUN/'nonrl_models'/task
   ok,reason=ready(dict(sector=sec,variant=variant))
   if not ok:record(task,'blocked',kind='nonrl_fit',reason=reason);continue
   data=load_prepared(sec,variant).subset('2022-01-01','2022-12-31')
   if not cache_valid(folder,ctx):
    start=time.perf_counter();x=data.states[:-2].reshape(len(data.dates)-2,-1);y=data.prices[2:]/data.prices[1:-1]-1;model=make_pipeline(StandardScaler(),Ridge(alpha=1.));model.fit(x,y);folder.mkdir(parents=True,exist_ok=True);joblib.dump(model,folder/'ridge.joblib');save_json(folder/'metadata.json',dict(alpha=1,k=3,samples=len(x),first_feature_date=data.dates[0],last_feature_date=data.dates[-3],last_target_date=data.dates[-1],training_only_scaler=True,target='raw close[t+2]/close[t+1]-1',accounting_scope=SCOPE));finish_receipt(folder,ctx,{});record(task,'completed',kind='nonrl_fit',seconds=time.perf_counter()-start)
  for strategy in ['equal_buy_hold','equal_daily','Ridge_Vanilla']+['Ridge_'+v for v in E1_VARIANTS]:
   for fee in FEES:
    task='Baseline_%s_%s_%s'%(sec,strategy,fee);out=RUN/'results/nonrl'/task;cctx=dict(ctx);model=None
    if strategy.startswith('Ridge'):
     variant=strategy.split('_',1)[1];folder=RUN/'nonrl_models'/('Ridge_%s_%s'%(sec,variant))
     if not cache_valid(folder,ctx):continue
     model=joblib.load(folder/'ridge.joblib');cctx['checkpoint_sha256']=sha(folder/'ridge.joblib')
    else:cctx['checkpoint_sha256']='no_learning_baseline'
    if cache_valid(out,cctx):continue
    data=load_prepared(sec,'Vanilla').subset('2024-01-01','2024-12-31');env=RealPortfolioEnv(data,FEES[fee]);done=False;start=time.perf_counter()
    while not done:
     if model is not None:
      scores=model.predict(data.states[env.i].reshape(1,-1))[0];ix=np.argsort(-scores,kind='mergesort')[:3];w=np.zeros(len(scores));w[ix]=1/3
     else:w=np.ones(len(data.tickers))/len(data.tickers) if strategy=='equal_daily' or env.i==0 else None
     _,_,done,_,_=env.advance(w)
    meta=dict(run_id=task,protocol_id=RUN.name+'_unadjusted_price_cost',sector=sec,variant=strategy,algorithm='ridge' if model is not None else 'none',seed=-1,fee=fee,phase=4,accounting_scope=SCOPE,result_status='cost_analysis_scope_limited',economic_return_validated=False,computation_completed=True,diagnostic_only=False,paper_eligible=False,input_hash=ctx['input_sha256'],code_hash=ctx['code_sha256'],checkpoint_sha256=cctx['checkpoint_sha256'])
    export_trace(env,out,meta);m={**stats_from_env(env),**meta};save_json(out/'metrics.json',m);finish_receipt(out,cctx,dict(metrics=m));record(task,'completed',kind='nonrl_eval',seconds=time.perf_counter()-start)
 if RULES:forecast_rules(ctx)
 print('Non-RL baselines/Ridge same-training-input controls and forecast-ranking rules completed.',flush=True)

FC_MODELS=['A0_Full_MGT','A1_Transformer_only','A2_MGNN_only','A3_SH_only','A4_QA_only','A5_KD_only','A6_SH_QA','A7_SH_KD','A8_QA_KD','A9_Fixed_fusion','B1_GCN_Transformer','B2_GAT_Transformer','B3_GNN_GRU','B4_GCN_GRU','B5_GAT_GRU','A10_No_cross_stock','Ridge']

def forecast_rules(ctx):
 """Hold, in equal weights, the three stocks with the highest predicted log return from the close of
 t+1 to the close of t+2 (predicted c2 - c1, fold-2024 forecasts made at t), executed at the close of t+1
 under the same ledger and costs as every other strategy.  Forecasts trained on 2018-2022, early stopping on 2023."""
 for sec in ['bank','healthcare']:
  data=load_prepared(sec,'Vanilla').subset('2024-01-01','2024-12-31')
  for name in FC_MODELS:
   src=FORECASTS/'runs'/sec/'2024'/'Ridge'/'pred.npz' if name=='Ridge' else FORECASTS/'ensembles'/('%s_2024_%s.npz'%(sec,name))
   if not src.exists():record('Forecast_%s_%s'%(sec,name),'blocked',kind='nonrl_eval',reason='forecast not yet available');continue
   with np.load(src,allow_pickle=False) as z:F=z['test'];fdates=[str(x) for x in z['test_dates']];ftick=[str(t) for t in z['tickers']]
   ftick=['PAB' if (sec=='bank' and t=='PAD') else t for t in ftick];order=[ftick.index(t) for t in data.tickers];score=(F[:,:,5]-F[:,:,2])[:,order];row={d:k for k,d in enumerate(fdates)}
   cctx=dict(ctx);cctx['checkpoint_sha256']=sha(src)
   for fee in FEES:
    task='Baseline_%s_Forecast_%s_%s'%(sec,name,fee);out=RUN/'results/nonrl'/task
    if cache_valid(out,cctx):continue
    env=RealPortfolioEnv(data,FEES[fee]);done=False;start=time.perf_counter()
    while not done:
     d=str(data.dates[env.i].date())
     if d not in row:raise ValueError('No forecast made at '+d)
     s=score[row[d]];ix=np.argsort(-s,kind='mergesort')[:3];w=np.zeros(len(s));w[ix]=1/3
     _,_,done,_,_=env.advance(w)
    meta=dict(run_id=task,protocol_id=RUN.name+'_unadjusted_price_cost',sector=sec,variant='Forecast_'+name,algorithm='forecast_rank',seed=-1,fee=fee,phase=4,accounting_scope=SCOPE,result_status='cost_analysis_scope_limited',economic_return_validated=False,computation_completed=True,diagnostic_only=False,paper_eligible=False,input_hash=ctx['input_sha256'],code_hash=ctx['code_sha256'],checkpoint_sha256=cctx['checkpoint_sha256'],rule='top-3 equal weight by predicted log(C[t+2]/C[t+1]); fold-2024 forecasts')
    export_trace(env,out,meta);m={**stats_from_env(env),**meta};save_json(out/'metrics.json',m);finish_receipt(out,cctx,dict(metrics=m));record(task,'completed',kind='nonrl_eval',seconds=time.perf_counter()-start)
