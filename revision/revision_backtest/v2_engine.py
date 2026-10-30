"""Executable training/evaluation pipeline with immutable task receipts and dependency gates."""
import inspect,math,pathlib,shutil,time,traceback
import numpy as np
import pandas as pd
from v2_audit import ROOT,PROJECT,RUN,sha,canonical,save_json,save_csv,load_json,dependencies,protocol,code_manifest,integrity
from v2_data import build_data,load_prepared,RealPortfolioEnv,export_trace,forecast_contract,EpisodeData,panel
from backtest import FEES,metrics,validate_splits

ALGORITHMS={}

def algorithms():
 global ALGORITHMS
 if not ALGORITHMS:
  from stable_baselines3 import A2C,PPO,SAC,TD3,DDPG
  ALGORITHMS={c.__name__.lower():c for c in [A2C,PPO,SAC,TD3,DDPG]}
 return ALGORITHMS

def records():
 import json
 f=RUN/'execution_records.jsonl'
 if not f.exists():return []
 return [json.loads(l) for l in f.read_text(encoding='utf-8').splitlines() if l.strip()]

def record(task,status,**kwargs):
 import json,datetime
 row=dict(task_id=task,status=status,recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs)
 with (RUN/'execution_records.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(row,ensure_ascii=False,default=str)+'\n');out.flush()
 return row

def plan():
 rows=[];p=load_json(RUN/'protocol.json');priority={tuple(x) for x in p['phase1']}
 for sec in p['market']:
  for v in ['Vanilla','A0']:
   for algo in p['all_algorithms']:
    for seed in p['seeds']:
     t='E1_%s_%s_%s_s%d'%(sec,v,algo,seed);phase=1 if (sec,algo) in priority else 2
     rows.append(dict(task_id=t,kind='train',sector=sec,variant=v,algorithm=algo,seed=seed,phase=phase,parent='',fee='',steps=p['steps'][algo]))
     for fee in FEES:rows.append(dict(task_id=t+'_'+fee,kind='evaluate',sector=sec,variant=v,algorithm=algo,seed=seed,phase=phase,parent=t,fee=fee,steps=0))
  for v in p['ablation']:
   for seed in p['seeds']:
    t='E2_%s_%s_ppo_s%d'%(sec,v,seed);rows.append(dict(task_id=t,kind='train',sector=sec,variant=v,algorithm='ppo',seed=seed,phase=3,parent='',fee='',steps=p['steps']['ppo']))
    for fee in ['F0','F1']:rows.append(dict(task_id=t+'_'+fee,kind='evaluate',sector=sec,variant=v,algorithm='ppo',seed=seed,phase=3,parent=t,fee=fee,steps=0))
  for v in ['equal_buy_hold','equal_daily','top3']:
   for fee in FEES:rows.append(dict(task_id='E4_%s_%s_%s'%(sec,v,fee),kind='baseline',sector=sec,variant=v,algorithm='none',seed=-1,phase=4,parent='',fee=fee,steps=0))
 rows.sort(key=lambda x:(x['phase'],x['sector'],x['algorithm'],x['seed'],x['variant'],x['kind']))
 save_csv(RUN/'task_plan.csv',rows)
 return rows

def current_context():
 return dict(protocol_sha256=sha(RUN/'protocol.json'),input_sha256=sha(RUN/'input_manifest.csv'),code_sha256=canonical(code_manifest()),environment_sha256=sha(RUN/'environment.json'))

def receipt(directory):return pathlib.Path(directory)/'receipt.json'
def cache_valid(directory,context):
 f=receipt(directory)
 if not f.exists():return False
 j=load_json(f)
 return j.get('status')=='completed' and j.get('context')==context and all((pathlib.Path(directory)/n).is_file() and sha(pathlib.Path(directory)/n)==h for n,h in j.get('artifacts',{}).items()) and bool(j.get('artifacts'))

def finish_receipt(directory,context,metadata):
 directory=pathlib.Path(directory)
 artifacts={f.relative_to(directory).as_posix():sha(f) for f in directory.rglob('*') if f.is_file() and f.name!='receipt.json' and not f.name.endswith('.tmp')}
 save_json(receipt(directory),dict(status='completed',context=context,artifacts=artifacts,**metadata))

def assert_locked_context():
 p=load_json(RUN/'protocol.json')
 integrity()
 if p['code']!=code_manifest() or p['input_manifest_sha256']!=sha(RUN/'input_manifest.csv') or p['environment_sha256']!=sha(RUN/'environment.json'):raise RuntimeError('Locked code/input/environment mismatch; no model/trajectory reuse')
 return p

def formal_gate(task,p):
 sec=task['sector'];m=p['market'][sec]
 if not m['structural_valid']:return False,'Market panel invalid'
 if not m['price']['formal']:return False,'Price basis not certified: '+m['price']['reason']
 if task['variant'] not in ['Vanilla','equal_buy_hold','equal_daily']:
  c=forecast_contract(sec,m,'A0' if task['variant']=='top3' else task['variant'])
  if not c['valid']:return False,c['reason']
 if task['variant']=='top3':return False,'Need certified next-day-close numeric ranking signal, independent target/available dates and horizon contract; four-row indicator bundle alone is not a ranking forecast'
 return True,'Verified dependencies under disclosed fractional delayed-close approximation'

def prepare(status):
 from v2_audit import dependencies,protocol
 RUN.joinpath('prepared').mkdir(exist_ok=True)
 # Feature preparation can proceed for B inputs, but never authorizes B-class RL training.
 prep=[]
 for sec,m in status.items():
  try:
   d=build_data(sec,m,'Vanilla')
   a=d.subset('2022-01-01','2022-12-31');b=d.subset('2023-01-01','2023-12-31');c=d.subset('2024-01-01','2024-12-31');validate_splits(a.dates,b.dates,c.dates)
   prep.append(dict(sector=sec,variant='Vanilla',status='completed',first_state=d.dates[0],last_state=d.dates[-1],state_rows=d.states.shape[1],stocks=len(d.tickers),train_dates=len(a.dates),validation_dates=len(b.dates),test_dates=len(c.dates),paper_eligible=m['price']['formal'],reason='Causal raw state prepared; input validity independently gates training'))
  except Exception as e:prep.append(dict(sector=sec,variant='Vanilla',status='failed',reason=repr(e)))
  for variant in ['A0','A1','A2','A6','A7','A8','A9']:
   fc=forecast_contract(sec,m,variant)
   if fc['valid']:
    try:build_data(sec,m,variant);prep.append(dict(sector=sec,variant=variant,status='completed',paper_eligible=m['price']['formal'],reason=fc['reason']))
    except Exception as e:prep.append(dict(sector=sec,variant=variant,status='failed',reason=repr(e)))
   else:prep.append(dict(sector=sec,variant=variant,status='blocked',paper_eligible=False,reason=fc['reason']))
 save_csv(RUN/'preparation_status.csv',prep)
 # Resolve all default parameters without fitting any model.
 pars={}
 for name,cls in algorithms().items():
  pars[name]={k:str(v.default) for k,v in inspect.signature(cls.__init__).parameters.items() if k not in ['self','policy','env']}
  settings=next(iter(load_json(ROOT/'original_settings.json').values()))
  pars[name].update(settings['MODEL_PARAMS'][name]);pars[name].update(device='cpu',seed='paired 1/2/3',policy='MlpPolicy',torch_threads=1,network='installed SB3 default; exported actual policy architecture per model',gamma=float(inspect.signature(cls.__init__).parameters['gamma'].default))
 save_json(RUN/'resolved_parameters.json',pars)
 protocol(status);dependencies(status);plan()
 print('Prepared causal real-data states; formal authorization remains dependency-specific.',flush=True)
 return prep

def stats_from_env(env):
 df=pd.DataFrame(env.trace);r=df.daily_return.iloc[1:].to_numpy(float);m=metrics(r);m.update(SR_rf02=metrics(r,.02)['SR'],final_nav=float(df.nav.iloc[-1]),initial_capital=float(df.nav.iloc[0]),cumulative_cost=float(df.cost.sum()),commission=float(df.commission.sum()),slippage=float(df.slippage.sum()),sell_tax=float(df.tax.sum()),total_buy=float(df.buy.sum()),total_sell=float(df.sell.sum()),mean_gross_turnover=float(df.gross_turnover.iloc[1:].mean()),annualized_gross_turnover=float(df.gross_turnover.iloc[1:].mean()*252),first_date=df.date.iloc[0],last_date=df.date.iloc[-1],cash_end=float(df.cash.iloc[-1]))
 return m

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
   before=model.num_timesteps;model.learn(total_timesteps=step-before,reset_num_timesteps=False,progress_bar=False)
   ckpt=directory/('checkpoint_%06d.zip'%model.num_timesteps);model.save(str(ckpt))
   vstart=time.perf_counter();ev=rollout_model(model,val,'F1');vm=stats_from_env(ev);seconds=time.perf_counter()-vstart
   row=dict(actual_steps=int(model.num_timesteps),requested_threshold=step,validation_SR=vm['SR'],validation_AR=vm['AR'],validation_MDD=vm['MDD'],validation_seconds=seconds,checkpoint=ckpt.name,checkpoint_sha256=sha(ckpt));validations.append(row)
   score=vm['SR'] if np.isfinite(vm['SR']) else -np.inf
   if best_path is None or score>best_score:best_score=score;best_path=ckpt
   log.write(str(row)+'\n');log.flush();print('  learned',task['task_id'],model.num_timesteps,'validation SR',score,flush=True)
  shutil.copy2(str(best_path),str(directory/'best_policy.zip'));model.save(str(directory/'final_policy.zip'))
 meta=dict(task=task,formal=not short,paper_eligible=not short,num_timesteps=int(model.num_timesteps),requested_steps=target,rounded_budget=actual_target,rollout_steps=nsteps,train_seconds=time.perf_counter()-start,validation_selection='net Sharpe F1, fixed thresholds; earliest tie',selected_checkpoint=best_path.name,checkpoint_sha256=sha(directory/'best_policy.zip'),algorithm_parameters=model._get_constructor_parameters() if hasattr(model,'_get_constructor_parameters') else params,actual_gamma=model.gamma,policy=str(model.policy),observation_shape=list(env.observation_space.shape),action_mapping='float64 stable softmax [-1,1] outputs to long-only simplex',reward='net daily return*.1',episode_transitions=len(train.dates)-1,train_dates=[str(train.dates[0].date()),str(train.dates[-1].date())],validation_dates=[str(val.dates[0].date()),str(val.dates[-1].date())],test_dates_not_evaluated=True)
 save_json(directory/'checkpoint_metadata.json',meta);save_csv(directory/'validation.csv',validations)
 return meta


def execute_training(phase='all',sector=None,algorithm=None,seed=None):
 p=assert_locked_context();tasks=pd.read_csv(RUN/'task_plan.csv',keep_default_na=False).to_dict('records');context=current_context()
 budget_count=sum(1 for r in records() if r.get('kind')=='train' and r['status']=='completed' and not r.get('short_run',False))
 for t in tasks:
  if t['kind']!='train' or (phase=='priority' and int(t['phase'])!=1) or (phase=='remaining' and int(t['phase'])==1) or (sector and t['sector']!=sector) or (algorithm and t['algorithm']!=algorithm) or (seed and int(t['seed'])!=seed):continue
  directory=RUN/'models'/t['task_id']
  if cache_valid(directory,context):continue
  if directory.exists() and receipt(directory).exists():
   record(t['task_id'],'failed',kind='train',reason='Existing training receipt mismatch; checkpoint protected, not overwritten',paper_eligible=False);continue
  valid,reason=formal_gate(t,p)
  if not valid:record(t['task_id'],'blocked',kind='train',phase=t['phase'],sector=t['sector'],variant=t['variant'],reason=reason,actual_steps=0,paper_eligible=False);continue
  if budget_count>=p['cap']:record(t['task_id'],'blocked',kind='train',reason='96-training cap reached');continue
  # Only one qualifying real-configuration API short run in a distinct directory; never paper data.
  smoke=RUN/'diagnostics/api_short_run'
  if not receipt(smoke).exists():
   try:
    meta=train_model(t,p,smoke,short=True);finish_receipt(smoke,context,meta);record('real_api_short_run','completed',kind='real_api_short_run',actual_steps=meta['num_timesteps'],seconds=meta['train_seconds'],paper_eligible=False)
   except Exception as e:record('real_api_short_run','failed',kind='real_api_short_run',reason=repr(e));continue
  record(t['task_id'],'running',kind='train',phase=t['phase'],sector=t['sector'],variant=t['variant'])
  try:
   print('TRAIN',t['task_id'],flush=True);meta=train_model(t,p,directory);finish_receipt(directory,context,meta);budget_count+=1
   record(t['task_id'],'completed',kind='train',phase=t['phase'],sector=t['sector'],variant=t['variant'],algorithm=t['algorithm'],seed=t['seed'],actual_steps=meta['num_timesteps'],seconds=meta['train_seconds'],paper_eligible=True,checkpoint_sha256=meta['checkpoint_sha256'],directory=str(directory.relative_to(RUN)))
  except Exception as e:
   directory.mkdir(parents=True,exist_ok=True);(directory/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');record(t['task_id'],'failed',kind='train',reason=repr(e),paper_eligible=False)


def baseline_data(sector,p):
 d,prices=panel(sector);prices=prices[prices.index.year==2024];dates=prices.index;tickers=list(prices.columns);n=len(tickers)
 states=np.zeros((len(dates),1,n),np.float32);allowed=np.ones(prices.shape,bool);mult=np.ones(prices.shape);div=np.zeros(prices.shape);basis=p['market'][sector]['price'];source=basis.get('source',{})
 if basis['formal'] and basis['price_basis']=='verified_unadjusted_with_actions':
  a=pd.read_csv(PROJECT/source['actions_file']);a.date=pd.to_datetime(a.date)
  for row in a.to_dict('records'):
   if row['date'] in dates and row['tic'] in tickers:
    i=dates.get_loc(row['date']);j=tickers.index(row['tic']);mult[i,j]=row['share_multiplier'];div[i,j]=row['cash_per_old_share']
 if source.get('tradability_file'):
  f=PROJECT/source['tradability_file']
  if sha(f)!=source.get('tradability_sha256'):raise ValueError('Tradability file changed')
  z=pd.read_csv(f);z.date=pd.to_datetime(z.date);z=z.pivot(index='date',columns='tic',values='tradable').reindex(index=dates,columns=tickers)
  if z.isna().any().any():raise ValueError('Missing explicit tradability entries')
  allowed=z.astype(bool).to_numpy()
 unit='total_return_index_fractional_units' if basis['price_basis']=='verified_total_return_index' else ('actual_fractional_equity_shares' if basis['formal'] else 'unverified_price_sequence_units')
 return EpisodeData(dates,prices.to_numpy(float),states,tickers,allowed,mult,div,unit)


def evaluate(phase='all',sector=None,algorithm=None,seed=None):
 p=assert_locked_context();context=current_context();tasks=pd.read_csv(RUN/'task_plan.csv',keep_default_na=False).to_dict('records')
 for t in tasks:
  if t['kind'] not in ['evaluate','baseline'] or (phase=='priority' and int(t['phase'])!=1) or (phase=='remaining' and int(t['phase'])==1) or (sector and t['sector']!=sector) or (algorithm and t['kind']=='evaluate' and t['algorithm']!=algorithm) or (seed and t['kind']=='evaluate' and int(t['seed'])!=seed):continue
  ok,reason=formal_gate(t,p);kind='formal_evaluation';diagnostic=False
  if t['kind']=='baseline' and t['variant']!='top3' and p['market'][t['sector']]['structural_valid']:
   diagnostic=not ok;kind='price_sequence_diagnostic' if diagnostic else 'formal_baseline'
  elif not ok:record(t['task_id'],'blocked',kind='evaluate',reason=reason,paper_eligible=False);continue
  ckpt=None;train_directory=RUN/'models'/t['parent'] if t['parent'] else None
  ctx=dict(context)
  if t['kind']=='evaluate':
   if not cache_valid(train_directory,context):record(t['task_id'],'blocked',kind='evaluate',reason='No completed matching parent training checkpoint',paper_eligible=False);continue
   ckpt=train_directory/'best_policy.zip';ctx['checkpoint_sha256']=sha(ckpt)
  directory=RUN/('diagnostics/baselines' if diagnostic else 'formal/evaluations')/t['task_id']
  if cache_valid(directory,ctx):continue
  if directory.exists() and receipt(directory).exists():record(t['task_id'],'failed',kind=kind,reason='Existing receipt mismatch; protected, not overwritten');continue
  start=time.perf_counter();record(t['task_id'],'running',kind=kind)
  try:
   if t['kind']=='baseline':
    data=baseline_data(t['sector'],p);env=RealPortfolioEnv(data,FEES[t['fee']]);done=False
    while not done:
     weights=np.ones(len(data.tickers),dtype=float)/len(data.tickers) if t['variant']=='equal_daily' or env.i==0 else None
     _,_,done,_,_=env.advance(weights)
   else:
    data=load_prepared(t['sector'],t['variant']).subset(*p['splits']['test']);model=algorithms()[t['algorithm']].load(str(ckpt),device='cpu');env=rollout_model(model,data,t['fee'])
   meta=dict(run_id=t['task_id'],protocol_id=p['protocol_id'],sector=t['sector'],variant=t['variant'],algorithm=t['algorithm'],seed=t['seed'],fee=t['fee'],phase=t['phase'],paper_eligible=not diagnostic,diagnostic_only=diagnostic,price_basis=p['market'][t['sector']]['price']['price_basis'],limitation='Given-price-sequence numerical accounting only; not economic investment returns or R1.2 evidence' if diagnostic else 'Fractional daily delayed-close approximation; see protocol constraints',checkpoint_sha256=sha(ckpt) if ckpt else 'no_learning_baseline',input_hash=context['input_sha256'],code_hash=context['code_sha256'],environment_hash=context['environment_sha256'])
   trace=export_trace(env,directory,meta);m={**stats_from_env(env),**meta,**ctx};m['seconds']=time.perf_counter()-start
   save_json(directory/'metrics.json',m);finish_receipt(directory,ctx,dict(task=t,metrics=m,paper_eligible=not diagnostic,diagnostic_only=diagnostic))
   record(t['task_id'],'completed',kind=kind,phase=t['phase'],sector=t['sector'],variant=t['variant'],algorithm=t['algorithm'],seed=t['seed'],fee=t['fee'],paper_eligible=not diagnostic,diagnostic_only=diagnostic,transitions=len(trace)-1,seconds=m['seconds'],directory=str(directory.relative_to(RUN)),checkpoint_sha256=meta['checkpoint_sha256'])
   print('EVALUATED',t['task_id'],kind,'n=',m['n'],'cost=',round(m['cumulative_cost'],2),flush=True)
  except Exception as e:
   directory.mkdir(parents=True,exist_ok=True);(directory/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');record(t['task_id'],'failed',kind=kind,reason=repr(e),paper_eligible=False)
