"""Versioned evidence, causal state preparation, and immutable input selection."""
import hashlib,json,pathlib,sys,importlib.metadata as md,subprocess,platform,os
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent
PROJECT=ROOT.parent
RUN=ROOT/'runs/v2'
PYTHON=r'C:\Users\perfe\.conda\envs\finrl-meta\python.exe'
SECTORS={'bank':('bank sectordata','banktrade20241231'),'healthcare':('haelthcaresectordata','techtrade20241231')}


def sha(f):
 h=hashlib.sha256()
 with pathlib.Path(f).open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()

def canonical(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()
def save_json(f,obj):
 f=pathlib.Path(f);f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8');tmp.replace(f)
def save_csv(f,rows,columns=None):
 f=pathlib.Path(f);f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.tmp');pd.DataFrame(rows,columns=columns).to_csv(tmp,index=False,encoding='utf-8-sig');tmp.replace(f)
def load_json(f):return json.loads(pathlib.Path(f).read_text(encoding='utf-8-sig'))
def find_market(sector):
 folder,stem=SECTORS[sector];hits=[f for f in (PROJECT/folder).rglob('*') if f.is_file() and f.stem.lower()==stem.lower() and f.suffix.lower()=='.csv']
 if len(hits)!=1:raise ValueError('%s exact independent raw candidates: %s'%(sector,hits))
 return hits[0]
def market_long(f,strict=True):
 d=pd.read_csv(f,dtype={'tic':str});required={'date','tic','close','high','low'}
 if not required.issubset(d):raise ValueError('Missing market fields '+str(required-set(d)))
 d['date']=pd.to_datetime(d.date,errors='raise');d['tic']=d.tic.str.strip()
 if d[['date','tic']].isna().any().any() or d.duplicated(['date','tic']).any():raise ValueError('Invalid market keys')
 for c in ['close','high','low']:d[c]=pd.to_numeric(d[c],errors='raise')
 if strict and (not np.isfinite(d[['close','high','low']]).all().all() or (d[['close','high','low']]<=0).any().any()):raise ValueError('Invalid raw HLC')
 return d.sort_values(['date','tic']).reset_index(drop=True)
def panel(sector):
 d=market_long(find_market(sector));p=d.pivot(index='date',columns='tic',values='close').sort_index()
 from backtest import validate_prices
 validate_prices(p);return d,p

def selected_inputs():
 result={}
 for sector in SECTORS:
  f=find_market(sector);result[f]='market_raw'
 for f in PROJECT.rglob('*'):
  if not f.is_file() or ROOT in f.parents:continue
  rel=f.relative_to(PROJECT).as_posix()
  if f.suffix=='.ipynb' and f.parent==PROJECT/'code':result[f]='source_provenance'
  elif f.suffix.lower()=='.npy':result[f]='frozen_array_provenance'
  elif f.suffix.lower()=='.csv' and ('_pred' in f.stem or '_true' in f.stem or 'finrl_csv' in rel):result[f]='frozen_forecast_or_merge_provenance'
 # Only concrete evidence inputs under this directory may upgrade a price basis; none are invented.
 ev=PROJECT/'backtest_source_evidence'
 if ev.exists():
  for f in ev.rglob('*'):
   if f.is_file():result[f]='author_source_evidence'
 return result

def delta(current,old):
 new={r['path']:r['sha256'] for r in current};rows=[]
 for p in sorted(set(new)|set(old)):
  state='added' if p not in old else ('deleted' if p not in new else ('unchanged' if new[p]==old[p] else 'modified'))
  rows.append(dict(path=p,change=state,old_sha256=old.get(p,''),new_sha256=new.get(p,'')))
 return rows

def verify_price_basis(sector,raw):
 """Specific source contract, not an allow_unverified switch.
 No inference from a static unrelated table header or equality to a merge.
 """
 e=PROJECT/'backtest_source_evidence'/('%s_price_basis.json'%sector)
 if not e.exists():return dict(formal=False,price_basis='unverified',reason='No source/export record linking this HLC table to adjustment, dividends and split treatment',evidence='Project notebook/source search + absent '+e.relative_to(PROJECT).as_posix())
 try:
  j=load_json(e)
  if j['market_sha256']!=sha(raw):raise ValueError('evidence raw hash mismatch')
  cited=PROJECT/j['source_record']
  if not cited.is_file() or j['source_sha256']!=sha(cited):raise ValueError('missing or mismatched source record')
  if j.get('security_kind')!='mainland_A_equity' or not j.get('security_mapping'):raise ValueError('security identity/mapping missing')
  if not set(market_long(raw).tic).issubset(j['security_mapping']):raise ValueError('security mapping does not cover all raw tickers')
  if j['price_basis'] not in ['verified_total_return_index','verified_unadjusted_with_actions']:raise ValueError('Supported accounting basis not established')
  if not j.get('method_description') or not j.get('dividend_split_treatment'):raise ValueError('Adjustment/dividend treatment description required')
  if j['price_basis']=='verified_unadjusted_with_actions':
   af=PROJECT/j['actions_file']
   if j['actions_sha256']!=sha(af):raise ValueError('corporate action source hash mismatch')
   actions=pd.read_csv(af)
   if not {'date','tic','share_multiplier','cash_per_old_share'}.issubset(actions):raise ValueError('Corporate action schema missing')
  return dict(formal=True,price_basis=j['price_basis'],reason='Explicit source record verified by hashes; simplified fractional daily model',evidence=str(e.relative_to(PROJECT)),source=j)
 except (KeyError,ValueError,OSError) as err:return dict(formal=False,price_basis='unverified',reason=str(err),evidence=str(e.relative_to(PROJECT)))

def audit():
 RUN.mkdir(parents=True,exist_ok=True)
 inputs=[dict(path=f.relative_to(PROJECT).as_posix(),role=role,bytes=f.stat().st_size,sha256=sha(f)) for f,role in sorted(selected_inputs().items())]
 mf=RUN/'input_manifest.csv'
 if mf.exists():
  previous=pd.read_csv(mf).to_dict('records')
  if {x['path']:x['sha256'] for x in previous}!={x['path']:x['sha256'] for x in inputs}:
   save_csv(RUN/'input_delta_after_lock.csv',delta(inputs,{x['path']:x['sha256'] for x in previous}));raise RuntimeError('V2 research inputs changed after lock. Preserve completed runs; use a new protocol/run version.')
 else:save_csv(mf,inputs)
 old=pd.read_csv(ROOT/'archive/v1/input_hashes.csv').set_index('path').sha256.to_dict()
 # Compare only corresponding research inputs; v1 result images and instruction text are out of scope.
 relevant=set(x['path'] for x in inputs)|{p for p in old if p.lower().endswith(('.npy','.ipynb')) or '_pred' in p or '_true' in p or 'finrl_csv' in p or p.endswith('banktrade20241231.CSV')}
 save_csv(RUN/'input_delta.csv',delta(inputs,{p:h for p,h in old.items() if p in relevant}))
 documents=[]
 for f in PROJECT.glob('*'):
  if f.is_file() and f.suffix.lower() in ['.txt','.md','.docx']:
   documents.append(dict(path=f.name,sha256=sha(f),role='instruction_document_not_market_input',delta='added' if f.name not in old else ('unchanged' if old[f.name]==sha(f) else 'modified')))
 for p in old:
  if '/' not in p and p.lower().endswith(('.txt','.md','.docx')) and not (PROJECT/p).exists():documents.append(dict(path=p,delta='deleted',role='prior_instruction_archived_v1'))
 save_csv(RUN/'instruction_delta.csv',documents)
 env={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{},'threads':1}
 for pkg,module in [('finrl','finrl'),('stable-baselines3','stable_baselines3'),('torch','torch'),('gym','gym'),('gymnasium','gymnasium'),('numpy','numpy'),('pandas','pandas'),('stockstats','stockstats')]:
  version=md.version(pkg);spec=__import__('importlib.util',fromlist=['find_spec']).find_spec(module)
  env['packages'][pkg]={'version':version,'import_source':spec.origin}
 import torch
 env['cuda_available']=torch.cuda.is_available();env['torch_threads']=1
 target=RUN/'environment.json'
 if target.exists() and canonical(load_json(target))!=canonical(env):raise RuntimeError('Environment changed after v2 lock')
 save_json(target,env)
 if not (RUN/'pip_freeze.txt').exists():
  proc=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True);(RUN/'pip_freeze.txt').write_text(proc.stdout,encoding='utf-8')
 audit_rows=[];diffs=[];mapping=[];qualities=[];forecast_rows=[];status={}
 for sector in SECTORS:
  f=find_market(sector);d,p=panel(sector);basis=verify_price_basis(sector,f)
  structural=bool(not p.isna().any().any())
  status[sector]=dict(raw=f.relative_to(PROJECT).as_posix(),sha256=sha(f),rows=len(d),stocks=list(p.columns),first=str(p.index.min().date()),last=str(p.index.max().date()),dates=len(p),structural_valid=structural,price=basis,has_open='open' in d)
  audit_rows.append(dict(sector=sector,raw=f.relative_to(PROJECT).as_posix(),rows=len(d),stocks=p.shape[1],dates=len(p),first=p.index.min(),last=p.index.max(),duplicate_keys=0,missing_hlc=int(d[['high','low','close']].isna().sum().sum()),volume_unit='unknown',open_present='open' in d,price_basis=basis['price_basis'],paper_eligible=basis['formal']))
  for t,g in d.groupby('tic'):
   initial=0;cols=['high','low','close']+(['volume'] if 'volume' in g else [])
   for _,row in g[cols].iterrows():
    if row.equals(g[cols].iloc[0]):initial+=1
    else:break
   qualities.append(dict(sector=sector,tic=t,initial_identical_rows=initial,invalid_volume=int(pd.to_numeric(g.volume,errors='coerce').isna().sum()) if 'volume' in g else None))
  for merge in sorted((PROJECT/SECTORS[sector][0]).rglob('*')):
   if not merge.is_file() or merge.suffix.lower()!='.csv' or 'finrl_csv' not in str(merge):continue
   m=market_long(merge,strict=False);j=m.merge(d,on=['date','tic'],suffixes=('_merge','_raw'),how='outer',indicator=True,validate='one_to_one')
   mapping.append(dict(sector=sector,merge=merge.relative_to(PROJECT).as_posix(),market_raw=f.relative_to(PROJECT).as_posix(),unique_keys=True,nonpositive_merge_close=int((m.close<=0).sum()),unmatched_keys=int((j['_merge']!='both').sum()),forecast_target_coverage='2022 artifacts only; compare differences does not certify forecast',available_at='not established'))
   for year,g in j.groupby(j.date.dt.year):
    for c in ['high','low','close']:
     a,b=g[c+'_merge'],g[c+'_raw'];valid=a.notna()&b.notna();diffs.append(dict(sector=sector,merge=merge.relative_to(PROJECT).as_posix(),raw=f.relative_to(PROJECT).as_posix(),year=year,channel=c,paired=int(valid.sum()),different=int(((a!=b)&valid).sum()),max_abs_diff=float((a[valid]-b[valid]).abs().max()),unmatched_keys=int((g['_merge']!='both').sum())))
  for pf in (PROJECT/SECTORS[sector][0]).rglob('*_pred*.csv'):
   x=pd.read_csv(pf);dated='date' in x
   if dated:dates=pd.to_datetime(x.date,errors='raise')
   forecast_rows.append(dict(sector=sector,path=pf.relative_to(PROJECT).as_posix(),rows=len(x),columns=len(x.columns),has_target_date=dated,target_first=str(dates.min().date()) if dated else '',target_last=str(dates.max().date()) if dated else '',stock_columns=json.dumps([c for c in x if c!='date']),actual_generated_at='unrecorded',derived_available_at='not established',scale='price claim' if '_raw' in pf.stem else 'unverified normalized',strict_oos='not established; bank scaler before split evidence',supports_2024=False))
 save_csv(RUN/'data_profiles.csv',audit_rows);save_csv(RUN/'merge_channel_differences.csv',diffs);save_csv(RUN/'input_mapping.csv',mapping);save_csv(RUN/'quality_flags.csv',qualities);save_csv(RUN/'forecast_inventory.csv',forecast_rows);save_json(RUN/'market_status.json',status)
 print('Incremental audit: %s selected research inputs; medical raw used directly.'%len(inputs),flush=True)
 return status


def code_manifest():
 names=['run_revision.py','backtest.py','v2_audit.py','v2_data.py','v2_engine.py','v2_reports.py','test_v2.py']
 return {n:sha(ROOT/n) for n in names if (ROOT/n).exists()}

def dependencies(status):
 rows=[]
 for sec,v in status.items():
  for task in ['train','evaluate','baseline','top3','ablation','bootstrap']:
   for variant in (['Vanilla','A0'] if task in ['train','evaluate'] else (['buy_hold','equal_daily'] if task=='baseline' else (['A0','A1','A2','A6','A7','A8','A9'] if task=='ablation' else ['A0']))):
    uses_pred=variant.startswith('A');checks=[];evidence=[]
    if not v['structural_valid']:checks.append('market date/ticker panel invalid')
    if not v['price']['formal']:checks.append('unverified price adjustment/dividend/split treatment')
    if uses_pred:
     from v2_data import forecast_contract
     fc=forecast_contract(sec,v,'A0' if task in ['top3','bootstrap'] else variant)
     if not fc['valid']:checks.append(fc['reason'])
    if task=='top3':checks.append('target-day horizon incompatible unless explicitly aligned before execution')
    if task=='bootstrap':checks.append('requires valid paired trained/test trajectories')
    st='ready' if not checks else ('diagnostic_ready' if task=='baseline' and v['structural_valid'] else 'blocked')
    rows.append(dict(sector=sec,variant=variant,window='train2022_val2023_test2024' if task!='baseline' else 'test2024',task=task,required_inputs='raw HLC + price basis'+(' + verified frozen forecast dates/scale/timing' if uses_pred else ''),status=st,paper_eligible=st=='ready',reason='; '.join(checks) if checks else 'All required checked inputs available',evidence=v['raw']+'; market_status.json; archive/v1/notebook_evidence.csv',prediction_required=uses_pred))
 save_csv(RUN/'dependency_status.csv',rows)
 return rows

def protocol(status):
 from backtest import FEES
 settings=load_json(ROOT/'original_settings.json');selected=next(iter(settings.values()))
 value=dict(protocol_id='v2_2024_delayed_close',input_manifest_sha256=sha(RUN/'input_manifest.csv'),environment_sha256=sha(RUN/'environment.json'),code=code_manifest(),splits={'train':['2022-01-01','2022-12-31'],'validation':['2023-01-01','2023-12-31'],'test':['2024-01-01','2024-12-31']},market=status,lookback=252,initial_amount=1_000_000,reward_scaling=.1,reward='net single period return * 0.1',observation='N x N real historical return covariance plus 4 x N macd/rsi_30/cci_30/dx_30; no cash/nav or optional valuation/text fields',forecasts='MGT disabled until frozen feature bundle has explicit decision/target dates, scale, training/scaler/graph cutoffs and nonleakage evidence; no raw future replacement',execution='close t decision, actual close t+1 execution; old holdings bear price move',fractional_units=True,leverage=False,short=False,terminal_liquidation=False,unmodeled=['market impact','minimum commissions','unobserved limit/suspension state','volume capacity'],fees={n:vars(c) for n,c in FEES.items()},training_fee='F1',seeds=[1,2,3],steps=selected['TOTAL_TIMESTEPS'],model_params=selected['MODEL_PARAMS'],gamma='installed SB3 default resolved in resolved_parameters.json',checkpoint_rule='validation net Sharpe at 20/40/60/80/100% budget, same rule across representations; ties earliest; no test selection',rollout_rounding='On-policy A2C/PPO actual timesteps ceil(target/n_steps)*n_steps; validation thresholds use actual rounded budget; off-policy exact target',phase1=[['bank','a2c'],['healthcare','sac'],['healthcare','ppo']],all_algorithms=['a2c','ppo','sac','td3','ddpg'],ablation=['A1','A2','A6','A7','A8','A9'],cap=96,bootstrap={'B':2000,'blocks':[10,5,20],'seed':1729},test_history='2024 already examined in original research; fixed historical evaluation, not unseen',rf=[0.,.02],diagnostic_scope='only two nonlearning baselines for unverified price basis; never RL/paper metrics')
 f=RUN/'protocol.json'
 if f.exists() and canonical(load_json(f))!=canonical(value):raise RuntimeError('V2 protocol or code changed after lock. No silent cache reuse; preserve old protocol and start a new version.')
 if not f.exists():save_json(f,value)
 save_json(RUN/'protocol_alternative.json',dict(status='not_approved_not_launched',candidate='2022 chronological subwindows',reason='Cannot cure preprocessing fitted on held-out data, unavailable bank dates/ticker scale or graph/availability gaps by recutting year. Does not answer 2024 hypothesis.'))
 return value

def integrity():
 x=pd.read_csv(RUN/'input_manifest.csv');rows=[]
 for record in x.to_dict('records'):
  f=PROJECT/record['path'];rows.append(dict(path=record['path'],unchanged=f.exists() and sha(f)==record['sha256']))
 save_csv(RUN/'input_integrity.csv',rows)
 if not all(r['unchanged'] for r in rows):raise RuntimeError('V2 research input integrity failure')
