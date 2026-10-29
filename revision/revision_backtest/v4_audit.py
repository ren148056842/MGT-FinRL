"""v4: training-only augmentation with walk-forward one-step-ahead forecasts (revision_v4/forecasts)."""
import pathlib,json,sys,os,importlib.metadata as md
import numpy as np
import pandas as pd
from v2_audit import ROOT,PROJECT,PYTHON,sha,canonical,save_json,save_csv,load_json,find_market,market_long
import os as _os
RUN=ROOT/('runs/'+_os.environ.get('MGT_V4_RUN','v4_forecast_train_only'))
V3RUN=ROOT/'runs/v3_train_only'
FORECASTS=PROJECT/'revision_v4/forecasts'
SCOPE='unadjusted_price_cost_sensitivity'
VARIANTS=_os.environ.get('MGT_V4_VARIANTS','Vanilla,A0,A1,A2,A6,A7,A8,A9').split(',')
E1_VARIANTS=[v for v in _os.environ.get('MGT_V4_E1','A0').split(',') if v and v!='none']   # 'none': no E1 tasks
ABLATION=[v for v in _os.environ.get('MGT_V4_ABLATION','A1,A2,A6,A7,A8,A9').split(',') if v]
RULES=_os.environ.get('MGT_V4_RULES','1')=='1'

def code_manifest():
 c={n:sha(ROOT/n) for n in ['run_revision_v4.py','v4_audit.py','v4_data.py','v3_data.py','v3_ledger.py','v4_engine.py','v4_reports.py','test_v4.py','backtest.py','v2_data.py','v2_audit.py','v2_engine.py']}
 c.update({'forecaster/'+n:sha(PROJECT/'revision_v4/forecaster'/n) for n in ['fc_data.py','fc_models.py','fc_run.py']});return c

def resolve_merge(sector,variant):
 folder=FORECASTS/'merges'
 hits=[f for f in folder.iterdir() if f.is_file() and f.suffix.lower()=='.csv' and f.name.startswith(sector+'_'+variant+'_') and f.name.endswith('_2022.csv')]
 if len(hits)!=1:raise ValueError('Expected unique canonical merge '+str((sector,variant,hits)))
 return hits[0]

def names():
 bank=pd.read_csv(PROJECT/'bank sectordata/bankcloseshizhi.csv');raw=market_long(find_market('bank'));aliases=sorted(raw.tic.unique());rows=[]
 for t in aliases:
  canonical_t='PAB' if t in ['PAD','PAB'] else t;hit=bank[bank.NAME.astype(str).str.strip()==canonical_t]
  if len(hit)!=1:raise ValueError('Bank mapping not one-to-one: '+t)
  rows.append(dict(sector='bank',raw_tic=t,tic=canonical_t,name_from_table=str(hit.iloc[0,1]),author_name={'PAB':'平安银行','CMBC':'民生银行','CMB':'招商银行'}.get(canonical_t,''),source='bankcloseshizhi NAME/name columns; Wind auxiliary only',market_source='Tushare author_confirmed; unadjusted',mapping_note='PAD/PAB same entity' if canonical_t=='PAB' else 'distinct alias preserved'))
 assert len(set(x['tic'] for x in rows))==15 and {x['tic'] for x in rows}>={'CMB','CMBC','PAB'}
 tech=pd.read_csv(PROJECT/'haelthcaresectordata/tech.csv',dtype={'code':str});medical=market_long(find_market('healthcare'))
 for t in sorted(medical.tic.unique()):
  hit=tech[tech.code.astype(str).str.strip()==t]
  if len(hit)!=1:raise ValueError('Medical security mapping ambiguous: '+t)
  rows.append(dict(sector='healthcare',raw_tic=t,tic=t,name_from_table=str(hit.iloc[0,1]),source='tech.csv (Wind auxiliary table)',market_source='Tushare author_confirmed; unadjusted',listing_date=str(hit.iloc[0,2])))
 save_csv(RUN/'security_mapping.csv',rows);return rows

def canonicalize(d,sector):
 d=d.copy()
 if sector=='bank':d['tic']=d.tic.replace({'PAD':'PAB'})
 return d

def audit():
 RUN.mkdir(parents=True,exist_ok=True);mapping=names();profiles=[];diffs=[];inputs={};issues=[]
 author=PROJECT/'Codex_MGT_TrainingOnly_V3_ZH.txt';inputs[author]='author_clarification'
 for f in [PROJECT/'bank sectordata/bankcloseshizhi.csv',PROJECT/'haelthcaresectordata/tech.csv']:inputs[f]='auxiliary_security_mapping'
 selected=[]
 for sec in ['bank','healthcare']:
  f=find_market(sec);raw=canonicalize(market_long(f),sec);inputs[f]='market_raw';profiles.append(dict(sector=sec,raw=f.relative_to(PROJECT).as_posix(),rows=len(raw),first=raw.date.min(),last=raw.date.max(),stocks=raw.tic.nunique(),data_source='Tushare',price_basis='unadjusted',basis_evidence='author_confirmed',accounting_scope=SCOPE))
  for variant in VARIANTS[1:]:
   f=resolve_merge(sec,variant);inputs[f]='frozen_training_augmentation';m=canonicalize(market_long(f,strict=False),sec);j=m.merge(raw,on=['date','tic'],suffixes=('_merge','_raw'),how='outer',indicator=True,validate='one_to_one')
   unmatched=int((j['_merge']!='both').sum());selected.append(dict(sector=sec,variant=variant,merge=f.relative_to(PROJECT).as_posix(),sha256=sha(f),choice_reason='v4 walk-forward forecaster, fold 2022 (train 2018-2020, validation 2021), ensemble of seeds 1-3; one-step-ahead H/L/C made at t-1; no selection using test returns',unmatched=unmatched,augmentation_scope='train_2022_only',validation_input='raw with raw 2022 warmup',test_input='raw with all-raw warmup'))
   if unmatched:issues.append(dict(sector=sec,variant=variant,type='key_mismatch',severity='blocking',count=unmatched))
   for year,g in j.groupby(j.date.dt.year):
    for c in ['high','low','close']:
     a,b=g[c+'_merge'],g[c+'_raw'];diffs.append(dict(sector=sec,variant=variant,year=year,channel=c,different=int((a!=b).sum()),paired=len(g),nonpositive_merge=int((a<=0).sum()),nonfinite_merge=int((~np.isfinite(a)).sum())))
   bad=m[(m.date.dt.year==2022)&((m.close<=0)|~np.isfinite(m.close))]
   for z in bad[['date','tic','close']].to_dict('records'):issues.append(dict(sector=sec,variant=variant,type='nonpositive_prediction_feature',severity='record_not_clipped',**z))
 save_csv(RUN/'data_profiles.csv',profiles);save_csv(RUN/'selected_inputs.csv',selected);save_csv(RUN/'merge_differences.csv',diffs);save_csv(RUN/'data_issues.csv',issues,columns=None if issues else ['sector','variant','type','severity','count'])
 # Existing source/graph fingerprints are provenance inputs only, never executed or rebuilt.
 for f in (PROJECT/'code').glob('*.ipynb'):inputs[f]='source_provenance_only'
 for sec in ['bank','healthcare']:
  for variant in VARIANTS[1:]:
   r=resolve_merge(sec,variant).with_suffix('.json');inputs[r]='v4_merge_receipt';inputs[FORECASTS/load_json(r)['source']]='v4_forecast_ensemble_2022'
 for secfolder in ['bank sectordata','haelthcaresectordata']:
  for f in (PROJECT/secfolder).glob('*.npy'):inputs[f]='frozen_graph_or_array_provenance_only'
 rows=[dict(path=f.relative_to(PROJECT).as_posix(),role=role,sha256=sha(f),bytes=f.stat().st_size) for f,role in sorted(inputs.items())]
 manifest=RUN/'input_manifest.csv'
 if manifest.exists() and {r['path']:r['sha256'] for r in pd.read_csv(manifest).to_dict('records')}!={r['path']:r['sha256'] for r in rows}:raise RuntimeError('V4 inputs changed after freeze; retain this run and start a new version')
 if not manifest.exists():save_csv(manifest,rows)
 save_json(RUN/'author_clarification.json',dict(date='2026-10-25',source=str(author.relative_to(PROJECT)),source_sha256=sha(author),evidence_class='author_confirmed_not_third_party_certification',augmentation_scope='train_2022_only',validation_input='raw',test_input='raw',market_source='Tushare',price_basis='unadjusted',allow_RL_price_cost_analysis=True,company_actions='not supplied; no preexisting Tushare environment credential available',interpretation='Effect of training on one-step-ahead forecast-augmented 2022 observations on subsequent raw-observation policies; not online test-time forecasts',cutoff_interpretation='Offline 2022 augmentation allowed; evaluation-period contamination blocks only if affirmatively established'))
 env={'executable':sys.executable,'python':sys.version,'packages':{x:md.version(x) for x in ['finrl','stable-baselines3','torch','gym','gymnasium','numpy','pandas','stockstats','scikit-learn']},'device':'cpu','threads':1}
 save_json(RUN/'environment.json',env)
 old=V3RUN/'pip_freeze.txt'
 if old.exists() and not (RUN/'pip_freeze.txt').exists():(RUN/'pip_freeze.txt').write_bytes(old.read_bytes())
 save_json(RUN/'company_action_status.json',dict(status='missing_scope_limited_continue',existing_tushare_credential=bool(os.environ.get('TUSHARE_TOKEN')),cash_dividends=False,bonus_splits=False,cash_payment_dates=False,note='No token logged or requested. Author-authorized unadjusted fixed-unit cost experiment proceeds. No total-return/economic validation claim.'))
 print('V4 audit complete; forecast merges registered.',flush=True)

def protocol():
 settings=next(iter(load_json(ROOT/'original_settings.json').values()))
 p=dict(protocol_id=RUN.name+'_unadjusted_price_cost',accounting_scope=SCOPE,result_status='cost_analysis_scope_limited',economic_return_validated=False,input_manifest_sha256=sha(RUN/'input_manifest.csv'),environment_sha256=sha(RUN/'environment.json'),code=code_manifest(),initial_amount=1_000_000,reward_scaling=.1,steps=settings['TOTAL_TIMESTEPS'],model_params=settings['MODEL_PARAMS'],splits={'train':['2022-01-01','2022-12-31'],'validation':['2023-01-01','2023-12-31'],'test':['2024-01-01','2024-12-31']},augmentation_scope='2022 train observations only',warmup='pre2022 raw for all training; full independently raw history for validation/test',state='N x N return covariance plus macd/rsi_30/cci_30/dx_30; same N+4 x N dimensions',market_accounting='raw Tushare unadjusted close in all stages; no dividend/split events supplied; fixed fractional units',checkpoint_rule='20/40/60/80/100% of rollout-rounded target; 2023 raw F1 net SR; earliest tie',execution='t close decision, t+1 actual raw close execution; initial all cash; no terminal liquidation',phase1=[['bank','a2c'],['healthcare','sac'],['healthcare','ppo']],algorithms=['a2c','ppo','sac','td3','ddpg'],seeds=[1,2,3],ablation=ABLATION,e1_variants=E1_VARIANTS,variants=VARIANTS,cap=96,ridge={'alpha':1.,'k':3,'target':'raw close[t+2]/raw close[t+1]-1; labels entirely in2022','fit_scaler':'training rows only'},bootstrap={'blocks':[10,5,20],'B':2000,'seed':1729},forecast_provenance='v4 walk-forward forecaster: 20-day window of returns, ranges, trading value and valuation yields; targets H/L/C at t+1,t+2; fold 2022 trained on 2018-2020, early stopping on 2021, scalers fitted on training rows only; relation graphs as archived. Vanilla RL runs are reused from v3 (identical code path and inputs).',vanilla_source='runs/v3_train_only',test_seen='2024 previously examined; fixed historical evaluation')
 f=RUN/'protocol_v4.json'
 if f.exists() and canonical(load_json(f))!=canonical(p):raise RuntimeError('V4 protocol/code already locked')
 if not f.exists():save_json(f,p)
 return p

def integrity():
 rows=[]
 for z in pd.read_csv(RUN/'input_manifest.csv').to_dict('records'):
  f=PROJECT/z['path'];rows.append(dict(path=z['path'],unchanged=f.exists() and sha(f)==z['sha256']))
 save_csv(RUN/'input_integrity.csv',rows)
 if not all(x['unchanged'] for x in rows):raise RuntimeError('V4 input integrity failure')
