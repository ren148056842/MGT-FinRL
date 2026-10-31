"""v4 preparation: v3 state construction; training states from v4 forecast merges."""
import numpy as np
import pandas as pd
from stockstats import StockDataFrame
from v4_audit import ROOT,PROJECT,RUN,V3RUN,VARIANTS,SCOPE,canonicalize,resolve_merge,find_market,market_long,save_json,save_csv,load_json,sha,protocol
from v2_data import EpisodeData,RealPortfolioEnv as BaseEnv,export_trace,TECH
from v3_ledger import Ledger
from backtest import FEES,validate_splits

from v3_data import RealPortfolioEnv,sequence_states,clean_history,train_sequence,save_episode,episode,load_episode
class _Unused(BaseEnv):
 def reset(self,seed=None,options=None):
  import gymnasium as gym
  gym.Env.reset(self,seed=seed);self.i=0;self.ledger=Ledger(len(self.data.tickers),self.initial,self.costs);self.trace=[self._initial_row()];return self.data.states[0].copy(),{}

def sequence_states(d,lookback=252):
 d=d.sort_values(['date','tic']);tickers=sorted(d.tic.unique());prices=d.pivot(index='date',columns='tic',values='close').sort_index().reindex(columns=tickers);dates=prices.index;pr=prices.to_numpy(float)
 if not np.isfinite(pr).all():raise ValueError('Nonfinite or missing feature close; no fill allowed')
 tech=[]
 for t in tickers:
  z=d[d.tic==t].set_index('date').reindex(dates);s=StockDataFrame.retype(z[['high','low','close']].copy());tech.append(np.stack([np.asarray(s[k],float) for k in TECH],axis=1))
 tech=np.stack(tech,axis=2);states=[];valid=[]
 for i in range(lookback,len(dates)):
  with np.errstate(divide='ignore',invalid='ignore'):r=pr[i-lookback+1:i+1]/pr[i-lookback:i]-1
  s=np.concatenate([np.cov(r,rowvar=False,ddof=1),tech[i]],axis=0)
  if not np.isfinite(s).all():
   where=np.argwhere(~np.isfinite(s));raise ValueError('Nonfinite feature on %s at state(row,stock)=%s; no clipping/substitution'%(dates[i],where[:8].tolist()))
  states.append(s);valid.append(i)
 return pd.DatetimeIndex(dates[valid]),np.asarray(states,np.float32),tickers


def clean_history(sec):
 d=canonicalize(market_long(find_market(sec)),sec)
 if sec=='bank':d=d[d.date>=pd.Timestamp('2019-12-10')].copy()
 return d

def train_sequence(raw,merge):
 # raw pre2022, exactly selected merge HLC during2022, and no later rows at all.
 base=raw[raw.date<'2023-01-01'].copy();m=merge[merge.date.dt.year==2022]
 j=base.merge(m[['date','tic','high','low','close']],on=['date','tic'],how='left',suffixes=('','_m'),validate='one_to_one')
 mask=j.date.dt.year==2022
 for c in ['high','low','close']:
  if j.loc[mask,c+'_m'].isna().any():raise ValueError('Missing explicit2022 augmentation key '+c)
  j.loc[mask,c]=j.loc[mask,c+'_m']
 return j.drop(columns=['high_m','low_m','close_m'])

def save_episode(f,data):
 f.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(f,dates=data.dates.to_numpy(),prices=data.prices,states=data.states,tickers=np.array(data.tickers),tradable=data.tradable,multipliers=data.multipliers,dividends=data.dividends)

def episode(raw,dates,states,tickers,start,end):
 mask=(dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end));ds=dates[mask];market=raw.pivot(index='date',columns='tic',values='close').reindex(index=ds,columns=tickers).to_numpy(float)
 if not np.isfinite(market).all() or (market<=0).any():raise ValueError('Invalid actual market accounting prices')
 return EpisodeData(ds,market,states[mask],list(tickers),np.ones(market.shape,bool),np.ones(market.shape),np.zeros(market.shape),'fixed_fractional_units_of_unadjusted_raw_price')

def prepare():
 RUN.joinpath('prepared').mkdir(exist_ok=True);rows=[];checks=[]
 for sec in ['bank','healthcare']:
  raw=clean_history(sec);dates,states,tickers=sequence_states(raw)
  va=episode(raw,dates,states,tickers,'2023-01-01','2023-12-31');te=episode(raw,dates,states,tickers,'2024-01-01','2024-12-31');save_episode(RUN/'prepared'/(sec+'_validation_raw.npz'),va);save_episode(RUN/'prepared'/(sec+'_test_raw.npz'),te)
  for variant in VARIANTS:
   try:
    if variant=='Vanilla':tdates,tstates=dates,states
    else:
     merged=canonicalize(market_long(resolve_merge(sec,variant),strict=False),sec);seq=train_sequence(raw,merged);tdates,tstates,ttickers=sequence_states(seq)
     if list(ttickers)!=list(tickers):raise ValueError('Ticker order changed')
    tr=episode(raw,tdates,tstates,tickers,'2022-01-01','2022-12-31');validate_splits(tr.dates,va.dates,te.dates)
    f=RUN/'prepared'/(sec+'_'+variant+'_train.npz');save_episode(f,tr)
    meta=dict(sector=sec,variant=variant,train=f.name,validation=sec+'_validation_raw.npz',test=sec+'_test_raw.npz',train_sha256=sha(f),validation_sha256=sha(RUN/'prepared'/(sec+'_validation_raw.npz')),test_sha256=sha(RUN/'prepared'/(sec+'_test_raw.npz')),augmentation_scope='train_2022_only',accounting_scope=SCOPE,states_shape=list(tr.states.shape),tickers=list(tickers),training_market_raw=True,evaluation_history_all_raw=True)
    save_json(RUN/'prepared'/(sec+'_'+variant+'_metadata.json'),meta);rows.append(dict(status='ready',reason='Finite2022 observations, raw accounting; shared independently raw2023/2024 state files; unknown enhancer provenance disclosed, no established evaluation contamination',**meta))
    checks.append(dict(sector=sec,variant=variant,validation_identical_to_vanilla=True,test_identical_to_vanilla=True,validation_hash=meta['validation_sha256'],test_hash=meta['test_sha256']))
   except Exception as e:rows.append(dict(sector=sec,variant=variant,status='blocked_numerical',reason=repr(e)))
 save_csv(RUN/'preparation_status.csv',rows);save_csv(RUN/'raw_eval_equivalence.csv',checks);protocol()
 print('V4 prepared:',[(x['sector'],x['variant'],x['status']) for x in rows],flush=True)

def load_episode(f):
 with np.load(f,allow_pickle=False) as z:return EpisodeData(pd.DatetimeIndex(z['dates']),z['prices'],z['states'],list(z['tickers']),z['tradable'],z['multipliers'],z['dividends'],'fixed_fractional_units_of_unadjusted_raw_price')

def load_prepared(sector,variant='Vanilla'):
 m=load_json(RUN/'prepared'/(sector+'_'+variant+'_metadata.json'));parts=[]
 for k in ['train','validation','test']:
  f=RUN/'prepared'/m[k]
  if sha(f)!=m[k+'_sha256']:raise ValueError('Prepared phase checksum changed')
  parts.append(load_episode(f))
 a=parts[0];return EpisodeData(parts[0].dates.append([parts[1].dates,parts[2].dates]),np.concatenate([x.prices for x in parts]),np.concatenate([x.states for x in parts]),a.tickers,np.concatenate([x.tradable for x in parts]),np.concatenate([x.multipliers for x in parts]),np.concatenate([x.dividends for x in parts]),a.unit_label)
