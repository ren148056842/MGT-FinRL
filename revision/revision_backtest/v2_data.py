"""Causal real-market adapter and Gymnasium environment; distinct from v1 fixture."""
from dataclasses import dataclass
import pathlib
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stockstats import StockDataFrame
from backtest import Ledger,FEES,Costs,validate_prices
from v2_audit import PROJECT,RUN,panel,load_json,save_json,save_csv,sha

TECH=['macd','rsi_30','cci_30','dx_30']
KNOWN_LISTING={'bank':{'CZB':'2019-11-26','PSBC':'2019-12-10'},'healthcare':{}}

@dataclass
class EpisodeData:
 dates: pd.DatetimeIndex
 prices: np.ndarray
 states: np.ndarray
 tickers: list
 tradable: np.ndarray
 multipliers: np.ndarray
 dividends: np.ndarray
 unit_label: str='unverified_price_sequence_units'
 def subset(self,start,end):
  mask=(self.dates>=pd.Timestamp(start))&(self.dates<=pd.Timestamp(end));ix=np.flatnonzero(mask)
  if len(ix)<2:raise ValueError('Insufficient nonoverlapping episode dates')
  return EpisodeData(self.dates[ix],self.prices[ix],self.states[ix],self.tickers,self.tradable[ix],self.multipliers[ix],self.dividends[ix],self.unit_label)


def causal_states(long_df,lookback=252):
 """Same fields/dimension as documented portfolio state; no all-period fitted scaler.
 Remove known prelisting rows BEFORE any indicator or covariance. Compute indicators per ticker.
 """
 d=long_df.copy().sort_values(['date','tic']);tickers=sorted(d.tic.unique().tolist())
 counts=d.groupby('date').tic.nunique();dates=pd.DatetimeIndex(counts[counts==len(tickers)].index)
 if len(dates)!=d.date.nunique():raise ValueError('Unbalanced dates; no forward/backward fill allowed')
 p=d.pivot(index='date',columns='tic',values='close').reindex(index=dates,columns=tickers);validate_prices(p)
 features=[]
 for t in tickers:
  z=d[d.tic==t].set_index('date').reindex(dates)
  sd=StockDataFrame.retype(z[['high','low','close']].copy())
  features.append(np.stack([np.asarray(sd[k],float) for k in TECH],axis=1))
 # time x 4 x stocks
 tech=np.stack(features,axis=2);pr=p.to_numpy(float);states=[];valid_dates=[];positions=[]
 for i in range(lookback,len(dates)):
  returns=pr[i-lookback+1:i+1]/pr[i-lookback:i]-1.
  cov=np.cov(returns,rowvar=False,ddof=1);state=np.concatenate([cov,tech[i]],axis=0)
  if not np.isfinite(state).all():raise ValueError('Nonfinite causal state on '+str(dates[i]))
  states.append(state);valid_dates.append(dates[i]);positions.append(i)
 if not states:raise ValueError('No valid 252-return warmup')
 return pd.DatetimeIndex(valid_dates),pr[positions],np.asarray(states,dtype=np.float32),tickers


def build_data(sector,market_status,variant='Vanilla'):
 d,p=panel(sector);excluded=[]
 for t,date in KNOWN_LISTING.get(sector,{}).items():
  mask=(d.tic==t)&(d.date<pd.Timestamp(date));excluded.append(dict(sector=sector,tic=t,before=date,excluded_rows=int(mask.sum()),purpose='Never used in state warmup/training/execution'));d=d[~mask]
 # For a fixed-stock panel use the common real history starting only once all known listings exist.
 common_start=max([d[d.tic==t].date.min() for t in d.tic.unique()]);d=d[d.date>=common_start]
 save_csv(RUN/'prepared'/(sector+'_listing_exclusions.csv'),excluded,columns=['sector','tic','before','excluded_rows','purpose'])
 dates,prices,states,tickers=causal_states(d)
 if variant!='Vanilla':
  contract=forecast_contract(sector,market_status,variant)
  if not contract['valid']:raise ValueError(contract['reason'])
  with np.load(PROJECT/contract['bundle'],allow_pickle=False) as z:
   ix=pd.DatetimeIndex(z['decision_dates']).get_indexer(dates)
   # Only dates used in declared train/validation/test need forecast state; no future price filling.
   needed=dates>=pd.Timestamp('2022-01-01')
   if (ix[needed]<0).any():raise ValueError('Missing MGT decisions')
   feat=np.empty((len(dates),4,len(tickers)),dtype=np.float32);feat[:]=np.nan
   feat[needed]=z['values'][ix[needed]]
   states[needed,-4:,:]=feat[needed]
 # Known status may be supplied, but never inferred from missing volume.
 tradable=np.ones(prices.shape,bool);multipliers=np.ones(prices.shape,float);dividends=np.zeros(prices.shape,float)
 basis=market_status['price'];unit='total_return_index_fractional_units' if basis['price_basis']=='verified_total_return_index' else ('actual_fractional_equity_shares' if basis['formal'] else 'unverified_price_sequence_units')
 source=basis.get('source',{})
 if basis['formal'] and basis['price_basis']=='verified_unadjusted_with_actions':
  a=pd.read_csv(PROJECT/source['actions_file']);a['date']=pd.to_datetime(a.date)
  if a.duplicated(['date','tic']).any():raise ValueError('Duplicate corporate actions; aggregate explicitly upstream')
  for row in a.to_dict('records'):
   if row['tic'] not in tickers:raise ValueError('Action ticker not in market')
   if row['date'] in dates:
    i=dates.get_loc(row['date']);j=tickers.index(row['tic']);mult=float(row['share_multiplier']);div=float(row['cash_per_old_share'])
    if mult<=0 or not np.isfinite(mult) or not np.isfinite(div):raise ValueError('Invalid corporate action')
    multipliers[i,j]=mult;dividends[i,j]=div
 if source.get('tradability_file'):
  f=PROJECT/source['tradability_file']
  if sha(f)!=source.get('tradability_sha256'):raise ValueError('Trading mask evidence hash mismatch')
  t=pd.read_csv(f);t['date']=pd.to_datetime(t.date)
  pp=t.pivot(index='date',columns='tic',values='tradable').reindex(index=dates,columns=tickers)
  if pp.isna().any().any():raise ValueError('Explicit tradability mask coverage incomplete')
  tradable=pp.astype(bool).to_numpy()
 data=EpisodeData(dates,prices,states,tickers,tradable,multipliers,dividends,unit)
 np.savez_compressed(RUN/'prepared'/(sector+'_'+variant+'.npz'),dates=dates.to_numpy(),prices=prices,states=states,tickers=np.array(tickers),tradable=tradable,multipliers=multipliers,dividends=dividends)
 save_json(RUN/'prepared'/(sector+'_'+variant+'_metadata.json'),dict(common_history_start=common_start,first_state=dates[0],last_state=dates[-1],lookback=252,state_shape=list(states.shape[1:]),tickers=tickers,fields=['real_return_covariance']+TECH,normalization='none; no global fitted scaler or bfill',unit_label=unit,paper_eligible=basis['formal'],input_hash=market_status['sha256'],prepared_sha256=sha(RUN/'prepared'/(sector+'_'+variant+'.npz'))))
 return data


def load_prepared(sector,variant='Vanilla'):
 meta=load_json(RUN/'prepared'/(sector+'_'+variant+'_metadata.json'))
 if sha(RUN/'prepared'/(sector+'_'+variant+'.npz'))!=meta['prepared_sha256']:raise ValueError('Prepared data checksum changed')
 with np.load(RUN/'prepared'/(sector+'_'+variant+'.npz'),allow_pickle=False) as z:
  return EpisodeData(pd.DatetimeIndex(z['dates']),z['prices'],z['states'],list(z['tickers']),z['tradable'],z['multipliers'],z['dividends'],meta['unit_label'])


def forecast_contract(sector,status,variant='A0'):
 f=PROJECT/'backtest_source_evidence'/(sector+'_'+variant+'_frozen_features.json')
 if variant=='A0' and not f.exists():f=PROJECT/'backtest_source_evidence'/(sector+'_frozen_features.json')
 if not f.exists():
  inv=RUN/'forecast_inventory.csv';x=pd.read_csv(inv);x=x[x.sector==sector]
  ranges=sorted(set(str(v) for v in x.target_last.dropna()))
  return dict(valid=False,reason='Existing dated target ends: '+str(ranges)+'; undated matrices require certified mapping/scale; no 2023/2024 causal feature record or scaler/graph cutoff proof',evidence='forecast_inventory.csv; archive/v1/notebook_evidence.csv')
 try:
  j=load_json(f);source=PROJECT/j['temporal_proof_record'];bundle=PROJECT/j['bundle']
  if sha(source)!=j['temporal_proof_sha256'] or sha(bundle)!=j['bundle_sha256']:raise ValueError('Frozen source/bundle hash mismatch')
  if not j.get('strict_oos') or not j.get('no_heldout_scaler_fit'):raise ValueError('Preprocessing/forecast future leakage not cleared by provenance')
  if pd.Timestamp(j['model_label_cutoff'])>=pd.Timestamp('2022-01-01') or pd.Timestamp(j['scaler_fit_cutoff'])>=pd.Timestamp('2022-01-01') or pd.Timestamp(j['graph_information_cutoff'])>=pd.Timestamp('2022-01-01'):raise ValueError('Cutoff after required earliest decision window')
  with np.load(bundle,allow_pickle=False) as z:
   ds=pd.DatetimeIndex(z['decision_dates']);target=pd.DatetimeIndex(z['target_dates']);avail=pd.DatetimeIndex(z['derived_available_at']);tickers=list(z['tickers'])
   if ds.has_duplicates or not ds.is_monotonic_increasing or ds.hasnans or target.hasnans or avail.hasnans or not (avail<=ds).all() or not (target>ds).all():raise ValueError('Invalid decision/target/availability alignment')
   if tickers!=status['stocks'] or z['values'].shape!=(len(ds),4,len(tickers)) or not np.isfinite(z['values']).all():raise ValueError('Feature fields/ticker order/shape/scale invalid')
   _,p=panel(sector);needed=p.index[p.index.year.isin([2022,2023,2024])]
   if not needed.isin(ds).all():raise ValueError('2022/2023/2024 decision coverage incomplete')
  return dict(valid=True,bundle=j['bundle'],reason='Frozen decision-date 4-row feature replacement validated against source proof; no prediction execution',evidence=str(f.relative_to(PROJECT)))
 except (KeyError,ValueError,OSError) as err:return dict(valid=False,reason=str(err),evidence=str(f.relative_to(PROJECT)))


class RealPortfolioEnv(gym.Env):
 """Real covariance+technical-indicator state. Accounting sees only market_raw."""
 metadata={'render_modes':[]}
 def __init__(self,data,costs=FEES['F1'],initial=1_000_000.,reward_scaling=.1):
  super().__init__();self.data=data;self.costs=costs;self.initial=initial;self.reward_scaling=reward_scaling;n=len(data.tickers)
  if len(data.dates)<2 or data.states.shape[0]!=len(data.dates) or not np.isfinite(data.states).all():raise ValueError('Invalid episode states')
  self.action_space=spaces.Box(-1,1,shape=(n,),dtype=np.float32);self.observation_space=spaces.Box(-np.inf,np.inf,shape=data.states.shape[1:],dtype=np.float32)
  self.reset()
 def _initial_row(self):return dict(date=str(self.data.dates[0].date()),decision_date='',nav=self.initial,pre_nav=self.initial,cash=self.initial,daily_return=np.nan,buy=0.,sell=0.,commission=0.,slippage=0.,tax=0.,cost=0.,gross_turnover=0.,shares=self.ledger.q.copy(),target_weights=np.zeros(len(self.data.tickers)),actual_weights=np.zeros(len(self.data.tickers)),unfilled_buy_value=0.,blocked_symbols=0,dividend_cash=0.)
 def reset(self,seed=None,options=None):
  super().reset(seed=seed);self.i=0;self.ledger=Ledger(len(self.data.tickers),self.initial,self.costs);self.trace=[self._initial_row()];return self.data.states[0].copy(),{}
 def advance(self,weights):
  if self.i>=len(self.data.dates)-1:raise RuntimeError('Episode already terminated')
  decision=self.data.dates[self.i];oldnav=self.ledger.nav(self.data.prices[self.i]);self.i+=1;i=self.i;p=self.data.prices[i]
  div=float(self.ledger.q@self.data.dividends[i]);self.ledger.cash+=div;self.ledger.q*=self.data.multipliers[i]
  if weights is None:
   nav=self.ledger.nav(p);r=dict(date=str(self.data.dates[i].date()),pre_nav=nav,nav=nav,cash=self.ledger.cash,buy=0.,sell=0.,commission=0.,slippage=0.,tax=0.,cost=0.,gross_turnover=0.,shares=self.ledger.q.copy(),target_weights=np.full(len(p),np.nan),actual_weights=self.ledger.q*p/nav,unfilled_buy_value=0.,blocked_symbols=int((~self.data.tradable[i]).sum()))
  else:r=self.ledger.rebalance(self.data.dates[i],p,weights,self.data.tradable[i])
  r.update(decision_date=str(decision.date()),daily_return=r['nav']/oldnav-1.,dividend_cash=div);self.trace.append(r)
  done=self.i==len(self.data.dates)-1
  return self.data.states[i].copy(),float(r['daily_return']*self.reward_scaling),done,False,{'nav':r['nav'],'cost':r['cost']}
 def step(self,action):
  a=np.asarray(action,dtype=np.float64).reshape(-1)
  if not np.isfinite(a).all():raise ValueError('Nonfinite policy action')
  w=np.exp(a-a.max());w/=w.sum();return self.advance(w)


def export_trace(env,directory,metadata):
 directory=pathlib.Path(directory);directory.mkdir(parents=True,exist_ok=True);rows=[];positions=[]
 for r in env.trace:
  rows.append({**{k:v for k,v in r.items() if k not in ['shares','target_weights','actual_weights']},**metadata})
  for j,t in enumerate(env.data.tickers):positions.append(dict(date=r['date'],tic=t,quantity=float(r['shares'][j]),target_weight=float(r['target_weights'][j]),actual_weight=float(r['actual_weights'][j]),unit_label=env.data.unit_label,**metadata))
 save_csv(directory/'trajectory.csv',rows);save_csv(directory/'positions.csv',positions)
 save_csv(directory/'trades.csv',[r for r in rows if r['buy']>1e-7 or r['sell']>1e-7],columns=list(rows[0]))
 return pd.DataFrame(rows)
