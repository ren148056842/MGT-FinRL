"""Optional event-aware fixed-checkpoint review, never a replacement for unadjusted training.
Receivables recognized on ex date; cash unavailable until pay date; bonus units locked until listing.
Only complete source events are accepted; no invented payment/listing dates.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import pathlib,sys,json
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from v3_audit import RUN,PROJECT,save_json,save_csv,sha,load_json,canonical
from v3_ledger import Ledger
from backtest import FEES,metrics


def event_run(data,action_fn,costs,events,initial=1e6):
 ledger=Ledger(len(data.tickers),initial,costs);pending=[];receivables=[];entitlements={};rows=[];holdings=[]
 def snapshot(day):
  for eid,e in enumerate(events):
   if e['record_date']==day:entitlements[eid]=float(ledger.q[e['j']]+sum(v['q'] for v in pending if v['j']==e['j']))
 def values(price):
  stock=np.zeros(len(price))
  for x in pending:stock[x['j']]+=x['q']
  rec=sum(x['cash'] for x in receivables);return stock,rec
 day=str(data.dates[0].date());snapshot(day);rows.append(dict(date=day,decision_date='',nav=initial,liquid_nav=initial,cash=initial,dividend_receivable=0.,pending_stock_value=0.,dividend_cash_paid=0.,buy=0.,sell=0.,cost=0.,gross_turnover=0.,daily_return=np.nan));previous=initial
 for i in range(1,len(data.dates)):
  day=str(data.dates[i].date());p=data.prices[i];released=0.
  for eid,e in enumerate(events):
   if e['ex_date']==day:
    qty=entitlements.get(eid,0.)
    if e['cash_per_share']>0:receivables.append(dict(cash=qty*e['cash_per_share'],pay=e['pay_date']))
    if e['stock_per_share']>0:pending.append(dict(j=e['j'],q=qty*e['stock_per_share'],listed=e['listing_date']))
  waiting=[]
  for x in receivables:
   if x['pay']<=day:ledger.cash+=x['cash'];released+=x['cash']
   else:waiting.append(x)
  receivables=waiting;waiting=[]
  for x in pending:
   if x['listed']<=day:ledger.q[x['j']]+=x['q']
   else:waiting.append(x)
  pending=waiting;stock,rec=values(p);locked=float(stock@p);w=action_fn(i-1)
  if w is None:
   nav=ledger.nav(p);row=dict(date=day,pre_nav=nav,nav=nav,cash=ledger.cash,buy=0.,sell=0.,commission=0.,slippage=0.,tax=0.,cost=0.,gross_turnover=0.)
  else:row=ledger.rebalance(data.dates[i],p,w)
  liquid=row['nav'];nav=liquid+rec+locked;row.update(date=day,decision_date=str(data.dates[i-1].date()),liquid_nav=liquid,nav=nav,dividend_receivable=rec,pending_stock_value=locked,dividend_cash_paid=released,gross_turnover=(row['buy']+row['sell'])/(row.get('pre_nav',liquid)+rec+locked),daily_return=nav/previous-1)
  if ledger.cash < -1e-7 or not np.isclose(nav,ledger.cash+float(ledger.q@p)+rec+locked,rtol=1e-12):raise ArithmeticError('Event ledger identity failed')
  rows.append({k:v for k,v in row.items() if k not in ['shares','target_weights','actual_weights']});snapshot(day);previous=nav
  for j,tic in enumerate(data.tickers):holdings.append(dict(date=day,tic=tic,available_quantity=ledger.q[j],pending_quantity=stock[j],economic_security_weight=(ledger.q[j]+stock[j])*p[j]/nav))
 return pd.DataFrame(rows),pd.DataFrame(holdings)


def parse_date(x):
 if pd.isna(x):return None
 s=str(x).replace('.0','')
 try:return str(pd.to_datetime(s,format='%Y%m%d').date())
 except ValueError:return None

def load_events(sector,tickers):
 cache=ROOT/'local_event_cache/v3_tushare';ref=pd.read_csv(cache/'security_reference.csv')
 bank={'PAB':'000001.SZ','SPDB':'600000.SH','HXB':'600015.SH','CMBC':'600016.SH','CMB':'600036.SH','CIB':'601166.SH','ABC':'601288.SH','BOCOM':'601328.SH','ICBC':'601398.SH','PSBC':'601658.SH','CEB':'601818.SH','CZB':'601916.SH','CCB':'601939.SH','BOC':'601988.SH','ECITIC':'601998.SH'}
 events=[];issues=[];sources={}
 for j,tic in enumerate(tickers):
  code=bank[tic] if sector=='bank' else tic;f=cache/(code.replace('.','_')+'_events2021_2024.csv')
  if not f.exists():issues.append(dict(tic=tic,issue='event query/cache incomplete'));continue
  sources[f.name]=sha(f);d=pd.read_csv(f).drop_duplicates()
  for row in d.to_dict('records'):
   ex=parse_date(row.get('ex_date'))
   if ex is None or not ('2024-01-01'<=ex<='2024-12-31'):continue
   record=parse_date(row.get('record_date'));pay=parse_date(row.get('pay_date'));listed=parse_date(row.get('div_listdate'));cash=row.get('cash_div_tax');stock=row.get('stk_div')
   if pd.isna(cash) or pd.isna(stock):issues.append(dict(tic=tic,ex_date=ex,issue='cash or stock amount missing'));continue
   cash=float(cash);stock=float(stock)
   if record is None or record>=ex or (cash>0 and (pay is None or pay<ex)) or (stock>0 and (listed is None or listed<ex)) or cash<0 or stock<0:
    issues.append(dict(tic=tic,ex_date=ex,issue='record/payment/stock-list timing or amount incomplete'));continue
   events.append(dict(j=j,tic=tic,record_date=record,ex_date=ex,pay_date=pay,listing_date=listed,cash_per_share=cash,stock_per_share=stock))
 return events,issues,sources


def review():
 import torch
 torch.set_num_threads(1)
 from v3_data import load_prepared
 from v3_engine import algorithms,plan
 out=RUN/'event_accounting_review';out.mkdir(exist_ok=True);summaries=[];status=[];sourcehash={};accounting={}
 for sec in ['bank','healthcare']:
  data=load_prepared(sec,'Vanilla').subset('2024-01-01','2024-12-31');events,issues,sources=load_events(sec,data.tickers);sourcehash.update(sources);accounting[sec]=(events,issues)
  status.append(dict(sector=sec,events2024=len(events),unresolved=len(issues),status='ready' if not issues else 'incomplete_do_not_validate'))
  if issues:save_csv(out/(sec+'_unresolved_events.csv'),issues)
 for task in plan():
  if task['phase']!=1:continue
  parent=RUN/'models'/task['task_id'];ckpt=parent/'best_policy.zip';events,issues=accounting[task['sector']]
  if issues or not (parent/'receipt.json').exists():continue
  data=load_prepared(task['sector'],task['variant']).subset('2024-01-01','2024-12-31');model=algorithms()[task['algorithm']].load(str(ckpt),device='cpu')
  for fee in ['F0','F1']:
   name=task['task_id']+'_'+fee;directory=out/name;receipt=directory/'metadata.json';ctx=dict(checkpoint_sha256=sha(ckpt),event_sources=sourcehash,code_sha256=sha(__file__),accounting='event_aware_gross_fixed_policy_review_v3')
   if receipt.exists() and load_json(receipt).get('context')==ctx:
    summaries.append(load_json(directory/'metrics.json'));continue
   def action(i):
    a,_=model.predict(data.states[i],deterministic=True);a=np.asarray(a,float);w=np.exp(a-a.max());return w/w.sum()
   tr,pos=event_run(data,action,FEES[fee],events);directory.mkdir(exist_ok=True);meta=dict(run_id=name,protocol_id='v3_fixed_policy_event_aware_gross',accounting_scope='event_aware_gross_fixed_policy_review',training_rewards='unadjusted_without_events_NOT_retrained',sector=task['sector'],algorithm=task['algorithm'],variant=task['variant'],seed=task['seed'],fee=fee,economic_return_validated=False,cash_bonus_event_accounting_validated=True,economic_scope='Gross dividend and fractional stock entitlements; no investor-specific dividend withholding tax, unobserved execution constraints remain',pending_policy='Apply target weights to liquid deployable NAV; locked stock/receivables cannot finance purchases',checkpoint_sha256=sha(ckpt),event_code_sha256=sha(__file__))
   for k,v in meta.items():tr[k]=v
   save_csv(directory/'trajectory.csv',tr);save_csv(directory/'positions.csv',pos);m={**metrics(tr.daily_return.iloc[1:]),**meta,'final_nav':tr.nav.iloc[-1],'cumulative_cost':tr.cost.sum()};save_json(directory/'metrics.json',m);save_json(receipt,dict(context=ctx,**meta));summaries.append(m)
 save_csv(out/'event_source_status.csv',status);save_csv(out/'summary.csv',summaries,columns=None if summaries else ['run_id','accounting_scope','economic_return_validated'])
 save_json(out/'protocol_event_review.json',dict(protocol_id='v3_fixed_policy_event_aware_gross',source_document='https://tushare.pro/document/2?doc_id=103',sources=sourcehash,code_sha256=sha(__file__),primary_training_unchanged=True,scope='Only fixed-priority-checkpoint supplementary evaluation. No event-aware training claim.',status=status))
 print('Fixed-policy event review outputs',len(summaries),status,flush=True)

if __name__=='__main__':review()
