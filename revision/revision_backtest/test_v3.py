import unittest
import numpy as np
import pandas as pd
from v3_ledger import Ledger
from backtest import Ledger as Ref,Costs
from v3_audit import RUN,names,canonicalize,find_market,market_long
from v3_data import sequence_states,train_sequence,clean_history,episode,load_prepared,RealPortfolioEnv
class V3Tests(unittest.TestCase):
 def test_preparation_status_is_ready(self):
  d=pd.read_csv(RUN/'preparation_status.csv');self.assertEqual(len(d),16);self.assertTrue((d.status=='ready').all(),d.to_string())
 def test_fast_ledger_matches_reference(self):
  rng=np.random.default_rng(7);a=Ledger(15);b=Ref(15)
  for i in range(60):
   p=rng.uniform(10,30,15);w=rng.dirichlet(np.ones(15));date=pd.Timestamp('2024-01-01')+pd.Timedelta(days=i);x=a.rebalance(date,p,w);y=b.rebalance(date,p,w)
   for k in ['nav','cash','buy','sell','cost']:np.testing.assert_allclose(x[k],y[k],rtol=1e-10,atol=1e-6)
 def test_pad_pab_and_distinct_cmb_cmbc(self):
  m=names();bank={z['raw_tic']:z['tic'] for z in m if z['sector']=='bank'};self.assertEqual(bank['PAD'],'PAB');self.assertEqual(bank['CMB'],'CMB');self.assertEqual(bank['CMBC'],'CMBC');self.assertEqual(len(set(bank.values())),15)
 def test_eval_inputs_and_full_warmup_identical(self):
  for sec in ['bank','healthcare']:
   a=load_prepared(sec,'Vanilla')
   for var in ['A0','A1','A2','A6','A7','A8','A9']:
    if not (RUN/'prepared'/(sec+'_'+var+'_metadata.json')).exists():continue
    b=load_prepared(sec,var)
    for year in [2023,2024]:
     mask=a.dates.year==year;np.testing.assert_array_equal(a.states[mask],b.states[mask]);np.testing.assert_array_equal(a.prices[mask],b.prices[mask])
 def test_merge_training_only_and_raw_reward(self):
  a=load_prepared('bank','Vanilla');b=load_prepared('bank','A0');mask=a.dates.year==2022
  self.assertFalse(np.allclose(a.states[mask],b.states[mask]));np.testing.assert_array_equal(a.prices,b.prices)
  ea=RealPortfolioEnv(a.subset('2022-01-01','2022-12-31'));eb=RealPortfolioEnv(b.subset('2022-01-01','2022-12-31'))
  for _ in range(6):
   x=ea.step(np.zeros(15));y=eb.step(np.zeros(15));self.assertAlmostEqual(x[1],y[1]);self.assertAlmostEqual(x[4]['nav'],y[4]['nav'])
 def test_future2024_cannot_change_training_state(self):
  raw=clean_history('bank');d1,s1,t=sequence_states(raw);changed=raw.copy();changed.loc[changed.date.dt.year==2024,['high','low','close']]*=3;d2,s2,t=sequence_states(changed);np.testing.assert_array_equal(s1[d1.year==2022],s2[d2.year==2022])
 def test_pre2022_merge_replacement_not_used(self):
  raw=clean_history('bank');merge=raw.copy();merge.loc[merge.date.dt.year<2022,['high','low','close']]*=30;seq=train_sequence(raw,merge);part=seq.date.dt.year<2022
  expected=raw[(raw.date<'2022-01-01')];np.testing.assert_allclose(seq.loc[part,'close'],expected.close)
 def test_ridge_labels_within_train(self):
  d=load_prepared('healthcare','A0').subset('2022-01-01','2022-12-31');y=d.prices[2:]/d.prices[1:-1]-1;self.assertEqual(len(y),len(d.states)-2);self.assertEqual(d.dates[-1].year,2022)
 def test_initial_cost_delay_and_no_terminal_liquidation(self):
  d=load_prepared('bank','Vanilla').subset('2024-01-01','2024-12-31');e=RealPortfolioEnv(d);_,_,_,_,x=e.advance(np.ones(15)/15);self.assertGreater(x['cost'],0);self.assertAlmostEqual(x['nav'],1e6-x['cost']);self.assertEqual(e.trace[-1]['decision_date'],str(d.dates[0].date()))
if __name__=='__main__':unittest.main()
