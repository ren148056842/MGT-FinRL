import unittest
import numpy as np
import pandas as pd
from v3_ledger import Ledger
from backtest import Ledger as Ref
from v4_audit import RUN,V3RUN,VARIANTS,resolve_merge,canonicalize,market_long,find_market
from v4_data import load_prepared,load_episode,RealPortfolioEnv
class V4Tests(unittest.TestCase):
 def test_preparation_status_is_ready(self):
  d=pd.read_csv(RUN/'preparation_status.csv');self.assertEqual(len(d),2*len(VARIANTS));self.assertTrue((d.status=='ready').all(),d.to_string())
 def test_vanilla_identical_to_v3(self):
  # Vanilla RL runs are reused from v3: the prepared Vanilla arrays must be identical.
  for sec in ['bank','healthcare']:
   for part in ['Vanilla_train','validation_raw','test_raw']:
    a=load_episode(RUN/'prepared'/('%s_%s.npz'%(sec,part)));b=load_episode(V3RUN/'prepared'/('%s_%s.npz'%(sec,part)))
    np.testing.assert_array_equal(a.states,b.states);np.testing.assert_array_equal(a.prices,b.prices);self.assertTrue((a.dates==b.dates).all());self.assertEqual(a.tickers,b.tickers)
 def test_fast_ledger_matches_reference(self):
  rng=np.random.default_rng(7);a=Ledger(15);b=Ref(15)
  for i in range(60):
   p=rng.uniform(10,30,15);w=rng.dirichlet(np.ones(15));date=pd.Timestamp('2024-01-01')+pd.Timedelta(days=i);x=a.rebalance(date,p,w);y=b.rebalance(date,p,w)
   for k in ['nav','cash','buy','sell','cost']:np.testing.assert_allclose(x[k],y[k],rtol=1e-10,atol=1e-6)
 def test_merges_change_only_2022(self):
  for sec in ['bank','healthcare']:
   raw=canonicalize(market_long(find_market(sec)),sec)
   for var in VARIANTS[1:]:
    m=canonicalize(market_long(resolve_merge(sec,var),strict=False),sec);j=m.merge(raw,on=['date','tic'],suffixes=('_m','_r'),validate='one_to_one')
    self.assertEqual(len(j),len(raw))
    out=j.date.dt.year!=2022
    for c in ['high','low','close']:np.testing.assert_array_equal(j.loc[out,c+'_m'].to_numpy(),j.loc[out,c+'_r'].to_numpy())
    self.assertTrue((j.loc[~out,'close_m']!=j.loc[~out,'close_r']).mean()>0.9)
    self.assertTrue((j.loc[~out,['high_m','low_m','close_m']]>0).all().all())
 def test_eval_inputs_identical(self):
  for sec in ['bank','healthcare']:
   a=load_prepared(sec,'Vanilla')
   for var in VARIANTS[1:]:
    b=load_prepared(sec,var)
    for year in [2023,2024]:
     mask=a.dates.year==year;np.testing.assert_array_equal(a.states[mask],b.states[mask]);np.testing.assert_array_equal(a.prices[mask],b.prices[mask])
    mask=a.dates.year==2022;self.assertFalse(np.allclose(a.states[mask],b.states[mask]));np.testing.assert_array_equal(a.prices,b.prices)
 def test_raw_reward(self):
  a=load_prepared('bank','Vanilla');b=load_prepared('bank',VARIANTS[1])
  ea=RealPortfolioEnv(a.subset('2022-01-01','2022-12-31'));eb=RealPortfolioEnv(b.subset('2022-01-01','2022-12-31'))
  for _ in range(6):
   x=ea.step(np.zeros(15));y=eb.step(np.zeros(15));self.assertAlmostEqual(x[1],y[1]);self.assertAlmostEqual(x[4]['nav'],y[4]['nav'])
if __name__=='__main__':unittest.main()
