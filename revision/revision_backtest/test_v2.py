"""Incremental regressions: actual adapters/state and execution routing, no synthetic RL retraining."""
import pathlib,tempfile,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from v2_audit import ROOT,RUN,delta,sha,save_json,load_json,panel
from v2_data import causal_states,EpisodeData,RealPortfolioEnv,forecast_contract
from v2_engine import formal_gate,cache_valid,finish_receipt,train_model,rollout_model
from v2_reports import collect_metrics,cost_differences
from backtest import Costs

class V2Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.raw,_=panel('bank')
 def fixture(self):
  dates=pd.to_datetime(['2022-01-04','2022-01-05','2022-01-06','2023-01-03','2023-01-04','2023-01-05','2024-01-02','2024-01-03','2024-01-04'])
  prices=np.tile([10.,20.],(9,1));states=np.zeros((9,6,2),np.float32)
  return EpisodeData(dates,prices,states,['a','b'],np.ones((9,2),bool),np.ones((9,2)),np.zeros((9,2)))
 def test_new_raw_path_and_schema(self):
  from v2_audit import find_market,market_long
  f=find_market('healthcare');d=market_long(f)
  self.assertEqual(f.suffix,'.CSV');self.assertIn('close',d);self.assertFalse(d.duplicated(['date','tic']).any())
 def test_manifest_added_is_not_modified(self):
  d=delta([dict(path='old',sha256='a'),dict(path='new',sha256='b')],{'old':'a'})
  self.assertEqual({r['path']:r['change'] for r in d},{'new':'added','old':'unchanged'})
 def test_manifest_detects_modified_and_deleted(self):
  d=delta([dict(path='old',sha256='b')],{'old':'a','gone':'c'});self.assertEqual({r['change'] for r in d},{'modified','deleted'})
 def test_causal_real_state_prefix(self):
  raw=self.raw[(self.raw.date>='2020-01-02')&(self.raw.date<='2022-12-31')].copy();dates=sorted(raw.date.unique());cut=dates[330]
  a,p,s,t=causal_states(raw);b,pb,sb,tb=causal_states(raw[raw.date<=cut])
  np.testing.assert_allclose(s[:len(sb)],sb,rtol=1e-7,atol=1e-8);self.assertEqual(s.shape[1:],(len(t)+4,len(t)));self.assertEqual(t,tb)
 def test_future_raw_changes_do_not_change_past_state(self):
  raw=self.raw[(self.raw.date>='2020-01-02')&(self.raw.date<='2022-12-31')].copy();dates=sorted(raw.date.unique());cut=dates[330]
  a,p,s,t=causal_states(raw);changed=raw.copy();changed.loc[changed.date>cut,['high','low','close']]*=5;_,_,b,_=causal_states(changed)
  mask=a<=cut;np.testing.assert_allclose(s[mask],b[mask],rtol=1e-7,atol=1e-8)
 def test_vanilla_does_not_require_forecast(self):
  task={'sector':'bank','variant':'Vanilla'};p={'market':{'bank':{'structural_valid':True,'price':{'formal':True,'reason':'fixture'}}}}
  with patch('v2_engine.forecast_contract',side_effect=AssertionError('Vanilla must not read forecasts')):self.assertTrue(formal_gate(task,p)[0])
 def test_price_gate_not_bypassed(self):
  p={'market':{'bank':{'structural_valid':True,'price':{'formal':False,'reason':'no source'}}}}
  self.assertFalse(formal_gate({'sector':'bank','variant':'Vanilla'},p)[0])
 def test_mgt_separate_missing_forecast(self):
  p={'market':{'bank':{'structural_valid':True,'price':{'formal':True,'reason':'fixture'}}}}
  with patch('v2_engine.forecast_contract',return_value={'valid':False,'reason':'missing actual coverage'}):self.assertFalse(formal_gate({'sector':'bank','variant':'A0'},p)[0])
 def test_env_reward_uses_market_not_state(self):
  a=self.fixture().subset('2024-01-01','2024-12-31');b=self.fixture().subset('2024-01-01','2024-12-31');b.states[:]=100000
  e1=RealPortfolioEnv(a);e2=RealPortfolioEnv(b)
  for _ in range(2):
   x=e1.step(np.array([.5,-.5]));y=e2.step(np.array([.5,-.5]));self.assertAlmostEqual(x[1],y[1]);self.assertAlmostEqual(x[4]['nav'],y[4]['nav'])
 def test_real_delay_and_corporate_action(self):
  d=self.fixture().subset('2024-01-01','2024-12-31');d.prices[:]=[[10,10],[20,10],[10,10]];d.multipliers[2,0]=2;d.dividends[2,0]=1
  e=RealPortfolioEnv(d,Costs(0,0,False));e.advance(np.array([1.,0.]));self.assertEqual(e.trace[-1]['nav'],1_000_000)
  e.advance(None);self.assertEqual(e.trace[-1]['nav'],1_050_000);self.assertEqual(e.ledger.q[0],100000)
 def test_cache_rejects_checkpoint_change(self):
  with tempfile.TemporaryDirectory() as name:
   f=pathlib.Path(name);(f/'checkpoint.zip').write_bytes(b'TEST ONLY');context={'input':'a'};finish_receipt(f,context,{})
   self.assertTrue(cache_valid(f,context));self.assertFalse(cache_valid(f,{'input':'b'}));(f/'checkpoint.zip').write_bytes(b'changed');self.assertFalse(cache_valid(f,context))
 def test_aggregation_does_not_overwrite_success(self):
  with tempfile.TemporaryDirectory() as name:
   f=pathlib.Path(name)/'evaluation';f.mkdir();(f/'trajectory.csv').write_text('date,nav\n2024-01-02,1000000\n',encoding='utf-8');save_json(f/'metrics.json',{'AR':.1,'paper_eligible':False,'diagnostic_only':True});finish_receipt(f,{'source':'test'},{});before=sha(f/'trajectory.csv')
   x=collect_metrics(pathlib.Path(name));self.assertEqual(len(x),1);self.assertEqual(sha(f/'trajectory.csv'),before)
 def test_actual_train_api_invocation_and_checkpoint_selection(self):
  # Test spy proves executable train path calls learn/save; no learned model or paper result is produced.
  class Spy:
   def __init__(self,*a,**k):self.num_timesteps=0;self.n_steps=10;self.gamma=.99;self.policy='TEST SPY';self.calls=[]
   def learn(self,total_timesteps,**k):self.calls.append(total_timesteps);self.num_timesteps+=total_timesteps;return self
   def save(self,path):pathlib.Path(path).write_bytes(('TEST SPY '+str(self.num_timesteps)).encode())
  data=self.fixture();task={'task_id':'unit_test_not_research','sector':'bank','variant':'Vanilla','algorithm':'a2c','seed':1,'steps':50};p={'splits':{'train':['2022-01-01','2022-12-31'],'validation':['2023-01-01','2023-12-31'],'test':['2024-01-01','2024-12-31']},'model_params':{'a2c':{}}}
  with tempfile.TemporaryDirectory() as folder,patch('v2_engine.algorithms',return_value={'a2c':Spy}),patch('v2_engine.load_prepared',return_value=data),patch('v2_engine.rollout_model',return_value=None),patch('v2_engine.stats_from_env',return_value={'SR':1.,'AR':.1,'MDD':.1}):
   m=train_model(task,p,pathlib.Path(folder));self.assertEqual(m['num_timesteps'],50);self.assertEqual(m['selected_checkpoint'],'checkpoint_000010.zip');self.assertTrue((pathlib.Path(folder)/'best_policy.zip').exists())
 def test_model_prediction_drives_complete_eval(self):
  class Model:
   def __init__(self):self.calls=0
   def predict(self,obs,deterministic=True):self.calls+=1;return np.array([0.,0.]),None
  m=Model();e=rollout_model(m,self.fixture().subset('2024-01-01','2024-12-31'),'F1');self.assertEqual(m.calls,2);self.assertEqual(len(e.trace),3)
 def test_fee_differences_are_pp_and_fixed_checkpoint(self):
  a=dict(sector='bank',variant='Vanilla',algorithm='a2c',seed=1,first_date='2024-01-02',last_date='2024-12-31',checkpoint_sha256='same',fee='F0',AR=.2,SR=1.,MDD=.1)
  b=dict(a,fee='F1',AR=.18,SR=.9,MDD=.12);x=cost_differences(pd.DataFrame([a,b]));r=x[x.fee=='F1'].iloc[0]
  self.assertAlmostEqual(r.delta_AR_vs_F0_pp,-2);self.assertAlmostEqual(r.delta_MDD_vs_F0_pp,2)
  b['checkpoint_sha256']='different'
  with self.assertRaises(ValueError):cost_differences(pd.DataFrame([a,b]))
if __name__=='__main__':unittest.main()
