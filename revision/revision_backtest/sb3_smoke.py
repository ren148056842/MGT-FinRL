"""Synthetic interface verification only. No paper experiment, no forecasting model."""
import json,time,pathlib
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C,PPO,SAC,TD3,DDPG
from stable_baselines3.common.env_checker import check_env
from backtest import Ledger,Costs,simulate
ROOT=pathlib.Path(__file__).resolve().parent

class DelayedCloseEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,market,features=None,initial=1_000_000.,costs=Costs()):
        super().__init__();self.market=np.asarray(market,float);self.features=self.market.copy() if features is None else np.asarray(features,float)
        if self.features.shape!=self.market.shape or not np.isfinite(self.features).all():raise ValueError('Invalid feature channel')
        self.n=self.market.shape[1];self.initial=initial;self.costs=costs
        self.action_space=spaces.Box(-1,1,shape=(self.n,),dtype=np.float32)
        # Minimal interface fixture, not the original research state; never used for E1/E2.
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(self.n*2+1,),dtype=np.float32)
    def _obs(self):return np.r_[self.features[self.t]/self.features[0],self.l.q*self.market[self.t]/self.l.nav(self.market[self.t]),self.l.cash/self.initial].astype(np.float32)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);self.t=0;self.l=Ledger(self.n,self.initial,self.costs);return self._obs(),{}
    def step(self,action):
        w=np.exp(np.asarray(action,dtype=np.float64)-np.max(action));w/=w.sum();oldnav=self.l.nav(self.market[self.t]);self.t+=1
        r=self.l.rebalance(pd.Timestamp('2000-01-01')+pd.Timedelta(days=self.t),self.market[self.t],w)
        return self._obs(),float((r['nav']/oldnav-1)*.1),self.t==len(self.market)-1,False,{'nav':r['nav']}

def main():
    rng=np.random.default_rng(4);market=10*np.exp(np.cumsum(rng.normal(0,.01,(35,3)),axis=0));rows=[]
    check_env(DelayedCloseEnv(market),warn=True)
    for cls in [A2C,PPO,SAC,TD3,DDPG]:
        start=time.perf_counter();kwargs={'n_steps':8} if cls is A2C else ({'n_steps':16,'batch_size':8,'n_epochs':1} if cls is PPO else {'buffer_size':100,'learning_starts':8,'batch_size':8})
        model=cls('MlpPolicy',DelayedCloseEnv(market),seed=1,device='cpu',verbose=0,policy_kwargs={'net_arch':[16,16]},**kwargs);model.learn(total_timesteps=32)
        obs,_=model.env.envs[0].reset();env=model.env.envs[0]
        while True:
            action,_=model.predict(obs,deterministic=True);obs,reward,done,_,info=env.step(action)
            assert np.isfinite(reward) and np.isfinite(obs).all()
            if done:break
        rows.append(dict(task='synthetic_'+cls.__name__,experiment='synthetic_interface_only',status='completed',actual_steps=model.num_timesteps,seconds=time.perf_counter()-start,reason='Synthetic market + fixture state; not a scientific training run'))
    pd.DataFrame(rows).to_csv(ROOT/'results/smoke_status.csv',index=False)
    history=ROOT/'results/smoke_history.csv'
    new=pd.DataFrame(rows);new['verification_batch']='repeat_'+str(time.time_ns());new['task']=new['task']+'_'+new['verification_batch']
    if history.exists():new=pd.concat([pd.read_csv(history),new],ignore_index=True)
    new.to_csv(history,index=False)
    dates=pd.to_datetime(['2024-01-02','2024-01-03','2024-01-04','2024-01-05']);d=pd.DataFrame({'a':[10.,20.,40.,20.],'b':[10.,10.,10.,20.]},index=dates)
    d.to_csv(ROOT/'results/synthetic_market.csv',index_label='date')
    trace=simulate(d,lambda o:np.array([1.,0.]),Costs(0,0,False),1000)
    trace.to_csv(ROOT/'results/synthetic_delay_trace.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False));print(trace[['date','decision_date','nav','cash','buy','sell','daily_return']].to_string(index=False))
if __name__=='__main__':main()

