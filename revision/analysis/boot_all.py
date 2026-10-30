import pathlib, sys
REV = pathlib.Path(__file__).resolve().parents[1]  # the revision/ folder
sys.path.insert(0, str(REV / 'revision_backtest'))
import numpy as np, pandas as pd, sys
from backtest import metrics
E=str(REV/'revision_backtest/runs/v3_train_only/results/evaluations')  # extract evaluations.tar.gz there first
def load(sec,v,a,s,fee='F1'):
    t=pd.read_csv(f'{E}/E1_{sec}_{v}_{a}_s{s}_{fee}/trajectory.csv')
    return t.set_index('date').daily_return.iloc[1:]
def boot(A,B,block=10,Bn=2000,seed=1729):
    a=np.asarray(A,float);b=np.asarray(B,float);rng=np.random.default_rng(seed);n=len(a);keys=['AR','SR','MDD']
    def est(ix):
        d=[]
        for s in range(a.shape[1]):
            ma,mb=metrics(a[ix,s]),metrics(b[ix,s]);d.append([ma[k]-mb[k] for k in keys])
        return np.mean(d,axis=0)
    point=est(np.arange(n));S=[]
    for _ in range(Bn):
        st=rng.integers(0,n,size=int(np.ceil(n/block)));ix=((st[:,None]+np.arange(block))%n).ravel()[:n];S.append(est(ix))
    S=np.array(S);lo,hi=np.percentile(S,[2.5,97.5],axis=0)
    return point,lo,hi
rows=[]
for sec in ['bank','healthcare']:
  for a in ['a2c','ddpg','ppo','sac','td3']:
    A=pd.concat([load(sec,'A0',a,s) for s in [1,2,3]],axis=1);B=pd.concat([load(sec,'Vanilla',a,s) for s in [1,2,3]],axis=1)
    assert (A.index==B.index).all()
    p,lo,hi=boot(A.values,B.values)
    # seed-level sign consistency
    wins=sum(metrics(A.values[:,s])['SR']>metrics(B.values[:,s])['SR'] for s in range(3))
    for k,m in enumerate(['AR','SR','MDD']):
        rows.append(dict(sector=sec,algorithm=a,metric=m,effect=p[k],ci_low=lo[k],ci_high=hi[k],seeds_A0_higher_SR=wins))
d=pd.DataFrame(rows);d.to_csv('paired_bootstrap_all_F1.csv',index=False)
pd.set_option('display.width',200);print(d.round(4).to_string())
