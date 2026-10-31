import pathlib, sys
REV = pathlib.Path(__file__).resolve().parents[1]  # the revision/ folder
sys.path.insert(0, str(REV / 'revision_backtest'))
import sys; import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from backtest import FEES, metrics
from v3_ledger import Ledger
U=str(REV)+'/'; E=U+'revision_backtest/runs/v3_train_only/results/evaluations'  # extract evaluations.tar.gz there first
BLUE,ORANGE,INK,MUTED='#2a78d6','#eb6834','#0b0b0b','#52514e'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9994','axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,'font.family':'sans-serif','font.sans-serif':['Helvetica'],'pdf.use14corefonts':True,'pdf.compression':9,'axes.unicode_minus':False})
fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),sharex=True)
for ax,(sec,f,title) in zip(axes,[('bank',U+'bank sectordata/banktrade20241231.CSV','Banking'),('healthcare',U+'haelthcaresectordata/techtrade20241231.CSV','Healthcare innovation')]):
    r=pd.read_csv(f);r['date']=pd.to_datetime(r.date);P=r.pivot(index='date',columns='tic',values='close').sort_index()
    P=P[(P.index>='2024-01-01')&(P.index<='2024-12-31')]
    # equal-weight daily rebalancing, same delayed execution and F1 costs
    led=Ledger(P.shape[1],1e6,FEES['F1']);nav=[1e6]
    for i in range(1,len(P)):
        rec=led.rebalance(P.index[i],P.values[i],np.ones(P.shape[1])/P.shape[1]);nav.append(rec['nav'])
    ew=pd.Series(np.array(nav)/1e6,index=P.index)
    print(sec,'EW AR',metrics(ew.pct_change().dropna().values)['AR'])
    for v,c,lab in [('Vanilla',BLUE,'Vanilla agents'),('A0',ORANGE,'MGT-augmented agents')]:
        curves=[]
        for a in ['a2c','ddpg','ppo','sac','td3']:
            for s in [1,2,3]:
                t=pd.read_csv(f'{E}/E1_{sec}_{v}_{a}_s{s}_F1/trajectory.csv');t['date']=pd.to_datetime(t.date)
                curves.append(t.set_index('date').nav/1e6)
        C=pd.concat(curves,axis=1)
        ax.fill_between(C.index,C.min(1),C.max(1),color=c,alpha=0.15,linewidth=0)
        ax.plot(C.index,C.mean(1),color=c,lw=1.6,label=lab)
    ax.plot(ew.index,ew.values,color=INK,lw=1.0,ls='--',label='Equal weight, daily')
    ax.set_title(title,fontsize=9,color=INK,loc='left');ax.grid(axis='y',color='#e6e5e0',lw=0.6)
    ax.set_ylabel('Account value (initial = 1)' if sec=='bank' else '')
    ax.tick_params(axis='x',labelrotation=0,labelsize=8)
    import matplotlib.dates as mdates; ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10])); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[0].legend(frameon=False,fontsize=8,loc='upper left')
fig.tight_layout()
fig.savefig('fig8.pdf');
