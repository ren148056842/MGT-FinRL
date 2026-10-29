import pathlib, sys
REV = pathlib.Path(__file__).resolve().parents[1]  # the revision/ folder
sys.path.insert(0, str(REV / 'revision_backtest'))
import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=str(REV)+'/'
raw={'bank':U+'bank sectordata/banktrade20241231.CSV','healthcare':U+'haelthcaresectordata/techtrade20241231.CSV'}
mer={'bank':U+'bank sectordata/ablation/finrl_csv/banktrademergeGNNtr20241231_{}.CSV','healthcare':U+'haelthcaresectordata/ablation_tech/finrl_csv/techtrade20241231_{}.csv'}
V=['A0_Full_MGT','A1_Transformer_only','A2_MGNN_only','A3_SH_only','A4_QA_only','A5_KD_only','A6_SH_QA','A7_SH_KD','A8_QA_KD','A9_Fixed_fusion']
rows=[]
def dir_ic(pr,ar):
    nz=(ar.values!=0)&(pr.values!=0)&np.isfinite(pr.values)&np.isfinite(ar.values)
    da=np.mean(np.sign(pr.values[nz])==np.sign(ar.values[nz]))
    ics=np.array([spearmanr(pr.loc[d],ar.loc[d]).statistic for d in pr.index if pr.loc[d].notna().sum()>3])
    ics=ics[np.isfinite(ics)]
    return da, ics.mean(), ics.mean()/ics.std(ddof=1)*np.sqrt(len(ics))
for sec in ['bank','healthcare']:
    r=pd.read_csv(raw[sec]);r['date']=pd.to_datetime(r.date)
    R=r.pivot(index='date',columns='tic',values='close').sort_index()
    W=R[(R.index>='2018-01-01')&(R.index<='2022-12-31')]
    half=(W.max()-W.min())/2   # MinMaxScaler(-1,1) unit = half range
    mask=(R.index.year==2022)
    A=R[mask]; prev=R.shift(1)[mask]; ar=A/prev-1
    # naive: previous close
    e=(prev-A); 
    rows.append(dict(sector=sec,model='Naive (previous close)',RMSE_CNY=np.sqrt(np.nanmean(e.values**2)),MAE_CNY=np.nanmean(np.abs(e.values)),MAPE=np.nanmean(np.abs(e.values)/A.values),bias_pct=np.nanmean((prev/A-1).values),RMSE_scaled=np.sqrt(np.nanmean((e/half).values**2)),dir_acc=np.nan,rank_IC=np.nan,IC_t=np.nan))
    lag=(R/R.shift(1)-1).shift(1)[mask]
    da,ic,t=dir_ic(lag,ar)
    rows.append(dict(sector=sec,model="Naive (yesterday's return sign)",dir_acc=da,rank_IC=ic,IC_t=t))
    for v in V:
        f=mer[sec].format(v)
        if not os.path.exists(f): print('missing',f); continue
        m=pd.read_csv(f);m['date']=pd.to_datetime(m.date)
        M=m.pivot(index='date',columns='tic',values='close').reindex(index=R.index,columns=R.columns)
        P=M[mask]; e=P-A
        pr=P/prev-1; da,ic,t=dir_ic(pr,ar)
        rows.append(dict(sector=sec,model=v,RMSE_CNY=np.sqrt(np.nanmean(e.values**2)),MAE_CNY=np.nanmean(np.abs(e.values)),MAPE=np.nanmean(np.abs(e.values)/A.values),bias_pct=np.nanmean((P/A-1).values),RMSE_scaled=np.sqrt(np.nanmean((e/half).values**2)),dir_acc=da,rank_IC=ic,IC_t=t))
d=pd.DataFrame(rows); d.to_csv('forecast_eval_2022_all.csv',index=False)
pd.set_option('display.width',250); print(d.round(4).to_string())
