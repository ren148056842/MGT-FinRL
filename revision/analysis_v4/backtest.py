"""Strict delayed-close ledger. No prediction, training or raw-data mutation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Costs:
    commission: float = .0003
    slippage: float = .0002
    historical_tax: bool = True
    def tax(self, date):
        return (0.001 if pd.Timestamp(date) < pd.Timestamp('2023-08-28') else .0005) if self.historical_tax else 0.

FEES = {f'F{i}': Costs(c*1e-4,s*1e-4,t) for i,(c,s,t) in enumerate([(0,0,False),(3,2,True),(5,2,True),(10,2,True),(20,2,True),(3,0,True),(3,5,True)])}

def validate_prices(prices):
    if not isinstance(prices.index,pd.DatetimeIndex) or not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError('Prices require unique increasing actual market dates')
    if prices.columns.has_duplicates or len(prices)<2 or prices.shape[1]<1:
        raise ValueError('Invalid market panel')
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices<=0).any().any():
        raise ValueError('Missing/nonpositive market prices: no filling permitted')

def validate_splits(train, validation, test):
    groups=[pd.DatetimeIndex(x) for x in (train,validation,test)]
    if any(len(x)==0 for x in groups) or not (max(groups[0])<min(groups[1]) and max(groups[1])<min(groups[2])):
        raise ValueError('Require nonempty chronological disjoint train/validation/test')

def validate_forecasts(pred, availability, decision_dates, tickers):
    if not isinstance(pred.index,pd.DatetimeIndex) or pred.index.has_duplicates or list(pred.columns)!=list(tickers):
        raise ValueError('Explicit target dates and verified ticker order required')
    if availability is None or len(availability)!=len(pred) or len(decision_dates)!=len(pred):
        raise ValueError('Verified availability required; no inferred dates')
    available=pd.DatetimeIndex(availability); decisions=pd.DatetimeIndex(decision_dates)
    if available.hasnans or (available>decisions).any() or (pred.index<=decisions).any() or not np.isfinite(pred.to_numpy()).all():
        raise ValueError('Unavailable, expired or invalid forecasts')

class Ledger:
    """Fractional-share research ledger; optional verified per-symbol lot sizes and trading mask.
    Slippage is a cash debit, never also changed execution price. One rebalance per date.
    Caller must use actual market prices and separately verified corporate-action treatment.
    """
    def __init__(self, n, initial=1_000_000., costs=Costs(), lots=None):
        self.cash=float(initial); self.q=np.zeros(n); self.costs=costs
        self.lots=None if lots is None else np.asarray(lots,dtype=float)
        if self.lots is not None and (self.lots.shape!=(n,) or (self.lots<=0).any()):raise ValueError('Invalid lot sizes')
        if initial<=0:raise ValueError('Positive initial capital required')
        self.last_date=None
    def nav(self,p):return self.cash+float(self.q@p)
    def rebalance(self,date,p,weights,tradable=None):
        p=np.asarray(p,dtype=float); w=np.asarray(weights,dtype=float)
        if p.shape!=self.q.shape or w.shape!=p.shape or not np.isfinite(p).all() or (p<=0).any():raise ValueError('Invalid prices/shape')
        if not np.isfinite(w).all() or (w<0).any() or w.sum()>1+1e-12:raise ValueError('Long-only unlevered target required')
        if self.last_date is not None and pd.Timestamp(date)<=self.last_date:raise ValueError('Only one rebalance per increasing date')
        self.last_date=pd.Timestamp(date)
        allowed=np.ones(len(p),dtype=bool) if tradable is None else np.asarray(tradable,dtype=bool)
        if allowed.shape!=p.shape:raise ValueError('Invalid tradability mask')
        old=self.q.copy(); pre=self.nav(p); rate=self.costs.commission+self.costs.slippage; tax=self.costs.tax(date)
        def desired(v):
            q=w*v/p
            if self.lots is not None:q=np.floor(q/self.lots+1e-12)*self.lots
            return np.where(allowed,q,old)
        def remaining(v):
            diff=(desired(v)-old)*p; b=np.maximum(diff,0).sum(); s=np.maximum(-diff,0).sum()
            return pre-v-rate*(b+s)-tax*s
        # Fee-aware target NAV; conservative bisection also supports discrete lots.
        lo,hi=0.,pre
        for _ in range(100):
            mid=(lo+hi)/2
            if remaining(mid)>=0:lo=mid
            else:hi=mid
        target=desired(lo)
        sell=np.maximum(old-target,0); self.q-=sell
        S=float(sell@p); self.cash+=S*(1-rate-tax)
        buys=np.maximum(target-self.q,0); requested=float(buys@p)
        scale=min(1.,max(0.,self.cash)/(requested*(1+rate))) if requested>0 else 1.
        buys*=scale
        if self.lots is not None:buys=np.floor(buys/self.lots+1e-12)*self.lots
        B=float(buys@p); self.q+=buys; self.cash-=B*(1+rate)
        commission=self.costs.commission*(B+S); slip=self.costs.slippage*(B+S); stamp=tax*S
        cost=commission+slip+stamp; post=self.nav(p)
        if self.cash < -1e-7 or (self.q < -1e-10).any() or not np.isclose(post,pre-cost,rtol=1e-10,atol=1e-7):raise ArithmeticError('Self-financing invariant failed')
        if abs(self.cash)<1e-8:self.cash=0.
        return dict(date=str(pd.Timestamp(date).date()),pre_nav=pre,nav=post,cash=self.cash,buy=B,sell=S,commission=commission,slippage=slip,tax=stamp,cost=cost,gross_turnover=(B+S)/pre,shares=self.q.copy(),target_weights=w.copy(),actual_weights=self.q*p/post,unfilled_buy_value=max(0.,requested-B),blocked_symbols=int((~allowed).sum()))

def simulate(market_raw,policy,costs=Costs(),initial=1_000_000.,features_mgt=None,lots=None,tradable=None):
    validate_prices(market_raw)
    if features_mgt is not None and (not features_mgt.index.equals(market_raw.index) or not features_mgt.columns.equals(market_raw.columns)):
        raise ValueError('Feature alignment must be explicit; no fill/relabel')
    ledger=Ledger(market_raw.shape[1],initial,costs,lots); rows=[]; previous=initial
    rows.append(dict(date=str(market_raw.index[0].date()),nav=initial,cash=initial,daily_return=np.nan,buy=0.,sell=0.,cost=0.,gross_turnover=0.,shares=ledger.q.copy()))
    for t in range(len(market_raw)-1):
        obs=dict(date=market_raw.index[t],market=market_raw.iloc[:t+1].copy(),features=None if features_mgt is None else features_mgt.iloc[:t+1].copy(),shares=ledger.q.copy(),cash=ledger.cash,nav=previous)
        w=policy(obs)
        p=market_raw.iloc[t+1].to_numpy(float)
        if w is None: # hold: no artificial rebalance or fee
            row=dict(date=str(market_raw.index[t+1].date()),pre_nav=ledger.nav(p),nav=ledger.nav(p),cash=ledger.cash,buy=0.,sell=0.,commission=0.,slippage=0.,tax=0.,cost=0.,gross_turnover=0.,shares=ledger.q.copy(),target_weights=np.full(len(p),np.nan),actual_weights=ledger.q*p/ledger.nav(p),unfilled_buy_value=0.,blocked_symbols=0)
        else:row=ledger.rebalance(market_raw.index[t+1],p,w,None if tradable is None else tradable.iloc[t+1].to_numpy())
        row['decision_date']=str(market_raw.index[t].date()); row['daily_return']=row['nav']/previous-1
        previous=row['nav']; rows.append(row)
    return pd.DataFrame(rows)

def metrics(returns,rf_annual=0.):
    r=np.asarray(returns,float)
    if len(r)==0 or not np.isfinite(r).all() or (r<=-1).any():raise ValueError('Finite actual daily returns required')
    curve=np.r_[1.,np.cumprod(1+r)]; ar=curve[-1]**(252/len(r))-1
    std=np.std(r,ddof=1) if len(r)>1 else np.nan
    mdd=np.max(1-curve/np.maximum.accumulate(curve)); rf=(1+rf_annual)**(1/252)-1
    return dict(n=len(r),AR=ar,SR=np.sqrt(252)*np.mean(r-rf)/std if std>0 else np.nan,Vol=np.sqrt(252)*std,MDD=mdd,Calmar=ar/mdd if mdd>0 else np.nan)

def paired_bootstrap(a,b,block=10,B=2000,seed=1729):
    """Rows dates, columns paired RL seeds. Same sampled dates across all seeds/strategies.
    Estimator: mean across seeds of recomputed paired AR/SR/MDD differences. CI only.
    """
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2 or len(a)<2:raise ValueError('Aligned day x seed returns required')
    rng=np.random.default_rng(seed); n=len(a); keys=['AR','SR','MDD']
    def estimate(ix):
        diffs=[]
        for s in range(a.shape[1]):
            ma,mb=metrics(a[ix,s]),metrics(b[ix,s]); diffs.append([ma[k]-mb[k] for k in keys])
        return np.mean(diffs,axis=0)
    samples=[]
    for _ in range(B):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)));ix=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        samples.append(estimate(ix))
    ci=np.nanpercentile(samples,[2.5,97.5],axis=0); point=estimate(np.arange(n))
    return pd.DataFrame([dict(metric=k,effect=point[i],ci_low=ci[0,i],ci_high=ci[1,i],block=block,B=B,stat_seed=seed) for i,k in enumerate(keys)])
