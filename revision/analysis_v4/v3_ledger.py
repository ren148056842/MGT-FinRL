"""Fast self-financing continuous-share ledger; accounting identical to v1 bisection."""
import numpy as np
import pandas as pd
from backtest import Ledger as ReferenceLedger
class Ledger(ReferenceLedger):
 def rebalance(self,date,p,weights,tradable=None):
  p=np.asarray(p,dtype=float);w=np.asarray(weights,dtype=float)
  if self.lots is not None or (tradable is not None and not np.asarray(tradable,bool).all()):return super().rebalance(date,p,w,tradable)
  if p.shape!=self.q.shape or w.shape!=p.shape or not np.isfinite(p).all() or (p<=0).any() or not np.isfinite(w).all() or (w<0).any() or w.sum()>1+1e-12:raise ValueError('Invalid prices or long-only target')
  if self.last_date is not None and pd.Timestamp(date)<=self.last_date:raise ValueError('Only one rebalance per date')
  self.last_date=pd.Timestamp(date);old=self.q.copy();u=old*p;pre=self.nav(p);rate=self.costs.commission+self.costs.slippage;tax=self.costs.tax(date);x=pre
  for _ in range(8):
   sell=u>w*x;ws=w[sell].sum();us=u[sell].sum();wb=w[~sell].sum();ub=u[~sell].sum();xnew=(pre-rate*(us-ub)-tax*us)/(1+rate*(wb-ws)-tax*ws)
   if np.array_equal(sell,u>w*xnew):x=xnew;break
   x=xnew
  else:
   lo,hi=0.,pre
   for _ in range(48):
    mid=(lo+hi)/2;delta=w*mid-u;fee=rate*np.abs(delta).sum()+tax*np.maximum(-delta,0).sum()
    if mid+fee<=pre:lo=mid
    else:hi=mid
   x=lo
  target=w*x/p;sells=np.maximum(old-target,0);S=float(sells@p);self.q-=sells;self.cash+=S*(1-rate-tax);buys=np.maximum(target-self.q,0);requested=float(buys@p)
  scale=min(1.,max(0.,self.cash)/(requested*(1+rate))) if requested>0 else 1.;buys*=scale;B=float(buys@p);self.q+=buys;self.cash-=B*(1+rate)
  commission=self.costs.commission*(B+S);slip=self.costs.slippage*(B+S);stamp=tax*S;cost=commission+slip+stamp;post=self.nav(p)
  if self.cash < -1e-7 or not np.isclose(post,pre-cost,rtol=1e-10,atol=1e-7):raise ArithmeticError('Self-financing invariant failed')
  if abs(self.cash)<1e-8:self.cash=0.
  return dict(date=str(pd.Timestamp(date).date()),pre_nav=pre,nav=post,cash=self.cash,buy=B,sell=S,commission=commission,slippage=slip,tax=stamp,cost=cost,gross_turnover=(B+S)/pre,shares=self.q.copy(),target_weights=w.copy(),actual_weights=self.q*p/post,unfilled_buy_value=max(0.,requested-B),blocked_symbols=0)
