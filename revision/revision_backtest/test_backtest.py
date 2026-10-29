import unittest
import numpy as np
import pandas as pd
from backtest import *

class LedgerTests(unittest.TestCase):
    def panel(self):return pd.DataFrame([[10.,10.],[20.,10.],[40.,10.],[20.,20.]],index=pd.to_datetime(['2024-01-02','2024-01-03','2024-01-04','2024-01-05']),columns=['a','b'])
    def test_initial_buy_and_fee_once(self):
        l=Ledger(1,1000,Costs(.001,.002,False));r=l.rebalance('2024-01-02',[10.],[1.])
        self.assertAlmostEqual(r['buy'],1000/1.003);self.assertAlmostEqual(r['cost'],r['buy']*.003);self.assertAlmostEqual(r['nav']+r['cost'],1000);self.assertAlmostEqual(l.cash,0)
    def test_sell_tax_date(self):
        for date,tax in [('2023-08-25',.001),('2023-08-28',.0005)]:
            l=Ledger(1,1000,Costs(0,0,True));l.rebalance('2023-08-24',[10],[1]);r=l.rebalance(date,[10],[0]);self.assertAlmostEqual(r['tax'],1000*tax);self.assertEqual(r['commission'],0)
    def test_drift_turnover(self):
        l=Ledger(2,1000,Costs(0,0,False));l.rebalance('2024-01-02',[10,10],[.5,.5]);r=l.rebalance('2024-01-03',[20,10],[.5,.5]);self.assertAlmostEqual(r['buy'],250);self.assertAlmostEqual(r['sell'],250);self.assertAlmostEqual(r['gross_turnover'],1/3)
    def test_no_trade_no_fee(self):
        l=Ledger(1,1000);r=l.rebalance('2024-01-02',[10],[0]);self.assertEqual(r['cost'],0)
    def test_cash_conservation_random(self):
        rng=np.random.default_rng(8);l=Ledger(3,10000)
        for i in range(80):
            p=rng.uniform(1,100,3);w=rng.dirichlet([1,1,1])*rng.uniform(.1,1);r=l.rebalance(pd.Timestamp('2024-01-01')+pd.Timedelta(days=i),p,w)
            self.assertGreaterEqual(l.cash,-1e-8);np.testing.assert_allclose(r['nav'],r['pre_nav']-r['cost'],rtol=1e-12,atol=1e-8)
    def test_execution_delay(self):
        d=self.panel()[['a']];r=simulate(d,lambda obs:np.ones(1),Costs(0,0,False),1000)
        self.assertEqual(r.nav.iloc[1],1000);self.assertEqual(r.nav.iloc[2],2000);self.assertEqual(r.decision_date.iloc[1],'2024-01-02');self.assertEqual(r.date.iloc[1],'2024-01-03')
    def test_real_price_isolation(self):
        d=self.panel();p=lambda o:np.array([.5,.5]);a=simulate(d,p,features_mgt=d*100);b=simulate(d,p,features_mgt=d*.001)
        np.testing.assert_allclose(a.nav,b.nav);np.testing.assert_allclose(a.buy,b.buy)
    def test_features_change_actions_without_changing_execution_price(self):
        d=self.panel();p=lambda o:np.array([1.,0.]) if o['features'].iloc[-1,0]>100 else np.array([0.,1.])
        a=simulate(d,p,Costs(0,0,False),1000,features_mgt=d*100);self.assertAlmostEqual(a.shares.iloc[1][0],50.)
    def test_splits_reject_overlap(self):
        with self.assertRaises(ValueError):validate_splits(['2022-01-03'],['2022-01-03'],['2024-01-03'])
        validate_splits(['2022-01-03'],['2023-01-03'],['2024-01-03'])
    def test_unknown_forecasts_rejected(self):
        d=self.panel()
        with self.assertRaises(ValueError):validate_forecasts(d,None,d.index,d.columns)
        with self.assertRaises(ValueError):validate_forecasts(d,d.index+pd.Timedelta(days=1),d.index-pd.Timedelta(days=1),d.columns)
        validate_forecasts(d,d.index-pd.Timedelta(days=1),d.index-pd.Timedelta(days=1),d.columns)
    def test_no_fill_missing_prices(self):
        d=self.panel();d.iloc[1,0]=np.nan
        with self.assertRaises(ValueError):simulate(d,lambda o:[.5,.5])
    def test_tradability_mask_and_lots(self):
        l=Ledger(2,1000,Costs(),lots=[10,20]);r=l.rebalance('2024-01-02',[10,10],[.5,.5],[False,True]);self.assertEqual(l.q[0],0);self.assertEqual(l.q[1]%20,0);self.assertEqual(r['blocked_symbols'],1)
    def test_long_only_no_leverage_or_double_day(self):
        l=Ledger(1)
        with self.assertRaises(ValueError):l.rebalance('2024-01-02',[10],[1.1])
        l.rebalance('2024-01-02',[10],[1.])
        with self.assertRaises(ValueError):l.rebalance('2024-01-02',[10],[0.])
    def test_metrics_include_initial_capital(self):
        m=metrics([-.1,.1]);self.assertAlmostEqual(m['MDD'],.1);self.assertAlmostEqual(m['AR'],.99**126-1);self.assertAlmostEqual(m['SR'],0);self.assertTrue(np.isnan(metrics([0.,0.])['Calmar']))
    def test_bootstrap_paired_shared_indices(self):
        a=np.array([[-.01,.01],[.02,-.02],[.01,.02],[-.02,.01]])
        b=paired_bootstrap(a,a,block=2,B=50)
        np.testing.assert_allclose(b[['effect','ci_low','ci_high']],0)
    def test_terminal_no_free_liquidation(self):
        r=simulate(self.panel(),lambda o:np.array([.5,.5]) if len(o['market'])==1 else None)
        self.assertEqual(r.sell.iloc[-1],0);self.assertGreater(r.shares.iloc[-1].sum(),0)
if __name__=='__main__':unittest.main()

