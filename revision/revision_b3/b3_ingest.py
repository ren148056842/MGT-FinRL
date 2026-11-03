"""Plan B3: build the extended (2018-2025) panels from the raw downloads, with the registered checks.

python b3_ingest.py <project_root> <raw_dir> <out_dir>

Inputs : <project_root>/{bank sectordata/banktrade20241231.CSV, haelthcaresectordata/techtrade20241231.CSV}
         <raw_dir>/{trade_cal.csv, daily.csv (Tushare), daily_basic.csv (Tushare), valuation_em.csv (Eastmoney)}
Outputs: <out_dir>/{bank,healthcare}_panel_20251231.csv  (existing rows unchanged + 2025 rows, same columns)
         <out_dir>/ingest_report.json, ratio_agreement_2024.csv, price_check_2024.csv
Checks (PROTOCOL_B3.md, section 2 and amendment 1): 2024 trading days identical to the panels; over 2024,
H/L/C within 1e-6 relative error and trading value within 0.5% on >= 99% of stock-days (else stop);
for each ratio, the source with the lower 2024 median absolute relative difference is used for 2025.
Suspended days: H = L = C = previous close, trading value '--', ratios carried forward.
"""
import sys, json, pathlib
import numpy as np
import pandas as pd

ROOT, RAW, OUT = map(pathlib.Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
PANELS = {'bank': ('bank sectordata', 'banktrade20241231.CSV', 'bank_panel_20251231.csv'),
          'healthcare': ('haelthcaresectordata', 'techtrade20241231.CSV', 'healthcare_panel_20251231.csv')}
BANK = {'PAD': '000001.SZ', 'SPDB': '600000.SH', 'HXB': '600015.SH', 'CMBC': '600016.SH', 'CMB': '600036.SH',
        'CIB': '601166.SH', 'ABC': '601288.SH', 'BOCOM': '601328.SH', 'ICBC': '601398.SH', 'PSBC': '601658.SH',
        'CEB': '601818.SH', 'CZB': '601916.SH', 'CCB': '601939.SH', 'BOC': '601988.SH', 'ECITIC': '601998.SH'}
RATIOS = {'TTM': [('em', 'PE_TTM'), ('ts', 'pe_ttm')], 'PBR': [('em', 'PB_MRQ'), ('ts', 'pb')],
          'PS': [('em', 'PS_TTM'), ('ts', 'ps_ttm'), ('em', 'RB_PS_TOI'), ('em', 'RB_PS_OI')],
          'PCF': [('em', 'PCF_OCF_TTM'), ('em', 'RB_PCF_CCE')]}
# RB_*: total market value / annual statement item of fiscal year (calendar year - 1) (amendment 2)
report = dict(sectors={})


def grid(df, col, dates, tics):
    return df.pivot(index='date', columns='tic', values=col).reindex(index=dates, columns=tics)


cal = pd.read_csv(RAW / 'trade_cal.csv', dtype=str)
cal = pd.DatetimeIndex(sorted(pd.to_datetime(cal[cal.is_open.astype(int) == 1].cal_date)))
ts_daily = pd.read_csv(RAW / 'daily.csv', dtype={'ts_code': str, 'trade_date': str, 'tic': str})
ts_daily['date'] = pd.to_datetime(ts_daily.trade_date)
ts_basic = pd.read_csv(RAW / 'daily_basic.csv', dtype={'ts_code': str, 'trade_date': str, 'tic': str}) if (RAW / 'daily_basic.csv').exists() else None
if ts_basic is not None:
    ts_basic['date'] = pd.to_datetime(ts_basic.trade_date)
em = pd.read_csv(RAW / 'valuation_em.csv', dtype={'tic': str, 'SECURITY_CODE': str})
em['date'] = pd.to_datetime(em.TRADE_DATE.str[:10])


def annual(fname, cols):
    f = RAW / fname
    if not f.exists():
        return None
    a = pd.read_csv(f, dtype={'tic': str}, low_memory=False)
    a['FY'] = a.REPORT_DATE.astype(str).str[:4].astype(int)
    a = a.drop_duplicates(['tic', 'FY'], keep='first')
    return a[['tic', 'FY'] + [c for c in cols if c in a]]


em['FY'] = em.date.dt.year - 1
cfa = annual('em_cashflow_annual.csv', ['FBCCE_ADD', 'CCE_ADD'])
if cfa is not None:
    cfa['CCE'] = cfa.FBCCE_ADD.where(cfa.FBCCE_ADD.notna(), cfa.CCE_ADD) if 'FBCCE_ADD' in cfa else cfa.CCE_ADD
    em = em.merge(cfa[['tic', 'FY', 'CCE']], on=['tic', 'FY'], how='left')
    em['RB_PCF_CCE'] = em.TOTAL_MARKET_CAP / em.CCE
ina = annual('em_income_annual.csv', ['TOTAL_OPERATE_INCOME', 'OPERATE_INCOME'])
if ina is not None:
    em = em.merge(ina, on=['tic', 'FY'], how='left')
    for c, k in [('TOTAL_OPERATE_INCOME', 'RB_PS_TOI'), ('OPERATE_INCOME', 'RB_PS_OI')]:
        if c in em:
            em[k] = em.TOTAL_MARKET_CAP / em[c]

stop = []
agree_rows, price_rows = [], []
for sec, (folder, fname, outname) in PANELS.items():
    f = ROOT / folder / fname
    raw_text = f.read_text(encoding='utf-8-sig').splitlines()
    p = pd.read_csv(f, dtype={'tic': str}); p['tic'] = p.tic.str.strip()
    p['dt'] = pd.to_datetime(p.date)
    n = p.tic.nunique(); tics = list(p.tic.iloc[:n])
    pdates = pd.DatetimeIndex(p.dt.iloc[::n])
    d24 = cal[(cal >= '2024-01-01') & (cal <= '2024-12-31')]
    d25 = cal[(cal >= '2025-01-01') & (cal <= '2025-12-31')]
    p24 = pdates[pdates.year == 2024]
    same_days = list(d24) == list(p24)
    if not same_days:
        stop.append('%s: 2024 trading days differ (calendar %d, panel %d)' % (sec, len(d24), len(p24)))
    td = ts_daily[ts_daily.sector == sec]
    H, L, C = (grid(td, c, cal, tics) for c in ['high', 'low', 'close'])
    V = grid(td, 'amount', cal, tics) * 1000.0                     # thousand CNY -> CNY
    # 2024 price check against the panel (stock-days on which both sources report trading)
    q = p[p.dt.dt.year == 2024].copy()
    pv = pd.to_numeric(q.volume, errors='coerce')
    q['V'] = pv
    chk = []
    for c, G in [('high', H), ('low', L), ('close', C)]:
        a = q.pivot(index='dt', columns='tic', values=c).reindex(index=d24, columns=tics)
        b = G.reindex(index=d24)
        traded = q.pivot(index='dt', columns='tic', values='V').reindex(index=d24, columns=tics).notna() & b.notna()
        rel = (a - b).abs() / a
        share = float((rel[traded] <= 1e-6).sum().sum() / traded.sum().sum())
        chk.append(dict(sector=sec, channel=c, stock_days=int(traded.sum().sum()), share_within_tol=share,
                        max_rel=float(rel[traded].max().max())))
    a = q.pivot(index='dt', columns='tic', values='V').reindex(index=d24, columns=tics)
    b = V.reindex(index=d24)
    both = a.notna() & b.notna()
    rel = (a - b).abs() / a
    chk.append(dict(sector=sec, channel='trading_value', stock_days=int(both.sum().sum()),
                    share_within_tol=float((rel[both] <= 0.005).sum().sum() / both.sum().sum()), max_rel=float(rel[both].max().max())))
    susp_panel = int(a.isna().sum().sum()); susp_ts = int(b.isna().sum().sum())
    for r in chk:
        price_rows.append(r)
        if r['share_within_tol'] < 0.99:
            stop.append('%s %s: only %.4f of 2024 stock-days within tolerance' % (sec, r['channel'], r['share_within_tol']))
    # ratio agreement over 2024 and choice of source
    choice = {}
    ems = em[em.sector == sec]
    for col, cands in RATIOS.items():
        a = q.pivot(index='dt', columns='tic', values=col).reindex(index=d24, columns=tics).astype(float)
        best = None
        for src, field in cands:
            base = ems if src == 'em' else (ts_basic[ts_basic.sector == sec] if ts_basic is not None else None)
            if base is None or field not in base:
                continue
            b = grid(base, field, cal, tics).reindex(index=d24).astype(float)
            m = a.notna() & b.notna() & (a != 0)
            rel = ((a - b).abs() / a.abs())[m]
            mard = float(np.nanmedian(rel.to_numpy()))
            row = dict(sector=sec, ratio=col, source=src, field=field, compared=int(m.sum().sum()),
                       coverage=float(m.sum().sum() / a.notna().sum().sum()), median_abs_rel_diff=mard,
                       p90_abs_rel_diff=float(np.nanpercentile(rel.to_numpy(), 90)),
                       sign_agreement=float((np.sign(a[m]) == np.sign(b[m])).sum().sum() / m.sum().sum()))
            agree_rows.append(row)
            if row['coverage'] >= 0.95 and (best is None or mard < best[2]):
                best = (src, field, mard)
        if best is None:
            stop.append('%s %s: no source covers 2024' % (sec, col))
            continue
        choice[col] = best[:2]
    if stop:
        continue
    # 2025 rows
    last = p[p.dt == pdates[-1]].set_index('tic').reindex(tics)
    prev_c = last.close.astype(float).to_numpy().copy()
    prev_r = {col: last[col].astype(float).to_numpy().copy() for col in RATIOS}
    R = {}
    for col, (src, field) in choice.items():
        base = ems if src == 'em' else ts_basic[ts_basic.sector == sec]
        R[col] = grid(base, field, cal, tics).reindex(index=d25).astype(float)
    lines = []; n_susp = 0; n_ffill = {c: 0 for c in RATIOS}
    for dt in d25:
        h, l, c, v = (G.loc[dt].to_numpy(float) for G in (H, L, C, V))
        for k, t in enumerate(tics):
            if np.isfinite(c[k]) and np.isfinite(v[k]) and v[k] > 0:
                hi, lo, cl, vol = h[k], l[k], c[k], '%d' % int(round(v[k]))
            else:
                hi = lo = cl = prev_c[k]; vol = '--'; n_susp += 1
            vals = []
            for col in ['TTM', 'PBR', 'PS', 'PCF']:
                x = R[col].loc[dt].iloc[k]
                if not np.isfinite(x):
                    x = prev_r[col][k]; n_ffill[col] += 1
                prev_r[col][k] = x; vals.append('%.4f' % x)
            prev_c[k] = cl
            row = [t, '%d/%d/%d' % (dt.year, dt.month, dt.day), '%.2f' % hi, '%.2f' % lo, '%.2f' % cl, vol] + vals
            if sec == 'bank':
                row.append(str(dt.weekday()))
            lines.append(','.join(row))
    out = OUT / outname
    text = '\n'.join(raw_text + lines) + '\n'
    out.write_text(text, encoding='utf-8')
    chk_new = pd.read_csv(out, dtype={'tic': str})
    report['sectors'][sec] = dict(panel_in=str(f), panel_out=str(out), rows_in=len(p), rows_2025=len(lines),
                                  days_2025=len(d25), first_2025=str(d25[0].date()), last_2025=str(d25[-1].date()),
                                  same_2024_days=same_days, suspended_stock_days_2025=n_susp,
                                  ratio_carried_forward_2025=n_ffill, ratio_source={k: list(v) for k, v in choice.items()},
                                  suspended_2024_panel=susp_panel, missing_2024_tushare=susp_ts,
                                  rows_out=len(chk_new), columns_out=list(chk_new.columns))
pd.DataFrame(price_rows).to_csv(OUT / 'price_check_2024.csv', index=False)
pd.DataFrame(agree_rows).to_csv(OUT / 'ratio_agreement_2024.csv', index=False)
report['stop'] = stop
(OUT / 'ingest_report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
print(json.dumps(report, indent=1, default=str))
print(pd.DataFrame(price_rows).round(6).to_string(index=False))
print(pd.DataFrame(agree_rows).round(4).to_string(index=False))
if stop:
    sys.exit('STOPPED: ' + '; '.join(stop))
