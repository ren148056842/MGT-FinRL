"""Plan B3: download the raw 2024-2025 inputs from Tushare Pro (no processing; see b3_ingest.py).

python b3_fetch_tushare.py <token_file> <out_dir> [apis]     (apis: comma list, default daily)

The token is read from <token_file> (kept outside the repository; never written to any output).
Writes <out_dir>/{trade_cal,daily,daily_basic,income,cashflow,balancesheet}.csv and fetch_log.json.
"""
import sys, json, time, pathlib, urllib.request
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]                 # D:/paper/2/xiu
URL = 'https://api.tushare.pro'
BANK = {'PAD': '000001.SZ', 'SPDB': '600000.SH', 'HXB': '600015.SH', 'CMBC': '600016.SH', 'CMB': '600036.SH',
        'CIB': '601166.SH', 'ABC': '601288.SH', 'BOCOM': '601328.SH', 'ICBC': '601398.SH', 'PSBC': '601658.SH',
        'CEB': '601818.SH', 'CZB': '601916.SH', 'CCB': '601939.SH', 'BOC': '601988.SH', 'ECITIC': '601998.SH'}
FIELDS = {
    'daily': 'ts_code,trade_date,open,high,low,close,pre_close,vol,amount',
    'daily_basic': 'ts_code,trade_date,close,pe,pe_ttm,pb,ps,ps_ttm,total_share,float_share,total_mv,circ_mv',
    'income': 'ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,revenue,total_revenue,n_income,n_income_attr_p,update_flag',
    'cashflow': 'ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,n_cashflow_act,update_flag',
    'balancesheet': 'ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,total_hldr_eqy_exc_min_int,total_hldr_eqy_inc_min_int,update_flag',
}
LOG = dict(calls=[], errors=[])


def call(token, api, params, fields=''):
    body = json.dumps(dict(api_name=api, token=token, params=params, fields=fields)).encode('utf-8')
    last = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=60) as r:
                j = json.loads(r.read().decode('utf-8'))
            if j.get('code') != 0:
                msg = str(j.get('msg'))
                if '权限' in msg or 'permission' in msg.lower():
                    raise PermissionError(msg)
                if '每天' in msg or '每日' in msg:
                    raise PermissionError(msg)                 # daily quota: do not retry
                raise RuntimeError(msg)
            d = j['data']
            LOG['calls'].append(dict(api=api, params=params, rows=len(d['items'])))
            return pd.DataFrame(d['items'], columns=d['fields'])
        except PermissionError:
            raise
        except Exception as e:
            last = e
            print('retry', api, params.get('ts_code', ''), repr(e)[:200], flush=True)
            time.sleep(61 if ('分钟' in str(e)) else min(65, 8 * (attempt + 1)))
    raise RuntimeError('%s %s: %r' % (api, params, last))


def codes():
    f = ROOT / 'haelthcaresectordata' / 'techtrade20241231.CSV'
    d = pd.read_csv(f, dtype={'tic': str}, usecols=['tic'])
    hc = list(dict.fromkeys(d.tic.str.strip()))
    return [('bank', k, v) for k, v in BANK.items()] + [('healthcare', c, c) for c in hc]


def main():
    token = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').strip()
    out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    apis = (sys.argv[3] if len(sys.argv) > 3 else 'daily').split(',')
    if not any((p / 'trade_cal.csv').exists() for p in [out, out.parent / 'raw']):
        call(token, 'trade_cal', dict(exchange='SSE', start_date='20240101', end_date='20251231')).to_csv(out / 'trade_cal.csv', index=False)
    frames = {k: [] for k in apis}
    allc = codes(); tsmap = {ts: (sector, tic) for sector, tic, ts in allc}
    for api in apis:
        if api in ('daily', 'daily_basic'):                 # several codes per call (row cap 6000)
            groups = [allc[i:i + 8] for i in range(0, len(allc), 8)]
            for g in groups:
                params = dict(ts_code=','.join(ts for _, _, ts in g), start_date='20240101', end_date='20251231')
                try:
                    d = call(token, api, params, FIELDS[api])
                except Exception as e:
                    LOG['errors'].append(dict(api=api, ts_code=params['ts_code'], error=str(e))); print('ERROR', api, e, flush=True); continue
                if len(d) >= 6000:
                    raise RuntimeError('row cap reached; use smaller groups')
                d.insert(0, 'tic', [tsmap[t][1] for t in d.ts_code]); d.insert(0, 'sector', [tsmap[t][0] for t in d.ts_code])
                frames[api].append(d); print(api, params['ts_code'], len(d), flush=True)
                time.sleep(1.0)
        else:
            for sector, tic, ts in allc:
                try:
                    d = call(token, api, dict(ts_code=ts, start_date='20220101', end_date='20251231'), FIELDS[api])
                except Exception as e:
                    LOG['errors'].append(dict(api=api, ts_code=ts, error=str(e))); print('ERROR', api, ts, e, flush=True); continue
                d.insert(0, 'tic', tic); d.insert(0, 'sector', sector); frames[api].append(d)
                print(api, sector, tic, ts, len(d), flush=True); time.sleep(0.4)
    for api, fs in frames.items():
        if fs:
            pd.concat(fs, ignore_index=True).to_csv(out / (api + '.csv'), index=False, encoding='utf-8')
    LOG['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    (out / 'fetch_log.json').write_text(json.dumps(LOG, indent=1, ensure_ascii=False), encoding='utf-8')
    (out / 'DONE').write_text(LOG['finished'], encoding='utf-8')
    print('DONE', {k: sum(len(f) for f in v) for k, v in frames.items()}, 'errors', len(LOG['errors']), flush=True)


if __name__ == '__main__':
    main()
