"""Plan B3: download the raw 2024-2025 inputs from Tushare Pro (no processing; see b3_ingest.py).

python b3_fetch_tushare.py <token_file> <out_dir>

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
                raise RuntimeError(msg)
            d = j['data']
            LOG['calls'].append(dict(api=api, params=params, rows=len(d['items'])))
            return pd.DataFrame(d['items'], columns=d['fields'])
        except PermissionError:
            raise
        except Exception as e:
            last = e
            time.sleep(min(65, 8 * (attempt + 1)))
    raise RuntimeError('%s %s: %r' % (api, params, last))


def codes():
    f = ROOT / 'haelthcaresectordata' / 'techtrade20241231.CSV'
    d = pd.read_csv(f, dtype={'tic': str}, usecols=['tic'])
    hc = list(dict.fromkeys(d.tic.str.strip()))
    return [('bank', k, v) for k, v in BANK.items()] + [('healthcare', c, c) for c in hc]


def main():
    token = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').strip()
    out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    call(token, 'trade_cal', dict(exchange='SSE', start_date='20240101', end_date='20251231')).to_csv(out / 'trade_cal.csv', index=False)
    frames = {k: [] for k in FIELDS}
    for sector, tic, ts in codes():
        for api in FIELDS:
            if api in ('daily', 'daily_basic'):
                params = dict(ts_code=ts, start_date='20240101', end_date='20251231')
            else:
                params = dict(ts_code=ts, start_date='20220101', end_date='20251231')
            try:
                d = call(token, api, params, FIELDS[api])
            except Exception as e:
                LOG['errors'].append(dict(api=api, ts_code=ts, error=str(e)))
                print('ERROR', api, ts, e, flush=True)
                continue
            d.insert(0, 'tic', tic); d.insert(0, 'sector', sector)
            frames[api].append(d)
            print(api, sector, tic, ts, len(d), flush=True)
            time.sleep(0.4)
    for api, fs in frames.items():
        if fs:
            pd.concat(fs, ignore_index=True).to_csv(out / (api + '.csv'), index=False, encoding='utf-8')
    LOG['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    (out / 'fetch_log.json').write_text(json.dumps(LOG, indent=1, ensure_ascii=False), encoding='utf-8')
    (out / 'DONE').write_text(LOG['finished'], encoding='utf-8')
    print('DONE', {k: sum(len(f) for f in v) for k, v in frames.items()}, 'errors', len(LOG['errors']), flush=True)


if __name__ == '__main__':
    main()
