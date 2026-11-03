"""Plan B3: daily valuation ratios (PE TTM, PB MRQ, PS TTM, PCF OCF TTM) from the public Eastmoney data
centre (report RPT_VALUEANALYSIS_DET), 2024-01-01 to 2025-12-31, for the 30 stocks (no processing).

python b3_fetch_em.py <out_dir>
Writes <out_dir>/valuation_em.csv and fetch_em_log.json.  Used because the Tushare account has no access to
the financial statements needed to rebuild negative PE(TTM) and PCF (see PROTOCOL_B3.md, amendment 1).
"""
import sys, json, time, pathlib, urllib.request, urllib.parse
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
BANK = {'PAD': '000001', 'SPDB': '600000', 'HXB': '600015', 'CMBC': '600016', 'CMB': '600036', 'CIB': '601166',
        'ABC': '601288', 'BOCOM': '601328', 'ICBC': '601398', 'PSBC': '601658', 'CEB': '601818', 'CZB': '601916',
        'CCB': '601939', 'BOC': '601988', 'ECITIC': '601998'}
LOG = dict(calls=[], errors=[])


def get(code, page):
    q = dict(sortColumns='TRADE_DATE', sortTypes='1', pageSize='500', pageNumber=str(page), reportName='RPT_VALUEANALYSIS_DET',
             columns='ALL', source='WEB', client='WEB',
             filter='(SECURITY_CODE="%s")(TRADE_DATE>=\'2024-01-01\')(TRADE_DATE<=\'2025-12-31\')' % code)
    req = urllib.request.Request(URL + '?' + urllib.parse.urlencode(q), headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://data.eastmoney.com/'})
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                j = json.loads(r.read().decode('utf-8'))
            if not j.get('success') or j.get('result') is None:
                raise RuntimeError(str(j.get('message')))
            LOG['calls'].append(dict(code=code, page=page, rows=len(j['result']['data'])))
            return j['result']
        except Exception as e:
            last = e; time.sleep(5 * (attempt + 1))
    raise RuntimeError('%s page %d: %r' % (code, page, last))


def main():
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    hc = pd.read_csv(ROOT / 'haelthcaresectordata' / 'techtrade20241231.CSV', dtype={'tic': str}, usecols=['tic'])
    codes = [('bank', k, v) for k, v in BANK.items()] + [('healthcare', t, t.split('.')[0]) for t in dict.fromkeys(hc.tic.str.strip())]
    frames = []
    for sector, tic, code in codes:
        try:
            r = get(code, 1); rows = list(r['data'])
            for p in range(2, int(r.get('pages') or 1) + 1):
                rows += get(code, p)['data']
            d = pd.DataFrame(rows); d.insert(0, 'tic', tic); d.insert(0, 'sector', sector); frames.append(d)
            print(sector, tic, code, len(d), flush=True)
        except Exception as e:
            LOG['errors'].append(dict(code=code, error=str(e))); print('ERROR', code, e, flush=True)
        time.sleep(0.5)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / 'valuation_em.csv', index=False, encoding='utf-8')
    LOG['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    (out / 'fetch_em_log.json').write_text(json.dumps(LOG, indent=1, ensure_ascii=False), encoding='utf-8')
    print('DONE rows', sum(len(f) for f in frames), 'errors', len(LOG['errors']), flush=True)


if __name__ == '__main__':
    main()
