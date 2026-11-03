"""Plan B3: annual cash-flow and income statements (Eastmoney F10, public), FY2016 onwards, for the 30 stocks.
Used only to identify and extend the annual denominators of the panels' PCF and PS (PROTOCOL_B3.md, amendment 2).

python b3_fetch_em_fin.py <out_dir>   ->  em_cashflow_annual.csv, em_income_annual.csv, fetch_em_fin_log.json
"""
import sys, gzip, json, time, pathlib, urllib.request, urllib.parse
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = 'https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/'
BANK = {'PAD': 'SZ000001', 'SPDB': 'SH600000', 'HXB': 'SH600015', 'CMBC': 'SH600016', 'CMB': 'SH600036', 'CIB': 'SH601166',
        'ABC': 'SH601288', 'BOCOM': 'SH601328', 'ICBC': 'SH601398', 'PSBC': 'SH601658', 'CEB': 'SH601818', 'CZB': 'SH601916',
        'CCB': 'SH601939', 'BOC': 'SH601988', 'ECITIC': 'SH601998'}
LOG = dict(calls=[], errors=[])


def get(name, params):
    req = urllib.request.Request(BASE + name + '?' + urllib.parse.urlencode(params),
                                 headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://emweb.securities.eastmoney.com/'})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
                if b[:2] == b'\x1f\x8b':
                    b = gzip.decompress(b)
                j = json.loads(b.decode('utf-8'))
            LOG['calls'].append(dict(name=name, params=params, n=len(j.get('data') or [])))
            return j.get('data') or []
        except Exception as e:
            last = e; time.sleep(4 * (attempt + 1))
    raise RuntimeError('%s %s: %r' % (name, params, last))


def statements(code, kind):
    """kind: xjllb (cash flow) or lrb (income); annual reports, consolidated."""
    for ctype in ['4', '3', '1', '2']:
        dates = get(kind + 'DateAjaxNew', dict(companyType=ctype, reportDateType='1', code=code))
        if not dates:
            continue
        ds = [str(d['REPORT_DATE'])[:10] for d in dates if str(d['REPORT_DATE'])[:4] >= '2016']
        rows = []
        for i in range(0, len(ds), 5):
            rows += get(kind + 'AjaxNew', dict(companyType=ctype, reportDateType='1', reportType='1', dates=','.join(ds[i:i + 5]), code=code))
            time.sleep(0.3)
        if rows:
            for r in rows:
                r['companyType'] = ctype
            return rows
    return []


def main():
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    hc = pd.read_csv(ROOT / 'haelthcaresectordata' / 'techtrade20241231.CSV', dtype={'tic': str}, usecols=['tic'])
    codes = [('bank', k, v) for k, v in BANK.items()] + \
            [('healthcare', t, t.split('.')[1] + t.split('.')[0]) for t in dict.fromkeys(hc.tic.str.strip())]
    for kind, fname in [('xjllb', 'em_cashflow_annual.csv'), ('lrb', 'em_income_annual.csv')]:
        frames = []
        for sector, tic, code in codes:
            try:
                rows = statements(code, kind)
                if not rows:
                    raise RuntimeError('no data')
                d = pd.DataFrame(rows); d.insert(0, 'tic', tic); d.insert(0, 'sector', sector); frames.append(d)
                print(kind, sector, tic, code, len(d), flush=True)
            except Exception as e:
                LOG['errors'].append(dict(kind=kind, code=code, error=str(e))); print('ERROR', kind, code, e, flush=True)
            time.sleep(0.3)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / fname, index=False, encoding='utf-8')
    LOG['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    (out / 'fetch_em_fin_log.json').write_text(json.dumps(LOG, indent=1, ensure_ascii=False), encoding='utf-8')
    print('DONE errors', len(LOG['errors']), flush=True)


if __name__ == '__main__':
    main()
