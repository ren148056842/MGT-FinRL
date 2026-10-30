"""Checks of the archived inputs that are reported in the revised manuscript.

Reads the archives in the repository root (banksector.zip, healthcare.zip) and the
matrices in revision/; nothing is written. Run from any directory:
    python revision/analysis/verify_inputs.py
"""
import io, pathlib, zipfile
import numpy as np, pandas as pd

REV = pathlib.Path(__file__).resolve().parents[1]
TOP = REV.parent

def read_zip_csv(zname, member, **kw):
    with zipfile.ZipFile(TOP / zname) as z:
        return pd.read_csv(io.BytesIO(z.read(member)), **kw)

# 1. Kendall graphs: pandas tau_b on daily volume, 2018-01-02 .. 2022-12-30, missing values -> stock mean
bank_v = read_zip_csv('banksector.zip', 'banksector/bankvolume.CSV').drop(columns='Date')
bank_k = bank_v.fillna(bank_v.mean()).corr(method='kendall').values
print('bank Kendall max |diff| vs archived:', np.abs(bank_k - np.load(REV / 'bank sectordata/bankvolume_kendall.npy', allow_pickle=True)).max())
print('bank missing volumes (pre-listing):', bank_v.isna().sum()[lambda s: s > 0].to_dict())
tech_v = read_zip_csv('healthcare.zip', 'healthcare/techvolume20221231.CSV').drop(columns='Date')
tech_k = tech_v.corr(method='kendall').values
print('healthcare Kendall max |diff| vs archived:', np.abs(tech_k - np.load(REV / 'haelthcaresectordata/techvolume_kendall.npy', allow_pickle=True)).max())

# 2. Q&A corpora: size and period
for zname, member in [('banksector.zip', 'banksector/bankqa.csv'), ('healthcare.zip', 'healthcare/techqa.csv')]:
    q = read_zip_csv(zname, member)
    t = pd.to_datetime(q.iloc[:, 0], errors='coerce')
    print(member, 'records:', len(q), 'period:', t.min(), '->', t.max())

# 3. Shareholder graph: isolated banks and CMBC self-loop
order = ['PAB', 'SPDB', 'HXB', 'CMBC', 'CMB', 'CIB', 'ABC', 'BOCOM', 'ICBC', 'PSBC', 'CEB', 'CZB', 'CCB', 'BOC', 'ECITIC']
A = np.load(REV / 'bank sectordata/bankshareholding_relation_adjacent_matrix_tongyi.npy', allow_pickle=True).astype(float)
print('bank SH isolated:', [order[i] for i in range(15) if np.allclose(A[i], 0)], 'CMBC self-loop: %.4f' % A[3, 3])
sh = read_zip_csv('banksector.zip', 'banksector/bankstockholder.csv')
print('bank shareholder records:', len(sh), 'short names:', sh.Name.tolist())

# 4. Valuation ratios embed the same-day close: implied EPS = close / PE-TTM is unchanged on most days
r = pd.read_csv(REV / 'bank sectordata/banktrade20241231.CSV'); r['date'] = pd.to_datetime(r.date)
r = r[(r.date >= '2018-01-01') & (r.date <= '2022-12-31')].sort_values(['tic', 'date'])
eps = r.close / r.TTM
chg = eps.groupby(r.tic).pct_change().abs()
print('share of bank-days with unchanged implied EPS (|change|<0.1%%): %.3f' % (chg.dropna() < 1e-3).mean())
