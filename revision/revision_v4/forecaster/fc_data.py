"""Data pipeline of the v4 walk-forward forecaster (numpy/pandas only).

Inputs per stock and day t (all known at the close of t):
  r1  = log(C_t / C_{t-1})                    daily log return
  hc  = log(H_t / C_t), lc = log(L_t / C_t)    intraday range relative to the close
  rv  = log((V_t + 1) / (mean(V_{t-20..t-1}) + 1)), clipped to [-5, 5]; V = trading value (CNY)
  ep, bp, sp, cfp = 1/PE-TTM, 1/PB, 1/PS, 1/PCF  valuation yields of day t
Targets for origin t and horizons k = 1, 2:
  log(H_{t+k}/C_t), log(L_{t+k}/C_t), log(C_{t+k}/C_t)   (channel order h1, l1, c1, h2, l2, c2)
Masks: features of a stock before its A-share listing are set to the training mean (0 after
standardisation); a target is masked if the origin precedes the listing or the stock does not
trade on the target day (suspension; trading value reported as '--').
Graph node order equals the per-date row order of the trade panel (the order used to build the
relation matrices); PAD is the same entity as PAB.
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]          # D:/paper/2/xiu
SECTORS = {
    'bank': dict(folder='bank sectordata', panel='banktrade20241231.CSV',
                 graphs=dict(KD='bankvolume_kendall.npy', QA='bank_qa_graph.npy',
                             SH='bankshareholding_relation_adjacent_matrix_tongyi.npy'),
                 listing={'CZB': '2019-11-26', 'PSBC': '2019-12-10'}),
    'healthcare': dict(folder='haelthcaresectordata', panel='techtrade20241231.CSV',
                       graphs=dict(KD='techvolume_kendall.npy', QA='tech_qa_graph.npy', SH='techstockholder.npy'),
                       listing={}),
}
GRAPH_ORDER = ['KD', 'QA', 'SH']          # same order as Fixed_Matrices = [S, I, T] in the notebooks
FEATURES = ['r1', 'hc', 'lc', 'rv', 'ep', 'bp', 'sp', 'cfp']
TARGETS = ['h1', 'l1', 'c1', 'h2', 'l2', 'c2']
WINDOW = 20
FOLDS = {2022: dict(train_end='2020-12-31', val=2021, test=2022),
         2023: dict(train_end='2021-12-31', val=2022, test=2023),
         2024: dict(train_end='2022-12-31', val=2023, test=2024)}


def load_panel(sector, root=None):
    root = ROOT if root is None else root
    cfg = SECTORS[sector]
    f = pathlib.Path(root) / cfg['folder'] / cfg['panel']
    d = pd.read_csv(f, dtype={'tic': str})
    d['tic'] = d.tic.str.strip()
    d['date'] = pd.to_datetime(d.date)
    n = d.tic.nunique()
    order = list(d.tic.iloc[:n])
    arr = np.asarray(d.tic.values, dtype=object).reshape(-1, n)
    if not (arr == np.asarray(order, dtype=object)).all():
        raise ValueError('Row order of tickers changes across dates')
    dates = pd.DatetimeIndex(d.date.iloc[::n])
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Dates not strictly increasing')

    def grid(col, numeric=True):
        x = pd.to_numeric(d[col], errors='coerce') if numeric else d[col]
        return np.asarray(x, dtype=float).reshape(-1, n)

    H, L, C = grid('high'), grid('low'), grid('close')
    V = grid('volume')                        # NaN where the panel reports '--' (no trading)
    ratios = np.stack([grid(c) for c in ['TTM', 'PBR', 'PS', 'PCF']], axis=2)
    listed = np.ones_like(C, dtype=bool)
    for tic, day in cfg['listing'].items():
        listed[:, order.index(tic)] = dates >= pd.Timestamp(day)
    traded = listed & np.isfinite(V) & (np.nan_to_num(V) > 0)
    if not (np.isfinite(H).all() and np.isfinite(L).all() and np.isfinite(C).all() and (C > 0).all()):
        raise ValueError('Invalid H/L/C')
    graphs = [np.load(pathlib.Path(root) / cfg['folder'] / cfg['graphs'][g], allow_pickle=True).astype(np.float32)
              for g in GRAPH_ORDER]
    return dict(sector=sector, dates=dates, tickers=order, H=H, L=L, C=C, V=V, ratios=ratios,
                listed=listed, traded=traded, graphs=graphs, panel_file=f, frame=d)


def features(p):
    H, L, C, V, R = p['H'], p['L'], p['C'], p['V'], p['ratios']
    T, N = C.shape
    X = np.full((T, N, len(FEATURES)), np.nan)
    X[1:, :, 0] = np.log(C[1:] / C[:-1])
    X[:, :, 1] = np.log(H / C)
    X[:, :, 2] = np.log(L / C)
    v = np.where(p['listed'], np.nan_to_num(V, nan=0.0), np.nan)
    trail = pd.DataFrame(v).rolling(20, min_periods=1).mean().shift(1).to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        X[:, :, 3] = np.clip(np.log((v + 1.0) / (trail + 1.0)), -5, 5)
        X[:, :, 4:8] = 1.0 / R
    X[~p['listed']] = np.nan
    return X


def targets(p):
    H, L, C = p['H'], p['L'], p['C']
    T, N = C.shape
    Y = np.full((T, N, 6), np.nan)
    M = np.zeros((T, N, 6), dtype=bool)
    for j, k in enumerate([1, 2]):
        for c, P in enumerate([H, L, C]):
            Y[:-k, :, 3 * j + c] = np.log(P[k:] / C[:-k])
            M[:-k, :, 3 * j + c] = p['listed'][:-k] & p['traded'][k:]
    Y[~M] = np.nan
    return Y, M


def split_indices(dates, fold, window=WINDOW):
    """Origins t (row indices).  train: t+2 <= train_end; val: t+1 and t+2 in the validation
    year; test: t+1 in the test year (t+2 may fall after the test year or be missing)."""
    cfg = FOLDS[fold]
    T = len(dates)
    t = np.arange(window - 1, T - 1)
    d1 = dates[t + 1]
    d2 = pd.DatetimeIndex([dates[i + 2] if i + 2 < T else pd.NaT for i in t])
    train = t[(d2 <= pd.Timestamp(cfg['train_end']))]
    val = t[(d1.year == cfg['val']) & (d2.year == cfg['val'])]
    test = t[d1.year == cfg['test']]
    if not (train.max() < val.min() and val.max() < test.min()):
        raise ValueError('Folds overlap')
    return dict(train=train, val=val, test=test)


class Scaler:
    """Feature and target standardisation fitted on training rows only."""
    def __init__(self, X, Y, M, train_idx, clip=5.0):
        last = train_idx.max() + 2                  # last day whose data enter training targets
        rows = X[: train_idx.max() + 1].reshape(-1, X.shape[2])
        self.mu = np.nanmean(rows, axis=0)
        self.sd = np.nanstd(rows, axis=0)
        self.sd[~(self.sd > 0)] = 1.0
        yt = Y[train_idx].reshape(-1, Y.shape[2])
        self.ymu = np.nanmean(yt, axis=0)
        self.ysd = np.nanstd(yt, axis=0)
        self.clip = clip
        self.last_feature_row = int(train_idx.max())
        self.last_target_row = int(last)

    def x(self, X):
        Z = (X - self.mu) / self.sd
        Z = np.clip(Z, -self.clip, self.clip)
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)

    def y(self, Y):
        return np.nan_to_num((Y - self.ymu) / self.ysd, nan=0.0).astype(np.float32)

    def y_inv(self, Z):
        return Z * self.ysd + self.ymu

    def state(self):
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist(), ymu=self.ymu.tolist(), ysd=self.ysd.tolist(),
                    clip=self.clip, last_feature_row=self.last_feature_row, last_target_row=self.last_target_row)


def windows(Z, idx, window=WINDOW):
    """[n, window, N, F] tensors ending at each origin."""
    return np.stack([Z[i - window + 1: i + 1] for i in idx]).astype(np.float32)


def naive_forecast(p, idx):
    """Previous-day values: H_{t+k} = H_t, L_{t+k} = L_t, C_{t+k} = C_t."""
    H, L, C = p['H'][idx], p['L'][idx], p['C'][idx]
    h, l = np.log(H / C), np.log(L / C)
    z = np.zeros_like(h)
    return np.stack([h, l, z, h, l, z], axis=2)


def to_prices(p, idx, F):
    """Convert log-ratio forecasts at origins idx into price forecasts (horizon 1 and 2)."""
    C = p['C'][idx][:, :, None]
    P = C * np.exp(F)
    for k in range(2):
        h, l, c = P[:, :, 3 * k], P[:, :, 3 * k + 1], P[:, :, 3 * k + 2]
        P[:, :, 3 * k] = np.maximum(h, np.maximum(c, l))
        P[:, :, 3 * k + 1] = np.minimum(l, np.minimum(c, h))
    return P
