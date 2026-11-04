"""Result figures for v4.0: forecasting evidence (fig_forecast_b3.pdf) and 2025 portfolio paths (fig_portfolio_b3.pdf)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'   # keep text as editable <text> in the SVG versions
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RES = '/home/claude/mgt-finrl/revision/revision_b3/results'
RL = '/mnt/user-data/uploads/paper--2--xiu/revision_b3/rl'
OUT = '/home/claude/v40/figs'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#c98500'
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 8.5,
                     'axes.titlecolor': INK, 'legend.fontsize': 7.5, 'pdf.fonttype': 42})
SEC = [('bank', 'Banking'), ('healthcare', 'Healthcare')]


def style(a, ygrid=True):
    if ygrid:
        a.grid(axis='y', color=GRID, lw=0.6)
    a.set_axisbelow(True)


# ------------------------------------------------------------------ Figure: forecasting evidence
fm = pd.read_csv(RES + '/forecast_metrics.csv')
YEARS = [2022, 2023, 2024, 2025]
NEURAL = ['MGAT_T', 'A1_R', 'GCN_R', 'A10_R', 'MGT_R']
fig, ax = plt.subplots(2, 2, figsize=(6.4, 4.5))
for c, (sec, name) in enumerate(SEC):
    # (a)/(b): RMSE reduction of MGAT for the next-day high and low, with the range of the other neural forecasters
    a = ax[0, c]; x = np.arange(len(YEARS)); w = 0.34
    for k, (ch, lab, col) in enumerate([('RMSE_h1_rel_naive', 'High', BLUE), ('RMSE_l1_rel_naive', 'Low', AQUA)]):
        g = fm[(fm.sector == sec) & (fm.model == 'MGAT_T')].set_index('test_year')[ch].reindex(YEARS)
        oth = fm[(fm.sector == sec) & fm.model.isin(NEURAL[1:])].pivot_table(index='test_year', columns='model', values=ch).reindex(YEARS)
        xx = x + (k - 0.5) * w
        a.bar(xx, -100 * g.values, w * 0.92, color=col, label='MGAT, next-day ' + lab.lower(), zorder=2)
        a.vlines(xx, -100 * oth.max(axis=1).values, -100 * oth.min(axis=1).values, color=INK, lw=1.0, zorder=3)
        top = np.maximum(-100 * g.values, -100 * oth.min(axis=1).values)
        for xi, v, tp in zip(xx, g.values, top):
            a.text(xi, tp + 0.6, '%.0f' % (-100 * v), ha='center', va='bottom', fontsize=6.5, color=INK2)
    a.axvspan(2.5, 3.5, color='#f3f2ee', zorder=0)
    a.text(3, 35.5, 'evaluation year', ha='center', va='top', fontsize=6.5, color=MUTED)
    a.set_xticks(x); a.set_xticklabels([str(y) for y in YEARS]); a.set_ylim(0, 36)
    a.set_title('(%s) %s: next-day high and low' % ('ab'[c], name), loc='left')
    if c == 0:
        a.set_ylabel('RMSE reduction vs\nprevious-day values (%)')
    style(a)
    # (c)/(d): rank IC at the tradable horizon with 95% intervals
    a = ax[1, c]
    for k, (m, lab, col, mk) in enumerate([('MGAT_T', 'MGAT', BLUE, 'o'), ('A1_R', 'Graph-free Transformer', ORANGE, 's'), ('Ridge_R', 'Ridge', YELLOW, 'D')]):
        g = fm[(fm.sector == sec) & (fm.model == m)].set_index('test_year').reindex(YEARS)
        se = (g.IC2 / g.IC2_t).abs()
        xx = x + (k - 1) * 0.22
        a.errorbar(xx, g.IC2, yerr=1.96 * se, fmt=mk, ms=4.5, color=col, mec='white', mew=0.8, elinewidth=1.2, capsize=0, label=lab, zorder=3)
    a.axhline(0, color=MUTED, lw=0.8, zorder=1)
    a.axvspan(2.5, 3.5, color='#f3f2ee', zorder=0)
    a.set_xticks(x); a.set_xticklabels([str(y) for y in YEARS]); a.set_ylim(-0.075, 0.13)
    a.set_title('(%s) %s: IC$_2$ of the tradable return' % ('cd'[c], name), loc='left')
    if c == 0:
        a.set_ylabel('mean daily rank IC (95% interval)')
    if sec == 'healthcare':
        a.annotate('registered H3: $p$ = 0.012', xy=(2.78, 0.052), xytext=(1.25, 0.112), fontsize=6.5, color=INK2,
                   arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7))
    style(a)
h1, l1 = ax[0, 0].get_legend_handles_labels()
h1 += [Line2D([], [], color=INK, lw=1.0)]; l1 += ['range of the other neural forecasters']
h2, l2 = ax[1, 0].get_legend_handles_labels()
fig.tight_layout(h_pad=1.8, w_pad=1.6, rect=(0, 0.05, 1, 0.93))
fig.legend(h1, l1, loc='upper center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
fig.legend(h2, l2, loc='lower center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
fig.savefig(OUT + '/fig_forecast_b3.pdf'); fig.savefig(OUT + '/fig_forecast_b3.svg'); fig.savefig(OUT + '/fig_forecast_b3_prev.png', dpi=110)
plt.close(fig)


# ------------------------------------------------------------------ Figure: 2025 portfolio paths and per-agent differences
def nav(path):
    t = pd.read_csv(path)
    return pd.Series(t.nav.values / t.nav.values[0], index=pd.to_datetime(t.date))


ALG = ['ppo', 'a2c', 'sac', 'td3', 'ddpg']
fig = plt.figure(figsize=(6.4, 4.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.22, top=0.86)
for c, (sec, name) in enumerate(SEC):
    a = fig.add_subplot(gs[0, c])
    for src, lab, col in [('Naive', 'Naive agents', ORANGE), ('MGAT_T', 'MGAT agents', BLUE)]:
        P = pd.concat([nav('%s/results/B3_%s_%s_%s_s%d_F1/trajectory.csv' % (RL, sec, src, al, s)) for al in ALG for s in range(1, 6)], axis=1)
        lo, hi = P.quantile(0.1, axis=1), P.quantile(0.9, axis=1)
        a.fill_between(P.index, lo, hi, color=col, alpha=0.14, lw=0)
        a.plot(P.index, P.mean(axis=1), color=col, lw=1.8, label=lab + ' (mean, 10–90%)')
    ew = nav('%s/references/%s_equal_daily_2025_F1/trajectory.csv' % (RL, sec))
    a.plot(ew.index, ew.values, color=INK2, lw=1.2, ls='--', label='Equal weight, daily')
    r0 = nav('%s/references/%s_Forecast_MGAT_T_2025_F0/trajectory.csv' % (RL, sec))
    r1 = nav('%s/references/%s_Forecast_MGAT_T_2025_F1/trajectory.csv' % (RL, sec))
    a.plot(r0.index, r0.values, color=AQUA, lw=1.2, ls=':', label='Top-3 MGAT rule, no costs')
    a.plot(r1.index, r1.values, color=AQUA, lw=1.2, label='Top-3 MGAT rule, F1')
    a.set_title('(%s) %s, 2025: account value' % ('ab'[c], name), loc='left')
    a.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b'))
    if c == 0:
        a.set_ylabel('value (initial = 1)')
    style(a)
    if c == 0:
        fig.legend(*a.get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
# (c) per-agent paired differences
ba = pd.read_csv(RES + '/rl_by_algorithm.csv'); tests = pd.read_csv(RES + '/rl_tests.csv')
a = fig.add_subplot(gs[1, :])
rows = [('All agents (H1)', None)] + [(al.upper(), al) for al in ALG]
for k, (sec, name) in enumerate(SEC):
    col = [INK2, BLUE][k]
    for i, (lab, al) in enumerate(rows):
        if al is None:
            t = tests[(tests.sector == sec) & (tests.hypothesis == 'H1') & (tests.block == 10)].iloc[0]
        else:
            t = ba[(ba.sector == sec) & (ba.algorithm == al) & (ba.comparison == 'MGAT_T - Naive') & (ba.fee == 'F1')].iloc[0]
        y = i + (k - 0.5) * 0.32
        a.plot([t.ci90_low, t.ci90_high], [y, y], color=col, lw=1.6, solid_capstyle='round')
        a.plot(t.dSR, y, 'o' if k else 's', ms=5, color=col, mec='white', mew=0.8)
a.axvline(0, color=MUTED, lw=0.8)
a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows]); a.invert_yaxis()
a.set_xlabel('paired difference in Sharpe ratio, MGAT − naive (90% block-bootstrap interval)')
a.set_title('(c) Paired differences from the naive source, 2025 (F1)', loc='left')
a.grid(axis='x', color=GRID, lw=0.6); a.set_axisbelow(True)
a.legend([Line2D([], [], marker='s', color=INK2, ms=5, lw=1.6), Line2D([], [], marker='o', color=BLUE, ms=5, lw=1.6)],
         ['Banking', 'Healthcare'], loc='lower right', bbox_to_anchor=(1.0, 0.995), frameon=False, ncol=2, borderaxespad=0.1, handlelength=1.6)
fig.savefig(OUT + '/fig_portfolio_b3.pdf', bbox_inches='tight'); fig.savefig(OUT + '/fig_portfolio_b3.svg', bbox_inches='tight'); fig.savefig(OUT + '/fig_portfolio_b3_prev.png', dpi=110, bbox_inches='tight')
print('ok')


# SVG copies for editing: plain font fallbacks (Inter if installed, otherwise Arial/Helvetica)
import re as _re
for _f in ['fig_forecast_b3.svg', 'fig_portfolio_b3.svg']:
    _s = open(OUT + '/' + _f).read()
    _s = _re.sub(r"font-family: [^;\"]+", "font-family: 'Inter', Arial, Helvetica, sans-serif", _s)
    open(OUT + '/' + _f, 'w').write(_s)
