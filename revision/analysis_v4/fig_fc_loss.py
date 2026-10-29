"""Validation-loss curves of the v4 forecasters (2022 fold, three seeds), Appendix figure."""
import sys, pathlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
FC, OUT = pathlib.Path(sys.argv[1]), sys.argv[2]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e6e5e0'
MODELS = [('A0_Full_MGT', 'MGT (A0)', '#eb6834'), ('A1_Transformer_only', 'Transformer only (A1)', '#2a78d6'),
          ('A10_No_cross_stock', 'No cross-stock propagation (A10)', '#7a5195'), ('B1_GCN_Transformer', 'Multi-GCN-Transformer', '#3a9d5d'),
          ('B2_GAT_Transformer', 'Multi-GAT-Transformer', '#bc5090'), ('B3_GNN_GRU', 'Multi-GNN-GRU', '#8c6d31'),
          ('B4_GCN_GRU', 'Multi-GCN-GRU', '#17becf'), ('B5_GAT_GRU', 'Multi-GAT-GRU', '#7f7f7f')]
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9994',
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'font.family': 'sans-serif',
                     'pdf.use14corefonts': True, 'axes.unicode_minus': False})
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
YL = {('bank','loss'):(0.55,1.4),('bank','val_loss'):(0.78,1.1),('healthcare','loss'):(0.85,1.3),('healthcare','val_loss'):(0.93,1.15)}
for col, (sec, title) in enumerate([('bank', 'Banking'), ('healthcare', 'Healthcare innovation')]):
    for row, (key, lab) in enumerate([('loss', 'Training loss'), ('val_loss', 'Validation loss')]):
        ax = axes[row, col]
        for name, label, c in MODELS:
            for s in [1, 2, 3]:
                f = FC / 'runs' / sec / '2022' / name / ('seed%d' % s) / 'history.csv'
                if not f.exists(): continue
                h = pd.read_csv(f)
                ax.plot(h.epoch, h[key], color=c, lw=0.9, alpha=0.85, label=label if s == 1 else None)
        ax.set_title('%s: %s' % (title, lab.lower()), fontsize=8, loc='left', color=INK)
        ax.grid(axis='y', color=GRID, lw=0.6); ax.set_ylim(*YL[(sec, key)])
        if col == 0: ax.set_ylabel('MSE (standardised targets)')
        if row == 1: ax.set_xlabel('Epoch')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, ncol=4, loc='lower center')
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT)
print('saved', OUT)
