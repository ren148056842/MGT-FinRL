import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'   # keep text as editable <text> in the SVG versions
B='/mnt/user-data/uploads/paper--2--xiu/revision_b3/forecasts/runs'
M=[('B2_GAT_Transformer_R','MGAT','#2a78d6'),('B1_GCN_Transformer_R','GCN','#eb6834'),('A10_No_cross_stock_R','No cross-stock','#1baf7a'),('A0_Full_MGT_R','MGNN','#c98500'),('A1_Transformer_only_R','Graph-free','#d55181')]
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8a8984','axes.labelcolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e'})
fig,ax=plt.subplots(2,5,figsize=(10,4.2),sharex=False)
for r,(sec,lab) in enumerate([('bank','Banking'),('healthcare','Healthcare')]):
    for c,(m,n,col) in enumerate(M):
        a=ax[r,c]
        for s in (1,2,3):
            h=pd.read_csv(f'{B}/{sec}/2025/{m}/seed{s}/history.csv')
            a.plot(h.epoch,h.loss,color=col,lw=1.0,ls='--',alpha=0.55)
            a.plot(h.epoch,h.val_loss,color=col,lw=1.6)
            k=h.val_loss.idxmin(); a.plot(h.epoch[k],h.val_loss[k],'o',ms=4,color=col,mec='white',mew=1)
        a.grid(axis='y',color='#e6e5e0',lw=0.6); a.set_axisbelow(True)
        if r==0: a.set_title(n,fontsize=9,color='#0b0b0b')
        if c==0: a.set_ylabel(lab+'\nloss')
        if r==1: a.set_xlabel('epoch')
from matplotlib.lines import Line2D
fig.legend([Line2D([],[],color='#52514e',lw=1.6),Line2D([],[],color='#52514e',lw=1.0,ls='--'),Line2D([],[],marker='o',ls='',color='#52514e',ms=4)],
           ['validation loss','training loss','early-stopping epoch'],loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(0.5,1.02))
fig.tight_layout(rect=(0,0,1,0.95))
fig.savefig('/home/claude/v40/figs/fig9_b3.pdf'); fig.savefig('/home/claude/v40/figs/fig9_b3.svg'); fig.savefig('/home/claude/v40/figs/fig9_b3_prev.png',dpi=90)
