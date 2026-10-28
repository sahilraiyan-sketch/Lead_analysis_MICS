"""Figure 2 (primary indicators, children and women) and Figure 3 (all indicators) on the common per-SD scale."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.lines as ml
p=pd.read_csv(os.path.join(OUT_DIR,'pooled_final_with_patterns.csv'),index_col=0); W=pd.read_csv(os.path.join(OUT_DIR,'women_results.csv')); sd=pd.read_csv(os.path.join(OUT_DIR,'sdH.csv'),index_col=0).iloc[:,0]
pc=lambda b:100*(np.exp(b)-1); os.makedirs(os.path.join(OUT_DIR,'figures'),exist_ok=True)
PRIM=['W01','S01','H04','E01','A03','D02','C01','N01','N03','M01']
short={'W01':'Piped water','S01':'Improved sanitation','H04':'Finished floor, roof, wall','E01':'Clean cooking','A03':'Wealth score','D02':'Mother secondary or higher','C01':'Pica','N01':'Lead activity nearby','N03':'Lead-related occupation','M01':'Soil lead'}
wp=W[(W.domain=='pooled')&(W.spec=='base')].set_index('id'); y=np.arange(len(PRIM))[::-1]
fig,axs=plt.subplots(1,2,figsize=(9.2,5.3),sharey=True)
for ax,(b,se,ttl,scale) in zip(axs,[('bW','seW','Within neighborhood\n(% per SD of the household value)',True),('g','seg','Between neighborhoods (contextual)\n(% per SD of the PSU mean)',False)]):
    for yy,i in zip(y,PRIM):
        r=p.loc[i]; k=sd[i] if scale else 1.0; off=0.17 if i in wp.index else 0
        ax.plot([pc((r[b]-1.96*r[se])*k),pc((r[b]+1.96*r[se])*k)],[yy+off]*2,color='#1f3a5f',lw=1.8); ax.plot(pc(r[b]*k),yy+off,'o',color='#1f3a5f',ms=5.5)
        if i in wp.index:
            w=wp.loc[i]; ax.plot([pc((w[b]-1.96*w[se])*k),pc((w[b]+1.96*w[se])*k)],[yy-off]*2,color='#d4731c',lw=1.8); ax.plot(pc(w[b]*k),yy-off,'D',color='#d4731c',ms=5)
    ax.axvline(0,color='grey',lw=.8,ls='--'); ax.set_title(ttl,fontsize=9.5); ax.set_xlabel('% difference in geometric-mean blood lead',fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([short[i] for i in PRIM],fontsize=9); ax.tick_params(axis='x',labelsize=8.5); ax.set_xlim(-18,20)
fig.legend(handles=[ml.Line2D([],[],color='#1f3a5f',marker='o',label='Children'),ml.Line2D([],[],color='#d4731c',marker='D',label='Pregnant women (household-level indicators only)')],loc='lower center',ncol=2,fontsize=8.5,frameon=False)
plt.tight_layout(rect=(0,0.06,1,1)); plt.savefig(os.path.join(OUT_DIR,'figures','Figure2_primary_indicators.png'),dpi=220); plt.close()
lab={'W01':'Piped water','W02':'Own piped connection','W04':'Tubewell','E01':'Clean cooking','E02':'Solid-fuel cooking','H07':'Owns dwelling','S04':'Shared toilet','H01':'Finished floor','H06':'Crowding','A03':'Wealth score','A06':'Smartphone','A04':'Refrigerator','A08':'Bank account','D02':'Mother secondary+','D04':'Household size >=6','A09':'Agricultural land','A10':'Livestock','D06':'Child is female','C01':'Pica','M01':'Soil lead','S01':'Improved sanitation'}
col={'Contextual detected; household not':'#27ae60','Household detected; contextual not':'#2e86c1','Both detected':'#c0392b','Neither detected':'#b8c2cc'}
X=pc(p.bW*p.sdH); Yv=pc(p.g)
fig,ax=plt.subplots(figsize=(8.2,6.6))
for k,c in col.items():
    m=p.pat2==k; ax.scatter(X[m],Yv[m],s=42,c=c,edgecolor='white',lw=.6,label=f'{k} ({int(m.sum())})',zorder=3)
ax.axhline(0,color='grey',lw=.7); ax.axvline(0,color='grey',lw=.7); ax.set_xlim(-5,5); ax.set_ylim(-6.4,6.9)
offs={'W01':(6,4),'W02':(-112,3),'E01':(6,-5),'W04':(-50,-3),'E02':(-98,1),'H07':(6,-12),'S04':(5,3),'H01':(6,1),'H06':(6,-9),'A03':(-62,4),'A06':(-62,1),'A04':(-58,6),'A08':(6,-10),'D02':(-8,-14),'D04':(6,3),'A09':(6,-3),'A10':(6,-9),'D06':(6,3),'C01':(-4,8),'M01':(6,-4),'S01':(6,5)}
for i,t in lab.items(): ax.annotate(t,(X[i],Yv[i]),fontsize=7.8,xytext=offs.get(i,(4,3)),textcoords='offset points',zorder=4)
ax.set_xlabel('Within-neighborhood component, % per SD of the household value',fontsize=9.5); ax.set_ylabel('Between-neighborhood (contextual) component,\n% per SD of the PSU mean',fontsize=9.5)
ax.legend(fontsize=8.2,loc='upper center',bbox_to_anchor=(0.5,-0.12),ncol=2,frameon=False); ax.tick_params(labelsize=8.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'figures','Figure3_all_indicators.png'),dpi=220,bbox_inches='tight'); plt.close(); print('figures written')
