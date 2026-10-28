"""Holm/BH adjustment, strict pre-declared share classification (with bootstrap CIs) -> final_pooled.csv"""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import pandas as pd, numpy as np
from statsmodels.stats.multitest import multipletests
r=pd.read_csv(os.path.join(OUT_DIR,'atlas_raw.csv')); st=pd.read_csv(os.path.join(OUT_DIR,'status.csv')); b=pd.read_csv(os.path.join(OUT_DIR,'boot_partial.csv'))
P=['W01','S01','H04','E01','A03','D02','C01','N01','N03','M01']
r=r.merge(st[['id','label']],on='id')
cls_ci=lambda lo,hi:(lambda c:c(lo) if c(lo)==c(hi) else None)(lambda x:'N' if x>=2/3 else ('M' if x>=1/3 else 'H'))
pool=r[(r.domain=='pooled')&(r.outcome=='lin')&(~r.nowq)&(~r.eb)].set_index('id'); eb=r[(r.domain=='pooled')&(r.outcome=='lin')&(r.eb)].set_index('id')
prim=[i for i in P if i in pool.index]; atl=[i for i in pool.index if i not in P]; padj={}
for grp,meth in [(prim,'holm'),(atl,'fdr_bh')]:
    padj.update(dict(zip(grp,multipletests(pool.loc[grp,'pg'].fillna(1).values,method=meth)[1])))
out=[]
for i in pool.index:
    x=pool.loc[i]; bb=b[(b.id==i)&(b.domain=='pooled')]
    ns,lo,hi=(np.nan,)*3 if bb.empty else (bb.NS.iloc[0],bb.NS_lo.iloc[0],bb.NS_hi.iloc[0])
    nsE=(eb.loc[i,'bT']-eb.loc[i,'bW'])/eb.loc[i,'bT']; k3=lambda v:'N' if v>=2/3 else ('M' if v>=1/3 else 'H')
    if x.pT>=0.05: c='Indeterminate (no clear total association)'
    else:
        k=cls_ci(lo,hi)
        if k is None: c='Indeterminate (share CI spans classes)'
        elif k3(nsE)!=k3(ns): c='Indeterminate (raw vs shrunken disagree)'
        elif k=='N' and padj[i]>=0.05: c='Indeterminate (contextual effect n.s.)'
        elif k=='H' and x.pW>=0.05: c='Indeterminate (within effect n.s.)'
        else: c={'N':'Neighborhood-carried','M':'Mixed','H':'Household-carried'}[k]
    out.append(dict(id=i,label=x.label,primary=i in P,n=int(x.n),bT=x.bT,pT=x.pT,bW=x.bW,pW=x.pW,gamma=x.g,p_gamma=x.pg,p_adj=padj[i],NS=ns,NS_lo=lo,NS_hi=hi,cls=c))
pd.DataFrame(out).to_csv(os.path.join(OUT_DIR,'final_pooled.csv'),index=False); print(pd.DataFrame(out).cls.value_counts())
