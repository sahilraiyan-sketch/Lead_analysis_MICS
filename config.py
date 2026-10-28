"""Adjusted p-values, descriptive patterns, household-value SDs and common-scale within components for the 63 indicators."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
r=pd.read_csv(os.path.join(OUT_DIR,'atlas_raw.csv')); st=pd.read_csv(os.path.join(OUT_DIR,'status.csv')).set_index('id'); U=pd.read_csv(os.path.join(OUT_DIR,'universe_stats.csv'),index_col=0)
f0=pd.read_pickle(os.path.join(OUT_DIR,'cohort.pkl')); H=pd.read_pickle(os.path.join(OUT_DIR,'H.pkl')); C=pd.read_pickle(os.path.join(OUT_DIR,'C.pkl'))
p=r[(r.domain=='pooled')&(r.outcome=='lin')&(~r.nowq)&(~r.eb)].set_index('id').copy()
p['p_adjW']=multipletests(p.pW.fillna(1),method='fdr_bh')[1]; p['p_adjG']=multipletests(p.pg.fillna(1),method='fdr_bh')[1]
names={(True,True):'Both detected',(True,False):'Household detected; contextual not',(False,True):'Contextual detected; household not',(False,False):'Neither detected'}
p['pat2']=[names[(bool(a<0.05),bool(b<0.05))] for a,b in zip(p.p_adjW,p.p_adjG)]
p['label']=st.loc[p.index,'label']; p['rel']=U.loc[p.index,'rel']; p['icc']=U.loc[p.index,'icc_u']
for a,s_ in [('bT','seT'),('bW','seW'),('g','seg')]:
    p[a+'_lo']=p[a]-1.96*p[s_]; p[a+'_hi']=p[a]+1.96*p[s_]
sd={}
for i in p.index:
    m=(f0.me!=9)&H[i].notna()&C[i].notna(); sd[i]=1.0 if i in ('A03','M01') else float(H.loc[m,i].std())
p['sdH']=pd.Series(sd); p['bWsd']=p.bW*p.sdH; p['seWsd']=p.seW*p.sdH
p.to_csv(os.path.join(OUT_DIR,'pooled_final_with_patterns.csv')); pd.Series(sd).to_csv(os.path.join(OUT_DIR,'sdH.csv'))
print(p.pat2.value_counts())
