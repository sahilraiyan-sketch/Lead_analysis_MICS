"""Fits the total and within-between models for each indicator in the pooled sample, Dhaka and elsewhere."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import sys, warnings, json; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from engine import Svy
from concurrent.futures import ProcessPoolExecutor
f0=pd.read_pickle(os.path.join(OUT_DIR,'cohort.pkl')); H=pd.read_pickle(os.path.join(OUT_DIR,'H.pkl')); C=pd.read_pickle(os.path.join(OUT_DIR,'C.pkl')); CE=pd.read_pickle(os.path.join(OUT_DIR,'CE.pkl'))
U=pd.read_csv(os.path.join(OUT_DIR,'universe_stats.csv'),index_col=0)
meta=pd.read_json(os.path.join(OUT_DIR,'screen.json'))
PRIMARY=['W01','S01','H04','E01','A03','D02','C01','N01','N03','M01']
def status(r):
    i=r['id']
    if i in PRIMARY: return 'Primary'
    if i=='A01': return 'Alt'
    if i in ['W16','W17','W18']: return 'Subset'
    if i=='G01': return 'Const'
    if r['prev'] is None or (r['n'] or 0)<1500: return 'Few'
    if r['binary'] and (r['prev']<0.03 or r['prev']>0.97): return 'Sparse'
    if (r['inf_psu'] or 0)<150: return 'FewPSU'
    return 'Atlas'
meta['status']=meta.apply(status,axis=1)
RUN=meta[meta.status.isin(['Primary','Atlas','Alt'])].id.tolist()
if os.environ.get('ONLY'): RUN=[i for i in RUN if i in os.environ['ONLY'].split(',')]
CONT={'A03','M01'}
BASE=['age','female','rural','C(wq)','C(me)','soil_ln','soil_miss','C(bat)','C(occ)']
SWAP={'A03':['C(wq)'],'A01':['C(wq)'],'D02':['C(me)'],'M01':['soil_ln','soil_miss'],'N01':['C(bat)'],'N02':['C(bat)'],'N03':['C(occ)'],'N04':['C(occ)'],'N05':['C(occ)'],'D06':['female']}
WEALTHCOMP=lambda i: i[0] in 'WSHEA' and i not in ('A03','A01')
def cov(i,nowq=False):
    drop=set(SWAP.get(i,[]))|({'C(wq)'} if nowq else set())
    return ' + '.join(b for b in BASE if b not in drop)
def prep(i):
    d=f0.copy(); d['Hx']=H[i].values; d['Cx']=C[i].values; d['Ex']=CE[i].values
    d=d[(d.me!=9)&d.Hx.notna()&d.Cx.notna()].copy()
    if i in CONT: d['Hx']=(d.Hx-d.Hx.mean())/d.Hx.std()
    sd=d.Cx.std(); d['Cz']=(d.Cx-d.Cx.mean())/sd; d['Ez']=(d.Ex-d.Ex.mean())/d.Ex.std()
    return d
def one(d,i,dom,out,nowq=False,eb=False):
    cv=cov(i,nowq); fam='linear' if out=='lin' else 'logit'; yv='loglead' if out=='lin' else 'y'
    cz='Ez' if eb else 'Cz'
    try:
        mT=Svy(f'{yv} ~ Hx + C(dist) + {cv}',d,family=fam); mW=Svy(f'{yv} ~ Hx + {cz} + C(dist) + {cv}',d,family=fam)
    except Exception as e:
        return None
    t=mT.lincom(['Hx']); w=mW.lincom(['Hx']); g=mW.lincom([cz])
    z=1.96
    return dict(id=i,domain=dom,outcome=out,nowq=nowq,eb=eb,n=mT.n,bT=t['b'],seT=t['se'],pT=t['p'],bW=w['b'],seW=w['se'],pW=w['p'],g=g['b'],seg=g['se'],pg=g['p'],ddf=mT.ddf)
def job(i):
    d=prep(i); rows=[]
    doms={'pooled':d,'Dhaka':d[d.dhaka==1],'Elsewhere':d[d.dhaka==0]}
    for dom,dd in doms.items():
        if len(dd)<300: continue
        for out in ('lin','bin'):
            r=one(dd,i,dom,out)
            if r: rows.append(r)
        if dom in ('pooled','Dhaka'):
            r=one(dd,i,dom,'lin',eb=True)
            if r: rows.append(r)
    if WEALTHCOMP(i):
        r=one(d,i,'pooled','lin',nowq=True)
        if r: rows.append(r)
        r=one(d[d.dhaka==1],i,'Dhaka','lin',nowq=True)
        if r: rows.append(r)
    return rows
if __name__=='__main__':
    print('indicators to run',len(RUN),flush=True)
    import os
    done=set(); allrows=[]
    if os.path.exists(os.path.join(OUT_DIR,'atlas_partial.csv')):
        pr=pd.read_csv(os.path.join(OUT_DIR,'atlas_partial.csv')); allrows=pr.to_dict('records'); done=set(pr.id)
    for k,i in enumerate(RUN):
        if i in done: continue
        rows=job(i); allrows+=rows; print(k+1,i,len(rows),flush=True)
        pd.DataFrame(allrows).to_csv(os.path.join(OUT_DIR,'atlas_partial.csv'),index=False)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT_DIR,'atlas_raw.csv'),index=False)
    meta[['id','label','status']].to_csv(os.path.join(OUT_DIR,'status.csv'),index=False)
    print('done',len(allrows))
