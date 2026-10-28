"""Cluster bootstrap of the neighborhood share for primary indicators and indicators with a clear total association."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import sys, os, warnings, time; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, patsy
from run_atlas import prep, cov, PRIMARY
r=pd.read_csv(os.path.join(OUT_DIR,'atlas_raw.csv'))
base=r[(r.domain=='pooled')&(r.outcome=='lin')&(~r.nowq)&(~r.eb)]
atlas_ids=[i for i in base[(base.pT<0.05)].id if i not in PRIMARY]
jobs=[(i,'pooled',400) for i in PRIMARY]+[(i,'Dhaka',400) for i in PRIMARY]+[(i,'pooled',200) for i in atlas_ids]
OUT=os.path.join(OUT_DIR,'boot_partial.csv')
done=set()
if os.path.exists(OUT):
    pr=pd.read_csv(OUT); done=set(zip(pr.id,pr.domain)); rows=pr.to_dict('records')
else: rows=[]
def design(d,i,spec):
    cv=cov(i); rhs='Hx + Cz + ' if spec=='W' else 'Hx + '
    Y,X=patsy.dmatrices(f'loglead ~ {rhs}C(dist) + {cv}',d,return_type='dataframe')
    return Y.iloc[:,0].to_numpy(float),X
def stats_for(d,i,B,seed):
    rng=np.random.default_rng(seed)
    d=d.copy();
    yT,XT=design(d,i,'T'); yW,XW=design(d,i,'W'); idx=XT.index.intersection(XW.index)
    dd=d.loc[idx]; XT=XT.loc[idx]; XW=XW.loc[idx]; yT=dd.loglead.to_numpy(float); w=dd.w.to_numpy(float)
    psu=pd.factorize(dd.PSU)[0]; P=psu.max()+1
    strat=dd.groupby('PSU').stratum.first().reindex(pd.factorize(dd.PSU)[1]).to_numpy()
    order=np.argsort(psu,kind='stable'); bounds=np.searchsorted(psu[order],np.arange(P+1))
    def agg(X):
        Xv=X.to_numpy(float)[order]; wv=w[order]; yv=yT[order]; k=Xv.shape[1]
        A=np.zeros((P,k*k)); b=np.zeros((P,k))
        for p in range(P):
            s,e=bounds[p],bounds[p+1]; x=Xv[s:e]; ww=wv[s:e]
            A[p]=(x.T@(x*ww[:,None])).ravel(); b[p]=x.T@(ww*yv[s:e])
        return A,b,k
    AT,bT_,kT=agg(XT); AW,bW_,kW=agg(XW)
    iT=list(XT.columns).index('Hx'); iW=list(XW.columns).index('Hx'); iC=list(XW.columns).index('Cz')
    def est(c,A,b,k,ix):
        S=(c@A).reshape(k,k); t=c@b
        beta=np.linalg.lstsq(S,t,rcond=1e-10)[0]; return beta[ix]
    one=np.ones(P)
    pT=est(one,AT,bT_,kT,iT); pW=est(one,AW,bW_,kW,iW)

    sidx=[np.where(strat==h)[0] for h in np.unique(strat)]
    res=[]
    for b in range(B):
        c=np.zeros(P)
        for ix in sidx:
            m=len(ix)
            if m<2: c[ix]+=1; continue
            draw=rng.integers(0,m,m); np.add.at(c,ix[draw],1)
        bt=est(c,AT,bT_,kT,iT); bw=est(c,AW,bW_,kW,iW); res.append((bt,bw))
    res=np.array(res); ns=(res[:,0]-res[:,1])/res[:,0]
    return dict(bT=pT,bW=pW,NS=(pT-pW)/pT,NS_lo=np.nanpercentile(ns,2.5),NS_hi=np.nanpercentile(ns,97.5),NS_med=np.nanmedian(ns),
        bT_lo=np.percentile(res[:,0],2.5),bT_hi=np.percentile(res[:,0],97.5),bW_lo=np.percentile(res[:,1],2.5),bW_hi=np.percentile(res[:,1],97.5),
        diff_lo=np.percentile(res[:,0]-res[:,1],2.5),diff_hi=np.percentile(res[:,0]-res[:,1],97.5),B=B,nP=P)
t0=time.time(); BUDGET=float(os.environ.get('TIME_BUDGET','1e12'))
for (i,dom,B) in jobs:
    if (i,dom) in done: continue
    if time.time()-t0>BUDGET: print('time budget reached'); break
    d=prep(i)
    if dom=='Dhaka': d=d[d.dhaka==1]
    out=stats_for(d,i,B,42); out.update(id=i,domain=dom); rows.append(out)
    pd.DataFrame(rows).to_csv(OUT,index=False); print(i,dom,round(out['NS'],2),round(out['NS_lo'],2),round(out['NS_hi'],2),round(time.time()-t0),'s',flush=True)
print('jobs total',len(jobs),'done',len(rows))
