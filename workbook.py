"""Residual intraclass correlations by collection factors, within-PSU laboratory-block tests, drift, and leave-one-phlebotomist-out refits."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, patsy
from scipy.stats import spearmanr
from engine import Svy
import run_atlas as ra
d=ra.prep('W01')
form='loglead ~ C(dist) + age + female + rural + C(wq) + C(me) + soil_ln + soil_miss + C(bat) + C(occ)'
m=Svy(form,d,family='linear'); Y,X=patsy.dmatrices(form,d,return_type='dataframe'); d['res']=(Y.iloc[:,0]-X@m.beta.values).values
def icc(x,g):
    df=pd.DataFrame({'x':x,'g':g}); gs=df.groupby('g').x.agg(['count','mean']); N=len(df); k=len(gs)
    ssb=(gs['count']*(gs['mean']-df.x.mean())**2).sum(); ssw=((df.x-df.groupby('g').x.transform('mean'))**2).sum()
    msb=ssb/(k-1); msw=ssw/(N-k); n0=(N-(gs['count']**2).sum()/N)/(k-1); return (msb-msw)/(msb+(n0-1)*msw)
rows=[]
for nm in ['PSU','phleb','month','labblk','hourgrp','day']: rows.append(('Residual ICC by '+nm,round(icc(d.res,d[nm]),4)))
n2=d.groupby('PSU').res.transform('count'); dd=d[n2>=2].copy(); dd['rw']=dd.res-dd.groupby('PSU').res.transform('mean')
for nm in ['labblk','phleb','month']: rows.append(('Within-PSU residual ICC by '+nm,round(icc(dd.rw,dd[nm]),4)))
rows.append(('Children in PSUs with at least two children',len(dd)))
for nm,col in [('lab sample number','labno'),('hour','hour'),('month','month')]: rows.append(('Spearman, residual vs '+nm,round(float(spearmanr(d.res,d[col])[0]),3)))
ph=d.groupby('phleb').agg(nd=('dist','nunique'),nv=('div','nunique'))
rows.append(('Phlebotomists',len(ph))); rows.append(('Phlebotomists in 2 or more districts',int((ph.nd>=2).sum()))); rows.append(('Phlebotomists in 2 or more divisions',int((ph.nv>=2).sum())))
rows.append(('Laboratory blocks',int(d.labblk.nunique()))); rows.append(('PSUs with children in 2 or more blocks',int((d.groupby('PSU').labblk.nunique()>=2).sum())))
pd.DataFrame(rows,columns=['diagnostic','value']).to_csv(os.path.join(OUT_DIR,'artifact_diag.csv'),index=False)
targets=[('W01','g'),('E01','g'),('M01','g'),('A03','g'),('H04','g'),('D02','bW'),('C01','bW')]; out=[]
for i,key in targets:
    di=ra.prep(i); term='Cz' if key=='g' else 'Hx'; fm=f'loglead ~ Hx + Cz + C(dist) + {ra.cov(i)}'; vals=[]
    for p in sorted(di.phleb.unique()):
        try: vals.append(Svy(fm,di[di.phleb!=p],family='linear').lincom([term])['b'])
        except Exception: pass
    full=Svy(fm,di,family='linear').lincom([term])['b']; out.append(dict(id=i,coef=key,full=full,loo_min=min(vals),loo_max=max(vals),n_loo=len(vals)))
pd.DataFrame(out).to_csv(os.path.join(OUT_DIR,'loo_phleb.csv'),index=False); print(pd.DataFrame(rows).to_string(index=False))
