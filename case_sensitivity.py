"""Laboratory-block random-intercept models and leave-one-block-out and leave-one-district-out refits."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import sys, os, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from engine import Svy
import run_atlas as ra
PRIM=ra.PRIMARY
t0=time.time(); BUDGET=float(os.environ.get('TIME_BUDGET','1e12'))

OUTA=os.path.join(OUT_DIR,'batch_re.csv')
if not os.path.exists(OUTA):
    rows=[]
    for i in PRIM:
        d=ra.prep(i); form=f'loglead ~ Hx + Cz + C(dist) + {ra.cov(i)}'
        ols=smf.ols(form,d).fit()
        try:
            m=smf.mixedlm(form,d,groups=d['labblk']).fit(reml=True,method='powell',maxiter=500)
            vb=float(m.cov_re.iloc[0,0]); ve=float(m.scale)
            rows.append(dict(id=i,n=len(d),ols_w=ols.params['Hx'],ols_w_se=ols.bse['Hx'],ols_g=ols.params['Cz'],ols_g_se=ols.bse['Cz'],re_w=m.params['Hx'],re_w_se=m.bse['Hx'],re_g=m.params['Cz'],re_g_se=m.bse['Cz'],block_var=vb,resid_var=ve,block_share=vb/(vb+ve),converged=bool(m.converged)))
        except Exception as e: print('fail',i,str(e)[:80])
        print(i,round(time.time()-t0),'s',flush=True)
    pd.DataFrame(rows).to_csv(OUTA,index=False)

TARGETS=[('W01','g'),('E01','g'),('M01','g'),('A03','g'),('H04','g'),('D02','bW'),('C01','bW')]
OUT=os.path.join(OUT_DIR,'batch_loo.csv')
rows=pd.read_csv(OUT).to_dict('records') if os.path.exists(OUT) else []
done={(r['kind'],r['id'],r['unit']) for r in rows}
cache={}
for kind,col in [('block','labblk'),('district','dist')]:
    for i,key in TARGETS:
        if i not in cache: cache[i]=ra.prep(i)
        d=cache[i]
        for u in sorted(d[col].unique()):
            if (kind,i,int(u)) in done: continue
            if time.time()-t0>BUDGET: break
            x=d[d[col]!=u]
            try:
                m=Svy(f'loglead ~ Hx + Cz + C(dist) + {ra.cov(i)}',x,family='linear'); r=m.lincom(['Cz' if key=='g' else 'Hx'])
                rows.append(dict(kind=kind,id=i,unit=int(u),coef=r['b'],n=len(x)))
            except Exception as e: pass
        else: continue
        break
    else: continue
    break
pd.DataFrame(rows).to_csv(OUT,index=False)
print('rows saved',len(rows),round(time.time()-t0),'s')
