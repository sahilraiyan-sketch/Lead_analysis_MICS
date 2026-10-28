"""Refits the decomposition with phlebotomist, month, hour and laboratory-block fixed effects."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import sys, os, warnings, time; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from engine import Svy
import run_atlas as ra
base=pd.read_csv(os.path.join(OUT_DIR,'atlas_raw.csv')); b=base[(base.domain=='pooled')&(base.outcome=='lin')&(~base.nowq)&(~base.eb)]
ids=ra.PRIMARY+[i for i in b[b.pT<0.05].id if i not in ra.PRIMARY]
SPECS={'phleb':' + C(phleb)','phleb+month+hour':' + C(phleb) + C(month) + C(hourgrp)','labblk':' + C(labblk)','phleb+labblk':' + C(phleb) + C(labblk)'}
OUT=os.path.join(OUT_DIR,'artifact_fe.csv')
rows=pd.read_csv(OUT).to_dict('records') if os.path.exists(OUT) else []
done={(r['id'],r['spec']) for r in rows}; t0=time.time(); BUDGET=float(os.environ.get('TIME_BUDGET','1e12'))
for i in ids:
    d=None
    for sp,ex in SPECS.items():
        if (i,sp) in done: continue
        if time.time()-t0>BUDGET: break
        if d is None: d=ra.prep(i)
        cv=ra.cov(i)+ex
        try:
            mT=Svy(f'loglead ~ Hx + C(dist) + {cv}',d,family='linear'); mW=Svy(f'loglead ~ Hx + Cz + C(dist) + {cv}',d,family='linear')
        except Exception as e: print('fail',i,sp); continue
        t=mT.lincom(['Hx']); w=mW.lincom(['Hx']); g=mW.lincom(['Cz'])
        rows.append(dict(id=i,spec=sp,n=mT.n,bT=t['b'],pT=t['p'],bW=w['b'],seW=w['se'],pW=w['p'],g=g['b'],seg=g['se'],pg=g['p']))
        pd.DataFrame(rows).to_csv(OUT,index=False)
    else: continue
    break
print('done rows',len(rows),'of',len(ids)*len(SPECS),round(time.time()-t0),'s')
