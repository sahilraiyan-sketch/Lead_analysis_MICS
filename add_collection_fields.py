"""Sensitivity of the Dhaka coverage estimate to coverage definition, outcome threshold, exclusion of the urban-industrial core and completion weighting."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyreadstat, statsmodels.api as sm, patsy
from engine import Svy, make_cohort
f,s=make_cohort(); f=f[f.me!=9].copy(); f['covl10']=f.cov_loo*10; f['y35']=(f.lead>=3.5).astype(int)
COV='age + female + rural + C(wq) + C(me) + soil_ln + soil_miss + C(bat) + C(occ)'
rows=[]
def add(label,m,terms,n):
    r=m.lincom(terms); rows.append(dict(specification=label,n=n,OR=round(r['OR'],4),lo=round(r['lo'],4),hi=round(r['hi'],4)))
m=Svy(f'y ~ (piped + cov10)*dhaka + C(dist) + {COV}',f); add('Principal model (inclusive coverage)',m,['cov10','cov10:dhaka'],len(f))
m=Svy(f'y ~ (piped + covl10)*dhaka + C(dist) + {COV}',f); add('Leave-one-out coverage',m,['covl10','covl10:dhaka'],len(f))
m=Svy(f'y35 ~ (piped + cov10)*dhaka + C(dist) + {COV}',f); add('Threshold 3.5 ug/dL, district effects',m,['cov10','cov10:dhaka'],len(f))
m=Svy(f'y35 ~ (piped + cov10)*dhaka + {COV}',f); add('Threshold 3.5 ug/dL, Dhaka main effect instead of district effects',m,['cov10','cov10:dhaka'],len(f))
core=[20,25,26,33,67]; g=f[~f.dist.isin(core)]; m=Svy(f'y ~ (piped + cov10)*dhaka + C(dist) + {COV}',g); add('Excluding the Dhaka urban-industrial core',m,['cov10','cov10:dhaka'],len(g))
ch=pyreadstat.read_sav(os.path.join(MICS_DIR,'ch.sav'),apply_value_formats=False)[0]
hh=pyreadstat.read_sav(os.path.join(MICS_DIR,'hh.sav'),apply_value_formats=False,usecols=['HH1','HH2','WS1','windex5','HH7A'])[0]
mod=ch[ch.LT2.notna()].copy(); mod['valid']=((mod.C_labresult==1)&(mod.C_Lead>0)&(mod.C_Lead<100)).astype(int)
mod=mod.merge(hh[['HH1','HH2','WS1']].rename(columns={'WS1':'ws1h'}),on=['HH1','HH2'],how='left')
mod['piped']=mod.ws1h.isin([11,12,13,14]).astype(float); cm=hh.assign(p=hh.WS1.isin([11,12,13,14]).astype(float)).groupby('HH1').p.mean(); mod['cov']=mod.HH1.map(cm)
mod['age']=mod.CAGE; mod['agesq']=mod.CAGE**2; mod['female']=(mod.HL4==2).astype(float); mod['rural']=(mod.HH6==2).astype(float)
need=['valid','piped','cov','age','female','rural','windex5','melevel','HH7A']; mm=mod.dropna(subset=need).copy()
X=patsy.dmatrix('C(HH7A) + piped + cov + age + agesq + female + rural + C(windex5) + C(melevel)',mm,return_type='dataframe')
p=sm.GLM(mm.valid.to_numpy(float),X,family=sm.families.Binomial()).fit().predict(X); mm['p']=p.values if hasattr(p,'values') else p
key=mm[['HH1','HH2','LT3','p']].rename(columns={'LT3':'LN'}); tmp=s[['HH1','HH2']].copy()
ch2=ch[(ch.C_labresult==1)&(ch.C_Lead>0)&(ch.C_Lead<100)].reset_index(drop=True); tmp['LN']=ch2.LT3.values
tmp=tmp.merge(key,on=['HH1','HH2','LN'],how='left'); assert tmp.p.notna().all()
f2=make_cohort()[0]; f2['p']=tmp.p.values; f2=f2[f2.me!=9].copy(); f2['w']=f2.w/f2.p
m=Svy(f'y ~ (piped + cov10)*dhaka + C(dist) + {COV}',f2); add('Additional inverse-completion weighting',m,['cov10','cov10:dhaka'],len(f2))
out=pd.DataFrame(rows); out.to_csv(os.path.join(OUT_DIR,'case_sensitivity.csv'),index=False); print(out.to_string(index=False))
