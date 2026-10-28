"""Division-specific district-adjusted models of piped-water use and PSU coverage."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from engine import Svy, make_cohort
f,s=make_cohort(); f=f[f.me!=9].copy()
COV='age + female + rural + C(wq) + C(me) + soil_ln + soil_miss + C(bat) + C(occ)'
DIV={10:'Barishal',20:'Chattogram',30:'Dhaka',40:'Khulna',45:'Mymensingh',50:'Rajshahi',55:'Rangpur',60:'Sylhet'}
rows=[]
for k,v in DIV.items():
    d=f[f['div']==k]
    try:
        m=Svy(f'y ~ piped + cov10 + C(dist) + {COV}',d); a=m.lincom(['cov10']); b=m.lincom(['piped'])
        rows.append(dict(division=v,n=m.n,piped_children=int(d.piped.sum()),coverage_OR=a['OR'],coverage_lo=a['lo'],coverage_hi=a['hi'],household_OR=b['OR'],household_lo=b['lo'],household_hi=b['hi'],coverage_se=a['se'],household_se=b['se']))
    except Exception as e:
        rows.append(dict(division=v,n=len(d),piped_children=int(d.piped.sum()),note=str(e)[:60]))
out=pd.DataFrame(rows); out.to_csv(os.path.join(OUT_DIR,'division_models.csv'),index=False)
pd.set_option('display.width',250); print(out.round(3).to_string(index=False))
