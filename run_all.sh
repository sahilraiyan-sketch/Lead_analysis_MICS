"""Piped-water case study: Models 1 to 4 (survey-weighted logistic) and within-PSU conditional logistic models."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, patsy
from engine import Svy, make_cohort
from statsmodels.discrete.conditional_models import ConditionalLogit
f,s=make_cohort(); f=f[f.me!=9].copy()
COV='age + female + rural + C(wq) + C(me) + soil_ln + soil_miss + C(bat) + C(occ)'
rows=[]
def add(label,r): rows.append(dict(contrast=label,OR=round(r['OR'],4),lo=round(r['lo'],4),hi=round(r['hi'],4)))
m=Svy(f'y ~ piped*dhaka + {COV}',f)
add('M1 household, Dhaka',m.lincom(['piped','piped:dhaka'])); add('M1 household, elsewhere',m.lincom(['piped'])); add('M1 household x Dhaka interaction',m.lincom(['piped:dhaka']))
m=Svy(f'y ~ (piped + cov10)*dhaka + {COV}',f)
add('M2 coverage per 10 pp, Dhaka',m.lincom(['cov10','cov10:dhaka'])); add('M2 coverage per 10 pp, elsewhere',m.lincom(['cov10']))
add('M2 household, Dhaka',m.lincom(['piped','piped:dhaka'])); add('M2 household, elsewhere',m.lincom(['piped'])); add('M2 coverage interaction',m.lincom(['cov10:dhaka']))
m=Svy(f'y ~ (piped + cov10)*dhaka + C(dist) + {COV}',f)
add('M3 coverage per 10 pp, Dhaka',m.lincom(['cov10','cov10:dhaka'])); add('M3 coverage per 10 pp, elsewhere',m.lincom(['cov10']))
add('M3 household, Dhaka',m.lincom(['piped','piped:dhaka'])); add('M3 household, elsewhere',m.lincom(['piped'])); add('M3 coverage interaction',m.lincom(['cov10:dhaka']))
m=Svy(f'y ~ (piped + cov10 + psu_wq)*dhaka + fin_wall + fin_roof + fin_floor + C(dist) + {COV}',f)
add('M4 coverage per 10 pp, Dhaka',m.lincom(['cov10','cov10:dhaka'])); add('M4 coverage per 10 pp, elsewhere',m.lincom(['cov10']))
add('M4 household, Dhaka',m.lincom(['piped','piped:dhaka'])); add('M4 household, elsewhere',m.lincom(['piped'])); add('M4 coverage interaction',m.lincom(['cov10:dhaka']))
d=f[f.dhaka==1].copy(); d['cb']=pd.cut(d['cov'],[-.01,0,.25,.75,1.0],labels=['0','0-25','25-75','75-100'])
m=Svy(f'y ~ piped + C(cb) + C(dist) + {COV}',d)
for k in m.cols:
    if k.startswith('C(cb)[T.'): add('Dhaka coverage category '+k[8:-1]+' vs 0',m.lincom([k]))
inf=f.groupby('PSU').y.nunique(); g=f[f.PSU.isin(inf[inf>1].index)].copy()
for c in ['bat','occ']: g[c]=g[c].astype(int)
def fit(extra):
    X=patsy.dmatrix('0 + piped'+extra+' + age + female + C(wq) + C(me) + C(bat) + C(occ)',g,return_type='dataframe')
    D=X.groupby(g.PSU.to_numpy()).transform(lambda c:c-c.mean()); cols=[]; M=np.zeros((len(g),0))
    for c in X.columns:
        if D[c].abs().sum()<=1e-9: continue
        M2=np.column_stack([M,D[c].to_numpy()])
        if np.linalg.matrix_rank(M2,tol=1e-8)>M.shape[1]: M=M2; cols.append(c)
    return ConditionalLogit(g.y.to_numpy(),X[cols].to_numpy(float),groups=g.PSU.to_numpy()).fit(disp=0,method='bfgs',maxiter=3000),cols
r,cols=fit(''); i=cols.index('piped'); b,se=r.params[i],r.bse[i]
rows.append(dict(contrast='Conditional logit, household within PSU, pooled',OR=round(np.exp(b),4),lo=round(np.exp(b-1.96*se),4),hi=round(np.exp(b+1.96*se),4)))
r,cols=fit(' + piped:dhaka'); C=r.cov_params(); ip,idh=cols.index('piped'),cols.index('piped:dhaka')
for lab,idx in [('elsewhere',[ip]),('Dhaka',[ip,idh])]:
    v=np.zeros(len(cols)); v[idx]=1; b=v@r.params; se=np.sqrt(v@C@v)
    rows.append(dict(contrast='Conditional logit, household within PSU, '+lab,OR=round(np.exp(b),4),lo=round(np.exp(b-1.96*se),4),hi=round(np.exp(b+1.96*se),4)))
out=pd.DataFrame(rows); out.to_csv(os.path.join(OUT_DIR,'case_study_results.csv'),index=False); print(out.to_string(index=False))
