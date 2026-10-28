"""Sensitivity of the primary indicators to the results at the minimum value: exclusion, and unweighted left-censored (Tobit) models."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, patsy
from scipy.optimize import minimize
from scipy.stats import norm
from engine import Svy
import run_atlas as ra
rows=[]
for i in ra.PRIMARY:
    d=ra.prep(i); form=f'loglead ~ Hx + Cz + C(dist) + {ra.cov(i)}'
    floor=d.lead.min(); cens=(d.lead<=floor*1.0001).values; c=np.log(floor)
    m0=Svy(form,d,family='linear'); m1=Svy(form,d[~cens],family='linear')
    y,X=patsy.dmatrices(form,d,return_type='dataframe'); Xm=X.to_numpy(float); yv=y.iloc[:,0].to_numpy(float)
    b0=np.linalg.lstsq(Xm,yv,rcond=None)[0]; s0=np.log(np.std(yv-Xm@b0))
    def nll(th):
        mu=Xm@th[:-1]; s=np.exp(th[-1]); z=(yv-mu)/s
        return -np.where(cens,norm.logcdf((c-mu)/s),norm.logpdf(z)-np.log(s)).sum()
    def grad(th):
        mu=Xm@th[:-1]; s=np.exp(th[-1]); z=(yv-mu)/s; lam=np.exp(norm.logpdf((c-mu)/s)-norm.logcdf((c-mu)/s))
        gmu=np.where(cens,-lam/s,z/s); gs=np.where(cens,-lam*(c-mu)/s,(z**2-1)); return -np.concatenate([Xm.T@gmu,[gs.sum()]])
    r=minimize(nll,np.concatenate([b0,[s0]]),jac=grad,method='L-BFGS-B',options={'maxiter':2000})
    cols=list(X.columns); ih,ic=cols.index('Hx'),cols.index('Cz')
    rows.append(dict(id=i,n_floor=int(cens.sum()),w_main=m0.lincom(['Hx'])['b'],g_main=m0.lincom(['Cz'])['b'],w_excl=m1.lincom(['Hx'])['b'],g_excl=m1.lincom(['Cz'])['b'],w_ols=b0[ih],g_ols=b0[ic],w_tobit=r.x[ih],g_tobit=r.x[ic],conv=r.success))
R=pd.DataFrame(rows); R.to_csv(os.path.join(OUT_DIR,'censor_sens.csv'),index=False); print(R.round(4).to_string(index=False))
