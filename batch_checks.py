"""Survey-weighted estimation with Taylor-linearized variance over the full PSU frame, and construction of the analytic child cohort."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, patsy, statsmodels.api as sm, pyreadstat
from scipy.stats import t as tdist
R=MICS_DIR
def load_frame():
    hh=pyreadstat.read_sav(f'{R}/hh.sav',usecols=['PSU','stratum'])[0]
    fr=hh.drop_duplicates('PSU')[['PSU','stratum']]
    return fr
FRAME=load_frame()
class Svy:
    """Weighted GLM (binomial/gaussian) with Taylor-linearized variance retaining all frame PSUs (zero contributions outside the domain)."""
    def __init__(self,formula,df,w='w',family='logit',frame=FRAME,psu='PSU',strat='stratum'):
        Y,X=patsy.dmatrices(formula,df,return_type='dataframe'); self.cols=list(X.columns)
        y=Y.iloc[:,0].to_numpy(float); Xm=X.to_numpy(float); ww=df.loc[Y.index,w].to_numpy(float)
        if family=='logit':
            fit=sm.GLM(y,Xm,family=sm.families.Binomial(),freq_weights=ww).fit(); mu=fit.fittedvalues
            bread=np.linalg.pinv(Xm.T@((ww*mu*(1-mu))[:,None]*Xm)); sc=(ww*(y-mu))[:,None]*Xm; beta=fit.params
        else:
            XtWX=Xm.T@(ww[:,None]*Xm); bread=np.linalg.pinv(XtWX); beta=bread@(Xm.T@(ww*y)); mu=Xm@beta; sc=(ww*(y-mu))[:,None]*Xm
        S=pd.DataFrame(sc); S['p']=df.loc[Y.index,psu].to_numpy()
        tot=S.groupby('p').sum()
        T=frame.merge(tot,left_on=psu,right_index=True,how='left').fillna(0.0)
        meat=np.zeros((Xm.shape[1],)*2)
        for _,g in T.groupby(strat,sort=False):
            A=g.iloc[:,2:].to_numpy(float); m=A.shape[0]
            if m<2: continue
            C=A-A.mean(0,keepdims=True); meat+=(m/(m-1.0))*(C.T@C)
        self.beta=pd.Series(np.asarray(beta),index=self.cols); self.cov=pd.DataFrame(bread@meat@bread,index=self.cols,columns=self.cols)
        self.ddf=len(T)-T[strat].nunique(); self.crit=tdist.ppf(.975,self.ddf); self.n=len(y)
    def lincom(self,terms):
        v=pd.Series(0.0,index=self.cols)
        for k,c in (terms.items() if isinstance(terms,dict) else [(t,1.0) for t in terms]): v[k]=c
        b=float(v@self.beta); se=float(np.sqrt(max(v@self.cov@v,0.0)))
        p=2*(1-tdist.cdf(abs(b/se),self.ddf)) if se>0 else float('nan')
        return dict(b=b,se=se,OR=np.exp(b),lo=np.exp(b-self.crit*se),hi=np.exp(b+self.crit*se),p=p)
def cat3(df,cols):
    X=df[cols]; a=(X==1).any(axis=1); neg=(X==2).all(axis=1)
    return np.where(a,1,np.where(neg,0,2))
def make_cohort():
    ch=pyreadstat.read_sav(f'{R}/ch.sav')[0]; hh=pyreadstat.read_sav(f'{R}/hh.sav')[0]
    hh['pipedhh']=hh.WS1.isin([11,12,13,14]).astype(float).where(hh.WS1.notna())
    g=hh.groupby('HH1'); hh['cov_all']=g.pipedhh.transform('sum')/g.pipedhh.transform('count'); hh['psu_wq_all']=g.windex5.transform('sum')/g.windex5.transform('count')
    hh['psu_loo']=(g.pipedhh.transform('sum')-hh.pipedhh)/(g.pipedhh.transform('count')-1)
    s=ch[(ch.C_labresult==1)&(ch.C_Lead>0)&(ch.C_Lead<100)].merge(hh[['HH1','HH2','WS1','cov_all','psu_loo','psu_wq_all','HC4','HC5','HC6','HC8']],on=['HH1','HH2'],how='left',suffixes=('','_hh')).reset_index(drop=True)
    f=pd.DataFrame(index=s.index)
    f['y']=(s.C_Lead>=5).astype(int); f['lead']=s.C_Lead; f['loglead']=np.log(s.C_Lead)
    f['age']=s.CAGE; f['female']=(s.HL4==2).astype(float); f['rural']=(s.HH6==2).astype(float); f['wq']=s.windex5.astype(int); f['me']=s.melevel.astype(int)
    f['soil_ln']=np.log(s.S_Lead.clip(lower=0.01)); f['soil_miss']=s.S_Lead.isna().astype(float); f.loc[f.soil_miss==1,'soil_ln']=np.log(s.S_Lead.dropna().median())
    f['bat']=cat3(s,['LT17A','LT17B','LT17C']); f['occ']=cat3(s,['LT18A','LT18B','LT18C','LT18D','LT18E','LT18F'])
    f['PSU']=s.PSU; f['stratum']=s.stratum; f['w']=s.chblweight; f['ws1']=s.WS1; f['piped']=s.WS1.isin([11,12,13,14]).astype(int)
    f['div']=s.HH7.astype(int); f['dhaka']=(s.HH7==30).astype(int); f['dist']=s.HH7A.astype(int)
    f['cov']=s.cov_all; f['cov10']=f['cov']*10; f['psu_wq']=s.psu_wq_all; f['cov_loo']=s.psu_loo
    f['fin_wall']=s.HC6.between(31,36).astype(float); f['fin_roof']=s.HC5.between(31,36).astype(float); f['fin_floor']=s.HC4.between(31,35).astype(float)
    f['HH1']=s.HH1; f['HH2']=s.HH2
    return f,s
