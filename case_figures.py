"""Total associations of lead-specific items with blood lead in pregnant women."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, pyreadstat
from engine import Svy, cat3
from scipy.stats import spearmanr, pearsonr
R=MICS_DIR
hhI=pd.read_pickle(os.path.join(OUT_DIR,'hhI.pkl')); cmn=pd.read_pickle(os.path.join(OUT_DIR,'hh_psu_means.pkl'))
f0=pd.read_pickle(os.path.join(OUT_DIR,'cohort.pkl')); H=pd.read_pickle(os.path.join(OUT_DIR,'H.pkl')); C=pd.read_pickle(os.path.join(OUT_DIR,'C.pkl'))
st=pd.read_csv(os.path.join(OUT_DIR,'status.csv')); atlas=pd.read_csv(os.path.join(OUT_DIR,'pooled_final_with_patterns.csv'),index_col=0)
cols=['HH1','HH2','LN','PSU','stratum','W_Lead','W_labresult','wmblweight','WAGE','welevel','windex5','HH6','HH7','HH7A','ATW6','ATW7M','ATW7D','ATW8H','W_LAB_ID','W_Hb','ATW10','ATW12','ATW13A','ATW13B','ATW13C','ATW13D','ATW13E','ATW13F','ATW14A','ATW14B','ATW14C','ATW14D','ATW14E','ATW14F','ATW16']
wm=pyreadstat.read_sav(f'{R}/wm.sav',apply_value_formats=False,usecols=cols)[0]
wm=wm[(wm.W_labresult==1)&(wm.W_Lead>0)&(wm.W_Lead<900)].reset_index(drop=True)
print('women',len(wm),'PSUs',wm.PSU.nunique(),'PSU==HH1:',(wm.PSU==wm.HH1).all())
w=pd.DataFrame(index=wm.index)
w['loglead']=np.log(wm.W_Lead); w['lead']=wm.W_Lead; w['y']=(wm.W_Lead>=5).astype(int); w['w']=wm.wmblweight; w['PSU']=wm.PSU; w['stratum']=wm.stratum
w['dist']=wm.HH7A.astype(int); w['dhaka']=(wm.HH7==30).astype(int); w['rural']=(wm.HH6==2).astype(float)
w['agegrp']=wm.WAGE.astype(int); w['wq']=wm.windex5.astype(int); w['we']=wm.welevel.astype(int)
w['bat']=cat3(wm,['ATW13A','ATW13B','ATW13C']); w['occ']=cat3(wm,['ATW14A','ATW14B','ATW14C','ATW14D','ATW14E','ATW14F'])
w['phleb']=wm.ATW6.astype(int); w['month']=wm.ATW7M.astype(int)
w['HH1']=wm.HH1; w['HH2']=wm.HH2
BASE=['C(agegrp)','rural','C(wq)','C(we)','C(bat)','C(occ)']

def lt(s,yes=(2,3,4)): return np.where(s.isna()|(s==9),np.nan,s.isin(yes).astype(float))
def anyy(cols):
    X=wm[cols]; a=(X==1).any(axis=1); neg=(X==2).all(axis=1); return np.where(a,1.0,np.where(neg,0.0,np.nan))
items={'Pica (ATW16)':lt(wm.ATW16),'Any lead-related activity nearby (ATW13A-F)':anyy([f'ATW13{k}' for k in 'ABCDEF']),'Battery activity nearby (ATW13A-C)':anyy([f'ATW13{k}' for k in 'ABC']),'Lead-related work (ATW14A-F)':anyy([f'ATW14{k}' for k in 'ABCDEF']),'Smoking in last 3 months (ATW10)':lt(wm.ATW10),'Surma use (ATW12)':lt(wm.ATW12)}
res=[]
for k,x in items.items():
    d=w.copy(); d['Hx']=x; d=d[d.Hx.notna()].copy()
    cvv=' + '.join(b for b in BASE if not (('battery' in k.lower() or 'nearby' in k.lower()) and b=='C(bat)') and not ('work' in k.lower() and b=='C(occ)'))
    m=Svy(f'loglead ~ Hx + C(dist) + {cvv}',d,family='linear'); t=m.lincom(['Hx'])
    res.append(dict(item=k,n=m.n,prev=float(np.average(d.Hx,weights=d.w)),b=t['b'],lo=t['b']-1.96*t['se'],hi=t['b']+1.96*t['se'],p=t['p'],GMR=np.exp(t['b'])))
R_=pd.DataFrame(res); R_.to_csv(os.path.join(OUT_DIR,'women_leadmodule.csv'),index=False); print(R_.round(3).to_string(index=False))
