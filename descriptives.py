"""Repeats the decomposition for household-level indicators in pregnant women and compares it with children."""
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
SWAP={'A03':['C(wq)'],'A01':['C(wq)']}
def cv(i,extra=''):
    b=[x for x in BASE if x not in SWAP.get(i,[])]; return ' + '.join(b)+extra

ok=(f0.me!=9)
ids=[i for i in atlas.index if i in hhI.columns]
def std_params(i):
    m=ok&H[i].notna()&C[i].notna(); c=C.loc[m,i]; h=H.loc[m,i]
    return c.mean(),c.std(),h.mean(),h.std()
CONT={'A03'}
def prep_w(i):
    d=w.copy(); h=wm[['HH1','HH2']].merge(hhI[['HH1','HH2',i]],on=['HH1','HH2'],how='left')[i].values
    d['Hx']=h; d['Cx']=wm.HH1.map(cmn[i]).values
    d=d[d.Hx.notna()&d.Cx.notna()].copy()
    mc,sc,mh,sh=std_params(i)
    d['Cz']=(d.Cx-mc)/sc
    if i in CONT: d['Hx']=(d.Hx-mh)/sh
    return d
def fit(d,i,extra=''):
    try:
        mT=Svy(f'loglead ~ Hx + C(dist) + {cv(i,extra)}',d,family='linear'); mW=Svy(f'loglead ~ Hx + Cz + C(dist) + {cv(i,extra)}',d,family='linear')
    except Exception as e: return None
    t=mT.lincom(['Hx']); b=mW.lincom(['Hx']); g=mW.lincom(['Cz'])
    return dict(n=mT.n,bT=t['b'],seT=t['se'],pT=t['p'],bW=b['b'],seW=b['se'],pW=b['p'],g=g['b'],seg=g['se'],pg=g['p'])
rows=[]
for i in ids:
    d=prep_w(i)
    if len(d)<500: continue
    r=fit(d,i);
    if r: r.update(id=i,domain='pooled',spec='base'); rows.append(r)
    if i in ['W01','S01','H04','E01','A03','D02']:
        r=fit(d[d.dhaka==1],i)
        if r: r.update(id=i,domain='Dhaka',spec='base'); rows.append(r)
        r=fit(d,i,' + C(phleb) + C(month)')
        if r: r.update(id=i,domain='pooled',spec='phleb+month FE'); rows.append(r)
W=pd.DataFrame(rows); W.to_csv(os.path.join(OUT_DIR,'women_results.csv'),index=False)
print(len(W),'women models')

ch_=atlas.loc[ids,['label','bT','bW','g','seg','pg','p_adjG','pat2']].rename(columns={'bT':'bT_ch','bW':'bW_ch','g':'g_ch','seg':'seg_ch','pg':'pg_ch','p_adjG':'padjG_ch','pat2':'pattern_ch'})
wp=W[(W.domain=='pooled')&(W.spec=='base')].set_index('id').rename(columns=lambda c:c+'_w' if c not in ('id',) else c)
cmp=ch_.join(wp,how='inner'); cmp['same_sign_g']=np.sign(cmp.g_ch)==np.sign(cmp.g_w)
cmp.to_csv(os.path.join(OUT_DIR,'women_vs_children.csv'))
print('indicators compared',len(cmp))
print('Spearman gamma child vs women:',spearmanr(cmp.g_ch,cmp.g_w))
print('Pearson gamma:',pearsonr(cmp.g_ch,cmp.g_w))
print('Spearman beta_W:',spearmanr(cmp.bW_ch,cmp.bW_w)); print('Spearman beta_T:',spearmanr(cmp.bT_ch,cmp.bT_w))
sig=cmp[cmp.padjG_ch<0.05]; print('children-significant contextual (BH) n=',len(sig),' same sign in women:',sig.same_sign_g.mean().round(2),' women p<.05:',(sig.pg_w<0.05).mean().round(2))

import statsmodels.api as sm
m=sm.OLS(cmp.g_w,sm.add_constant(cmp.g_ch)).fit(); print('slope women~child gamma',m.params.round(3).to_dict(),'R2',round(m.rsquared,2))
pd.set_option('display.width',250)
P=['W01','S01','H04','E01','A03']
print(cmp.loc[[p for p in P],['g_ch','seg_ch','g_w','seg_w','pg_w','bW_ch','bW_w','pW_w','bT_ch','bT_w']].round(3))
print(W[(W.domain=='Dhaka')][['id','n','g','seg','pg','bW','pW']].round(3))
print(W[(W.spec=='phleb+month FE')][['id','n','g','seg','pg','bW','pW']].round(3))
