"""Computes household-level indicators for all households and their PSU means."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import sys, ast, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from engine import *
import pandas as pd, numpy as np, pyreadstat
ch=pyreadstat.read_sav(f'{R}/ch.sav',apply_value_formats=False)[0]
hh=pyreadstat.read_sav(f'{R}/hh.sav',apply_value_formats=False)[0]
src=open(os.path.join(HERE,'screen.py')).read(); tree=ast.parse(src)
specs=[]
for n in ast.walk(tree):
    if isinstance(n,ast.Call) and getattr(n.func,'id','')=='add' and len(n.args)==9 and isinstance(n.args[0],ast.Constant):
        specs.append(dict(id=n.args[0].value,expr=ast.get_source_segment(src,n.args[7])))

for i,v,lab in [('A04','HC9B','Refrigerator'),('A05','HC9A','Television'),('A06','HC12A','Smartphone'),('A07','HC13','Internet at home'),('A08','HC19','Bank account'),('A09','HC15','Agricultural land'),('A10','HC17','Livestock'),('A11','HC11','Computer')]:
    specs.append(dict(id=i,expr=f"yn(d['{v}'])"))
meta=pd.read_json(os.path.join(OUT_DIR,'screen.json')).set_index('id')
CHILD=['C01','C02','C03','C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C14','D02','D06','D07','D08','N01','N02','N03','N04','N05','K01','K02','M01','M02']
HELP='''
def yn(s,yes=(1,)):
    return pd.Series(np.where(s.isna()|s.isin([8,9,98,99]),np.nan,s.isin(yes).astype(float)),index=s.index)
def inset(s,yes,bad=(99,98,96,9)):
    return pd.Series(np.where(s.isna()|s.isin(bad),np.nan,s.isin(yes).astype(float)),index=s.index)
def any_yes(cols):
    X=d[cols]; a=(X==1).any(axis=1); allobs=X.notna().all(axis=1)&(X.isin([1,2])).all(axis=1)
    return pd.Series(np.where(a,1.0,np.where(allobs,0.0,np.nan)),index=d.index)
def lt(s,yes=(2,3,4)): return pd.Series(np.where(s.isna()|(s==9),np.nan,s.isin(yes).astype(float)),index=d.index)
def fin(s,lo,hi): return pd.Series(np.where(s.isna()|(s==96),np.nan,((s>=lo)&(s<=hi)).astype(float)),index=d.index)
'''
def evaluate(df,ids):
    ns={'pd':pd,'np':np,'d':df.copy(),'imp':[11,12,13,21,22,24,31,32]}
    exec(HELP,ns)
    ns['d']['_f']=ns['fin'](ns['d'].HC4,31,35) if 'HC4' in ns['d'] else np.nan
    ns['d']['_r']=ns['fin'](ns['d'].HC5,31,36) if 'HC5' in ns['d'] else np.nan
    ns['d']['_w']=ns['fin'](ns['d'].HC6,31,36) if 'HC6' in ns['d'] else np.nan
    out={}
    for sp in specs:
        if sp['id'] not in ids: continue
        try: out[sp['id']]=pd.Series(eval(sp['expr'],ns),index=df.index).astype(float)
        except Exception as e: out[sp['id']]=None; print('fail',sp['id'],str(e)[:80])
    return pd.DataFrame({k:v for k,v in out.items() if v is not None},index=df.index)

HH_IDS=[sp['id'] for sp in specs if sp['id'] not in CHILD and sp['id']!='G01']
hhI=evaluate(hh,HH_IDS)
hhI['HH1']=hh.HH1.values; hhI['HH2']=hh.HH2.values
hhI.to_pickle(os.path.join(OUT_DIR,'hhI.pkl'))
cm=hhI.drop(columns=['HH2']).groupby('HH1').mean(); cm.to_pickle(os.path.join(OUT_DIR,'hh_psu_means.pkl'))
print(hhI.shape,cm.shape)
