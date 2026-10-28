"""Cohort accounting, participant characteristics, completion of testing and descriptive numbers for the piped-water case study."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyreadstat
from engine import Svy, make_cohort
f,s=make_cohort()
ch=pyreadstat.read_sav(os.path.join(MICS_DIR,'ch.sav'),apply_value_formats=False)[0]
DIV={10:'Barishal',20:'Chattogram',30:'Dhaka',40:'Khulna',45:'Mymensingh',50:'Rajshahi',55:'Rangpur',60:'Sylhet'}
mod=ch.LT2.notna(); coll=ch.C_labresult.notna(); valid_code=(ch.C_labresult==1)
acc=[('Child records in the file',len(ch)),('Child records with the blood-collection module',int(mod.sum())),('Records with a laboratory result code',int(coll.sum())),
 ('Invalid laboratory results (code 2)',int((ch.C_labresult==2).sum())),('Valid-coded results with a rejected value (999)',int((valid_code&(ch.C_Lead>=100)).sum())),
 ('Valid blood-lead results',len(f)),('Unknown maternal education',int((f.me==9).sum())),('Principal adjusted cohort',int((f.me!=9).sum())),('Contributing PSUs in adjusted cohort',int(f[f.me!=9].PSU.nunique())),
 ('Soil lead missing',int(f[f.me!=9].soil_miss.sum())),('Nearby battery status unknown',int((f[f.me!=9].bat==2).sum())),('Occupational contact unknown',int((f[f.me!=9].occ==2).sum()))]
pd.DataFrame(acc,columns=['item','value']).to_csv(os.path.join(OUT_DIR,'cohort_accounting.csv'),index=False)
g=f[f.me!=9]; n=len(g); pct=lambda m:f"{int(m.sum()):,} ({100*m.mean():.1f})"
rows=[('Age, months: mean (SD)',f"{g.age.mean():.1f} ({g.age.std():.1f})"),('Female',pct(g.female==1)),('Rural residence',pct(g.rural==1))]
for k,v in DIV.items(): rows.append((f'Division: {v}',pct(g['div']==k)))
for q in range(1,6): rows.append((f'Wealth quintile {q}',pct(g.wq==q)))
for k,v in {0:'Pre-primary or none',1:'Primary',2:'Lower secondary',3:'Secondary',4:'Higher secondary',5:'Higher'}.items(): rows.append((f'Maternal education: {v}',pct(g.me==k)))
rows+= [('Soil lead measured',pct(g.soil_miss==0)),('Blood lead, geometric mean (unweighted)',f"{np.exp(np.log(g.lead).mean()):.2f}"),('Blood lead >=5 ug/dL, unweighted',pct(g.y==1)),('Blood lead >=5 ug/dL, weighted prevalence',f"{100*np.average(g.y,weights=g.w):.1f}%"),('Children per PSU: median (range)',f"{int(g.groupby('PSU').size().median())} ({int(g.groupby('PSU').size().min())}-{int(g.groupby('PSU').size().max())})")]
pd.DataFrame(rows,columns=['characteristic','value']).to_csv(os.path.join(OUT_DIR,'cohort_characteristics.csv'),index=False)
mr=ch[mod].copy(); mr['valid']=((mr.C_labresult==1)&(mr.C_Lead>0)&(mr.C_Lead<100)).astype(int); out=[]
for k,v in DIV.items():
    d=mr[mr.HH7==k]; out.append((v,len(d),int(d.valid.sum()),round(100*d.valid.mean(),1)))
for q in range(1,6):
    d=mr[mr.windex5==q]; out.append((f'Wealth quintile {q}',len(d),int(d.valid.sum()),round(100*d.valid.mean(),1)))
pd.DataFrame(out,columns=['group','module_records','valid_results','valid_pct']).to_csv(os.path.join(OUT_DIR,'completion.csv'),index=False)
def wmean(d,col):
    m=Svy(f'{col} ~ 1',d,family='linear'); b=m.beta.iloc[0]; se=float(np.sqrt(m.cov.iloc[0,0])); return 100*b,100*(b-m.crit*se),100*(b+m.crit*se)
f['y35']=(f.lead>=3.5).astype(int); desc=[]
for lab,d,col in [('Weighted prevalence >=5, all',f,'y'),('Weighted prevalence >=3.5, all',f,'y35')]+[(f'Weighted prevalence >=5, {v}',f[f['div']==k],'y') for k,v in DIV.items()]:
    desc.append((lab,)+tuple(round(x,1) for x in wmean(d,col)))
pd.DataFrame(desc,columns=['quantity','estimate','lo','hi']).to_csv(os.path.join(OUT_DIR,'prevalence_weighted.csv'),index=False)
rows=[('Piped-water households',int(f.piped.sum()),round(100*f.piped.mean(),1))]
for dh,lab in [(1,'Dhaka'),(0,'Elsewhere')]:
    for pp,pl in [(1,'piped'),(0,'non-piped')]:
        d=f[(f.dhaka==dh)&(f.piped==pp)]; rows.append((f'{lab}, {pl}: n and unweighted % >=5',len(d),round(100*d.y.mean(),1)))
d=f[(f.dhaka==1)&(f.piped==0)]
bins=[-.01,0,.2499,.4999,.7499,.9999,1.0]; labs=['0','1-24','25-49','50-74','75-99','100']
cb=pd.cut(d.cov_loo.clip(lower=0),bins,labels=labs)
for k,v in d.groupby(cb): rows.append((f'Dhaka non-piped, other-household coverage {k}%: n and unweighted % >=5',len(v),round(100*v.y.mean(),1)))
d2=f[(f.dhaka==1)&(f.piped==1)]; cb2=pd.cut(d2.cov_loo.clip(lower=0),bins,labels=labs)
for k,v in d2.groupby(cb2):
    if len(v): rows.append((f'Dhaka piped, other-household coverage {k}%: n and unweighted % >=5',len(v),round(100*v.y.mean(),1)))
pd.DataFrame(rows,columns=['quantity','n_or_count','percent']).to_csv(os.path.join(OUT_DIR,'case_descriptives.csv'),index=False)
for fn in ['cohort_accounting','completion','prevalence_weighted','case_descriptives']: print(pd.read_csv(os.path.join(OUT_DIR,fn+'.csv')).to_string(index=False)); print()
