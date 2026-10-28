"""Figure 1 (prevalence by PSU piped-water coverage and household use), Figure S1 (cohort flow) and Figure S2 (model sequence)."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from engine import Svy, make_cohort
FD=os.path.join(OUT_DIR,'figures'); os.makedirs(FD,exist_ok=True)
INK='#223142'; plt.rcParams.update({'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
f,s=make_cohort()
labs=['0','>0-25','>25-50','>50-75','>75-100']; f['cb']=pd.cut(f['cov'],[-.01,0,.25,.5,.75,1.0],labels=labs)
fig,axs=plt.subplots(1,2,figsize=(11,4.9),sharey=True)
for ax,(dh,title) in zip(axs,[(1,'A  Dhaka division'),(0,'B  Elsewhere')]):
    for pp,col,mk,ls,lab,off in [(0,'#2563a8','o','-','Non-piped household',-0.07),(1,'#d97b1f','s','--','Piped household',0.07)]:
        xs,ys,lo,hi=[],[],[],[]
        for i,cb in enumerate(labs):
            d=f[(f.dhaka==dh)&(f.cb==cb)&(f.piped==pp)]
            if len(d)>=15:
                m=Svy('y ~ 1',d,family='linear'); b=m.beta.iloc[0]; se=float(np.sqrt(m.cov.iloc[0,0]))
                xs.append(i+off); ys.append(100*b); lo.append(100*(b-m.crit*se)); hi.append(min(100*(b+m.crit*se),100))
        ys,lo,hi=np.array(ys),np.array(lo),np.array(hi)
        ax.errorbar(xs,ys,yerr=[ys-lo,hi-ys],color=col,marker=mk,ls=ls,capsize=3,lw=2,ms=7,label=lab)
    ax.set_xticks(range(5)); ax.set_xticklabels(labs); ax.set_title(title,fontweight='bold',loc='center'); ax.set_ylim(0,100); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.25)
    ax.set_xlabel('PSU piped-water coverage (%)')
axs[0].set_ylabel('Prevalence of blood lead >=5 ug/dL (%)'); axs[0].legend(frameon=False,loc='upper left',fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FD,'Figure1_piped_water_prevalence.png'),dpi=250); plt.close()
fig,ax=plt.subplots(figsize=(7,6.4)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,10)
def box(x,y,w,h,t,fc='#e7f0fa'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.15',fc=fc,ec='#2563a8',lw=1.4)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
ch_n=17132
box(1.2,8.2,4.2,1.3,'17,132 child records\nwith the blood-collection module'); box(1.2,5.7,4.2,1.3,'10,661 records with a\nlaboratory result code'); box(1.2,3.2,4.2,1.3,'10,622 valid blood-lead results\n(descriptive cohort)'); box(1.2,0.7,4.2,1.3,'10,619 adjusted observations\n2,984 contributing PSUs',fc='#dff3e6')
for y0,y1 in [(8.2,7.0),(5.7,4.5),(3.2,2.0)]: ax.annotate('',xy=(3.3,y1),xytext=(3.3,y0),arrowprops=dict(arrowstyle='->',color=INK,lw=1.4))
box(6.2,6.9,3.6,0.9,'6,471 without a\nlaboratory result',fc='#fbeee0'); box(6.2,4.3,3.6,0.9,'27 invalid results\n12 rejected values',fc='#fbeee0'); box(6.2,1.8,3.6,0.9,'3 unknown\nmaternal education',fc='#fbeee0')
for yy in (7.35,4.75,2.25): ax.annotate('',xy=(6.2,yy),xytext=(5.4,yy),arrowprops=dict(arrowstyle='->',color='#b35a00',lw=1.2))
plt.savefig(os.path.join(FD,'FigureS1_cohort_flow.png'),dpi=250,bbox_inches='tight'); plt.close()
g=f[f.me!=9].copy(); COV='age + female + rural + C(wq) + C(me) + soil_ln + soil_miss + C(bat) + C(occ)'
specs=[('Household + coverage',f'y ~ (piped + cov10)*dhaka + {COV}'),('+ District domains',f'y ~ (piped + cov10)*dhaka + C(dist) + {COV}'),('+ Contextual wealth\nand housing',f'y ~ (piped + cov10 + psu_wq)*dhaka + fin_wall + fin_roof + fin_floor + C(dist) + {COV}')]
res=[]
for lab,form in specs:
    m=Svy(form,g); res.append((lab,m.lincom(['cov10','cov10:dhaka']),m.lincom(['piped','piped:dhaka'])))
fig,axs=plt.subplots(1,2,figsize=(11,5),sharey=True); y=np.arange(3)[::-1]
for ax,(idx,ttl,col,xl) in zip(axs,[(1,'A  PSU coverage\nPer 10 percentage points','#00897b',(0.97,1.43)),(2,'B  Household use\nPiped versus non-piped','#7e57a8',(0.35,1.1))]):
    for yy,r in zip(y,res):
        e=r[idx]; ax.errorbar(e['OR'],yy,xerr=[[e['OR']-e['lo']],[e['hi']-e['OR']]],color=col,marker='o',capsize=4,lw=2,ms=8)
        ax.text(e['OR'],yy+0.2,f"{e['OR']:.2f} ({e['lo']:.2f}-{e['hi']:.2f})",ha='center',fontsize=9,color=col)
    ax.axvline(1,color='#6b7a8d',ls='--',lw=1.2); ax.set_xlim(*xl); ax.set_title(ttl,fontweight='bold'); ax.set_xlabel('Adjusted odds ratio (95% CI)'); ax.spines[['top','right']].set_visible(False); ax.grid(axis='x',alpha=.25)
axs[0].set_yticks(y); axs[0].set_yticklabels([r[0] for r in res]); axs[0].set_ylim(-0.6,2.7)
plt.tight_layout(); plt.savefig(os.path.join(FD,'FigureS2_model_sequence.png'),dpi=250); plt.close(); print('case figures written')
