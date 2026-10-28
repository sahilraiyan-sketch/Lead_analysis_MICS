"""Collects all estimates into one Excel workbook."""
import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import numpy as np, pandas as pd, json
J=lambda n:os.path.join(OUT_DIR,n); pc=lambda b:100*(np.exp(b)-1)
p=pd.read_csv(J('pooled_final_with_patterns.csv'),index_col=0); raw=pd.read_csv(J('atlas_raw.csv')); st=pd.read_csv(J('status.csv')); fp=pd.read_csv(J('final_pooled.csv'))
cat=pd.read_json(J('screen.json')); boot=pd.read_csv(J('boot_partial.csv')); wv=pd.read_csv(J('women_vs_children.csv'),index_col=0)
sh={}
t=p[['label','n','icc','rel','sdH','bT','bT_lo','bT_hi','pT','bW','bW_lo','bW_hi','pW','p_adjW','g','g_lo','g_hi','pg','p_adjG','pat2']].copy()
t['within_per_SD_H_pct']=pc(p.bW*p.sdH); t['between_pct_per_SD']=pc(p.g); t['total_pct']=pc(p.bT); t['within_pct']=pc(p.bW)
t=t.reset_index().rename(columns={'id':'ID','pat2':'pattern'}).merge(fp[['id','NS','NS_lo','NS_hi','cls']].rename(columns={'id':'ID','cls':'predeclared_class'}),on='ID',how='left'); sh['Pooled_linear_all']=t.round(4)
for dom in ['Dhaka','Elsewhere']:
    d=raw[(raw.domain==dom)&(raw.outcome=='lin')&(~raw.nowq)&(~raw.eb)].merge(st[['id','label']],on='id'); d['NS']=(d.bT-d.bW)/d.bT; sh[f'{dom}_linear']=d.round(4)
for nm,dom in [('Pooled_binary','pooled'),('Dhaka_binary','Dhaka'),('Elsewhere_binary','Elsewhere')]:
    sh[nm]=raw[(raw.domain==dom)&(raw.outcome=='bin')&(~raw.nowq)&(~raw.eb)].merge(st[['id','label']],on='id').round(4)
sh['Sensitivity_EB_noWealth']=raw[(raw.eb)|(raw.nowq)].merge(st[['id','label']],on='id').round(4)
sh['Bootstrap_NS']=boot.round(4); sh['Indicator_catalogue']=cat.round(4)
sh['Women_vs_children']=wv.round(4)
for nm,fn in [('Women_models','women_results.csv'),('Women_leadmodule','women_leadmodule.csv'),('Artifact_FE','artifact_fe.csv'),('Artifact_diagnostics','artifact_diag.csv'),('LOO_phlebotomist','loo_phleb.csv'),('Batch_random_intercept','batch_re.csv'),('LOBO_LODO','batch_loo.csv'),('Censoring_sensitivity','censor_sens.csv'),('Case_study','case_study_results.csv'),('Case_sensitivity','case_sensitivity.csv'),('Division_models','division_models.csv'),('Cohort_accounting','cohort_accounting.csv'),('Cohort_characteristics','cohort_characteristics.csv'),('Completion','completion.csv'),('Prevalence_weighted','prevalence_weighted.csv'),('Case_descriptives','case_descriptives.csv')]:
    if os.path.exists(J(fn)): sh[nm]=pd.read_csv(J(fn)).round(4)
with pd.ExcelWriter(J('Results_workbook.xlsx'),engine='openpyxl') as xw:
    for k,v in sh.items(): v.to_excel(xw,sheet_name=k[:31],index=False)
print('workbook sheets:',list(sh))
