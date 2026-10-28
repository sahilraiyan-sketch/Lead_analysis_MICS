"""Add phlebotomist, month, hour and laboratory-sample-block fields to the cohort file."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from config import *
import pandas as pd, numpy as np, pyreadstat
ch=pyreadstat.read_sav(os.path.join(MICS_DIR,'ch.sav'),apply_value_formats=False,usecols=['C_labresult','C_Lead','LT8','LT9D','LT9M','LT10H','C_LAB_ID','CorrNo'])[0]
ch=ch[(ch.C_labresult==1)&(ch.C_Lead>0)&(ch.C_Lead<100)].reset_index(drop=True)
f=pd.read_pickle(os.path.join(OUT_DIR,'cohort.pkl')); assert len(f)==len(ch) and np.allclose(f.lead.values,ch.C_Lead.values)
f['phleb']=ch.LT8.astype(int).values; f['month']=ch.LT9M.astype(int).values; f['day']=ch.LT9D.astype(int).values; f['hour']=ch.LT10H.astype(int).values
f['labno']=ch.C_LAB_ID.str.split('/').str[0].astype(int).values; f['corr']=ch.CorrNo.values
f['labblk']=(f.labno//100).astype(int); f['hourgrp']=pd.cut(f.hour,[0,11,13,15,17,19,24],labels=False)
f.to_pickle(os.path.join(OUT_DIR,'cohort.pkl')); print('cohort augmented:',len(f),'children;',f.phleb.nunique(),'phlebotomists;',f.labblk.nunique(),'lab blocks')
