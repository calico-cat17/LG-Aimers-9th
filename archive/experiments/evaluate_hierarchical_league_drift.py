"""Forecast next-season league environment and shrink its shift by pitcher history."""
import json
import numpy as np,pandas as pd
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive

raw=pd.read_csv('data/train.csv',low_memory=False);means=raw.groupby('season').control_success.mean();q={y:load_predictions(raw,y) for y in (2022,2023,2024)}
r22,y22,x22,h22=q[2022];r23,y23,x23,h23=q[2023];r24,y24,x24,h24=q[2024];g23=fit_adaptive(x22,y22,h22,2020023);base={2023:np.clip(h23+g23.predict(x23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']};rows={2023:r23,2024:r24};ys={2023:y23,2024:y24}
def forecast(target,k):
 s=means.loc[means.index<target].iloc[-k:];coef=np.polyfit(s.index.to_numpy(float),s.to_numpy(float),1);return float(np.polyval(coef,target)),float(s.iloc[-1])
def adjust(p,r,delta,strength,scale,floor,logit_mode):
 n=pd.to_numeric(r.asof_pitcher_n,errors='coerce').fillna(0).clip(lower=0).to_numpy(float);trust=floor+(1-floor)*strength/(n+strength);shift=scale*delta*trust
 if not logit_mode:return np.clip(p+shift,1e-6,1-1e-6)
 z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)));dlogit=shift/(p*(1-p)).clip(.12,None);return np.clip(1/(1+np.exp(-(z+dlogit))),1e-6,1-1e-6)
records=[]
for k in (3,4,5):
 for strength in (30,60,100,200,400,800):
  for scale in (.1,.2,.35,.5,.75,1):
   for floor in (0,.1,.25,.5,1):
    for lm in (False,True):
     pred={};scores=[]
     for year in (2023,2024):
      f,last=forecast(year,k);pred[year]=adjust(base[year],rows[year],f-last,strength,scale,floor,lm);scores.append(skill(ys[year],pred[year]))
     gains=np.array(scores)-np.array([skill(y23,base[2023]),skill(y24,base[2024])]);records.append((gains.min(),gains.mean(),k,strength,scale,floor,lm,*scores,pred[2023],pred[2024]))
best=max(records,key=lambda z:z[:2]);worst,avg,k,strength,scale,floor,lm,s23,s24,p23,p24=best;f25,last24=forecast(2025,k);print('best',dict(worst_gain=worst,mean_gain=avg,k=k,strength=strength,scale=scale,floor=floor,logit=lm,score23=s23,score24=s24,forecast_2025=f25,last_2024=last24,delta_2025=f25-last24))
np.savez_compressed('evaluation/hierarchical_league_drift.npz',y23=y23,p23=p23,base23=base[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=p24,base24=base[2024],pitcher24=r24.pitcher_id.to_numpy(),forecast_2025=f25,delta_2025=f25-last24,k=k,strength=strength,scale=scale,floor=floor,logit=lm)
json.dump(dict(k=int(k),strength=float(strength),scale=float(scale),floor=float(floor),logit=bool(lm),forecast_2025=f25,delta_2025=f25-last24),open('evaluation/hierarchical_league_drift_config.json','w'),indent=2)
