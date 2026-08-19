"""Forward evaluation for context-adjusted pitcher psychological profiles."""
import os,numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive,standardize
from src.context_adjusted_psych import attach_context_adjusted_psych

raw=pd.read_csv('data/train.csv',low_memory=False);parts={y:load_predictions(raw,y) for y in (2022,2023,2024)};r22,y22,g22,b22=parts[2022];r23,y23,g23,b23=parts[2023];r24,y24,g24,b24=parts[2024];gate23=fit_adaptive(g22,y22,b22,1820023);base={2022:b22,2023:np.clip(b23+gate23.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']};rows={2022:r22,2023:r23,2024:r24};ys={2022:y22,2023:y23,2024:y24};scales=(.02,.05,.08,.1,.15,.2,.25,.3,.4,.5)
alpha_values=(float(os.environ['PSYCH_ALPHA']),) if 'PSYCH_ALPHA' in os.environ else (100.,300.,800.,1500.)
for ap in alpha_values:
 X={y:attach_context_adjusted_psych(rows[y],raw.loc[raw.season.lt(y)],400.,ap) for y in rows}
 for ridge in (300,1000,3000,10000,30000,100000):
  z22,z23=standardize(X[2022],X[2023]);m23=Ridge(alpha=ridge,fit_intercept=False).fit(z22,y22-base[2022]);c23=m23.predict(z23);tr=pd.concat([X[2022],X[2023]],ignore_index=True);ztr,z24=standardize(tr,X[2024]);m24=Ridge(alpha=ridge,fit_intercept=False).fit(ztr,np.r_[y22-base[2022],y23-base[2023]],sample_weight=np.r_[np.full(len(y22),.55),np.ones(len(y23))]);c24=m24.predict(z24);s23=np.array([skill(y23,np.clip(base[2023]+s*c23,1e-6,1-1e-6)) for s in scales]);s24=np.array([skill(y24,np.clip(base[2024]+s*c24,1e-6,1-1e-6)) for s in scales]);gain=np.minimum(s23-skill(y23,base[2023]),s24-skill(y24,base[2024]));j=int(gain.argmax())
  if gain[j]>0:print('alpha',ap,'ridge',ridge,'scale',scales[j],'scores',round(s23[j],4),round(s24[j],4),'gains',round(s23[j]-skill(y23,base[2023]),4),round(s24[j]-skill(y24,base[2024]),4),flush=True)
  if ap==100 and ridge==1000:np.savez_compressed('evaluation/context_adjusted_psych.npz',y23=y23,p23=np.clip(base[2023]+scales[j]*c23,1e-6,1-1e-6),base23=base[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=np.clip(base[2024]+scales[j]*c24,1e-6,1-1e-6),base24=base[2024],pitcher24=r24.pitcher_id.to_numpy(),scale=scales[j],alpha=ap,ridge=ridge)
