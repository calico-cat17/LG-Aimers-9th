"""Fit final 2025 psychological/latent-pitch residual assets from temporal OOF."""
import pickle
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from evaluate_crossfit_psych_profiles import attach_profile,build_profile,load_predictions
from evaluate_psych_residual_on_adaptive import fit_adaptive
from evaluate_novel_residual_channels import latent_pitch_channel

OUT=Path('model_hierarchical_stack');raw=pd.read_csv('data/train.csv',low_memory=False);latent=pd.read_csv('artifacts/latent_pitch_context.csv',dtype={'pitcher_id':str,'batter_hand':str})
parts={y:load_predictions(raw,y) for y in (2022,2023,2024)}
r22,y22,x22,b22=parts[2022];r23,y23,x23,b23=parts[2023];r24,y24,x24,b24=parts[2024]
g23=fit_adaptive(x22,y22,b22,720023);a22=b22;a23=np.clip(b23+g23.predict(x23),1e-6,1-1e-6);a24=np.load('old/experiments/adaptive_gate_2024.npz')['p']
frames=[]
for year,rows in ((2022,r22),(2023,r23),(2024,r24)):
 psych=attach_profile(rows,raw.loc[raw.season.lt(year)],500.);lp=latent_pitch_channel(rows,latent);frames.append(pd.concat([psych,lp],axis=1))
x=pd.concat(frames,ignore_index=True);mean=x.mean();std=x.std().replace(0,1).fillna(1);z=((x-mean)/std).fillna(0)
target=np.r_[y22-a22,y23-a23,y24-a24];weights=np.r_[np.full(len(y22),.55**2),np.full(len(y23),.55),np.ones(len(y24))]
m=Ridge(alpha=10000,fit_intercept=False).fit(z,target,sample_weight=weights)
np.savez_compressed(OUT/'psych_latent_meta.npz',columns=np.array(x.columns),mean=mean.to_numpy(),std=std.to_numpy(),coef=m.coef_,scale=.4)
profile,league=build_profile(raw,500.)
with open(OUT/'psych_profile.pkl','wb') as f:pickle.dump({'profile':profile,'league':league},f,protocol=4)
latent.loc[latent.season.eq(2025)].to_csv(OUT/'latent_pitch_context.csv',index=False)
print('saved',x.shape,len(profile),len(latent.loc[latent.season.eq(2025)]))
