"""Strict forward calibration tests on temporal Adaptive predictions."""
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive

raw=pd.read_csv('data/train.csv',low_memory=False);parts={y:load_predictions(raw,y) for y in (2022,2023,2024)}
r22,y22,x22,b22=parts[2022];r23,y23,x23,b23=parts[2023];r24,y24,x24,b24=parts[2024]
g23=fit_adaptive(x22,y22,b22,820023);p={2022:b22,2023:np.clip(b23+g23.predict(x23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']};ys={2022:y22,2023:y23,2024:y24}

def features(q,kind):
 q=np.clip(q,1e-5,1-1e-5);z=np.log(q/(1-q))
 if kind=='affine':return np.c_[np.ones(len(q)),z]
 if kind=='curve':return np.c_[np.ones(len(q)),z,z*z,z*z*z]
 knots=np.array([.30,.35,.40,.45,.50,.55,.60,.65,.70])
 return np.c_[np.ones(len(q)),q,*[np.maximum(q-k,0) for k in knots]]

scales=(.1,.2,.3,.5,.75,1.0)
for kind in ('affine','curve','spline'):
 for ridge in (100,300,1000,3000,10000,30000,100000,300000):
  scores=[]
  for valid,tr_years in ((2023,(2022,)),(2024,(2022,2023))):
   X=np.concatenate([features(p[y],kind) for y in tr_years]);target=np.concatenate([ys[y]-p[y] for y in tr_years]);w=np.concatenate([np.full(len(ys[y]),.55**((valid-1)-y)) for y in tr_years])
   m=Ridge(alpha=ridge,fit_intercept=False).fit(X,target,sample_weight=w);corr=m.predict(features(p[valid],kind));scores.append([skill(ys[valid],np.clip(p[valid]+s*corr,1e-6,1-1e-6)) for s in scales])
  gains=np.minimum(np.array(scores[0])-skill(y23,p[2023]),np.array(scores[1])-skill(y24,p[2024]));j=int(gains.argmax())
  if gains[j]>0:print(kind,ridge,'scale',scales[j],'s23',round(scores[0][j],4),'s24',round(scores[1][j],4),'gains',round(scores[0][j]-skill(y23,p[2023]),4),round(scores[1][j]-skill(y24,p[2024]),4),flush=True)
