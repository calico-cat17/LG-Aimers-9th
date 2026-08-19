"""Fast ID-free numeric HGB diversity channel under forward validation."""
from __future__ import annotations
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive
from src.features import build_features
raw=pd.read_csv('data/train.csv',low_memory=False);z={y:load_predictions(raw,y) for y in (2022,2023,2024)};r22,y22,g22,o22=z[2022];r23,y23,g23,o23=z[2023];r24,y24,_,_=z[2024];gg=fit_adaptive(g22,y22,o22,520023);bases={2023:np.clip(o23+gg.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p'].astype(float)};ys={2023:y23,2024:y24}
for target in (2023,2024):
 fit=raw.season.lt(target);valid=raw.season.eq(target);X=build_features(raw.loc[fit]);V=build_features(raw.loc[valid]);yy=raw.loc[fit,'control_success'].to_numpy(float);season=raw.loc[fit,'season'].to_numpy();rng=np.random.default_rng(988000+target);idx=rng.choice(len(X),min(450000,len(X)),replace=False);m=HistGradientBoostingRegressor(loss='squared_error',learning_rate=.055,max_iter=260,max_leaf_nodes=31,max_depth=None,min_samples_leaf=120,l2_regularization=20,early_stopping=True,validation_fraction=.12,n_iter_no_change=30,random_state=target);m.fit(X.iloc[idx],yy[idx],sample_weight=np.power(.55,(target-1)-season[idx]));p=np.clip(m.predict(V),1e-6,1-1e-6);base=bases[target];print('\n',target,'iterations',m.n_iter_,'standalone',skill(ys[target],p),'base',skill(ys[target],base));best=(-1,None)
 for a in (0,.025,.05,.075,.1,.15,.2,.3,.4,.5):
  q=(1-a)*base+a*p;sc=skill(ys[target],q)
  if sc>best[0]:best=(sc,a)
 print('best blend',best,'gain',best[0]-skill(ys[target],base),flush=True)
