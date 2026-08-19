"""Forward validation of rational-denominator game-state residual channel."""
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive
from src.rational_game_state import build_rational_game_features

raw=pd.read_csv('data/train.csv',low_memory=False);q={y:load_predictions(raw,y) for y in (2022,2023,2024)}
r22,y22,g22,o22=q[2022];r23,y23,g23,o23=q[2023];r24,y24,g24,o24=q[2024];gate=fit_adaptive(g22,y22,o22,2810023)
rows={2022:r22,2023:r23,2024:r24};ys={2022:y22,2023:y23,2024:y24};base={2022:o22,2023:np.clip(o23+gate.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']}
X={y:build_rational_game_features(rows[y]) for y in rows};corr={}
for valid,trains in ((2023,(2022,)),(2024,(2022,2023))):
 x=pd.concat([X[y] for y in trains],ignore_index=True);t=np.concatenate([ys[y]-base[y] for y in trains]);latest=max(trains);w=np.concatenate([np.full(len(ys[y]),.55**(latest-y)) for y in trains])
 m=CatBoostRegressor(iterations=300,depth=5,learning_rate=.025,loss_function='RMSE',l2_leaf_reg=150,random_strength=.25,bootstrap_type='Bernoulli',subsample=.8,random_seed=2810000+valid,thread_count=6,allow_writing_files=False,verbose=False);m.fit(x,t,sample_weight=w);corr[valid]=m.predict(X[valid])
best=None
for s in (0,.05,.1,.15,.2,.3,.4,.5,.75,1):
 p={y:np.clip(base[y]+s*corr[y],1e-6,1-1e-6) for y in (2023,2024)};g={y:skill(ys[y],p[y])-skill(ys[y],base[y]) for y in p};row=(min(g.values()),sum(g.values())/2,s,p,g)
 if best is None or row[:2]>best[:2]:best=row
_,_,s,p,g=best;print('best scale',s,'gains',g,'scores',{y:skill(ys[y],p[y]) for y in p},flush=True)
np.savez_compressed('evaluation/rational_game_state.npz',y23=y23,p23=p[2023],base23=base[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=p[2024],base24=base[2024],pitcher24=r24.pitcher_id.to_numpy(),corr23=corr[2023],corr24=corr[2024],scale=s)
