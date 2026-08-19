"""Forward-season evaluation of pitcher arsenal × batter familiarity."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from evaluate_crossfit_psych_profiles import load_predictions, skill
from evaluate_psych_residual_on_adaptive import fit_adaptive
from src.trackman_matchup import build_trackman_matchup

raw = pd.read_csv("data/train.csv", low_memory=False)
parts = {y: load_predictions(raw, y) for y in (2022, 2023, 2024)}
r22,y22,g22,o22=parts[2022];r23,y23,g23,o23=parts[2023];r24,y24,g24,o24=parts[2024]
gate=fit_adaptive(g22,y22,o22,2240023)
base={2022:o22,2023:np.clip(o23+gate.predict(g23),1e-6,1-1e-6),2024:np.load("old/experiments/adaptive_gate_2024.npz")["p"]}
rows={2022:r22,2023:r23,2024:r24};ys={2022:y22,2023:y23,2024:y24}
X={y:build_trackman_matchup(rows[y],"artifacts/trackman_prior_features.csv","artifacts/batter_trackman_familiarity.csv") for y in rows}
print("coverage",{y:float(X[y].matchup_available.mean()) for y in X},flush=True)
def pred(train_years,valid):
 x=pd.concat([X[y] for y in train_years],ignore_index=True);t=np.concatenate([ys[y]-base[y] for y in train_years]);latest=max(train_years);w=np.concatenate([np.full(len(ys[y]),.55**(latest-y)) for y in train_years])
 m=CatBoostRegressor(iterations=350,depth=5,learning_rate=.025,loss_function="RMSE",l2_leaf_reg=100,random_strength=.25,bootstrap_type="Bernoulli",subsample=.8,random_seed=2240000+valid,thread_count=6,allow_writing_files=False,verbose=False)
 m.fit(x,t,sample_weight=w);return m.predict(X[valid])
c23=pred((2022,),2023);c24=pred((2022,2023),2024);trials=[]
for s in (0,.05,.1,.15,.2,.3,.4,.5,.75,1):
 p23=np.clip(base[2023]+s*c23,1e-6,1-1e-6);p24=np.clip(base[2024]+s*c24,1e-6,1-1e-6);g=(skill(y23,p23)-skill(y23,base[2023]),skill(y24,p24)-skill(y24,base[2024]));trials.append((min(g),s,g,p23,p24))
_,s,g,p23,p24=max(trials,key=lambda q:q[0]);print("scale",s,"gains",g,"scores",skill(y23,p23),skill(y24,p24),flush=True)
np.savez_compressed("evaluation/trackman_matchup_expert.npz",y23=y23,p23=p23,base23=base[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=p24,base24=base[2024],pitcher24=r24.pitcher_id.to_numpy(),scale=s,corr23=c23,corr24=c24)
