"""Train the two production correction assets added to the Regime FiLM ZIP."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from evaluate_crossfit_psych_profiles import load_predictions
from evaluate_psych_residual_on_adaptive import fit_adaptive
from src.context_adjusted_psych import attach_context_adjusted_psych,build_profiles
from src.context_pressure_features import build_context_pressure_features
from src.stable_experts import PLATOON_COLS

OUT=Path('model_hierarchical_stack');raw=pd.read_csv('data/train.csv',low_memory=False);parts={y:load_predictions(raw,y) for y in (2022,2023,2024)}
r22,y22,g22,o22=parts[2022];r23,y23,g23,o23=parts[2023];r24,y24,g24,o24=parts[2024];gate=fit_adaptive(g22,y22,o22,2440023)
rows={2022:r22,2023:r23,2024:r24};ys={2022:y22,2023:y23,2024:y24};base={2022:o22,2023:np.clip(o23+gate.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']}
weights=np.concatenate([np.full(len(ys[y]),.55**(2024-y)) for y in rows]);target=np.concatenate([ys[y]-base[y] for y in rows])

# Context-adjusted psychology Ridge. Candidate internal scale .25 × minimax .75.
frames=[attach_context_adjusted_psych(rows[y],raw.loc[raw.season.lt(y)],400.,100.) for y in rows];x=pd.concat(frames,ignore_index=True);mean=x.mean().to_numpy();std=x.std().replace(0,1).fillna(1).to_numpy();z=np.nan_to_num((x.to_numpy()-mean)/std);ridge=Ridge(alpha=1000,fit_intercept=False).fit(z,target,sample_weight=weights)
np.savez_compressed(OUT/'stable_context_ridge.npz',columns=np.array(x.columns),mean=mean,std=std,coef=ridge.coef_,scale=np.float32(.25*.75));build_profiles(raw,400.,100.).to_pickle(OUT/'stable_context_profile.pkl')

# Dedicated platoon/pitch-mix tree. Candidate internal scale .30 × minimax 1.00.
px=pd.concat([build_context_pressure_features(rows[y])[PLATOON_COLS] for y in rows],ignore_index=True);m=CatBoostRegressor(iterations=180,depth=3,learning_rate=.02,loss_function='RMSE',l2_leaf_reg=100,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=250000+len(PLATOON_COLS),thread_count=6,allow_writing_files=False,verbose=False);m.fit(px,target,sample_weight=weights);m.save_model(OUT/'stable_platoon.cbm')
print('saved stable final assets',x.shape,px.shape,flush=True)
