"""Measure seed averaging and capacity variants for the F residual expert."""
from __future__ import annotations
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
from evaluate_crossfit_psych_profiles import COEFFICIENTS,INTERCEPT,skill
from src.adaptive_gate import build_gate_features
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots

raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024;hist=raw.loc[tr];prior=float(y[tr].mean())
ps=build_snapshots(hist);bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,base=build_v3_features(raw,prior,ps,bs,ms,'model/trackman_prior_features.csv');target=y-base
p2=np.load('old/experiments/result_dirs/v2_multisplit/2024.npz')['p'];p55=np.load('old/experiments/model_v3_2024.npz')['p'];p30=np.load('old/experiments/model_v3_decay_30.npz')['p'];risks=np.load('old/experiments/v3_subtypes_2024.npz')['risks'];rows=raw.loc[va].reset_index(drop=True);vf=rows.game_type.eq('F').to_numpy();fit=tr&raw.game_type.eq('F').to_numpy();yv=y[va]
gate=CatBoostRegressor();gate.load_model('model_hierarchical_stack/adaptive_gate.cbm')
def final(q):
 pred=[p2,p55,q];main=.27358084*p2+.26512224*p55+.46129691*q;stack=np.clip(INTERCEPT+np.c_[main,risks]@COEFFICIENTS,1e-6,1-1e-6);gx=build_gate_features(rows,pred,[risks[:,i] for i in range(3)],stack);return np.clip(stack+gate.predict(gx),1e-6,1-1e-6)
print('base',skill(yv,final(p30)),flush=True)
members=[]
for j in range(8):
 m=CatBoostRegressor(iterations=123,depth=7,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=15,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=360001+j*101,thread_count=6,allow_writing_files=False,verbose=False)
 m.fit(x.loc[fit],target[fit],sample_weight=np.power(.30,2023-s[fit]),cat_features=CAT_V2)
 members.append(np.clip(base[va][vf]+m.predict(x.loc[va].loc[vf]),1e-6,1-1e-6));q=p30.copy();q[vf]=np.mean(members,axis=0);print('seeds',j+1,'score',skill(yv,final(q)),flush=True)
np.savez_compressed('f_expert_ensemble_2024.npz',y=yv,p_f=np.mean(members,axis=0),mask_f=vf)
