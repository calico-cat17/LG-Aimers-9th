"""F specialist trained only on the post-break 2023 regime, validated on 2024."""
from __future__ import annotations
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from evaluate_crossfit_psych_profiles import COEFFICIENTS,INTERCEPT,skill
from src.adaptive_gate import build_gate_features
from src.preprocessing_v2 import CAT_V2,build_v2_features,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();va=s==2024;hist=raw.loc[s<2024];prior=float(hist.control_success.mean());ps=build_snapshots(hist);bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x2,b2=build_v2_features(raw,prior,ps,'model/trackman_prior_features.csv');x3,b3=build_v3_features(raw,prior,ps,bs,ms,'model/trackman_prior_features.csv')
old=[np.load('old/experiments/result_dirs/v2_multisplit/2024.npz')['p'],np.load('old/experiments/model_v3_2024.npz')['p'],np.load('old/experiments/model_v3_decay_30.npz')['p']];risks=np.load('old/experiments/v3_subtypes_2024.npz')['risks'];rows=raw.loc[va].reset_index(drop=True);vf=rows.game_type.eq('F').to_numpy();fit=(s==2023)&raw.game_type.eq('F').to_numpy();yv=y[va];gate=CatBoostRegressor();gate.load_model('model_hierarchical_stack/adaptive_gate.cbm')
def final(P):
 main=.27358084*P[0]+.26512224*P[1]+.46129691*P[2];st=np.clip(INTERCEPT+np.c_[main,risks]@COEFFICIENTS,1e-6,1-1e-6);return np.clip(st+gate.predict(build_gate_features(rows,P,[risks[:,i] for i in range(3)],st)),1e-6,1-1e-6)
P=[q.copy() for q in old]; print('base',skill(yv,final(P)),flush=True);saved={}
for channel,(x,base,iters) in enumerate(((x2,b2,140),(x3,b3,220),(x3,b3,199))):
 members=[];best=(-1,0,None)
 for j in range(6):
  m=CatBoostRegressor(iterations=iters,depth=8,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=20,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=958000+channel*100+j,thread_count=6,allow_writing_files=False,verbose=False);m.fit(x.loc[fit],(y-base)[fit],cat_features=CAT_V2);members.append(np.clip(base[va][vf]+m.predict(x.loc[va].loc[vf]),1e-6,1-1e-6));q=np.mean(members,axis=0)
  for blend in (.25,.5,.75,1):
   trial=[a.copy() for a in P];trial[channel][vf]=(1-blend)*old[channel][vf]+blend*q;sc=skill(yv,final(trial))
   if sc>best[0]:best=(sc,j+1,(blend,q.copy()))
  print('channel',channel,'seeds',j+1,'best',best[:2], 'blend',best[2][0],flush=True)
 blend,q=best[2];P[channel][vf]=(1-blend)*old[channel][vf]+blend*q;saved[channel]=P[channel][vf];print('cumulative',channel,skill(yv,final(P)),flush=True)
np.savez_compressed('f_2023_only_2024.npz',y=yv,p0=saved[0],p1=saved[1],p2=saved[2],mask_f=vf,final=final(P))
