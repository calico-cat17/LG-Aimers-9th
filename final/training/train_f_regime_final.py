"""Train the validated multi-channel Futures regime assets for 2025."""
from __future__ import annotations
import json,pickle,time
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,CatBoostRegressor
from evaluate_crossfit_psych_profiles import load_predictions
from evaluate_psych_residual_on_adaptive import fit_adaptive
from evaluate_league_transition_gate import CAT as TRANSITION_CAT,features,prior_type_table
from src.label_recovery import recover_failure_labels
from src.preprocessing_v2 import CAT_V2,build_v2_features,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
OUT=Path('model_f_regime');OUT.mkdir(exist_ok=True);t0=time.time();raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();prior=float(y.mean());ps=build_snapshots(raw);bs=build_entity_snapshots(raw,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(raw,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x2,b2=build_v2_features(raw,prior,ps,'model/trackman_prior_features.csv');x3,b3=build_v3_features(raw,prior,ps,bs,ms,'model/trackman_prior_features.csv');f=raw.game_type.eq('F').to_numpy()
def regs(stem,x,base,mask,decay,iters,n,seed0):
 for j in range(n):
  m=CatBoostRegressor(iterations=iters,depth=8,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=20,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=seed0+j,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(decay,2024-s[mask]) if decay else None;m.fit(x.loc[mask],(y-base)[mask],sample_weight=w,cat_features=CAT_V2);m.save_model(OUT/f'{stem}_{j}.cbm');print(stem,j+1,flush=True)
regs('f_v2_all',x2,b2,f,.55,140,4,968000)
regs('f_v355_recent',x3,b3,f&(s==2024),None,220,6,968100)
regs('f_v330_all',x3,b3,f,.30,199,4,968200)
regs('f_v330_recent',x3,b3,f&(s==2024),None,199,2,968300)
labels,recovered=recover_failure_labels(raw);ok=f&recovered.astype(bool)
for i,(name,iters) in enumerate(zip(('middle','wild','reverse'),(100,190,230))):
 m=CatBoostClassifier(iterations=iters,depth=7,learning_rate=.04,loss_function='Logloss',l2_leaf_reg=20,random_strength=.4,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=968400+i,thread_count=6,allow_writing_files=False,verbose=False);m.fit(x3.loc[ok],labels[ok,i],sample_weight=np.power(.30,2024-s[ok]),cat_features=CAT_V2);m.save_model(OUT/f'f_subtype_{name}.cbm');print('subtype',name,flush=True)
# Forward predictions supply a legal correction target for the 2025 transition gate.
z={yr:load_predictions(raw,yr) for yr in (2022,2023,2024)};r22,y22,g22,o22=z[2022];r23,y23,g23,o23=z[2023];r24,y24,_,_=z[2024];gate23=fit_adaptive(g22,y22,o22,520023);a23=np.clip(o23+gate23.predict(g23),1e-6,1-1e-6);a24=np.load('old/experiments/adaptive_gate_2024.npz')['p'].astype(float);xt=pd.concat([features(r23,a23,raw,2023),features(r24,a24,raw,2024)],ignore_index=True);target=np.r_[y23-a23,y24-a24];weights=np.r_[np.full(len(y23),.30),np.ones(len(y24))];tg=CatBoostRegressor(iterations=250,depth=6,learning_rate=.025,loss_function='RMSE',l2_leaf_reg=100,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=968500,thread_count=6,allow_writing_files=False,verbose=False);tg.fit(xt,target,sample_weight=weights,cat_features=TRANSITION_CAT);tg.save_model(OUT/'transition_gate.cbm');lookup=prior_type_table(raw,2025).to_dict();pickle.dump(lookup,open(OUT/'prior_type.pkl','wb'),protocol=4)
json.dump({'v2_scale':2.0,'v355_scale':.5,'v330_scale':.5,'v330_all_weight':.25,'v330_recent_inner_scale':.25,'subtype_scale':.75,'transition_scale':.15},open(OUT/'f_regime_meta.json','w'),indent=2);print('saved',OUT,'seconds',time.time()-t0)
