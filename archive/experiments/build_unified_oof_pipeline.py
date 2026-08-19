"""Generate 2022-2024 forward predictions with one identical model recipe."""
from pathlib import Path
import time,gc,os
import numpy as np,pandas as pd
from catboost import CatBoostRegressor,CatBoostClassifier
from src.preprocessing_v2 import CAT_V2,build_v2_features,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
from src.label_recovery import recover_failure_labels

OUT=Path('artifacts/unified_oof');OUT.mkdir(parents=True,exist_ok=True);raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);season=raw.season.to_numpy();failure,valid_failure=recover_failure_labels(raw)
REG=[('v2_decay55',False,.55,140,260802),('v3_decay55',True,.55,220,260803),('v3_decay30',True,.30,199,260804)];SUB=[('middle',0,100,261000),('wild',1,190,261001),('reverse',2,230,261002)]
years=tuple(int(x) for x in os.environ.get('UNIFIED_YEARS','2022,2023,2024').split(','))
for vy in years:
 tr=season<vy;va=season==vy;hist=raw.loc[tr];prior=float(y[tr].mean());ps=build_snapshots(hist);bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'])
 x2,b2=build_v2_features(raw,prior,ps,'model_hierarchical_stack/trackman_prior_features.csv');pred=[]
 for name,use_v3,decay,it,seed in REG:
  if use_v3:continue
  x,base=x2,b2;m=CatBoostRegressor(iterations=it,depth=8,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=12,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=seed,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(decay,(vy-1)-season[tr]);t=time.time();m.fit(x.loc[tr],y[tr]-base[tr],sample_weight=w,cat_features=CAT_V2);p=np.clip(base[va]+m.predict(x.loc[va]),1e-6,1-1e-6);pred.append(p);print(vy,name,it,round(time.time()-t,1),flush=True);del m,x2,b2;gc.collect()
 x3,b3=build_v3_features(raw,prior,ps,bs,ms,'model_hierarchical_stack/trackman_prior_features.csv')
 for name,use_v3,decay,it,seed in REG:
  if not use_v3:continue
  m=CatBoostRegressor(iterations=it,depth=8,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=12,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=seed,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(decay,(vy-1)-season[tr]);t=time.time();m.fit(x3.loc[tr],y[tr]-b3[tr],sample_weight=w,cat_features=CAT_V2);p=np.clip(b3[va]+m.predict(x3.loc[va]),1e-6,1-1e-6);pred.append(p);print(vy,name,it,round(time.time()-t,1),flush=True);del m;gc.collect()
 risks=[];ok=tr&valid_failure.astype(bool)
 for name,j,it,seed in SUB:
  m=CatBoostClassifier(iterations=it,depth=7,learning_rate=.04,loss_function='Logloss',l2_leaf_reg=12,random_strength=.4,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=seed,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(.30,(vy-1)-season[ok]);t=time.time();m.fit(x3.loc[ok],failure[ok,j],sample_weight=w,cat_features=CAT_V2);risks.append(m.predict_proba(x3.loc[va])[:,1]);print(vy,'subtype_'+name,it,round(time.time()-t,1),flush=True)
 np.savez_compressed(OUT/f'{vy}.npz',y=y[va],p_v2=pred[0],p_v3_55=pred[1],p_v3_30=pred[2],risks=np.column_stack(risks),pitcher_id=raw.loc[va,'pitcher_id'].to_numpy());print('saved',vy,flush=True)
