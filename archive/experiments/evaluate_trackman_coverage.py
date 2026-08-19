"""Strict 2024 test of expanded high-confidence Trackman ID coverage."""
import gc,time
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
R=Path(__file__).resolve().parent;raw=pd.read_csv(R/'data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024;h=raw.loc[tr];prior=float(y[tr].mean());ps=build_snapshots(h);bs=build_entity_snapshots(h,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(h,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'])
def sc(p):q=y[va];return 1e5*(1-np.mean((q-p)**2)/(q.mean()*(1-q.mean())))
variants=[('strict',R/'model/trackman_prior_features.csv'),('cov80',R/'artifacts/trackman_prior_cov80.csv'),('cov89',R/'artifacts/trackman_prior_cov89.csv'),('cov96',R/'artifacts/trackman_prior_cov96.csv')]
import os
if os.environ.get('TM_SOFT_ONLY')=='1':variants=[('soft',R/'artifacts/trackman_prior_soft.csv')]
if os.environ.get('TM_PHYSICS_ONLY')=='1':variants=[('physics',R/'artifacts/trackman_prior_physics.csv')]
if os.environ.get('TM_ARSENAL_ONLY')=='1':variants=[('arsenal',R/'artifacts/trackman_prior_arsenal_core.csv')]
if os.environ.get('TM_CONSENSUS_ONLY')=='1':variants=[('consensus',R/'artifacts/trackman_prior_consensus.csv')]
for tag,path in variants:
 t=time.time();x,base=build_v3_features(raw,prior,ps,bs,ms,str(path));target=y-base;m=CatBoostRegressor(iterations=199,depth=8,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=12,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=260811,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(.30,2023-s[tr]);m.fit(x.loc[tr],target[tr],sample_weight=w,cat_features=CAT_V2);p=np.clip(base[va]+m.predict(x.loc[va]),1e-6,1-1e-6);print(tag,sc(p),time.time()-t,flush=True);np.savez_compressed(f'trackman_{tag}_2024.npz',y=y[va],p=p);m.save_model(f'trackman_{tag}_validation.cbm');del x,m;gc.collect()
