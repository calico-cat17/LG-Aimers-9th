"""Asymmetric leaf-wise/depth-wise CatBoost structures for rare contexts."""
import time,gc
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
R=Path(__file__).resolve().parent;raw=pd.read_csv(R/'data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024;h=raw.loc[tr];prior=float(y[tr].mean());ps=build_snapshots(h);bs=build_entity_snapshots(h,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(h,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,base=build_v3_features(raw,prior,ps,bs,ms,str(R/'model/trackman_prior_features.csv'));target=y-base;w=np.power(.30,2023-s[tr])
def sc(p):q=y[va];return 1e5*(1-np.mean((q-p)**2)/(q.mean()*(1-q.mean())))
specs=[('lossguide64',dict(grow_policy='Lossguide',max_leaves=64,min_data_in_leaf=100)),('depthwise8',dict(grow_policy='Depthwise',depth=8,min_data_in_leaf=100))]
for name,extra in specs:
 t=time.time();m=CatBoostRegressor(iterations=900,learning_rate=.03,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=20,random_strength=.3,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=910001,thread_count=6,allow_writing_files=False,verbose=False,**extra);m.fit(x.loc[tr],target[tr],sample_weight=w,cat_features=CAT_V2,eval_set=(x.loc[va],target[va]),early_stopping_rounds=130,use_best_model=True);p=np.clip(base[va]+m.predict(x.loc[va]),1e-6,1-1e-6);print(name,m.get_best_iteration()+1,sc(p),time.time()-t,flush=True);np.savez_compressed(f'asymmetric_{name}_2024.npz',y=y[va],p=p);m.save_model(f'asymmetric_{name}_validation.cbm');del m;gc.collect()
