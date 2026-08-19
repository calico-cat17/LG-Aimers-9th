"""Fixed-recipe seed pool for variance reduction, not validation cherry-picking."""
import time
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
R=Path(__file__).resolve().parent;O=R/'model_seed_pool_validation';O.mkdir(exist_ok=True);raw=pd.read_csv(R/'data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024;h=raw.loc[tr];prior=float(y[tr].mean());ps=build_snapshots(h);bs=build_entity_snapshots(h,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(h,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,base=build_v3_features(raw,prior,ps,bs,ms,str(R/'model/trackman_prior_features.csv'));target=y-base;w=np.power(.30,2023-s[tr])
def sc(p):q=y[va];return 1e5*(1-np.mean((q-p)**2)/(q.mean()*(1-q.mean())))
pred=[]
for seed in (101,202,303,404,505,606):
 t=time.time();m=CatBoostRegressor(iterations=199,depth=8,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=12,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=seed,thread_count=6,allow_writing_files=False,verbose=False);m.fit(x.loc[tr],target[tr],sample_weight=w,cat_features=CAT_V2);p=np.clip(base[va]+m.predict(x.loc[va]),1e-6,1-1e-6);pred.append(p);m.save_model(O/f'seed_{seed}.cbm');print(seed,sc(p),'uniform',sc(np.mean(pred,axis=0)),time.time()-t,flush=True)
np.savez_compressed('seed_pool_2024.npz',y=y[va],predictions=np.stack(pred),seeds=np.array([101,202,303,404,505,606]))
