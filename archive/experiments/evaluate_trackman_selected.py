import time
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024;hist=raw.loc[tr];prior=float(y[tr].mean());ps=build_snapshots(hist);bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,base=build_v3_features(raw,prior,ps,bs,ms,'artifacts/trackman_selected_features.csv');target=y-base
for depth in [7,8]:
 m=CatBoostRegressor(iterations=650,depth=depth,learning_rate=.035,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=15,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=350000+depth,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(.30,2023-s[tr]);t=time.time();m.fit(x.loc[tr],target[tr],sample_weight=w,cat_features=CAT_V2,eval_set=(x.loc[va],target[va]),early_stopping_rounds=100,use_best_model=True);p=np.clip(base[va]+m.predict(x.loc[va]),1e-6,1-1e-6);b=np.mean((y[va]-p)**2);r=y[va].mean();print(depth,m.get_best_iteration()+1,1e5*(1-b/(r*(1-r))),time.time()-t,flush=True)
