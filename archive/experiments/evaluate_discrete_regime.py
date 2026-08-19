"""Test a true discrete recent-regime model: train only 2023, validate 2024."""
import time,numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
full=pd.read_csv('data/train.csv',low_memory=False);hist=full.loc[full.season<2024];raw=full.loc[full.season>=2023].reset_index(drop=True);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s==2023;va=s==2024;prior=float(y[tr].mean());ps=build_snapshots(hist);bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,base=build_v3_features(raw,prior,ps,bs,ms,'model/trackman_prior_features.csv');target=y-base
def sc(p):q=y[va];return 1e5*(1-np.mean((q-p)**2)/(q.mean()*(1-q.mean())))
for depth in (6,):
 m=CatBoostRegressor(iterations=700,depth=depth,learning_rate=.035,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=30,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=1120000+depth,thread_count=6,allow_writing_files=False,verbose=False);t=time.time();m.fit(x.loc[tr],target[tr],cat_features=CAT_V2,eval_set=(x.loc[va],target[va]),early_stopping_rounds=100,use_best_model=True);p=np.clip(base[va]+m.predict(x.loc[va]),1e-6,1-1e-6);print(depth,m.get_best_iteration()+1,sc(p),time.time()-t,flush=True)
