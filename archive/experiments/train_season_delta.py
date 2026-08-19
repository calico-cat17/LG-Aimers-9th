import os,time,joblib
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.catboost_features import CAT_COLS,build_catboost_features,attach_trackman_features
from src.season_delta_features import build_snapshots,attach_season_delta

raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024;prior=float(y[tr].mean())
snap=build_snapshots(raw.loc[tr]);x=attach_trackman_features(build_catboost_features(raw,prior),'model/trackman_prior_features.csv');x=attach_season_delta(x,raw,snap,prior)
os.makedirs('model_season_delta',exist_ok=True);joblib.dump(snap,'model_season_delta/snapshots.joblib',compress=3)
for depth in [7,8]:
 m=CatBoostRegressor(iterations=1200,depth=depth,learning_rate=.04,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=8,random_strength=.4,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=4200+depth,thread_count=6,allow_writing_files=False,verbose=False)
 w=np.power(.65,2023-s[tr]);t=time.time();m.fit(x.loc[tr],y[tr],sample_weight=w,cat_features=CAT_COLS,eval_set=(x.loc[va],y[va]),early_stopping_rounds=150,use_best_model=True)
 p=np.clip(m.predict(x.loc[va]),1e-6,1-1e-6);b=np.mean((y[va]-p)**2);r=y[va].mean();print(depth,m.get_best_iteration()+1,b,1e5*(1-b/(r*(1-r))),time.time()-t,flush=True);np.savez_compressed(f'model_season_delta/d{depth}.npz',y=y[va],p=p)
