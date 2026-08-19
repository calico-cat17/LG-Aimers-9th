"""Test recency plus history-confidence weighting for pitcher residuals."""
import os,time
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.catboost_features import CAT_COLS,build_catboost_features,attach_trackman_features

raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024;prior=float(y[tr].mean())
n=raw.asof_pitcher_n.fillna(0).clip(lower=0).to_numpy();rate=raw.asof_pitcher_success_rate.fillna(prior).to_numpy();base=(rate*n+prior*75)/(n+75);target=y-base
recent=raw[['asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate']].to_numpy(float)
spread=np.nanstd(recent,axis=1);confidence=n/(n+150);stability=np.exp(-4*np.nan_to_num(spread,nan=.25));quality=.75+.25*confidence*stability
x=attach_trackman_features(build_catboost_features(raw,prior),'model/trackman_prior_features.csv')
for strength in [.25,.5,1.0]:
 w=np.power(.55,2023-s[tr])*(1+strength*(quality[tr]-quality[tr].mean()))
 m=CatBoostRegressor(iterations=1000,depth=7,learning_rate=.04,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=10,random_strength=.4,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=3700+int(strength*100),thread_count=6,allow_writing_files=False,verbose=False)
 t=time.time();m.fit(x.loc[tr],target[tr],sample_weight=w,cat_features=CAT_COLS,eval_set=(x.loc[va],target[va]),early_stopping_rounds=120,use_best_model=True)
 correction=m.predict(x.loc[va]);rv=recent[va];recent_mean=np.nanmean(rv,axis=1);recent_mean=np.where(np.isfinite(recent_mean),recent_mean,base[va]);p=np.clip(base[va]+correction+.05*(recent_mean-base[va]),1e-6,1-1e-6);b=np.mean((y[va]-p)**2);r=y[va].mean();score=1e5*(1-b/(r*(1-r)))
 print(strength,m.get_best_iteration()+1,b,score,time.time()-t,flush=True)
