"""Validate a true pitcher-baseline residual controller on the 2024 season."""
import os, time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from src.catboost_features import CAT_COLS, build_catboost_features, attach_trackman_features


def pitcher_base(raw, prior, strength=100.):
    n=pd.to_numeric(raw.asof_pitcher_n,errors='coerce').fillna(0).clip(lower=0).to_numpy()
    r=pd.to_numeric(raw.asof_pitcher_success_rate,errors='coerce').fillna(prior).to_numpy()
    return ((r*n+prior*strength)/(n+strength)).astype(np.float32)


def main():
 raw=pd.read_csv('data/train.csv',low_memory=False); y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024
 prior=float(y[tr].mean());base=pitcher_base(raw,prior);target=y-base
 x=attach_trackman_features(build_catboost_features(raw,prior),'model/trackman_prior_features.csv')
 os.makedirs('model_pitcher_residual',exist_ok=True)
 for depth in [7,8]:
  for decay in [.55,.65,.75]:
   m=CatBoostRegressor(iterations=1000,depth=depth,learning_rate=.04,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=10,random_strength=.4,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=2600+depth*10+int(decay*100),thread_count=6,allow_writing_files=False,verbose=False)
   w=np.power(decay,2023-s[tr]);t=time.time();m.fit(x.loc[tr],target[tr],sample_weight=w,cat_features=CAT_COLS,eval_set=(x.loc[va],target[va]),early_stopping_rounds=120,use_best_model=True)
   p=np.clip(base[va]+m.predict(x.loc[va]),1e-6,1-1e-6);b=np.mean((y[va]-p)**2);r=y[va].mean();score=1e5*(1-b/(r*(1-r)))
   fn=f'd{depth}_r{int(decay*100)}.npz';np.savez_compressed('model_pitcher_residual/'+fn,y=y[va],p=p,residual=m.predict(x.loc[va]))
   print(depth,decay,m.get_best_iteration()+1,b,score,time.time()-t,flush=True)

if __name__=='__main__':main()
