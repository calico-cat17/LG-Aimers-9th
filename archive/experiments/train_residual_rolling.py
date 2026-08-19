"""Separate season-level drift from per-pitch control discrimination."""

import os,time
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.catboost_features import CAT_COLS,build_catboost_features,attach_trackman_features
from src.features import TARGET_COL


def main():
 raw=pd.read_csv('data/train.csv',encoding='utf-8-sig',low_memory=False);y=raw[TARGET_COL].to_numpy(np.float32); seasons=raw.season.to_numpy();os.makedirs('residual_results',exist_ok=True)
 for vy in [2023,2024]:
  tr=seasons<vy;va=seasons==vy;means=raw.loc[tr].groupby('season')[TARGET_COL].mean().to_dict()
  residual=np.array([yy-means.get(int(s),0) for yy,s in zip(y,seasons)],dtype=np.float32)
  val_mean=float(y[va].mean());residual[va]=y[va]-val_mean
  prior=float(y[tr].mean());x=attach_trackman_features(build_catboost_features(raw,prior),'model/trackman_prior_features.csv')
  for decay in [.65,.8]:
   w=np.power(decay,(vy-1)-seasons[tr]);m=CatBoostRegressor(iterations=900,depth=8,learning_rate=.04,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=8,random_strength=.4,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=vy+int(decay*100),thread_count=6,allow_writing_files=False,verbose=False)
   t=time.time();m.fit(x.loc[tr],residual[tr],sample_weight=w,cat_features=CAT_COLS,eval_set=(x.loc[va],residual[va]),early_stopping_rounds=120,use_best_model=True)
   r=m.predict(x.loc[va]);p=np.clip(val_mean+r,1e-6,1-1e-6);b=float(np.mean((p-y[va])**2));base=val_mean*(1-val_mean);score=100000*(1-b/base)
   print({'year':vy,'decay':decay,'iter':m.get_best_iteration()+1,'brier':b,'score':score,'seconds':time.time()-t},flush=True)
   np.savez_compressed(f'residual_results/v{vy}_d{decay}.npz',y=y[va],residual_pred=r,pred=p)
  del x
if __name__=='__main__':main()
