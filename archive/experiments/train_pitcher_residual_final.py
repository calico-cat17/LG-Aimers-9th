"""Refit the selected pitcher-baseline residual controllers on 2019-2024."""
import json, os, time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from src.catboost_features import CAT_COLS, build_catboost_features, attach_trackman_features

def base(df, prior, strength=75.):
 n=pd.to_numeric(df.asof_pitcher_n,errors='coerce').fillna(0).clip(lower=0).to_numpy()
 r=pd.to_numeric(df.asof_pitcher_success_rate,errors='coerce').fillna(prior).to_numpy()
 return ((r*n+prior*strength)/(n+strength)).astype(np.float32)

def main():
 out='model_pitcher_controller';os.makedirs(out,exist_ok=True)
 raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();prior=float(y.mean());b=base(raw,prior);target=y-b
 x=attach_trackman_features(build_catboost_features(raw,prior),'model/trackman_prior_features.csv')
 specs=[(7,207,2707),(8,172,2808)];files=[]
 for depth,it,seed in specs:
  m=CatBoostRegressor(iterations=it,depth=depth,learning_rate=.04,loss_function='RMSE',l2_leaf_reg=10,random_strength=.4,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=seed,thread_count=6,allow_writing_files=False,verbose=50)
  w=np.power(.55,2024-s);t=time.time();m.fit(x,target,sample_weight=w,cat_features=CAT_COLS)
  fn=f'pitcher_residual_d{depth}.cbm';m.save_model(os.path.join(out,fn));files.append(fn);print(fn,time.time()-t,flush=True)
 meta={'models':files,'prior':prior,'smoothing_strength':75.0,'momentum_weight':0.05,'decay':0.55,'validation_score':791.49,'cat_cols':CAT_COLS}
 json.dump(meta,open(os.path.join(out,'pitcher_controller_meta.json'),'w'),indent=2)

if __name__=='__main__':main()
