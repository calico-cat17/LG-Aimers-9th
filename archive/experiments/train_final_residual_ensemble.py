"""Train diverse full-season residual models with monotone recency weights."""

import json,os,time
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.catboost_features import CAT_COLS,build_catboost_features,attach_trackman_features
from src.features import TARGET_COL


def main():
 out='model_ensemble';os.makedirs(out,exist_ok=True);raw=pd.read_csv('data/train.csv',encoding='utf-8-sig',low_memory=False);y=raw[TARGET_COL].to_numpy(np.float32);s=raw.season.to_numpy()
 means=raw.groupby('season')[TARGET_COL].mean().to_dict();target=np.array([yy-means[int(ss)] for yy,ss in zip(y,s)],np.float32)
 x=attach_trackman_features(build_catboost_features(raw,float(y.mean())),'model/trackman_prior_features.csv')
 configs=[]
 for depth in [7,8]:
  for decay in [.55,.65,.75]:configs.append((depth,decay))
 files=[]
 for i,(depth,decay) in enumerate(configs):
  m=CatBoostRegressor(iterations=200,depth=depth,learning_rate=.04,loss_function='RMSE',l2_leaf_reg=8,random_strength=.4,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=261000+i,thread_count=6,allow_writing_files=False,verbose=100)
  w=np.power(decay,2024-s);t=time.time();m.fit(x,target,sample_weight=w,cat_features=CAT_COLS)
  fn=f'residual_d{depth}_r{int(decay*100)}.cbm';m.save_model(os.path.join(out,fn));files.append(fn);print(fn,time.time()-t,flush=True)
 years=np.array(sorted(means));rates=np.array([means[k] for k in years]);k=4
 forecast=float(np.polyval(np.polyfit(years[-k:]-2025,rates[-k:],1),0))
 meta={'models':files,'prior':float(y.mean()),'base_rate_2025':forecast,'residual_weight':.4,'direct_weight':.6,
       'decays':[d for _,d in configs],'depths':[d for d,_ in configs],
       'weight_example':{'2019_at_decay_065':float(.65**5),'2023_at_decay_065':.65,'2024':1.0}}
 with open(os.path.join(out,'ensemble_meta.json'),'w') as f:json.dump(meta,f,indent=2)
 print(meta)
if __name__=='__main__':main()
