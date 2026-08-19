"""Evaluate and refit explicit previous-season target-history features."""

import json,os,time,joblib
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from src.catboost_features import CAT_COLS,build_catboost_features,attach_trackman_features
from src.recent_season_features import build_recent_tables,attach_recent_features
from src.features import TARGET_COL

def model(seed,iters=1200):
 return CatBoostRegressor(iterations=iters,depth=8,learning_rate=.04,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=7,random_strength=.4,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=seed,thread_count=6,allow_writing_files=False,verbose=100)

def main():
 out='model_recent';os.makedirs(out,exist_ok=True);raw=pd.read_csv('data/train.csv',encoding='utf-8-sig',low_memory=False);y=raw[TARGET_COL].to_numpy(np.float32);s=raw.season.to_numpy();va=s==2024;tr=s<2024
 tables=build_recent_tables(raw);x=build_catboost_features(raw,float(y[tr].mean()));x=attach_trackman_features(x,'model/trackman_prior_features.csv');x=attach_recent_features(x,raw,tables)
 m=model(260810);w=np.power(.65,2023-s[tr]);m.fit(x.loc[tr],y[tr],sample_weight=w,cat_features=CAT_COLS,eval_set=(x.loc[va],y[va]),early_stopping_rounds=150,use_best_model=True)
 p=np.clip(m.predict(x.loc[va]),1e-6,1-1e-6);cal=LogisticRegression().fit(p.reshape(-1,1),y[va]);pc=cal.predict_proba(p.reshape(-1,1))[:,1];base=y[va].mean()*(1-y[va].mean())
 for name,q in [('raw',p),('cal',pc)]:
  b=np.mean((q-y[va])**2);print(name,b,100000*(1-b/base))
 best=m.get_best_iteration()+1
 # Refit on every season. 2025 rows consume only the table generated from 2024.
 fm=model(260811,best);wf=np.power(.65,2024-s);fm.fit(x,y,sample_weight=wf,cat_features=CAT_COLS);fm.save_model(f'{out}/recent_catboost.cbm')
 joblib.dump(tables,f'{out}/recent_tables.joblib',compress=3)
 meta={'prior':float(y.mean()),'iterations':best,'decay':.65,'calibration_coef':float(cal.coef_[0,0]),'calibration_intercept':float(cal.intercept_[0])}
 json.dump(meta,open(f'{out}/recent_meta.json','w'),indent=2);np.savez_compressed(f'{out}/validation.npz',y=y[va],raw=p,calibrated=pc)
if __name__=='__main__':main()
