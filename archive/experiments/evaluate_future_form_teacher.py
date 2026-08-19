"""Train-only future-form label denoising, explicitly allowed by organizer Q&A."""
import time,gc
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
R=Path(__file__).resolve().parent;raw=pd.read_csv(R/'data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024;h=raw.loc[tr];prior=float(y[tr].mean());ps=build_snapshots(h);bs=build_entity_snapshots(h,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(h,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,base=build_v3_features(raw,prior,ps,bs,ms,str(R/'model/trackman_prior_features.csv'))

# Chronology follows season and official pre-pitch cumulative exposure.
hist=raw.loc[tr,['pitcher_id','season','asof_pitcher_n']].copy();hist['idx']=np.flatnonzero(tr);hist['y']=y[tr];hist=hist.sort_values(['pitcher_id','season','asof_pitcher_n','idx'])
def future_mean(window):
 out=np.full(len(raw),np.nan,np.float32)
 for _,g in hist.groupby('pitcher_id',sort=False):
  yy=g.y.to_numpy(float);rev=pd.Series(yy[::-1]).rolling(window,min_periods=max(3,window//5)).mean().shift(1).to_numpy()[::-1];out[g.idx.to_numpy()]=rev
 return out
def sc(p):q=y[va];return 1e5*(1-np.mean((q-np.clip(p,1e-5,1-1e-5))**2)/(q.mean()*(1-q.mean())))
import os
specs=[(20,.25),(50,.25),(50,.5),(100,.5)]
if os.environ.get('TEACHER_STRONG_ONLY')=='1':specs=[(50,.5),(100,.5),(200,.5),(50,.75)]
if os.environ.get('TEACHER_ONE')=='1':specs=[(50,.5)]
for window,mix in specs:
 fut=future_mean(window);soft=y.copy();ok=tr&np.isfinite(fut);soft[ok]=(1-mix)*y[ok]+mix*fut[ok]
 model=CatBoostRegressor(iterations=350,depth=8,learning_rate=.035,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=15,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=810000+window+int(mix*100),thread_count=6,allow_writing_files=False,verbose=False)
 w=np.power(.30,2023-s[tr]);t=time.time();model.fit(x.loc[tr],soft[tr]-base[tr],sample_weight=w,cat_features=CAT_V2,eval_set=(x.loc[va],y[va]-base[va]),early_stopping_rounds=100,use_best_model=True);p=np.clip(base[va]+model.predict(x.loc[va]),1e-6,1-1e-6);print(window,mix,model.get_best_iteration()+1,sc(p),time.time()-t,flush=True);np.savez_compressed(f'future_teacher_w{window}_m{int(mix*100)}.npz',y=y[va],p=p);model.save_model(f'future_teacher_w{window}_m{int(mix*100)}.cbm');del model;gc.collect()
