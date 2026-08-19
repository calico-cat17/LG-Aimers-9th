"""Logistic boosting initialized by the hierarchical probability baseline."""
import time
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
R=Path(__file__).resolve().parent;raw=pd.read_csv(R/'data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.int8);s=raw.season.to_numpy();tr=s<2024;va=s==2024;h=raw.loc[tr];prior=float(y[tr].mean());ps=build_snapshots(h);bs=build_entity_snapshots(h,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(h,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,base=build_v3_features(raw,prior,ps,bs,ms,str(R/'model/trackman_prior_features.csv'));logit=np.log(np.clip(base,1e-4,1-1e-4)/(1-np.clip(base,1e-4,1-1e-4)));w=np.power(.30,2023-s[tr]);pt=Pool(x.loc[tr],y[tr],cat_features=CAT_V2,weight=w,baseline=logit[tr,None]);pv=Pool(x.loc[va],y[va],cat_features=CAT_V2,baseline=logit[va,None])
def sc(p):q=y[va];return 1e5*(1-np.mean((q-p)**2)/(q.mean()*(1-q.mean())))
for depth,l2 in [(7,20),(8,20)]:
 t=time.time();m=CatBoostClassifier(iterations=1000,depth=depth,learning_rate=.03,loss_function='Logloss',eval_metric='Logloss',l2_leaf_reg=l2,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=960000+depth,thread_count=6,allow_writing_files=False,verbose=False);m.fit(pt,eval_set=pv,early_stopping_rounds=140,use_best_model=True);raw_total=m.predict(pv,prediction_type='RawFormulaVal');p=1/(1+np.exp(-raw_total));p_added=1/(1+np.exp(-(logit[va]+raw_total)));print(depth,m.get_best_iteration()+1,'pool_total',sc(p),'manual_added',sc(p_added),time.time()-t,flush=True);np.savez_compressed(f'logit_offset_d{depth}_2024.npz',y=y[va],p=p);m.save_model(f'logit_offset_d{depth}_validation.cbm')
