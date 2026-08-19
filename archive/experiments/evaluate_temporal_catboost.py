import time
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
R='old/experiments';raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024;hist=raw.loc[tr];prior=float(y[tr].mean());ps=build_snapshots(hist);bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,base=build_v3_features(raw,prior,ps,bs,ms,'model/trackman_prior_features.csv');target=y-base;ref=np.load(R+'/model_v3_decay_30.npz')['p']
def sc(p):r=y[va].mean();return 1e5*(1-np.mean((y[va]-p)**2)/(r*(1-r)))
for ordered in [False,True]:
 kw={'boosting_type':'Ordered'} if ordered else {}
 m=CatBoostRegressor(iterations=700,depth=8,learning_rate=.035,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=15,random_strength=.3,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=400000+ordered,thread_count=6,allow_writing_files=False,verbose=False,has_time=True,**kw);w=np.power(.30,2023-s[tr]);t=time.time();m.fit(x.loc[tr],target[tr],sample_weight=w,cat_features=CAT_V2,eval_set=(x.loc[va],target[va]),early_stopping_rounds=110,use_best_model=True);p=np.clip(base[va]+m.predict(x.loc[va]),1e-6,1-1e-6);print('ordered',ordered,m.get_best_iteration()+1,'single',sc(p),'corr',np.corrcoef(ref-y[va],p-y[va])[0,1],time.time()-t,flush=True);print('blend',max((sc((1-a)*ref+a*p),a) for a in np.arange(0,1.01,.05)),flush=True);np.savez_compressed(f'temporal_catboost_{ordered}.npz',y=y[va],p=p)
