import time
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
from src.label_recovery import recover_failure_labels
RPATH='old/experiments'
raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024;hist=raw.loc[tr];prior=float(y[tr].mean());ps=build_snapshots(hist);bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,_=build_v3_features(raw,prior,ps,bs,ms,'model/trackman_prior_features.csv');L,mask=recover_failure_labels(raw);ok=tr&mask.astype(bool)
for depth in [6,7,8]:
 m=CatBoostRegressor(iterations=600,depth=depth,learning_rate=.035,loss_function='MultiRMSE',eval_metric='MultiRMSE',l2_leaf_reg=15,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=340000+depth,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(.30,2023-s[ok]);t=time.time();m.fit(x.loc[ok],L[ok],sample_weight=w,cat_features=CAT_V2,eval_set=(x.loc[va],L[va]),early_stopping_rounds=100,use_best_model=True);Q=np.clip(m.predict(x.loc[va]),0,1);print('model',depth,m.get_best_iteration()+1,time.time()-t,flush=True)
 pa=np.load(f'{RPATH}/adaptive_gate_2024.npz')['p'];pg=np.load(f'{RPATH}/group_meta_alpha100.npz')['p2024'];cur=.6*pg+.4*pa;union=np.clip(1-Q.sum(1),1e-6,1-1e-6)
 def sc(p):r=y[va].mean();return 1e5*(1-np.mean((y[va]-p)**2)/(r*(1-r)))
 print('union',sc(union),'best',max((sc((1-a)*cur+a*union),a) for a in np.arange(0,.301,.01)),flush=True)
