import time
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
RPATH='old/experiments';raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024;hist=raw.loc[tr];prior=float(y[tr].mean());ps=build_snapshots(hist);bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,base=build_v3_features(raw,prior,ps,bs,ms,'model/trackman_prior_features.csv');target=y-base;base_p=np.load(f'{RPATH}/model_v3_decay_30.npz')['p'];new=base_p.copy()
def sc(mask,p):r=y[mask].mean();return 1e5*(1-np.mean((y[mask]-p)**2)/(r*(1-r)))
print('all base',sc(va,base_p))
for j,g in enumerate(['R','F']):
 a=tr&raw.game_type.eq(g).to_numpy();b=va&raw.game_type.eq(g).to_numpy();m=CatBoostRegressor(iterations=800,depth=7,learning_rate=.035,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=15,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=360000+j,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(.30,2023-s[a]);t=time.time();m.fit(x.loc[a],target[a],sample_weight=w,cat_features=CAT_V2,eval_set=(x.loc[b],target[b]),early_stopping_rounds=100,use_best_model=True);p=np.clip(base[b]+m.predict(x.loc[b]),1e-6,1-1e-6);new[np.flatnonzero(b)-np.flatnonzero(va)[0]]=p;print(g,a.sum(),b.sum(),m.get_best_iteration()+1,'base',sc(b,base_p[np.flatnonzero(b)-np.flatnonzero(va)[0]]),'expert',sc(b,p),time.time()-t,flush=True)
print('combined',sc(va,new));np.savez_compressed('game_type_experts_2024.npz',y=y[va],p=new)
