import time
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
R='old/experiments';raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();tr=s<2024;va=s==2024;hist=raw.loc[tr];prior=float(y[tr].mean());ps=build_snapshots(hist);bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,base=build_v3_features(raw,prior,ps,bs,ms,'model/trackman_prior_features.csv');target=y-base;ref=np.load(R+'/model_v3_decay_30.npz')['p'];new=ref.copy();vstart=np.flatnonzero(va)[0]
def sc(yy,p):r=yy.mean();return 1e5*(1-np.mean((yy-p)**2)/(r*(1-r)))
for balls in range(4):
 for strikes in range(3):
  state=(raw.balls_before.eq(balls)&raw.strikes_before.eq(strikes)).to_numpy();a=tr&state;b=va&state;idx=np.flatnonzero(b)-vstart
  m=CatBoostRegressor(iterations=650,depth=7,learning_rate=.035,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=20,random_strength=.3,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=390000+10*balls+strikes,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(.30,2023-s[a]);t=time.time();m.fit(x.loc[a],target[a],sample_weight=w,cat_features=CAT_V2,eval_set=(x.loc[b],target[b]),early_stopping_rounds=90,use_best_model=True);p=np.clip(base[b]+m.predict(x.loc[b]),1e-6,1-1e-6);new[idx]=p;print(f'{balls}-{strikes}',a.sum(),b.sum(),m.get_best_iteration()+1,'base',sc(y[b],ref[idx]),'expert',sc(y[b],p),time.time()-t,flush=True)
print('combined',sc(y[va],new),flush=True);np.savez_compressed('count_experts_2024.npz',y=y[va],p=new)
