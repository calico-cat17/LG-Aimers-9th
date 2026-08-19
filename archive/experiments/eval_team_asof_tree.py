"""Residual CatBoost with explicit train-only EB team rates as input features."""
import gc,numpy as np,pandas as pd
from catboost import CatBoostRegressor,Pool
from hypotheses import tree_inputs
from src.preprocessing_v2 import CAT_V2
def sc(y,p):r=y.mean();return 1e5*(1-np.mean((y-np.clip(p,1e-5,1-1e-5))**2)/(r*(1-r)))
for target in (2022,2023,2024):
 frame,x,base,_,prior=tree_inputs(target,'model/trackman_prior_features.csv');season=frame.season.to_numpy();fit=season<target;te=season==target;h=frame.loc[fit];alpha=3000
 rates=[]
 for key in ('pitcher_team_id','batter_team_id'):
  a=h.groupby(key).control_success.agg(['sum','count']);rate=(a['sum']+alpha*prior)/(a['count']+alpha);rates.append(frame[key].map(rate).fillna(prior).to_numpy())
 x=x.copy();x['asof_pitcher_team_success_rate']=rates[0];x['asof_batter_team_success_rate']=rates[1];x['team_matchup_gap']=rates[0]-rates[1];x['is_same_hand']=(frame.pitcher_hand.astype(str)==frame.batter_hand.astype(str)).astype('int8')
 m=CatBoostRegressor(iterations=220,depth=8,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=12,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=260803,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(.55,(target-1)-season[fit]);m.fit(Pool(x.loc[fit],frame.control_success.to_numpy()[fit]-base[fit],weight=w,cat_features=CAT_V2));p=np.clip(base[te]+m.predict(x.loc[te]),1e-5,1-1e-5);print(target,sc(frame.control_success.to_numpy()[te],p),flush=True);del x,m;gc.collect()
