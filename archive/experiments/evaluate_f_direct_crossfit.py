"""Strict forward F-only direct probability classifier, blended row-locally."""
import gc,numpy as np,pandas as pd
from catboost import CatBoostClassifier
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy();s=raw.season.to_numpy();strict=np.load('evaluation/season_state_crossfit.npz');saved={}
def skill(y,p):r=y.mean();return 1e5*(1-np.mean((y-np.clip(p,1e-6,1-1e-6))**2)/(r*(1-r)))
for t in (2022,2023,2024):
 tr=s<t;va=s==t;hist=raw.loc[tr];prior=float(y[tr].mean());ps=build_snapshots(hist);bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);frame=raw.loc[s<=t].reset_index(drop=True);sf=frame.season.to_numpy();fit=(sf<t)&frame.game_type.eq('F').to_numpy();te=sf==t;x,_=build_v3_features(frame,prior,ps,bs,ms,'model_hierarchical_stack/trackman_prior_features.csv');m=CatBoostClassifier(iterations=400,depth=7,learning_rate=.03,loss_function='Logloss',l2_leaf_reg=30,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=970000+t,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(.30,(t-1)-sf[fit]);m.fit(x.loc[fit],frame.control_success.to_numpy()[fit],sample_weight=w,cat_features=CAT_V2);rows=frame.loc[te].reset_index(drop=True);mask=rows.game_type.eq('F').to_numpy();direct=m.predict_proba(x.loc[te].loc[mask])[:,1];base=strict[f'base{str(t)[-2:]}'].astype(float);saved[t]=(frame.control_success.to_numpy()[te],base,mask,direct);print(t,'F n',mask.sum(),'direct mean',direct.mean(),flush=True);del x,m;gc.collect()
best=None
for a in (.02,.05,.08,.1,.15,.2,.3,.4,.5):
 gains={}
 for t,(yt,b,mask,d) in saved.items():
  p=b.copy();p[mask]=(1-a)*p[mask]+a*d;gains[t]=skill(yt,p)-skill(yt,b)
 row=(min(gains.values()),sum(gains.values()),a,gains);best=row if best is None or row[:2]>best[:2] else best
print('BEST',best)
