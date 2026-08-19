import numpy as np, pandas as pd, time
from catboost import CatBoostRegressor, Pool
from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
tr=pd.read_csv('data/train.csv',low_memory=False)
lm=tr.groupby('season').control_success.mean()
print('league mean by season:', {int(k):round(v,4) for k,v in lm.items()},flush=True)
for t in (2022,2023,2024):
    hist=tr[tr.season<t]
    ps=build_snapshots(hist)
    bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success')
    ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'])
    prior=float(hist.control_success.mean())
    frame=tr[tr.season<=t].reset_index(drop=True)
    x,base=build_v3_features(frame,prior,ps,bs,ms,'model/trackman_prior_features.csv')
    y=frame.control_success.to_numpy(float); season=frame.season.to_numpy()
    te=season==t; fit=~te; w=np.power(.55,t-season[fit])
    m=CatBoostRegressor(iterations=220,depth=8,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=12,
        random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,
        random_seed=260803,thread_count=6,allow_writing_files=False,verbose=0)
    m.fit(Pool(x[fit],(y-base)[fit],weight=w,cat_features=CAT_V2))
    p=np.clip(base[te]+m.predict(x[te]),1e-5,1-1e-5); yt=y[te]; V=yt.mean()*(1-yt.mean())
    bias=p.mean()-yt.mean(); Vp=p.var(); C=((p-p.mean())*(yt-yt.mean())).mean(); beta=C/Vp
    sc=1e5*(1-((p-yt)**2).mean()/V)
    print(f'{t}: prior_used={prior:.4f} actual={yt.mean():.4f}  base_bias={base[te].mean()-yt.mean():+.4f} '
          f'final_bias={bias:+.4f} beta={beta:.3f} | score={sc:9.1f} = refine {1e5*C*C/(Vp*V):8.1f} '
          f'- slope {1e5*Vp*(1-beta)**2/V:7.1f} - intercept {1e5*bias*bias/V:8.1f}',flush=True)
