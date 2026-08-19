"""Asymmetric specialisation: refit only the smaller, more distinct population.

Splitting 75% of rows into two role models lost 7.31 rho on 2022 - the sample
each model gives up outweighs the shape difference. F survived that trade
because it is 12% of rows and a genuinely separate league. So this keeps the
shared model everywhere and refits only relievers, then only the narrower slice
of relievers where the count-by-count gap actually opened (hitter's counts).
"""
from __future__ import annotations
import pickle
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool
from src.role_split import build_role_lookup, role_of
from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots

def rho(y,p):
    v=y.mean()*(1-y.mean()); c=((p-p.mean())*(y-y.mean())).mean(); return 1e5*c*c/(p.var()*v)
def bss(y,p):
    v=y.mean()*(1-y.mean()); return 1e5*(1-((np.clip(p,1e-5,1-1e-5)-y)**2).mean()/v)

tr = pd.read_csv('data/train.csv', low_memory=False)
CH=[('v2_decay55',.55,140),('v3_decay55',.55,220),('v3_decay30',.30,199)]
W=[0.27358084,0.26512224,0.46129691]
for t in (2024,):
    hist=tr[tr.season<t]; prior=float(hist.control_success.mean())
    ps=build_snapshots(hist)
    bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',
        ['asof_batter_success_rate','asof_batter_middle_rate'],'control_success')
    ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',
        ['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'])
    frame=tr[tr.season<=t].reset_index(drop=True)
    x,base=build_v3_features(frame,prior,ps,bs,ms,'model/trackman_prior_features.csv')
    y=frame.control_success.to_numpy(float); season=frame.season.to_numpy()
    te=season==t; fit=~te
    lp=f'/tmp/role_{t}.pkl'
    with open(lp,'wb') as f: pickle.dump(build_role_lookup(hist),f)
    role=role_of(frame,lp)
    balls=pd.to_numeric(frame.balls_before,errors='coerce').fillna(0).to_numpy()
    strikes=pd.to_numeric(frame.strikes_before,errors='coerce').fillna(0).to_numpy()
    behind=balls>strikes          # 타자 유리 카운트 = 차이가 벌어진 구간
    yt=y[te]
    def fit_ch(decay,iters,mask,seed):
        m=CatBoostRegressor(iterations=iters,depth=8,learning_rate=.035,loss_function='RMSE',
            l2_leaf_reg=12,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,
            one_hot_max_size=16,random_seed=seed,thread_count=6,allow_writing_files=False,verbose=0)
        m.fit(Pool(x[mask],(y-base)[mask],weight=np.power(decay,t-season[mask]),cat_features=CAT_V2))
        return m
    shared=[fit_ch(d,i,fit,260802+k) for k,(n,d,i) in enumerate(CH)]
    P=[np.clip(base+m.predict(x),1e-6,1-1e-6) for m in shared]
    p0=np.average(np.vstack(P),axis=0,weights=W)
    print(f'\n=== {t} ===',flush=True)
    print(f'  shared                     {bss(yt,p0[te]):8.1f}   rho={rho(yt,p0[te]):8.1f}')
    for tag,sel in [('불펜 전체', role=='reliever'),
                    ('불펜 3채널', role=='reliever')]:
        sub=sel&fit
        if sub.sum()<25000: print(f'  {tag}: 표본 부족'); continue
        P2=[p.copy() for p in P]
        chans=[2] if tag=='불펜 전체' else [0,1,2]
        for k in chans:
            m=fit_ch(CH[k][1],CH[k][2],sub,9100+k)
            P2[k][sel]=np.clip(base[sel]+m.predict(x[sel]),1e-6,1-1e-6)
        p=np.average(np.vstack(P2),axis=0,weights=W)
        print(f'  {tag:22s} ({sel[te].mean():5.1%}) {bss(yt,p[te]):8.1f} ({bss(yt,p[te])-bss(yt,p0[te]):+6.2f})'
              f'   rho={rho(yt,p[te]):8.1f} ({rho(yt,p[te])-rho(yt,p0[te]):+6.2f})',flush=True)
