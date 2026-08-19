"""Batter-side channels. The stack is pitcher-centric; the batter enters through
three asof columns and nothing else. On 2024 the pitcher-by-batter oracle was
+121.7, the largest of any axis measured, so the matchup carries signal the
current features never see."""
from __future__ import annotations
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool
from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots

def rho(y,p):
    v=y.mean()*(1-y.mean()); c=((p-p.mean())*(y-y.mean())).mean(); return 1e5*c*c/(p.var()*v)
def bss(y,p):
    v=y.mean()*(1-y.mean()); return 1e5*(1-((np.clip(p,1e-5,1-1e-5)-y)**2).mean()/v)

tr=pd.read_csv('data/train.csv',low_memory=False)
CH=[('v2',.55,140),('v355',.55,220),('v330',.30,199)]
W=[0.27358084,0.26512224,0.46129691]
for t in (2022,2023,2024):
    hist=tr[tr.season<t]; prior=float(hist.control_success.mean())
    ps=build_snapshots(hist)
    bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',
        ['asof_batter_success_rate','asof_batter_middle_rate'],'control_success')
    ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',
        ['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'])
    frame=tr[tr.season<=t].reset_index(drop=True)
    x,base=build_v3_features(frame,prior,ps,bs,ms,'model/trackman_prior_features.csv')
    y=frame.control_success.to_numpy(float); season=frame.season.to_numpy()
    te=season==t; fit=~te; yt=y[te]
    def ch(d,i,mask,seed):
        m=CatBoostRegressor(iterations=i,depth=8,learning_rate=.035,loss_function='RMSE',
            l2_leaf_reg=12,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,
            one_hot_max_size=16,random_seed=seed,thread_count=6,allow_writing_files=False,verbose=0)
        m.fit(Pool(x[mask],(y-base)[mask],weight=np.power(d,t-season[mask]),cat_features=CAT_V2))
        return m
    P=[np.clip(base+ch(d,i,fit,260802+k).predict(x),1e-6,1-1e-6) for k,(n,d,i) in enumerate(CH)]
    p0=np.average(np.vstack(P),axis=0,weights=W)
    print(f'\n=== {t} ===  shared {bss(yt,p0[te]):8.1f}  rho={rho(yt,p0[te]):8.1f}',flush=True)

    # batter-side history built from train only, joined on the row's own batter_id
    hb=hist.groupby('batter_id').agg(bn=('control_success','size'),br=('control_success','mean'))
    hb=hb[hb.bn>=300]
    hbh=hist.groupby(['batter_id','pitcher_hand']).control_success.agg(['size','mean'])
    hbh=hbh[hbh['size']>=100]
    lg=float(hist.control_success.mean())
    key=pd.MultiIndex.from_arrays([frame.batter_id.astype(str),frame.pitcher_hand.astype(str)])
    ex=pd.DataFrame({
        'base_prediction':p0,
        'batter_hist_rate':frame.batter_id.map(hb.br).fillna(lg).to_numpy(),
        'batter_hist_n':np.log1p(frame.batter_id.map(hb.bn).fillna(0).to_numpy()),
        'batter_vs_hand':key.map(hbh['mean']).to_numpy(float),
        'batter_vs_hand_n':np.log1p(np.nan_to_num(key.map(hbh['size']).to_numpy(float))),
        'batter_season_rate':pd.to_numeric(frame.asof_batter_success_rate,errors='coerce'),
        'batter_season_n':np.log1p(pd.to_numeric(frame.asof_batter_n,errors='coerce').fillna(0)),
        'batter_middle':pd.to_numeric(frame.asof_batter_middle_rate,errors='coerce'),
        'pitcher_rate':pd.to_numeric(frame.asof_pitcher_success_rate,errors='coerce'),
        'count':frame.balls_before.astype(str)+'-'+frame.strikes_before.astype(str),
        'hand':frame.pitcher_hand.astype(str)+'-'+frame.batter_hand.astype(str),
        'game_type':frame.game_type.astype(str),
        'li':pd.to_numeric(frame.li,errors='coerce'),
        'inning':pd.to_numeric(frame.inning,errors='coerce')})
    ex['batter_vs_hand_gap']=ex.batter_vs_hand-ex.batter_hist_rate
    CATB=['count','hand','game_type']
    ex[CATB]=ex[CATB].astype('string').fillna('__M__').astype(str)
    ex=ex.replace([np.inf,-np.inf],np.nan)
    for it,lr in ((250,.03),):
        m=CatBoostRegressor(iterations=it,depth=6,learning_rate=lr,loss_function='RMSE',
            l2_leaf_reg=20,random_strength=.4,bootstrap_type='Bernoulli',subsample=.85,
            random_seed=31,thread_count=6,allow_writing_files=False,verbose=0)
        m.fit(Pool(ex[fit],(y-p0)[fit],weight=np.power(.55,t-season[fit]),cat_features=CATB))
        d=m.predict(ex)
        for s in (0.25,0.5,0.75,1.0):
            p=np.clip(p0+s*d,1e-5,1-1e-5)
            print(f'  batter ch it={it} scale={s}  {bss(yt,p[te]):8.1f} ({bss(yt,p[te])-bss(yt,p0[te]):+6.2f})'
                  f'  rho={rho(yt,p[te]):8.1f} ({rho(yt,p[te])-rho(yt,p0[te]):+6.2f})',flush=True)
