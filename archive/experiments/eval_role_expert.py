"""Role-specific residual channels, in the exact shape that made F work."""
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
CH = [('v2_decay55',.55,140),('v3_decay55',.55,220),('v3_decay30',.30,199)]
W = [0.27358084, 0.26512224, 0.46129691]
for t in (2022, 2023, 2024):
    hist = tr[tr.season < t]; prior = float(hist.control_success.mean())
    ps = build_snapshots(hist)
    bs = build_entity_snapshots(hist,'batter_id','asof_batter_n',
        ['asof_batter_success_rate','asof_batter_middle_rate'],'control_success')
    ms = build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',
        ['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'])
    frame = tr[tr.season <= t].reset_index(drop=True)
    x, base = build_v3_features(frame, prior, ps, bs, ms, 'model/trackman_prior_features.csv')
    y = frame.control_success.to_numpy(float); season = frame.season.to_numpy()
    te = season == t; fit = ~te
    lp = f'/tmp/role_{t}.pkl'
    with open(lp,'wb') as f: pickle.dump(build_role_lookup(hist), f)
    role = role_of(frame, lp)
    yt = y[te]
    print(f"\n=== {t}  starter {(role[te]=='starter').mean():.1%} / "
          f"reliever {(role[te]=='reliever').mean():.1%} / unknown {(role[te]=='unknown').mean():.1%} ===", flush=True)

    def fit_channel(name, decay, iters, mask, seed):
        m = CatBoostRegressor(iterations=iters, depth=8, learning_rate=.035, loss_function='RMSE',
            l2_leaf_reg=12, random_strength=.35, bootstrap_type='Bernoulli', subsample=.85,
            one_hot_max_size=16, random_seed=seed, thread_count=6, allow_writing_files=False, verbose=0)
        m.fit(Pool(x[mask], (y-base)[mask], weight=np.power(decay, t-season[mask]), cat_features=CAT_V2))
        return m
    shared = [fit_channel(n,d,i,fit,260802+k) for k,(n,d,i) in enumerate(CH)]
    P = [np.clip(base + m.predict(x), 1e-6, 1-1e-6) for m in shared]
    p0 = np.average(np.vstack(P), axis=0, weights=W)
    print(f'  shared            {bss(yt,p0[te]):8.1f}   rho={rho(yt,p0[te]):8.1f}')

    for which in ([2], [0,1,2]):
        P2 = [p.copy() for p in P]
        for r in ('starter','reliever'):
            sub = (role == r) & fit
            if sub.sum() < 30000: continue
            for k in which:
                n,d,i = CH[k]
                m = fit_channel(n, d, i, sub, 5000+k)
                sel = role == r
                P2[k][sel] = np.clip(base[sel] + m.predict(x[sel]), 1e-6, 1-1e-6)
        p = np.average(np.vstack(P2), axis=0, weights=W)
        tag = 'v3_decay30만' if which==[2] else '3채널 전부'
        print(f'  role-only {tag:12s}{bss(yt,p[te]):8.1f} ({bss(yt,p[te])-bss(yt,p0[te]):+6.2f})'
              f'   rho={rho(yt,p[te]):8.1f} ({rho(yt,p[te])-rho(yt,p0[te]):+6.2f})', flush=True)
