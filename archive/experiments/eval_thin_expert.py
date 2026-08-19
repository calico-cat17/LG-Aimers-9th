"""Does a thin-history expert transfer across every forward season?"""
from __future__ import annotations
import pickle, time
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool
from src.thin_history_expert import CAT, membership, expert_features
from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots

def rho(y, p):
    v = y.mean()*(1-y.mean()); c = ((p-p.mean())*(y-y.mean())).mean()
    return 1e5*c*c/(p.var()*v)
def bss(y, p):
    v = y.mean()*(1-y.mean()); return 1e5*(1-((np.clip(p,1e-5,1-1e-5)-y)**2).mean()/v)

tr = pd.read_csv('data/train.csv', low_memory=False)
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
    te = season == t; fit = ~te; w = np.power(.55, t - season[fit])
    cb = CatBoostRegressor(iterations=220, depth=8, learning_rate=.035, loss_function='RMSE',
        l2_leaf_reg=12, random_strength=.35, bootstrap_type='Bernoulli', subsample=.85,
        one_hot_max_size=16, random_seed=260803, thread_count=6, allow_writing_files=False, verbose=0)
    cb.fit(Pool(x[fit], (y-base)[fit], weight=w, cat_features=CAT_V2))
    p0 = np.clip(base + cb.predict(x), 1e-5, 1-1e-5)

    seen_path = f'/tmp/seen_{t}.pkl'
    with open(seen_path,'wb') as f: pickle.dump(set(hist.pitcher_id.astype(str)), f)
    mem = membership(frame, seen_path)
    ex = expert_features(frame, p0, seen_path)
    train_mask = mem & fit
    yt = y[te]
    print(f'\n=== {t}  thin-history 행 {mem[te].mean():.1%} (학습용 {train_mask.sum():,}행) ===', flush=True)
    print(f'  base           {bss(yt,p0[te]):8.1f}   rho={rho(yt,p0[te]):8.1f}')
    for it, lr in ((150,.03),(300,.02)):
        m = CatBoostRegressor(iterations=it, depth=6, learning_rate=lr, loss_function='RMSE',
            l2_leaf_reg=20, random_strength=.4, bootstrap_type='Bernoulli', subsample=.85,
            random_seed=7, thread_count=6, allow_writing_files=False, verbose=0)
        m.fit(Pool(ex[train_mask], (y-p0)[train_mask],
                   weight=np.power(.55, t-season[train_mask]), cat_features=CAT))
        for scale in (0.3, 0.5, 0.8):
            p = p0.copy()
            p[mem] = np.clip(p0[mem] + scale*m.predict(ex[mem]), 1e-5, 1-1e-5)
            print(f'  it={it} lr={lr} scale={scale}  {bss(yt,p[te]):8.1f} ({bss(yt,p[te])-bss(yt,p0[te]):+6.2f})'
                  f'   rho={rho(yt,p[te]):8.1f} ({rho(yt,p[te])-rho(yt,p0[te]):+6.2f})', flush=True)
