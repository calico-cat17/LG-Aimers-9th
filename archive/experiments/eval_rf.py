"""Random forest as a blend partner, judged on rho rather than raw score.

A member does not need to win on its own to be worth carrying. What decides its
value is how much correlation the mix gains, which is why this measures the
affine ceiling (1e5*rho^2) - the site already applies the optimal shift and
stretch, so a candidate's raw score is not what it will be paid for.
"""
from __future__ import annotations
import time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor, Pool
from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots

def ceiling(y, p):
    v = y.mean()*(1-y.mean()); c = ((p-p.mean())*(y-y.mean())).mean()
    return 1e5*c*c/(p.var()*v)
def raw(y, p):
    v = y.mean()*(1-y.mean()); return 1e5*(1-((np.clip(p,1e-5,1-1e-5)-y)**2).mean()/v)

tr = pd.read_csv('data/train.csv', low_memory=False)
for t in (2024, 2023):
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
    p_cb = np.clip(base[te] + cb.predict(x[te]), 1e-5, 1-1e-5)

    num = [c for c in x.columns if c not in CAT_V2 and pd.api.types.is_numeric_dtype(x[c])]
    xn = x[num].fillna(-999).to_numpy(np.float32)
    t0 = time.time()
    # A deterministic temporal-stratified cap is sufficient to measure channel
    # diversity and avoids spending minutes fitting nearly identical trees.
    fit_idx=np.flatnonzero(fit); rng=np.random.default_rng(970000+t)
    take=rng.choice(fit_idx,min(320000,len(fit_idx)),replace=False)
    rf = RandomForestRegressor(n_estimators=64, max_depth=13, min_samples_leaf=160,
        max_features=0.35, n_jobs=6, random_state=0)
    rf.fit(xn[take], (y-base)[take], sample_weight=np.power(.55,t-season[take]))
    p_rf = np.clip(base[te] + rf.predict(xn[te]), 1e-5, 1-1e-5)
    yt = y[te]
    print(f'\n=== {t}  (RF 학습 {time.time()-t0:.0f}s, 피처 {len(num)}개) ===', flush=True)
    print(f'  CatBoost   raw={raw(yt,p_cb):8.1f}  아핀상한={ceiling(yt,p_cb):8.1f}')
    print(f'  RF         raw={raw(yt,p_rf):8.1f}  아핀상한={ceiling(yt,p_rf):8.1f}')
    print(f'  두 예측 상관 = {np.corrcoef(p_cb,p_rf)[0,1]:.4f}')
    best=(0,ceiling(yt,p_cb))
    for a in np.arange(0,1.01,.05):
        c=ceiling(yt,(1-a)*p_cb+a*p_rf)
        if c>best[1]: best=(a,c)
    print(f'  최적 혼합  RF 비중 {best[0]:.2f} → 아핀상한 {best[1]:8.1f}  ({best[1]-ceiling(yt,p_cb):+.1f})', flush=True)
