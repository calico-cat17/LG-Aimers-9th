"""How much of the prediction is seed noise, and what does averaging it away buy?

score is 1e5*(2*Cov(p,y) - Var(p) - bias^2)/V. Independent noise in p raises
Var(p) without touching Cov, so removing it is a pure gain of 1e5*Var(noise)/V -
a direction the channel surface cannot reach, because every channel is a fixed
vector while this changes the base itself.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool
from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
tr = pd.read_csv('data/train.csv', low_memory=False)
SEEDS = [260803, 11, 22, 33, 44, 55]
for t in (2024, 2023, 2022):
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
    pool = Pool(x[fit], (y-base)[fit], weight=w, cat_features=CAT_V2)
    preds = []
    for s in SEEDS:
        m = CatBoostRegressor(iterations=220, depth=8, learning_rate=.035, loss_function='RMSE',
            l2_leaf_reg=12, random_strength=.35, bootstrap_type='Bernoulli', subsample=.85,
            one_hot_max_size=16, random_seed=s, thread_count=6, allow_writing_files=False, verbose=0)
        m.fit(pool)
        preds.append(np.clip(base[te] + m.predict(x[te]), 1e-5, 1-1e-5))
    P = np.vstack(preds); yt = y[te]; V = yt.mean()*(1-yt.mean())
    sc = lambda q: 1e5*(1-((q-yt)**2).mean()/V)
    noise_sd = P.std(0, ddof=1).mean()
    singles = [sc(p) for p in P]
    print(f'\n=== {t} ===', flush=True)
    print(f'  seed별 점수: ' + ' '.join(f'{s:.1f}' for s in singles))
    print(f'  행별 seed 표준편차 = {noise_sd:.5f}   (예측 sd = {P[0].std():.5f})')
    for k in (2, 3, 6):
        print(f'  {k}-seed 평균: {sc(P[:k].mean(0)):9.1f}   (단일 평균 {np.mean(singles[:k]):9.1f}, '
              f'+{sc(P[:k].mean(0))-np.mean(singles[:k]):.1f})')
    print(f'  이론 상한(노이즈 완전제거) = +{1e5*noise_sd**2/V:.1f}')
