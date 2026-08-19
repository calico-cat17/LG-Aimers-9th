"""Forward-evaluate one architecture against every target season, not just 2024.

For target year t the snapshots, prior and model all see seasons < t only, so each
year is an honest one-year-ahead transfer. Selecting on the worst of four years
instead of on 2024 alone is the point.
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool
from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots

TARGETS = (2022, 2023, 2024)
tr = pd.read_csv('data/train.csv', low_memory=False)

def year_data(t):
    hist = tr[tr.season < t]
    ps = build_snapshots(hist)
    bs = build_entity_snapshots(hist, 'batter_id', 'asof_batter_n',
        ['asof_batter_success_rate', 'asof_batter_middle_rate'], 'control_success')
    ms = build_entity_snapshots(hist, 'pitcher_id', 'asof_pitcher_pitchmix_n',
        ['asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate'])
    prior = float(hist.control_success.mean())
    frame = tr[tr.season <= t].reset_index(drop=True)
    x, base = build_v3_features(frame, prior, ps, bs, ms, 'model/trackman_prior_features.csv')
    y = frame.control_success.to_numpy(float)
    season = frame.season.to_numpy()
    return x, base, y, season

CACHE = {}
def get(t):
    if t not in CACHE:
        t0 = time.time(); CACHE[t] = year_data(t)
        print(f'  [features {t}: {time.time()-t0:.0f}s]', flush=True)
    return CACHE[t]

def evaluate(name, iterations=220, depth=8, lr=.035, decay=.55, l2=12, seed=260803):
    out = {}
    for t in TARGETS:
        x, base, y, season = get(t)
        te = season == t; fit = ~te
        w = np.power(decay, t - season[fit])
        m = CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=lr,
            loss_function='RMSE', l2_leaf_reg=l2, random_strength=.35,
            bootstrap_type='Bernoulli', subsample=.85, one_hot_max_size=16,
            random_seed=seed, thread_count=6, allow_writing_files=False, verbose=0)
        m.fit(Pool(x[fit], (y - base)[fit], weight=w, cat_features=CAT_V2))
        p = np.clip(base[te] + m.predict(x[te]), 1e-5, 1 - 1e-5)
        yt = y[te]; V = yt.mean() * (1 - yt.mean())
        out[t] = 1e5 * (1 - ((p - yt) ** 2).mean() / V)
    print(f'{name:38s} ' + '  '.join(f'{t}={out[t]:8.1f}' for t in TARGETS)
          + f'   worst={min(out.values()):8.1f}  mean={np.mean(list(out.values())):8.1f}', flush=True)
    return out

print('=== single residual CatBoost, forward on every target season ===', flush=True)
evaluate('baseline it=220 lr=.035 decay=.55')
for it, lr in [(400, .035), (220, .02), (600, .02)]:
    evaluate(f'it={it} lr={lr} decay=.55', iterations=it, lr=lr)
for decay in (.30, .80, 1.0):
    evaluate(f'it=220 lr=.035 decay={decay}', decay=decay)
for depth in (6, 10):
    evaluate(f'it=220 lr=.035 depth={depth}', depth=depth)
