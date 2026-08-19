"""Does the expanded Trackman mapping help on every forward season?"""
from __future__ import annotations
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool
from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
tr = pd.read_csv('data/train.csv', low_memory=False)
TABLES = {'old (245 pitchers)': 'model/trackman_prior_features.csv',
          'new (765 pitchers)': 'artifacts/trackman_prior_features_v2.csv'}
out = {k: {} for k in TABLES}
for t in (2022, 2023, 2024):
    hist = tr[tr.season < t]; prior = float(hist.control_success.mean())
    ps = build_snapshots(hist)
    bs = build_entity_snapshots(hist,'batter_id','asof_batter_n',
        ['asof_batter_success_rate','asof_batter_middle_rate'],'control_success')
    ms = build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',
        ['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'])
    frame = tr[tr.season <= t].reset_index(drop=True)
    y = frame.control_success.to_numpy(float); season = frame.season.to_numpy()
    te = season == t; fit = ~te; w = np.power(.55, t - season[fit])
    for name, path in TABLES.items():
        x, base = build_v3_features(frame, prior, ps, bs, ms, path)
        m = CatBoostRegressor(iterations=220, depth=8, learning_rate=.035, loss_function='RMSE',
            l2_leaf_reg=12, random_strength=.35, bootstrap_type='Bernoulli', subsample=.85,
            one_hot_max_size=16, random_seed=260803, thread_count=6,
            allow_writing_files=False, verbose=0)
        m.fit(Pool(x[fit], (y - base)[fit], weight=w, cat_features=CAT_V2))
        p = np.clip(base[te] + m.predict(x[te]), 1e-5, 1 - 1e-5)
        yt = y[te]; V = yt.mean() * (1 - yt.mean())
        out[name][t] = 1e5 * (1 - ((p - yt) ** 2).mean() / V)
    print(f'{t}: ' + '   '.join(f'{k}={v[t]:8.1f}' for k, v in out.items())
          + f'   delta={out["new (765 pitchers)"][t]-out["old (245 pitchers)"][t]:+8.1f}', flush=True)
print()
for k, v in out.items():
    print(f'{k:22s} worst={min(v.values()):8.1f}  mean={np.mean(list(v.values())):8.1f}')
