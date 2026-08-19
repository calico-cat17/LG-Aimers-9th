"""Per-row league level as the shrinkage target itself, on every forward season."""
from __future__ import annotations
import os
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool
from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
from src.league_from_batters import league_level
tr = pd.read_csv('data/train.csv', low_memory=False)
res={}
targets=tuple(int(x) for x in os.environ.get('LEAGUE_TARGETS','2022,2023,2024').split(','))
for t in targets:
    hist = tr[tr.season < t]; prior = float(hist.control_success.mean())
    ps = build_snapshots(hist)
    bs = build_entity_snapshots(hist,'batter_id','asof_batter_n',
        ['asof_batter_success_rate','asof_batter_middle_rate'],'control_success')
    ms = build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',
        ['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'])
    frame = tr[tr.season <= t].reset_index(drop=True)
    y = frame.control_success.to_numpy(float); season = frame.season.to_numpy()
    te = season == t; fit = ~te; w = np.power(.55, t - season[fit])
    for tag, strength in (('fixed prior', None), ('league k=150', 150.), ('league k=400', 400.)):
        lvl = None if strength is None else league_level(frame, bs, prior, strength)
        x, base = build_v3_features(frame, prior, ps, bs, ms,
                                    'model/trackman_prior_features.csv', lvl)
        m = CatBoostRegressor(iterations=220, depth=8, learning_rate=.035, loss_function='RMSE',
            l2_leaf_reg=12, random_strength=.35, bootstrap_type='Bernoulli', subsample=.85,
            one_hot_max_size=16, random_seed=260803, thread_count=6, allow_writing_files=False, verbose=0)
        m.fit(Pool(x[fit], (y-base)[fit], weight=w, cat_features=CAT_V2))
        p = np.clip(base[te] + m.predict(x[te]), 1e-5, 1-1e-5)
        yt=y[te]; V=yt.mean()*(1-yt.mean()); sc=1e5*(1-((p-yt)**2).mean()/V)
        res.setdefault(tag,{})[t]=sc
        print(f'{t} {tag:14s} score={sc:9.1f}  base_bias={base[te].mean()-yt.mean():+.4f}', flush=True)
print()
for tag,d in res.items():
    print(f'{tag:14s} worst={min(d.values()):9.1f}  mean={np.mean(list(d.values())):9.1f}')
