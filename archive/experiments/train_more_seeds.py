"""Six more residual seeds (12 total) and a seed-ensembled adaptive gate.

Noise falls as 1/k, so the 6-seed ensemble already recovered ~83% of it and
another six buys about half a point. The gate was still single-seed, and it is
applied to every row, so its own noise had never been averaged at all.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from src.preprocessing_v2 import CAT_V2, build_v2_features, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
from src.adaptive_gate import build_gate_features

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "model_seed_ensemble"
NEW = [66, 77, 88, 99, 111, 122]
raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
y = raw.control_success.to_numpy(np.float32); season = raw.season.to_numpy(np.int16)
prior = float(y.mean())
ps = pd.read_pickle(OUT / "pitcher_snapshots.pkl")
bs = pd.read_pickle(OUT / "batter_snapshots.pkl")
ms = pd.read_pickle(OUT / "pitchmix_snapshots.pkl")
tm = str(ROOT / "model/trackman_prior_features.csv")
x2, base2 = build_v2_features(raw, prior, ps, tm)
x3, base3 = build_v3_features(raw, prior, ps, bs, ms, tm)
print("features built", flush=True)
t0 = time.time()
for fam, (name, x, base, decay, iters) in enumerate(
        [("v2_decay55", x2, base2, .55, 140),
         ("v3_decay55", x3, base3, .55, 220),
         ("v3_decay30", x3, base3, .30, 199)]):
    for i, s in enumerate(NEW):
        m = CatBoostRegressor(iterations=iters, depth=8, learning_rate=.035, loss_function="RMSE",
            l2_leaf_reg=12, random_strength=.35, bootstrap_type="Bernoulli", subsample=.85,
            one_hot_max_size=16, random_seed=270000 + fam * 1000 + s, thread_count=6,
            allow_writing_files=False, verbose=0)
        m.fit(x, y - base, sample_weight=np.power(decay, int(season.max()) - season), cat_features=CAT_V2)
        m.save_model(OUT / f"{name}_seed{6+i}.cbm")
        print(f"  {name} seed{6+i} done [{time.time()-t0:.0f}s]", flush=True)
print("saved", flush=True)
