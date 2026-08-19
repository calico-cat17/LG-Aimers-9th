"""Retrain the production residual stack with a seed ensemble.

Seed-to-seed prediction noise measured 0.0061-0.0079 per row, and averaging it
away is worth +15 to +25 on every forward season - almost exactly the theoretical
1e5*Var(noise)/V, so the averaging recovers essentially all of it. The shipped
models were single-seed, so that loss is currently in the live score.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from src.label_recovery import recover_failure_labels
from src.preprocessing_v2 import CAT_V2, build_v2_features, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "model_seed_ensemble"; OUT.mkdir(exist_ok=True)
SEEDS = [0, 11, 22, 33, 44, 55]

raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
y = raw.control_success.to_numpy(np.float32); season = raw.season.to_numpy(np.int16)
prior = float(y.mean())
ps = build_snapshots(raw)
bs = build_entity_snapshots(raw, "batter_id", "asof_batter_n",
    ["asof_batter_success_rate", "asof_batter_middle_rate"], "control_success")
ms = build_entity_snapshots(raw, "pitcher_id", "asof_pitcher_pitchmix_n",
    ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"])
for name, obj in (("pitcher_snapshots", ps), ("batter_snapshots", bs), ("pitchmix_snapshots", ms)):
    obj.to_pickle(OUT / f"{name}.pkl")
tm = str(ROOT / "model/trackman_prior_features.csv")
x2, base2 = build_v2_features(raw, prior, ps, tm)
x3, base3 = build_v3_features(raw, prior, ps, bs, ms, tm)
print("features built", flush=True)

def reg(name, x, base, decay, iterations, seed):
    m = CatBoostRegressor(iterations=iterations, depth=8, learning_rate=.035, loss_function="RMSE",
        l2_leaf_reg=12, random_strength=.35, bootstrap_type="Bernoulli", subsample=.85,
        one_hot_max_size=16, random_seed=seed, thread_count=6, allow_writing_files=False, verbose=0)
    m.fit(x, y - base, sample_weight=np.power(decay, int(season.max()) - season), cat_features=CAT_V2)
    m.save_model(OUT / f"{name}_s{seed}.cbm")

t0 = time.time()
for base_seed, (name, x, base, decay, iters) in enumerate(
        [("v2_decay55", x2, base2, .55, 140),
         ("v3_decay55", x3, base3, .55, 220),
         ("v3_decay30", x3, base3, .30, 199)]):
    for s in SEEDS:
        reg(name, x, base, decay, iters, 260802 + base_seed * 1000 + s)
        print(f"  {name} seed {s} done  [{time.time()-t0:.0f}s]", flush=True)

labels, recovered = recover_failure_labels(raw)
ok = recovered.astype(bool)
sw = np.power(.30, int(season.max()) - season[ok])
for index, (name, iterations) in enumerate([("middle", 100), ("wild", 190), ("reverse", 230)]):
    for s in SEEDS:
        m = CatBoostClassifier(iterations=iterations, depth=7, learning_rate=.04,
            loss_function="Logloss", l2_leaf_reg=12, random_strength=.4,
            bootstrap_type="Bernoulli", subsample=.85, one_hot_max_size=16,
            random_seed=261000 + index * 100 + s, thread_count=6,
            allow_writing_files=False, verbose=0)
        m.fit(x3.loc[ok], labels[ok, index], sample_weight=sw, cat_features=CAT_V2)
        m.save_model(OUT / f"subtype_{name}_s{s}.cbm")
    print(f"  subtype {name} done  [{time.time()-t0:.0f}s]", flush=True)
(OUT / "seeds.json").write_text(json.dumps(SEEDS))
print("saved", OUT, f"[{time.time()-t0:.0f}s]")
