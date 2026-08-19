"""Train the selected forward-season probability stack on every labelled season."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from src.label_recovery import recover_failure_labels
from src.preprocessing_v2 import CAT_V2, build_v2_features, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "model_hierarchical_stack"
OUT.mkdir(exist_ok=True)

raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
y = raw.control_success.to_numpy(np.float32)
season = raw.season.to_numpy(np.int16)
prior = float(y.mean())

pitcher_snap = build_snapshots(raw)
batter_snap = build_entity_snapshots(
    raw, "batter_id", "asof_batter_n",
    ["asof_batter_success_rate", "asof_batter_middle_rate"], "control_success")
mix_snap = build_entity_snapshots(
    raw, "pitcher_id", "asof_pitcher_pitchmix_n",
    ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"])
pitcher_snap.to_pickle(OUT / "pitcher_snapshots.pkl")
batter_snap.to_pickle(OUT / "batter_snapshots.pkl")
mix_snap.to_pickle(OUT / "pitchmix_snapshots.pkl")

trackman = ROOT / "model/trackman_prior_features.csv"
x2, base2 = build_v2_features(raw, prior, pitcher_snap, str(trackman))
x3, base3 = build_v3_features(raw, prior, pitcher_snap, batter_snap, mix_snap, str(trackman))

def reg(name, x, base, decay, iterations, depth=8, seed=260810):
    model = CatBoostRegressor(
        iterations=iterations, depth=depth, learning_rate=.035, loss_function="RMSE",
        l2_leaf_reg=12, random_strength=.35, bootstrap_type="Bernoulli", subsample=.85,
        one_hot_max_size=16, random_seed=seed, thread_count=6,
        allow_writing_files=False, verbose=50)
    weights = np.power(decay, int(season.max()) - season)
    model.fit(x, y - base, sample_weight=weights, cat_features=CAT_V2)
    model.save_model(OUT / f"{name}.cbm")

# Iteration counts are frozen from the untouched 2024 forward holdout.
reg("v2_decay55", x2, base2, .55, 140, seed=260802)
reg("v3_decay55", x3, base3, .55, 220, seed=260803)
reg("v3_decay30", x3, base3, .30, 199, seed=260804)

labels, recovered = recover_failure_labels(raw)
ok = recovered.astype(bool)
subtype_specs = [("middle", 100), ("wild", 190), ("reverse", 230)]
sub_weights = np.power(.30, int(season.max()) - season[ok])
for index, (name, iterations) in enumerate(subtype_specs):
    model = CatBoostClassifier(
        iterations=iterations, depth=7, learning_rate=.04, loss_function="Logloss",
        l2_leaf_reg=12, random_strength=.4, bootstrap_type="Bernoulli", subsample=.85,
        one_hot_max_size=16, random_seed=261000 + index, thread_count=6,
        allow_writing_files=False, verbose=50)
    model.fit(x3.loc[ok], labels[ok, index], sample_weight=sub_weights, cat_features=CAT_V2)
    model.save_model(OUT / f"subtype_{name}.cbm")

meta = {
    "version": 1,
    "prior": prior,
    "main_weights": [0.27358084, 0.26512224, 0.46129691],
    "stack_intercept": 0.0300329767,
    "stack_coefficients": [0.93505266, -0.00520129, 0.01091677, -0.02528331],
    "validation_2024_score": 903.2578,
    "notes": "OOF ridge stack; subtype channels are recovered training proxies, not causal probabilities.",
}
(OUT / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
(OUT / "trackman_prior_features.csv").write_bytes(trackman.read_bytes())
print("saved", OUT)
