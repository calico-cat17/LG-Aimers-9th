"""Forward validation of a row-local season-clock expert.

The numeric part of row_id preserves the released pitch order.  Unlike a
rolling feature, parsing the current row's identifier does not inspect any
other evaluation row.  Train IDs are global, while evaluation IDs reset, so
the transform explicitly supports both formats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from evaluate_crossfit_psych_profiles import load_predictions, skill
from evaluate_psych_residual_on_adaptive import fit_adaptive


raw = pd.read_csv("data/train.csv", low_memory=False)
parts = {y: load_predictions(raw, y) for y in (2022, 2023, 2024)}
r22, y22, g22, old22 = parts[2022]
r23, y23, g23, old23 = parts[2023]
r24, y24, g24, old24 = parts[2024]
gate23 = fit_adaptive(g22, y22, old22, 2140023)
base = {
    2022: old22,
    2023: np.clip(old23 + gate23.predict(g23), 1e-6, 1 - 1e-6),
    2024: np.load("old/experiments/adaptive_gate_2024.npz")["p"],
}
rows = {2022: r22, 2023: r23, 2024: r24}
ys = {2022: y22, 2023: y23, 2024: y24}

# Learned only from the public training table. Production TEST IDs reset at 1.
season_start = raw.groupby("season", sort=True).row_id.first().str.extract(r"(\d+)")[0].astype(int) - 1


def clock_features(d: pd.DataFrame) -> pd.DataFrame:
    r = d.reset_index(drop=True)
    number = r.row_id.str.extract(r"(\d+)")[0].astype(float)
    is_test = r.row_id.str.startswith("TEST_")
    offset = r.season.map(season_start).fillna(0).astype(float)
    within = np.where(is_test, number, number - offset)
    progress = np.clip(within / 255_000.0, 0, 1.1)
    month = pd.to_numeric(r.game_month, errors="coerce").fillna(6).to_numpy()
    x = pd.DataFrame({
        "season_progress": progress,
        "season_progress2": progress ** 2,
        "season_progress3": progress ** 3,
        "month_progress": month + progress,
        "progress_sin": np.sin(np.pi * progress),
        "progress_cos": np.cos(np.pi * progress),
        "month": r.game_month.astype(str),
        "game_type": r.game_type.astype(str),
        "pitcher_team": r.pitcher_team_id.astype(str),
        "pitcher": r.pitcher_id.astype(str),
        "clock_team": r.pitcher_team_id.astype(str) + "|" + pd.cut(progress, 10, labels=False).astype(str),
        "clock_pitcher": r.pitcher_id.astype(str) + "|" + pd.cut(progress, 5, labels=False).astype(str),
    })
    return x


X = {y: clock_features(rows[y]) for y in rows}
CATS = ["month", "game_type", "pitcher_team", "pitcher", "clock_team", "clock_pitcher"]


def fit_predict(train_years, valid_year):
    tx = pd.concat([X[y] for y in train_years], ignore_index=True)
    target = np.concatenate([ys[y] - base[y] for y in train_years])
    latest = max(train_years)
    weight = np.concatenate([np.full(len(ys[y]), 0.55 ** (latest - y)) for y in train_years])
    model = CatBoostRegressor(
        iterations=300, depth=5, learning_rate=.025, loss_function="RMSE",
        l2_leaf_reg=100, random_strength=.3, bootstrap_type="Bernoulli",
        subsample=.8, random_seed=2140000 + valid_year, thread_count=6,
        allow_writing_files=False, verbose=False,
    )
    model.fit(tx, target, sample_weight=weight, cat_features=CATS)
    return model.predict(X[valid_year])


c23 = fit_predict((2022,), 2023)
c24 = fit_predict((2022, 2023), 2024)
scales = (0, .05, .1, .15, .2, .3, .4, .5, .75, 1)
trials = []
for s in scales:
    p23 = np.clip(base[2023] + s * c23, 1e-6, 1 - 1e-6)
    p24 = np.clip(base[2024] + s * c24, 1e-6, 1 - 1e-6)
    gains = (skill(y23, p23) - skill(y23, base[2023]), skill(y24, p24) - skill(y24, base[2024]))
    trials.append((min(gains), s, gains, p23, p24))
_, scale, gains, p23, p24 = max(trials, key=lambda z: z[0])
print("scale", scale, "gains", gains, "scores", skill(y23, p23), skill(y24, p24), flush=True)
np.savez_compressed(
    "evaluation/rowid_clock_expert.npz",
    y23=y23, p23=p23, base23=base[2023], pitcher23=r23.pitcher_id.to_numpy(),
    y24=y24, p24=p24, base24=base[2024], pitcher24=r24.pitcher_id.to_numpy(),
    scale=scale, corr23=c23, corr24=c24,
)
