"""Forward-only R/F league calibration on top of the adaptive prediction.

The model uses only official training labels from seasons strictly before the
validation year.  It tests whether Regular/Futures deserves a separate level
and spread adjustment beyond CatBoost's existing categorical split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from evaluate_crossfit_psych_profiles import load_predictions, skill
from evaluate_psych_residual_on_adaptive import fit_adaptive


def design(rows: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    is_r = rows.game_type.eq("R").to_numpy(float)
    is_f = rows.game_type.eq("F").to_numpy(float)
    centered = p - 0.5
    # Separate level and calibration slope, plus promotion/demotion proxies that
    # are fully row-local. ID itself stays in the upstream CatBoost model.
    return np.nan_to_num(np.column_stack([
        is_r, is_f, is_r * centered, is_f * centered,
        is_r * np.log1p(rows.asof_pitcher_n.to_numpy(float)),
        is_f * np.log1p(rows.asof_pitcher_n.to_numpy(float)),
        is_r * (rows.asof_pitcher_prev3_game_success_rate.to_numpy(float) -
                rows.asof_pitcher_success_rate.to_numpy(float)),
        is_f * (rows.asof_pitcher_prev3_game_success_rate.to_numpy(float) -
                rows.asof_pitcher_success_rate.to_numpy(float)),
    ]), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> None:
    raw = pd.read_csv("data/train.csv", low_memory=False)
    parts = {year: load_predictions(raw, year) for year in (2022, 2023, 2024)}
    r22, y22, x22, p22 = parts[2022]
    r23, y23, x23, p23 = parts[2023]
    r24, y24, _, _ = parts[2024]
    gate23 = fit_adaptive(x22, y22, p22, 520023)
    a22 = p22
    a23 = np.clip(p23 + gate23.predict(x23), 1e-6, 1 - 1e-6)
    archived = np.load("old/experiments/adaptive_gate_2024.npz")
    a24 = archived["p"].astype(float)

    datasets = {2022: (r22, y22, a22), 2023: (r23, y23, a23),
                2024: (r24, y24, a24)}
    print("year type       n      rate    pred    residual")
    for year, (rows, y, p) in datasets.items():
        for kind in ("R", "F"):
            m = rows.game_type.eq(kind).to_numpy()
            print(f"{year} {kind:>4} {m.sum():8d} {y[m].mean():9.5f} "
                  f"{p[m].mean():9.5f} {(y[m]-p[m]).mean():+10.5f}")

    for alpha in (1e3, 3e3, 1e4, 3e4, 1e5):
        gains = []
        for valid, train_years in ((2023, (2022,)), (2024, (2022, 2023))):
            xx = np.concatenate([design(datasets[t][0], datasets[t][2]) for t in train_years])
            yy = np.concatenate([datasets[t][1] - datasets[t][2] for t in train_years])
            weights = np.concatenate([
                np.full(len(datasets[t][1]), 0.55 ** ((valid - 1) - t)) for t in train_years])
            model = Ridge(alpha=alpha, fit_intercept=False).fit(xx, yy, sample_weight=weights)
            rows, y, base = datasets[valid]
            correction = model.predict(design(rows, base))
            base_score = skill(y, base)
            scores = []
            for scale in (0.25, 0.5, 0.75, 1.0):
                scores.append(skill(y, np.clip(base + scale * correction, 1e-6, 1-1e-6)))
            gains.append(np.asarray(scores) - base_score)
        worst = np.minimum(*gains)
        j = int(np.argmax(worst))
        print(f"alpha={alpha:8.0f} scale={(.25,.5,.75,1.)[j]:.2f} "
              f"gain23={gains[0][j]:+.4f} gain24={gains[1][j]:+.4f} "
              f"worst={worst[j]:+.4f}")


if __name__ == "__main__":
    main()
