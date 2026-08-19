"""Add a strongly regularized psychological residual on top of Adaptive Gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

from evaluate_crossfit_psych_profiles import attach_profile, load_predictions, skill


def fit_adaptive(x, y, base, seed):
    model = CatBoostRegressor(
        iterations=73, depth=3, learning_rate=.025, loss_function="RMSE",
        l2_leaf_reg=30, random_strength=.2, bootstrap_type="Bernoulli",
        subsample=.8, random_seed=seed, thread_count=6,
        allow_writing_files=False, verbose=False,
    )
    model.fit(x, y-base)
    return model


def standardize(train, valid):
    mean = train.mean(axis=0)
    std = train.std(axis=0).replace(0, 1).fillna(1)
    return ((train-mean)/std).fillna(0), ((valid-mean)/std).fillna(0)


def main():
    raw = pd.read_csv("data/train.csv", low_memory=False)
    loaded = {year: load_predictions(raw, year) for year in (2022, 2023, 2024)}

    # Strictly forward Adaptive predictions: 2023 gate sees only 2022 residuals.
    r22, y22, x22, b22 = loaded[2022]
    r23, y23, x23, b23 = loaded[2023]
    r24, y24, x24, b24 = loaded[2024]
    gate23 = fit_adaptive(x22, y22, b22, 520023)
    a22 = b22.copy()  # no earlier OOF gate target is available
    a23 = np.clip(b23 + gate23.predict(x23), 1e-6, 1-1e-6)
    # Use the exact archived prediction of the current 912.72 validation model.
    archived = np.load("old/experiments/adaptive_gate_2024.npz")
    a24 = archived["p"].astype(float)
    print("adaptive", "2023", skill(y23, a23), "2024", skill(y24, a24))

    for alpha in (500., 800., 1200., 2000., 4000.):
        psych = {}
        for year, rows in ((2022, r22), (2023, r23), (2024, r24)):
            psych[year] = attach_profile(rows, raw.loc[raw.season.lt(year)], alpha)

        # Only active effects/reliabilities are allowed to change a row. Static
        # pitcher effects are deliberately excluded to avoid relearning ability.
        cols = [c for c in psych[2022] if c.endswith("active_effect") or
                c.endswith("active_reliability")]
        for ridge in (100., 300., 1000., 3000., 10000., 30000.):
            # 2023 audit: learn on 2022 hierarchical residual, apply on Adaptive.
            z22, z23 = standardize(psych[2022][cols], psych[2023][cols])
            m23 = Ridge(alpha=ridge, fit_intercept=False).fit(z22, y22-a22)
            c23 = m23.predict(z23)

            # 2024 audit: train on strict 2022/2023 predictions. Recent year gets
            # greater weight, matching the production gate's temporal weighting.
            tr = pd.concat([psych[2022][cols], psych[2023][cols]], ignore_index=True)
            ztr, z24 = standardize(tr, psych[2024][cols])
            target = np.r_[y22-a22, y23-a23]
            weights = np.r_[np.full(len(y22), .55), np.ones(len(y23))]
            m24 = Ridge(alpha=ridge, fit_intercept=False).fit(ztr, target, sample_weight=weights)
            c24 = m24.predict(z24)

            scores23=[];scores24=[]
            scales = (.02,.05,.08,.10,.15,.20,.25,.30,.35,.40,.50)
            for scale in scales:
                scores23.append(skill(y23,np.clip(a23+scale*c23,1e-6,1-1e-6)))
                scores24.append(skill(y24,np.clip(a24+scale*c24,1e-6,1-1e-6)))
            # Selection aid is deliberately the same fixed scale in both years.
            gains=np.minimum(np.asarray(scores23)-skill(y23,a23),
                             np.asarray(scores24)-skill(y24,a24))
            j=int(np.argmax(gains))
            if gains[j] > 0:
                print(f"alpha={alpha:.0f} ridge={ridge:.0f} scale={scales[j]:.2f} "
                      f"s23={scores23[j]:.4f} s24={scores24[j]:.4f} "
                      f"gains=({scores23[j]-skill(y23,a23):+.4f},"
                      f"{scores24[j]-skill(y24,a24):+.4f})",flush=True)


if __name__ == "__main__":
    main()
