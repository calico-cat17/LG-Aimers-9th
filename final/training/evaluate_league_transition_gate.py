"""Test a row-local R/F transition residual under forward-year validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from evaluate_crossfit_psych_profiles import load_predictions, skill
from evaluate_psych_residual_on_adaptive import fit_adaptive


CAT = ["game_type", "prior_type", "transition", "count", "hand", "team_type"]


def prior_type_table(history: pd.DataFrame, target: int) -> pd.Series:
    h = history[history.season < target]
    counts = h.groupby(["pitcher_id", "season", "game_type"], observed=True).size().rename("n").reset_index()
    dominant = counts.sort_values("n").groupby(["pitcher_id", "season"]).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id").tail(1)
    return latest.set_index(latest.pitcher_id.astype(str)).game_type


def features(rows: pd.DataFrame, base: np.ndarray, history: pd.DataFrame, target: int) -> pd.DataFrame:
    prior = prior_type_table(history, target)
    pid = rows.pitcher_id.astype(str)
    prev = pid.map(prior).fillna("NEW").astype(str)
    current = rows.game_type.astype(str)
    x = pd.DataFrame({
        "game_type": current,
        "prior_type": prev,
        "transition": prev + ">" + current,
        "count": rows.balls_before.astype(str) + "-" + rows.strikes_before.astype(str),
        "hand": rows.pitcher_hand.astype(str) + "-" + rows.batter_hand.astype(str),
        "team_type": rows.pitcher_team_id.astype(str) + "|" + current,
        "base_prediction": base,
        "log_pitcher_n": np.log1p(pd.to_numeric(rows.asof_pitcher_n, errors="coerce").fillna(0)),
        "career": pd.to_numeric(rows.asof_pitcher_success_rate, errors="coerce"),
        "recent1": pd.to_numeric(rows.asof_pitcher_prev1_game_success_rate, errors="coerce"),
        "recent3": pd.to_numeric(rows.asof_pitcher_prev3_game_success_rate, errors="coerce"),
        "recent5": pd.to_numeric(rows.asof_pitcher_prev5_game_success_rate, errors="coerce"),
        "middle": pd.to_numeric(rows.asof_pitcher_middle_rate, errors="coerce"),
        "reverse": pd.to_numeric(rows.asof_pitcher_reverse_rate, errors="coerce"),
        "li": pd.to_numeric(rows.li, errors="coerce"),
        "inning": pd.to_numeric(rows.inning, errors="coerce"),
        "runners": pd.to_numeric(rows.num_runners_on, errors="coerce"),
    })
    x[CAT] = x[CAT].astype("string").fillna("__MISSING__").astype(str)
    return x


def main() -> None:
    raw = pd.read_csv("data/train.csv", low_memory=False)
    z = {year: load_predictions(raw, year) for year in (2022, 2023, 2024)}
    r22, y22, gx22, b22 = z[2022]
    r23, y23, gx23, b23 = z[2023]
    r24, y24, _, _ = z[2024]
    g23 = fit_adaptive(gx22, y22, b22, 520023)
    a22 = b22
    a23 = np.clip(b23 + g23.predict(gx23), 1e-6, 1-1e-6)
    a24 = np.load("old/experiments/adaptive_gate_2024.npz")["p"].astype(float)
    data = {2022: (r22, y22, a22), 2023: (r23, y23, a23), 2024: (r24, y24, a24)}

    for train_year, valid_year in ((2022, 2023), (2023, 2024)):
        rt, yt, pt = data[train_year]
        rv, yv, pv = data[valid_year]
        xt = features(rt, pt, raw, train_year)
        xv = features(rv, pv, raw, valid_year)
        print(f"\n{train_year}->{valid_year} base={skill(yv,pv):.4f}")
        for depth in (2, 3, 4, 5, 6):
            model = CatBoostRegressor(
                iterations=250, depth=depth, learning_rate=.025, loss_function="RMSE",
                l2_leaf_reg=100, random_strength=.2, bootstrap_type="Bernoulli",
                subsample=.8, random_seed=918000+depth+valid_year,
                thread_count=6, allow_writing_files=False, verbose=False)
            model.fit(xt, yt-pt, cat_features=CAT)
            correction = model.predict(xv)
            vals=[]
            for scale in (.1,.2,.3,.4,.5,.75,1.0):
                vals.append(skill(yv,np.clip(pv+scale*correction,1e-6,1-1e-6))-skill(yv,pv))
            j=int(np.argmax(vals)); scales=(.1,.2,.3,.4,.5,.75,1.0)
            print(f"depth={depth} scale={scales[j]:.2f} gain={vals[j]:+.4f} all={[round(v,3) for v in vals]}")
            if valid_year == 2024:
                fbase = np.load("expanded_blend_2024.npz")["p"].astype(float)
                fvals=[]
                for scale in (.1,.2,.3,.4,.5,.75,1.0):
                    fvals.append(skill(yv,np.clip(fbase+scale*correction,1e-6,1-1e-6))-skill(yv,fbase))
                k=int(np.argmax(fvals))
                print(f"          over-F-expert scale={scales[k]:.2f} gain={fvals[k]:+.4f} all={[round(v,3) for v in fvals]}")
                if depth >= 4:
                    np.savez_compressed("league_transition_2024.npz",y=yv,correction=correction)


if __name__ == "__main__":
    main()
