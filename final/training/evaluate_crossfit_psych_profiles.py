"""Strict year-forward evaluation of shrunk pitcher pressure-response profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.adaptive_gate import build_gate_features

ROOT = "old/experiments"
COEFFICIENTS = np.array([0.93505266, -0.00520129, 0.01091677, -0.02528331])
INTERCEPT = 0.0300329767


def skill(y: np.ndarray, p: np.ndarray) -> float:
    rate = y.mean()
    return 1e5 * (1 - np.mean((y - p) ** 2) / (rate * (1 - rate)))


def condition_frame(raw: pd.DataFrame) -> pd.DataFrame:
    li = pd.to_numeric(raw.li, errors="coerce").fillna(0)
    inning = pd.to_numeric(raw.inning, errors="coerce").fillna(0)
    score = pd.to_numeric(raw.score_diff_pitcher_team, errors="coerce").fillna(0)
    balls = pd.to_numeric(raw.balls_before, errors="coerce").fillna(0)
    strikes = pd.to_numeric(raw.strikes_before, errors="coerce").fillna(0)
    runners = pd.to_numeric(raw.num_runners_on, errors="coerce").fillna(0)
    risp = raw.runner_on_2b.fillna(0).astype(bool) | raw.runner_on_3b.fillna(0).astype(bool)
    return pd.DataFrame({
        "high_li": li >= 1.5,
        "extreme_li": li >= 3.0,
        "traffic": runners > 0,
        "risp": risp,
        "late": inning >= 7,
        "close": score.abs() <= 1,
        "behind": score < 0,
        "three_ball": balls == 3,
        "two_strike": strikes == 2,
        "full_count": (balls == 3) & (strikes == 2),
        "compound_pressure": (li >= 1.5) & (risp | (balls == 3)) & (score.abs() <= 2),
    }, index=raw.index).astype("int8")


def build_profile(history: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, dict[str, float]]:
    h = history.reset_index(drop=True).copy()
    cond = condition_frame(h)
    y = h.control_success.astype(float)
    league = {}
    rows = []
    global_rate = float(y.mean())
    pitcher = h.pitcher_id.astype(str)
    overall = h.assign(_y=y).groupby(pitcher, sort=False)._y.agg(["sum", "count"])
    overall_rate = (overall["sum"] + alpha * global_rate) / (overall["count"] + alpha)

    out = pd.DataFrame(index=overall.index)
    out["psych_history_log_n"] = np.log1p(overall["count"])
    for name in cond:
        mask = cond[name].astype(bool)
        context_rate = float(y[mask].mean()) if mask.any() else global_rate
        league[name] = context_rate - global_rate
        agg = h.loc[mask].assign(_y=y[mask]).groupby(pitcher[mask], sort=False)._y.agg(["sum", "count"])
        sums = agg["sum"].reindex(out.index).fillna(0)
        counts = agg["count"].reindex(out.index).fillna(0)
        # Shrink the context rate toward that pitcher's stabilized overall ability.
        rate = (sums + alpha * overall_rate) / (counts + alpha)
        out[f"psych_{name}_effect"] = rate - overall_rate
        out[f"psych_{name}_reliability"] = counts / (counts + alpha)
    return out.reset_index(names="pitcher_id"), league


def attach_profile(raw: pd.DataFrame, history: pd.DataFrame, alpha: float) -> pd.DataFrame:
    profile, league = build_profile(history, alpha)
    conditions = condition_frame(raw.reset_index(drop=True))
    joined = pd.DataFrame({"pitcher_id": raw.pitcher_id.astype(str).to_numpy()}).merge(
        profile, on="pitcher_id", how="left", sort=False
    )
    joined = joined.drop(columns="pitcher_id")
    joined["psych_history_log_n"] = joined["psych_history_log_n"].fillna(0)
    for name in conditions:
        effect = joined[f"psych_{name}_effect"].fillna(league[name])
        reliability = joined[f"psych_{name}_reliability"].fillna(0)
        active = conditions[name].to_numpy(float)
        joined[f"psych_{name}_active_effect"] = effect * active
        joined[f"psych_{name}_active_reliability"] = reliability * active
        joined[f"psych_{name}_effect"] = effect
        joined[f"psych_{name}_reliability"] = reliability
    return joined.astype("float32")


def load_predictions(raw: pd.DataFrame, year: int):
    rows = raw.loc[raw.season.eq(year)].reset_index(drop=True)
    if year < 2024:
        z2 = np.load(f"{ROOT}/result_dirs/v2_multisplit/{year}.npz")
        y, p2 = z2["y"], z2["p"]
        p55 = np.load(f"{ROOT}/result_dirs/v3_multisplit/{year}.npz")["p"]
        p30 = np.load(f"{ROOT}/result_dirs/v3_multisplit/{year}_d30.npz")["p"]
        risks = np.load(f"{ROOT}/result_dirs/v3_subtype_oof/{year}.npz")["risks"]
    else:
        subtype = np.load(f"{ROOT}/v3_subtypes_2024.npz")
        y, risks = subtype["y"], subtype["risks"]
        p2 = np.load(f"{ROOT}/result_dirs/v2_multisplit/2024.npz")["p"]
        p55 = np.load(f"{ROOT}/model_v3_2024.npz")["p"]
        p30 = np.load(f"{ROOT}/model_v3_decay_30.npz")["p"]
    predictions = [p2, p55, p30]
    main = 0.27358084 * p2 + 0.26512224 * p55 + 0.46129691 * p30
    old = np.clip(INTERCEPT + np.column_stack([main, risks]) @ COEFFICIENTS, 1e-6, 1 - 1e-6)
    gate = build_gate_features(rows, predictions, [risks[:, i] for i in range(3)], old)
    return rows, y, gate, old


def main():
    raw = pd.read_csv("data/train.csv", low_memory=False)
    loaded = {year: load_predictions(raw, year) for year in (2022, 2023, 2024)}
    for alpha in (500.0, 800.0, 1200.0, 2000.0):
        parts = {}
        for year, (rows, y, base_x, old) in loaded.items():
            history = raw.loc[raw.season.lt(year)]
            psych = attach_profile(rows, history, alpha)
            parts[year] = (y, pd.concat([base_x.reset_index(drop=True), psych], axis=1), old)
        for valid_year, train_years in ((2023, (2022,)), (2024, (2022, 2023))):
            x_train = pd.concat([parts[y][1] for y in train_years], ignore_index=True)
            y_train = np.concatenate([parts[y][0] for y in train_years])
            old_train = np.concatenate([parts[y][2] for y in train_years])
            years = np.concatenate([np.full(len(parts[y][0]), y) for y in train_years])
            y_valid, x_valid, old_valid = parts[valid_year]
            model = CatBoostRegressor(
                iterations=350, depth=2, learning_rate=0.02, loss_function="RMSE",
                l2_leaf_reg=100, random_strength=0.2, bootstrap_type="Bernoulli",
                subsample=0.8, random_seed=410000 + valid_year + int(alpha),
                thread_count=6, allow_writing_files=False, verbose=False,
            )
            weights = np.power(0.55, (valid_year - 1) - years)
            model.fit(x_train, y_train - old_train, sample_weight=weights)
            correction = model.predict(x_valid)
            values = []
            for scale in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
                pred = np.clip(old_valid + scale * correction, 1e-6, 1 - 1e-6)
                values.append(round(skill(y_valid, pred), 4))
            print(
                f"alpha={alpha:.0f} valid={valid_year} base={skill(y_valid, old_valid):.4f} "
                f"scales={values}", flush=True
            )


if __name__ == "__main__":
    main()
