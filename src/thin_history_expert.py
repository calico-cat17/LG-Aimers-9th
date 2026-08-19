"""Expert for pitchers whose career sample is thin but not empty.

Forward-vs-in-year on 2024 splits the shortfall very unevenly. Returning
pitchers holding 200-800 career pitches are 5.8% of rows and run 680.5 against
1274.2 - a gap of 593.7, twice the 292.9 average. They are the awkward middle:
too much history to treat as a debut, too little for the career rate to be
trusted, and the dynamic shrinkage swings hardest exactly there.

Membership is decided from the row's own `asof_pitcher_n` plus a lookup built
from train, so nothing here reads another evaluation row.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

CAT = ["count", "hand", "base_state", "seen_before", "game_type"]
LOW, HIGH = 200, 800


def membership(rows: pd.DataFrame, seen_path) -> np.ndarray:
    """Rows whose pitcher carries a thin-but-present career history."""
    with open(seen_path, "rb") as f:
        seen = pickle.load(f)
    n = pd.to_numeric(rows.asof_pitcher_n, errors="coerce").fillna(0).to_numpy(float)
    known = rows.pitcher_id.astype(str).isin(seen).to_numpy()
    return known & (n > LOW) & (n <= HIGH)


def expert_features(rows: pd.DataFrame, prediction: np.ndarray, seen_path) -> pd.DataFrame:
    with open(seen_path, "rb") as f:
        seen = pickle.load(f)
    num = lambda c: pd.to_numeric(rows[c], errors="coerce")
    n = num("asof_pitcher_n").fillna(0)
    x = pd.DataFrame({
        "count": rows.balls_before.astype(str) + "-" + rows.strikes_before.astype(str),
        "hand": rows.pitcher_hand.astype(str) + "-" + rows.batter_hand.astype(str),
        "base_state": rows.base_state.astype(str),
        "seen_before": rows.pitcher_id.astype(str).isin(seen).map({True: "Y", False: "N"}),
        "game_type": rows.game_type.astype(str),
        "base_prediction": prediction,
        "log_pitcher_n": np.log1p(n),
        # how far into the thin band this pitcher sits - the shrinkage weight
        # moves fastest here, so the position within the band is the signal
        "band_position": ((n - LOW) / (HIGH - LOW)).clip(0, 1),
        "career": num("asof_pitcher_success_rate"),
        "career_minus_recent3": num("asof_pitcher_success_rate") - num("asof_pitcher_prev3_game_success_rate"),
        "recent1": num("asof_pitcher_prev1_game_success_rate"),
        "recent3": num("asof_pitcher_prev3_game_success_rate"),
        "recent5": num("asof_pitcher_prev5_game_success_rate"),
        "recent_spread": num("asof_pitcher_prev1_game_success_rate") - num("asof_pitcher_prev5_game_success_rate"),
        "middle": num("asof_pitcher_middle_rate"),
        "reverse": num("asof_pitcher_reverse_rate"),
        "ball": num("asof_pitcher_ball_rate"),
        "batter_n": np.log1p(num("asof_batter_n").fillna(0)),
        "batter_rate": num("asof_batter_success_rate"),
        "li": num("li"), "inning": num("inning"), "runners": num("num_runners_on"),
        "score_diff": num("score_diff_pitcher_team"),
    })
    x[CAT] = x[CAT].astype("string").fillna("__MISSING__").astype(str)
    return x.replace([np.inf, -np.inf], np.nan)
