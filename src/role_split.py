"""Starter / reliever split, decided from a train-only lookup.

game_type worked because F is a different data-generating process, not because
the column was a useful feature - the win came from refitting the residual
channels on F rows alone. Pitching role is the same kind of axis inside R: on
2024 the two groups differ by +0.0121 on average, but the count-by-count spread
around that is 0.0126, so the shape differs too. The gap opens in hitter's
counts (1-0 +0.026, 2-0 +0.029, 2-1 +0.025), closes in pitcher's counts
(0-2 +0.002), and flips at 3-0 (-0.014) - a starter pacing through a plate
appearance and a reliever airing it out behave differently once behind.

Role comes from the share of a pitcher's train pitches thrown in innings 1-3,
so an evaluation row only ever looks up its own pitcher_id.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

STARTER_CUT = 0.30
MIN_PITCHES = 300


def build_role_lookup(history: pd.DataFrame) -> dict[str, str]:
    early = (pd.to_numeric(history.inning, errors="coerce") <= 3).astype(float)
    agg = pd.DataFrame({"pid": history.pitcher_id.astype(str), "early": early}) \
        .groupby("pid")["early"].agg(["mean", "size"])
    agg = agg[agg["size"] >= MIN_PITCHES]
    return {p: ("starter" if v >= STARTER_CUT else "reliever")
            for p, v in agg["mean"].items()}


def role_of(rows: pd.DataFrame, lookup_path) -> np.ndarray:
    with open(lookup_path, "rb") as f:
        lookup = pickle.load(f)
    return rows.pitcher_id.astype(str).map(lookup).fillna("unknown").to_numpy()
