"""Recover per-pitch failure subtypes from the following training row.

Competition staff explicitly allowed referencing future rows inside train.csv.
For a cumulative pre-pitch rate R and count N, the current event is revealed by
the next pre-pitch state: event = (N+1)*R_next - N*R_current.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import TARGET_COL


def recover_failure_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    group = df.groupby("pitcher_id", sort=False)
    n = pd.to_numeric(df["asof_pitcher_n"], errors="coerce")
    next_n = pd.to_numeric(group["asof_pitcher_n"].shift(-1), errors="coerce")
    valid = next_n.eq(n + 1)
    recovered = []
    for col in ("asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_reverse_rate"):
        rate = pd.to_numeric(df[col], errors="coerce")
        next_rate = pd.to_numeric(group[col].shift(-1), errors="coerce")
        delta = next_n * next_rate - n * rate
        valid &= delta.notna()
        recovered.append((delta > 0.5).to_numpy(np.float32))

    middle, ball_result, reverse = recovered
    failure = 1 - df[TARGET_COL].to_numpy(np.float32)
    # An ordinary ball can still be a successful intended pitch. With no actual
    # coordinate, wild is operationally the residual unexplained failure.
    wild = failure * (1 - middle) * (1 - reverse)
    labels = np.stack([middle, wild, reverse], axis=1).astype(np.float32)
    mask = valid.to_numpy(np.float32)
    selected = mask.astype(bool)
    print("recovered subtype audit:", {
        "coverage": float(mask.mean()),
        "ball_result_rate": float(ball_result[selected].mean()),
        "middle_rate": float(middle[selected].mean()),
        "wild_residual_rate": float(wild[selected].mean()),
        "reverse_rate": float(reverse[selected].mean()),
    })
    return labels, mask

