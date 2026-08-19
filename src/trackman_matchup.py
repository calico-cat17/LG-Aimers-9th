"""Prior-only pitcher-arsenal versus batter-familiarity features."""
from __future__ import annotations

import numpy as np
import pandas as pd


PITCHES = ("fastball", "breaking", "offspeed")
METRICS = ("rel_speed", "spin_rate", "induced_vert_break", "horz_break")


def build_trackman_matchup(raw: pd.DataFrame, pitcher_path: str, batter_path: str) -> pd.DataFrame:
    key = raw[["pitcher_id", "batter_id", "season"]].copy()
    key["pitcher_id"] = key.pitcher_id.astype(str)
    key["batter_id"] = key.batter_id.astype(str)
    pitcher = pd.read_csv(pitcher_path, dtype={"pitcher_id": str})
    batter = pd.read_csv(batter_path, dtype={"batter_id": str})
    pcols = ["pitcher_id", "season", "tm_n", "tm_mapping_similarity"]
    bcols = ["batter_id", "season", "btm_n", "btm_mapping_similarity"]
    for pitch in PITCHES:
        pcols.append(f"tm_{pitch}_rate")
        bcols.append(f"btm_{pitch}_exposure")
        for metric in METRICS:
            pcols.append(f"tm_{pitch}_{metric}_mean")
            bcols.extend([f"btm_{pitch}_{metric}_mean", f"btm_{pitch}_{metric}_std"])
    z = key.merge(pitcher[pcols], on=["pitcher_id", "season"], how="left", sort=False)
    z = z.merge(batter[bcols], on=["batter_id", "season"], how="left", sort=False)
    out = pd.DataFrame(index=z.index)
    out["matchup_available"] = (z.tm_n.notna() & z.btm_n.notna()).astype("int8")
    out["matchup_reliability"] = np.sqrt(
        z.tm_n.fillna(0) / (z.tm_n.fillna(0) + 300)
        * z.btm_n.fillna(0) / (z.btm_n.fillna(0) + 500)
    )
    weighted_distance = 0.0
    for pitch in PITCHES:
        pr = z[f"tm_{pitch}_rate"]
        be = z[f"btm_{pitch}_exposure"]
        out[f"match_{pitch}_exposure_gap"] = pr - be
        pitch_distance = 0.0
        for metric in METRICS:
            pm = z[f"tm_{pitch}_{metric}_mean"]
            bm = z[f"btm_{pitch}_{metric}_mean"]
            bs = z[f"btm_{pitch}_{metric}_std"].abs().clip(lower=1.0)
            diff = pm - bm
            out[f"match_{pitch}_{metric}_diff"] = diff
            out[f"match_{pitch}_{metric}_z"] = diff / bs
            pitch_distance = pitch_distance + (diff / bs).clip(-10, 10).pow(2)
        out[f"match_{pitch}_distance"] = np.sqrt(pitch_distance)
        weighted_distance = weighted_distance + pr.fillna(0) * out[f"match_{pitch}_distance"].fillna(0)
    out["match_arsenal_familiarity_distance"] = weighted_distance
    return out.replace([np.inf, -np.inf], np.nan).astype("float32")
