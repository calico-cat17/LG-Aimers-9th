"""Leakage-safe previous-season target aggregates with Bayesian shrinkage."""

from __future__ import annotations
import numpy as np
import pandas as pd
from src.features import TARGET_COL

SPECS = [
    ("pitcher", ["pitcher_id"], 100.0),
    ("batter", ["batter_id"], 100.0),
    ("pitcher_count", ["pitcher_id", "count_state"], 50.0),
    ("pitcher_base", ["pitcher_id", "base_state"], 50.0),
    ("pitcher_inning", ["pitcher_id", "inning_bucket"], 50.0),
    ("pitcher_leverage", ["pitcher_id", "leverage_bucket"], 50.0),
    ("pitcher_team", ["pitcher_team_id"], 300.0),
    ("batter_team", ["batter_team_id"], 300.0),
    ("count_context", ["count_state", "base_state", "inning_bucket"], 300.0),
    ("hand_context", ["pitcher_hand", "batter_hand", "count_state"], 300.0),
]

def make_keys(df: pd.DataFrame) -> pd.DataFrame:
    k=df.copy()
    k["count_state"]=k["balls_before"].astype(str)+"-"+k["strikes_before"].astype(str)
    k["inning_bucket"]=pd.cut(k["inning"],[-np.inf,3,6,9,np.inf],labels=False).fillna(-1).astype(int)
    k["leverage_bucket"]=pd.cut(k["li"],[-np.inf,.7,1.5,3,np.inf],labels=False).fillna(-1).astype(int)
    return k

def build_recent_tables(train: pd.DataFrame) -> dict[str,pd.DataFrame]:
    k=make_keys(train); priors=k.groupby("season")[TARGET_COL].mean().to_dict(); tables={}
    for name,cols,strength in SPECS:
        g=k.groupby(["season",*cols],dropna=False)[TARGET_COL].agg(["size","sum"]).reset_index()
        g["prior"]=g["season"].map(priors)
        g[f"recent_{name}_rate"]=(g["sum"]+strength*g["prior"])/(g["size"]+strength)
        g[f"recent_{name}_log_n"]=np.log1p(g["size"])
        g["season"]=g["season"]+1
        tables[name]=g[["season",*cols,f"recent_{name}_rate",f"recent_{name}_log_n"]]
    return tables

def attach_recent_features(x: pd.DataFrame, raw: pd.DataFrame, tables: dict[str,pd.DataFrame]) -> pd.DataFrame:
    out=x.copy(); keys=make_keys(raw); out["inning_bucket"]=keys["inning_bucket"].to_numpy();out["leverage_bucket"]=keys["leverage_bucket"].to_numpy()
    out["_order"]=np.arange(len(out))
    for name,cols,_ in SPECS:
        table=tables[name].copy()
        for c in cols:
            # CatBoost feature frame stores several identifiers as strings.
            if c in out and out[c].dtype == object: table[c]=table[c].astype(str)
        out=out.merge(table,on=["season",*cols],how="left",sort=False)
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
