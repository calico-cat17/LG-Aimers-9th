"""Offline inference entry point for the hierarchical residual stack."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from src.preprocessing_v2 import build_v2_features, build_v3_features
from src.adaptive_gate import build_gate_features

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"

def main():
    started = time.time()
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    test = pd.read_csv(ROOT / "data/test.csv", low_memory=False)
    row_id = test["row_id"].copy()
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    x2, base2 = build_v2_features(test, meta["prior"], ps, tm)
    x3, base3 = build_v3_features(test, meta["prior"], ps, bs, ms, tm)

    predictions = []
    for filename, x, base in [
        ("v2_decay55.cbm", x2, base2),
        ("v3_decay55.cbm", x3, base3),
        ("v3_decay30.cbm", x3, base3),
    ]:
        model = CatBoostRegressor(); model.load_model(MODEL / filename)
        predictions.append(np.clip(base + model.predict(x), 1e-6, 1-1e-6))
    risks = []
    for name in ("middle", "wild", "reverse"):
        model = CatBoostClassifier(); model.load_model(MODEL / f"subtype_{name}.cbm")
        risks.append(model.predict_proba(x3)[:, 1])
    # Original strictly-forward meta prediction is also the reference signal
    # used to train the adaptive residual gate.
    original_main = np.average(np.vstack(predictions), axis=0, weights=[.27358084,.26512224,.46129691])
    original_z = np.column_stack([original_main] + risks)
    original_p = .0300329767 + original_z @ np.asarray([.93505266,-.00520129,.01091677,-.02528331])
    if "direct_coefficients" in meta:
        z = np.column_stack(predictions + risks)
        p = meta["stack_intercept"] + z @ np.asarray(meta["direct_coefficients"])
    else:
        main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
        z = np.column_stack([main_p] + risks)
        p = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])
    if meta.get("adaptive_gate", False):
        gate_x=build_gate_features(test,predictions,risks,np.clip(original_p,1e-6,1-1e-6))
        gate=CatBoostRegressor();gate.load_model(MODEL/'adaptive_gate.cbm')
        # Gate was validated against the strictly-forward original meta model.
        p=original_p+float(meta.get('gate_scale',1.0))*gate.predict(gate_x)
    p = np.clip(p, 1e-5, 1-1e-5)
    if len(p) != len(test) or not np.isfinite(p).all():
        raise RuntimeError("invalid predictions")
    out = ROOT / "output"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": row_id, "control_success": p}).to_csv(out / "submission.csv", index=False)
    print(f"predicted {len(p):,} rows in {time.time()-started:.1f}s")

if __name__ == "__main__":
    main()
