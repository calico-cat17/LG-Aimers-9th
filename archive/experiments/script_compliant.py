"""Rule-compliant offline inference.

Every prediction depends only on that row, fixed assets learned from the official
2019-2024 data, and prior-season Trackman summaries. No evaluation-set statistic,
ranking, rolling value, or cross-row group is used here.
"""
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

from src.adaptive_gate import build_gate_features
from src.preprocessing_v2 import build_v2_features, build_v3_features
from src.psych_latent import apply_linear_residual, build_production_features

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"


def _regressors(stem: str, seeds: list[int]):
    for seed in seeds:
        model = CatBoostRegressor()
        model.load_model(MODEL / f"{stem}_seed{seed}.cbm")
        yield model


def main() -> None:
    started = time.time()
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    test = pd.read_csv(ROOT / "data/test.csv", low_memory=False)
    row_id = test["row_id"].copy()

    pitcher = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    batter = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    pitchmix = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    trackman = str(MODEL / "trackman_prior_features.csv")
    x2, base2 = build_v2_features(test, meta["prior"], pitcher, trackman)
    x3, base3 = build_v3_features(test, meta["prior"], pitcher, batter, pitchmix, trackman)

    residual_seeds = list(meta["residual_seeds"])
    subtype_seeds = list(meta["subtype_seeds"])
    predictions = []
    for stem, x, base in (("v2_decay55", x2, base2),
                          ("v3_decay55", x3, base3),
                          ("v3_decay30", x3, base3)):
        members = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6)
                   for m in _regressors(stem, residual_seeds)]
        predictions.append(np.mean(members, axis=0))

    risks = []
    for name in ("middle", "wild", "reverse"):
        members = []
        for seed in subtype_seeds:
            model = CatBoostClassifier()
            model.load_model(MODEL / f"subtype_{name}_seed{seed}.cbm")
            members.append(model.predict_proba(x3)[:, 1])
        risks.append(np.mean(members, axis=0))

    main_p = np.average(np.vstack(predictions), axis=0,
                        weights=meta["main_weights"])
    z = np.column_stack([main_p] + risks)
    forward_p = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])

    gate_x = build_gate_features(test, predictions, risks,
                                 np.clip(forward_p, 1e-6, 1 - 1e-6))
    gate = CatBoostRegressor()
    gate.load_model(MODEL / "adaptive_gate.cbm")
    p = forward_p + gate.predict(gate_x)

    # This small residual was fitted only from 2022/23/24 temporal OOF labels.
    # Its scale is stored by the train-only temporal fit.
    residual_x = build_production_features(
        test, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv")
    p += apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")

    p = np.clip(p, 1e-5, 1 - 1e-5)
    if len(p) != len(test) or not np.isfinite(p).all():
        raise RuntimeError("invalid predictions")
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    pd.DataFrame({"row_id": row_id, "control_success": p}).to_csv(
        output / "submission.csv", index=False)
    print(f"predicted {len(p):,} independent rows in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
