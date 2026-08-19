"""Train a compact sklearn baseline and save an inference-ready bundle."""

from __future__ import annotations

import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.features import TARGET_COL, build_features


def score_report(y: np.ndarray, p: np.ndarray, label: str) -> None:
    brier = brier_score_loss(y, p)
    rate = float(np.mean(y))
    reference = rate * (1.0 - rate)
    skill = max(0.0, 100000.0 * (1.0 - brier / reference))
    print(f"{label}: brier={brier:.7f}, rate={rate:.5f}, score={skill:.2f}")


def make_model(args: argparse.Namespace) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2,
        early_stopping=True,
        validation_fraction=0.08,
        n_iter_no_change=30,
        random_state=args.seed,
        verbose=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--output", default="model/model.joblib")
    parser.add_argument("--valid-season", type=int, default=2024)
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    parser.add_argument("--l2", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    raw = pd.read_csv(args.data, encoding="utf-8-sig", low_memory=False)
    y = raw[TARGET_COL].to_numpy(dtype=np.int8)
    seasons = raw["season"].to_numpy()
    x = build_features(raw)
    feature_names = x.columns.tolist()
    valid = seasons == args.valid_season
    if not valid.any() or valid.all():
        raise ValueError("valid-season must leave non-empty train and validation sets")

    print(f"rows={len(x):,}, features={x.shape[1]}, validation={valid.sum():,}")
    model = make_model(args)
    model.fit(x.loc[~valid], y[~valid])
    raw_p = model.predict_proba(x.loc[valid])[:, 1]
    score_report(y[valid], raw_p, "raw validation")

    # Platt calibration is compact and directly aligned with probability quality.
    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(raw_p.reshape(-1, 1), y[valid])
    calibrated_p = calibrator.predict_proba(raw_p.reshape(-1, 1))[:, 1]
    score_report(y[valid], calibrated_p, "calibrated validation")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(os.path.join(os.path.dirname(args.output) or ".", "validation_hgb.npz"),
                        y=y[valid], raw=raw_p, calibrated=calibrated_p)

    # Also retain an all-season exploratory model. Production uses the temporal
    # model because its calibrator and blend weight were measured on its outputs.
    final_model = make_model(args)
    final_model.set_params(verbose=0)
    final_model.fit(x, y)
    bundle = {
        "model": model,
        "calibrator": calibrator,
        "features": feature_names,
        "valid_season": args.valid_season,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)
    joblib.dump({**bundle, "model": final_model},
                os.path.join(os.path.dirname(args.output) or ".", "model_all_seasons.joblib"),
                compress=3)
    print(f"saved {args.output} ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
