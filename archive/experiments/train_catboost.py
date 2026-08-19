"""Train CatBoost ordered boosting on a temporal holdout."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression

from src.catboost_features import CAT_COLS, build_catboost_features, attach_trackman_features
from src.features import TARGET_COL


def brier(y, p): return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--model-dir", default="model_catboost")
    ap.add_argument("--iterations", type=int, default=1800)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", choices=["classification", "regression"], default="classification")
    ap.add_argument("--recency-decay", type=float, default=1.0)
    ap.add_argument("--trackman-features", default="")
    args = ap.parse_args(); os.makedirs(args.model_dir, exist_ok=True)
    started = time.time(); raw = pd.read_csv(args.data, encoding="utf-8-sig", low_memory=False)
    valid = raw["season"].to_numpy() == 2024
    y = raw[TARGET_COL].to_numpy(np.int8)
    prior = float(y[~valid].mean())
    x = build_catboost_features(raw, prior)
    if args.trackman_features:
        x = attach_trackman_features(x, args.trackman_features)
    model_cls = CatBoostClassifier if args.task == "classification" else CatBoostRegressor
    losses = ({"loss_function": "Logloss", "eval_metric": "BrierScore"}
              if args.task == "classification" else
              {"loss_function": "RMSE", "eval_metric": "RMSE"})
    model = model_cls(
        iterations=args.iterations, depth=args.depth, learning_rate=0.05,
        **losses, l2_leaf_reg=5.0,
        random_strength=0.5, bootstrap_type="Bernoulli", subsample=0.85,
        one_hot_max_size=16, random_seed=args.seed, thread_count=6,
        allow_writing_files=False, verbose=100,
    )
    seasons = raw["season"].to_numpy()
    train_weight = np.power(args.recency_decay, 2023 - seasons[~valid])
    model.fit(x.loc[~valid], y[~valid], sample_weight=train_weight, cat_features=CAT_COLS,
              eval_set=(x.loc[valid], y[valid]), early_stopping_rounds=150,
              use_best_model=True)
    raw_p = (model.predict_proba(x.loc[valid])[:, 1] if args.task == "classification"
             else np.clip(model.predict(x.loc[valid]), 1e-6, 1-1e-6))
    cal = LogisticRegression(C=1.0, solver="lbfgs").fit(raw_p.reshape(-1,1), y[valid])
    calibrated = cal.predict_proba(raw_p.reshape(-1,1))[:,1]
    print(f"catboost raw={brier(y[valid],raw_p):.7f} calibrated={brier(y[valid],calibrated):.7f}")
    model.save_model(os.path.join(args.model_dir, "catboost.cbm"))
    np.savez_compressed(os.path.join(args.model_dir, "validation_catboost.npz"),
                        y=y[valid], raw=raw_p, calibrated=calibrated)
    meta = {"prior": prior, "calibration_coef": float(cal.coef_[0,0]),
            "calibration_intercept": float(cal.intercept_[0]), "cat_cols": CAT_COLS,
            "best_iteration": model.get_best_iteration(), "task": args.task,
            "recency_decay": args.recency_decay}
    meta["uses_trackman"] = bool(args.trackman_features)
    with open(os.path.join(args.model_dir, "catboost_meta.json"), "w") as f: json.dump(meta,f,indent=2)
    print(f"saved in {time.time()-started:.1f}s")


if __name__ == "__main__": main()
