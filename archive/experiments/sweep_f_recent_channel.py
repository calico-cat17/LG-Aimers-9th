"""Fast validation sweep for replacing only the F v2 residual channel."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from evaluate_crossfit_psych_profiles import COEFFICIENTS, INTERCEPT, skill
from src.adaptive_gate import build_gate_features

raw = pd.read_csv("data/train.csv", low_memory=False)
va = raw.season.to_numpy() == 2024
rows = raw.loc[va].reset_index(drop=True)
y = raw.loc[va, "control_success"].to_numpy(np.float32)
old = [
    np.load("old/experiments/result_dirs/v2_multisplit/2024.npz")["p"],
    np.load("old/experiments/model_v3_2024.npz")["p"],
    np.load("old/experiments/model_v3_decay_30.npz")["p"],
]
risks = np.load("old/experiments/v3_subtypes_2024.npz")["risks"]
full = np.load("full_f_residual_stack_2024.npz")
recent = np.load("f_2023_only_2024.npz")
mask = full["mask_f"].astype(bool)
gate = CatBoostRegressor()
gate.load_model("model_hierarchical_stack/adaptive_gate.cbm")

def final(preds):
    main = .27358084 * preds[0] + .26512224 * preds[1] + .46129691 * preds[2]
    stacked = np.clip(INTERCEPT + np.c_[main, risks] @ COEFFICIENTS, 1e-6, 1 - 1e-6)
    gf = build_gate_features(rows, preds, [risks[:, i] for i in range(3)], stacked)
    return np.clip(stacked + gate.predict(gf), 1e-6, 1 - 1e-6)

best = (-1.0, None)
for w in np.linspace(-0.5, 2.0, 26):
    preds = [p.copy() for p in old]
    preds[0][mask] = (1.0 - w) * full["p0"] + w * recent["p0"]
    preds[1][mask] = full["p1"]
    preds[2][mask] = full["p2"]
    score = skill(y, final(preds))
    print(f"w={w:5.2f} score={score:12.6f}")
    if score > best[0]:
        best = (score, w)
print(f"BEST score={best[0]:.8f} w={best[1]:.3f}")
