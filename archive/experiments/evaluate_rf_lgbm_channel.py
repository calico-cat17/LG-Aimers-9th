"""Independent R/F LightGBM residual channel on strict forward OOF predictions.

This is deliberately different from the production CatBoost stack: it uses
numeric row-local inputs only, leaf-wise boosting and separately fitted R/F
models. 2022 OOF trains the 2023 meta model; 2022+2023 OOF train 2024.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from evaluate_crossfit_psych_profiles import load_predictions, skill
from evaluate_psych_residual_on_adaptive import fit_adaptive


DROP = {"control_success", "pitch_id", "game_id", "pitcher_id", "batter_id"}


def numeric_frame(rows: pd.DataFrame, base: np.ndarray) -> pd.DataFrame:
    x = rows.select_dtypes(include=[np.number, "bool"]).drop(
        columns=[c for c in DROP if c in rows], errors="ignore"
    ).copy()
    # IDs encoded as integers are not stable baseball signals.
    x = x.drop(columns=[c for c in x if c.endswith("_id")], errors="ignore")
    x["base_prediction"] = np.asarray(base, dtype=np.float32)
    x["count_pressure"] = (
        pd.to_numeric(rows.balls_before, errors="coerce").fillna(0) * 3
        + pd.to_numeric(rows.strikes_before, errors="coerce").fillna(0)
    )
    x["runner_out_pressure"] = (
        pd.to_numeric(rows.num_runners_on, errors="coerce").fillna(0) * 3
        + pd.to_numeric(rows.outs_before, errors="coerce").fillna(0)
    )
    return x.replace([np.inf, -np.inf], np.nan).astype("float32")


def fit_predict(xtr, ytr, xva, yva, seed):
    med = xtr.median(numeric_only=True)
    xtr = xtr.fillna(med).fillna(0)
    xva = xva.reindex(columns=xtr.columns).fillna(med).fillna(0)
    model = LGBMRegressor(
        objective="regression_l2", n_estimators=1200, learning_rate=.025,
        num_leaves=31, max_depth=-1, min_child_samples=500,
        subsample=.8, colsample_bytree=.72, reg_alpha=1.0, reg_lambda=30.0,
        random_state=seed, n_jobs=6, verbosity=-1,
    )
    model.fit(xtr, ytr, eval_set=[(xva, yva)], eval_metric="l2",
              callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    return model.predict(xva), model.best_iteration_


def main():
    raw = pd.read_csv("data/train.csv", low_memory=False)
    loaded = {yr: load_predictions(raw, yr) for yr in (2022, 2023, 2024)}
    r22,y22,g22,b22=loaded[2022]; r23,y23,g23,b23=loaded[2023]; r24,y24,_,_=loaded[2024]
    gate23=fit_adaptive(g22,y22,b22,520023)
    a22=b22
    a23=np.clip(b23+gate23.predict(g23),1e-6,1-1e-6)
    a24=np.load("old/experiments/adaptive_gate_2024.npz")["p"].astype(float)
    data={2022:(r22,y22,a22),2023:(r23,y23,a23),2024:(r24,y24,a24)}
    saved={}
    for valid, train_years in ((2023,(2022,)),(2024,(2022,2023))):
        rv,yv,bv=data[valid]
        pred=np.zeros(len(yv),dtype=float)
        for league in ("R","F"):
            train_parts=[]; targets=[]
            for yr in train_years:
                rr,yy,bb=data[yr]; m=rr.game_type.astype(str).eq(league).to_numpy()
                train_parts.append(numeric_frame(rr.loc[m].reset_index(drop=True),bb[m]))
                targets.append(yy[m]-bb[m])
            xv_mask=rv.game_type.astype(str).eq(league).to_numpy()
            xtr=pd.concat(train_parts,ignore_index=True); yt=np.concatenate(targets)
            xv=numeric_frame(rv.loc[xv_mask].reset_index(drop=True),bv[xv_mask])
            corr,it=fit_predict(xtr,yt,xv,yv[xv_mask]-bv[xv_mask],826000+valid+(league=="F"))
            pred[xv_mask]=corr
            print(f"valid={valid} league={league} train={len(yt)} valid_n={xv_mask.sum()} iter={it}",flush=True)
        base_score=skill(yv,bv)
        print(f"valid={valid} base={base_score:.6f}")
        for scale in (.05,.10,.15,.20,.30,.40,.50,.75,1.0):
            p=np.clip(bv+scale*pred,1e-6,1-1e-6)
            print(f"  scale={scale:.2f} score={skill(yv,p):.6f} gain={skill(yv,p)-base_score:+.6f}")
        saved[valid]=(yv,bv,pred,rv.pitcher_id.astype(str).to_numpy())
    out={}
    # Use a deliberately fixed conservative scale; report script can compare it
    # identically across both forward years without tuning on 2024 alone.
    scale=.20
    for yr,(yy,bb,cc,pid) in saved.items():
        out[f"y{str(yr)[-2:]}"]=yy; out[f"base{str(yr)[-2:]}"]=bb
        out[f"p{str(yr)[-2:]}"]=np.clip(bb+scale*cc,1e-6,1-1e-6)
        out[f"pitcher{str(yr)[-2:]}"]=pid
    np.savez_compressed("evaluation/rf_lgbm_channel.npz",**out)


if __name__ == "__main__":
    main()
