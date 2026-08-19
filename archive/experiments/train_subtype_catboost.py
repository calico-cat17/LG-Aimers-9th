"""Privileged-label specialists: predict three recovered failure causes."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from src.catboost_features import CAT_COLS, build_catboost_features
from src.features import TARGET_COL
from src.label_recovery import recover_failure_labels


def main():
    out_dir = "model_subtype_catboost"; os.makedirs(out_dir, exist_ok=True)
    started=time.time(); raw=pd.read_csv("data/train.csv",encoding="utf-8-sig",low_memory=False)
    valid=raw["season"].to_numpy()==2024; y=raw[TARGET_COL].to_numpy(np.float32)
    prior=float(y[~valid].mean()); x=build_catboost_features(raw,prior)
    subtype, mask=recover_failure_labels(raw); models=[]; vp=[]
    names=["middle","wild","reverse"]
    train_ok=(~valid)&mask.astype(bool); valid_ok=valid&mask.astype(bool)
    for i,name in enumerate(names):
        model=CatBoostClassifier(
            iterations=1000,depth=8,learning_rate=0.05,loss_function="Logloss",
            eval_metric="Logloss",l2_leaf_reg=6,random_strength=0.5,
            bootstrap_type="Bernoulli",subsample=0.85,one_hot_max_size=16,
            random_seed=100+i,thread_count=6,allow_writing_files=False,verbose=100)
        model.fit(x.loc[train_ok],subtype[train_ok,i],cat_features=CAT_COLS,
                  eval_set=(x.loc[valid_ok],subtype[valid_ok,i]),
                  early_stopping_rounds=120,use_best_model=True)
        path=f"subtype_{name}.cbm"; model.save_model(os.path.join(out_dir,path)); models.append(path)
        vp.append(model.predict_proba(x.loc[valid])[:,1])
    risks=np.column_stack(vp)
    # Causes overlap, so use noisy-OR then calibrate its total failure probability.
    failure=1-np.prod(1-risks,axis=1); raw_success=np.clip(1-failure,1e-6,1-1e-6)
    cal=LogisticRegression(C=1.0,solver="lbfgs").fit(raw_success.reshape(-1,1),y[valid])
    pred=cal.predict_proba(raw_success.reshape(-1,1))[:,1]
    brier=lambda p:float(np.mean((p-y[valid])**2))
    print(f"subtype_union raw={brier(raw_success):.7f} calibrated={brier(pred):.7f}")
    np.savez_compressed(os.path.join(out_dir,"validation_subtype.npz"),y=y[valid],raw=raw_success,calibrated=pred,risks=risks)
    meta={"models":models,"names":names,"prior":prior,"calibration_coef":float(cal.coef_[0,0]),
          "calibration_intercept":float(cal.intercept_[0])}
    with open(os.path.join(out_dir,"subtype_meta.json"),"w") as f:json.dump(meta,f,indent=2)
    print(f"saved in {time.time()-started:.1f}s")


if __name__=="__main__":main()
