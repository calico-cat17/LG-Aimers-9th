"""Refit the temporally-selected CatBoost recipe on all 2019-2024 labels."""

import json, os, time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.catboost_features import CAT_COLS, build_catboost_features, attach_trackman_features
from src.features import TARGET_COL


def main():
    out="model_final";os.makedirs(out,exist_ok=True);t=time.time()
    raw=pd.read_csv("data/train.csv",encoding="utf-8-sig",low_memory=False)
    y=raw[TARGET_COL].to_numpy(np.float32);prior=float(y.mean())
    x=attach_trackman_features(build_catboost_features(raw,prior),"model/trackman_prior_features.csv")
    weight=np.power(.8,2024-raw.season.to_numpy())
    model=CatBoostRegressor(iterations=160,depth=8,learning_rate=.05,loss_function="RMSE",
        l2_leaf_reg=5,random_strength=.5,bootstrap_type="Bernoulli",subsample=.85,
        one_hot_max_size=16,random_seed=260810,thread_count=6,allow_writing_files=False,verbose=40)
    model.fit(x,y,sample_weight=weight,cat_features=CAT_COLS)
    model.save_model(os.path.join(out,"catboost_full.cbm"))
    temporal=json.load(open("model/catboost_track_meta.json"))
    meta={"prior":prior,"cat_cols":CAT_COLS,"iterations":160,
          "calibration_coef":temporal["calibration_coef"],
          "calibration_intercept":temporal["calibration_intercept"],
          "calibration_source":"2019-2023 -> 2024 temporal holdout"}
    with open(os.path.join(out,"catboost_full_meta.json"),"w") as f:json.dump(meta,f,indent=2)
    print("saved",time.time()-t)

if __name__=="__main__":main()
