"""Forward-2024 evaluation of Futures-only failure-cause experts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from evaluate_crossfit_psych_profiles import COEFFICIENTS, INTERCEPT, skill
from src.adaptive_gate import build_gate_features
from src.label_recovery import recover_failure_labels
from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots


def main() -> None:
    raw=pd.read_csv("data/train.csv",low_memory=False)
    y=raw.control_success.to_numpy(np.float32); season=raw.season.to_numpy(); valid=season==2024
    hist=raw.loc[~valid]; prior=float(y[~valid].mean())
    ps=build_snapshots(hist)
    bs=build_entity_snapshots(hist,"batter_id","asof_batter_n",
        ["asof_batter_success_rate","asof_batter_middle_rate"],"control_success")
    ms=build_entity_snapshots(hist,"pitcher_id","asof_pitcher_pitchmix_n",
        ["asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"])
    x,_=build_v3_features(raw,prior,ps,bs,ms,"model/trackman_prior_features.csv")
    labels,recovered=recover_failure_labels(raw)
    f=raw.game_type.eq("F").to_numpy(); fit=(~valid)&f&recovered.astype(bool)
    vrows=raw.loc[valid].reset_index(drop=True); vf=vrows.game_type.eq("F").to_numpy()

    old=np.load("old/experiments/v3_subtypes_2024.npz")["risks"].astype(float)
    risks=old.copy(); names=("middle","wild","reverse"); iterations=(100,190,230)
    for i,(name,iters) in enumerate(zip(names,iterations)):
        model=CatBoostClassifier(iterations=iters,depth=7,learning_rate=.04,
            loss_function="Logloss",l2_leaf_reg=20,random_strength=.4,
            bootstrap_type="Bernoulli",subsample=.85,one_hot_max_size=16,
            random_seed=928100+i,thread_count=6,allow_writing_files=False,verbose=False)
        weights=np.power(.30,2023-season[fit])
        model.fit(x.loc[fit],labels[fit,i],sample_weight=weights,cat_features=CAT_V2)
        risks[vf,i]=model.predict_proba(x.loc[valid].loc[vf])[:,1]
        print(name,"trained",fit.sum(),flush=True)

    p2=np.load("old/experiments/result_dirs/v2_multisplit/2024.npz")["p"]
    p55=np.load("old/experiments/model_v3_2024.npz")["p"]
    p30=np.load("old/experiments/model_v3_decay_30.npz")["p"]
    predictions=[p2,p55,p30]
    main=.27358084*p2+.26512224*p55+.46129691*p30
    old_stack=np.clip(INTERCEPT+np.column_stack([main,old])@COEFFICIENTS,1e-6,1-1e-6)
    new_stack=np.clip(INTERCEPT+np.column_stack([main,risks])@COEFFICIENTS,1e-6,1-1e-6)
    gate=CatBoostRegressor();gate.load_model("model_hierarchical_stack/adaptive_gate.cbm")
    old_x=build_gate_features(vrows,predictions,[old[:,i] for i in range(3)],old_stack)
    new_x=build_gate_features(vrows,predictions,[risks[:,i] for i in range(3)],new_stack)
    old_p=np.clip(old_stack+gate.predict(old_x),1e-6,1-1e-6)
    yv=y[valid]
    print("base",skill(yv,old_p))
    for blend in (.25,.5,.75,1.0):
        mixed=old+blend*(risks-old)
        stack=np.clip(INTERCEPT+np.column_stack([main,mixed])@COEFFICIENTS,1e-6,1-1e-6)
        gx=build_gate_features(vrows,predictions,[mixed[:,i] for i in range(3)],stack)
        p=np.clip(stack+gate.predict(gx),1e-6,1-1e-6)
        print("blend",blend,"score",skill(yv,p),"gain",skill(yv,p)-skill(yv,old_p))
    np.savez_compressed("f_subtype_experts_2024.npz",y=yv,risks=risks,old_risks=old)


if __name__=="__main__":main()
