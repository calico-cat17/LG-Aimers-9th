"""2024 OOF controller over diverse pre-2024 base-model predictions."""

import json, os, time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold

from src.catboost_features import CAT_COLS, build_catboost_features, attach_trackman_features
from src.features import TARGET_COL


BASE_FILES = {
    "pred_hgb": "model/validation_hgb.npz",
    "pred_deep": "model/validation_deep.npz",
    "pred_cat_track": "model_catboost_trackman/validation_catboost.npz",
    "pred_subtype": "model_subtype_catboost/validation_subtype.npz",
}


def make_model(seed, iterations=600):
    return CatBoostRegressor(iterations=iterations,depth=7,learning_rate=.035,
        loss_function="RMSE",eval_metric="RMSE",l2_leaf_reg=8,random_strength=.3,
        bootstrap_type="Bernoulli",subsample=.85,one_hot_max_size=16,
        random_seed=seed,thread_count=6,allow_writing_files=False,verbose=False)


def main():
    out="model_meta";os.makedirs(out,exist_ok=True);t=time.time()
    raw=pd.read_csv("data/train.csv",encoding="utf-8-sig",low_memory=False)
    raw=raw.loc[raw.season==2024].reset_index(drop=True);y=raw[TARGET_COL].to_numpy(np.float32)
    # Prior is saved by the matching base model and is not fitted on 2024.
    base_meta=json.load(open("model_catboost_trackman/catboost_meta.json")); prior=base_meta["prior"]
    x=attach_trackman_features(build_catboost_features(raw,prior),"artifacts/trackman_prior_features.csv")
    for name,path in BASE_FILES.items():
        z=np.load(path); assert np.array_equal(z["y"],y); x[name]=z["calibrated"]
    skf=StratifiedKFold(5,shuffle=True,random_state=260810);oof=np.zeros(len(y));best=[]
    for fold,(ti,vi) in enumerate(skf.split(x,y)):
        m=make_model(260810+fold)
        m.fit(x.iloc[ti],y[ti],cat_features=CAT_COLS,eval_set=(x.iloc[vi],y[vi]),
              early_stopping_rounds=80,use_best_model=True,verbose=100)
        oof[vi]=np.clip(m.predict(x.iloc[vi]),1e-6,1-1e-6);best.append(m.get_best_iteration()+1)
        print("fold",fold,"best",best[-1],"brier",np.mean((oof[vi]-y[vi])**2))
    brier=float(np.mean((oof-y)**2));base=float(y.mean()*(1-y.mean()));score=max(0,100000*(1-brier/base))
    print(f"META OOF brier={brier:.7f} score={score:.2f}")
    final_iter=max(50,int(np.median(best)));final=make_model(260900,final_iter)
    final.fit(x,y,cat_features=CAT_COLS,verbose=100);final.save_model(os.path.join(out,"meta_controller.cbm"))
    np.savez_compressed(os.path.join(out,"validation_meta_oof.npz"),y=y,pred=oof)
    meta={"prior":prior,"base_files":BASE_FILES,"iterations":final_iter,"cat_cols":CAT_COLS,
          "trackman_file":"trackman_prior_features.csv","oof_brier":brier,"oof_score":score}
    with open(os.path.join(out,"meta_controller.json"),"w") as f:json.dump(meta,f,indent=2)
    print("saved in",time.time()-t)

if __name__=="__main__":main()
