"""Rolling-origin search for recency decay; never mixes future seasons."""

import os,time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.catboost_features import CAT_COLS,build_catboost_features,attach_trackman_features
from src.features import TARGET_COL


def main():
    out="rolling_results";os.makedirs(out,exist_ok=True);raw=pd.read_csv("data/train.csv",encoding="utf-8-sig",low_memory=False)
    y=raw[TARGET_COL].to_numpy(np.float32); rows=[]
    for valid_year in [2023,2024]:
        train=raw.season.to_numpy()<valid_year; valid=raw.season.to_numpy()==valid_year
        prior=float(y[train].mean()); x=attach_trackman_features(build_catboost_features(raw,prior),"model/trackman_prior_features.csv")
        for decay in [.5,.65,.8,.95]:
            t=time.time();w=np.power(decay,(valid_year-1)-raw.loc[train,"season"].to_numpy())
            m=CatBoostRegressor(iterations=600,depth=8,learning_rate=.05,loss_function="RMSE",eval_metric="RMSE",
                l2_leaf_reg=5,random_strength=.5,bootstrap_type="Bernoulli",subsample=.85,one_hot_max_size=16,
                random_seed=260000+valid_year+int(decay*100),thread_count=6,allow_writing_files=False,verbose=False)
            m.fit(x.loc[train],y[train],sample_weight=w,cat_features=CAT_COLS,eval_set=(x.loc[valid],y[valid]),
                  early_stopping_rounds=100,use_best_model=True)
            p=np.clip(m.predict(x.loc[valid]),1e-6,1-1e-6);b=float(np.mean((p-y[valid])**2));base=float(y[valid].mean()*(1-y[valid].mean()))
            score=max(0,100000*(1-b/base));row={"year":valid_year,"decay":decay,"iteration":m.get_best_iteration()+1,
                "brier":b,"score":score,"pred_mean":float(p.mean()),"target_mean":float(y[valid].mean()),"seconds":time.time()-t}
            rows.append(row);print(row,flush=True)
            np.savez_compressed(f"{out}/v{valid_year}_d{decay}.npz",y=y[valid],pred=p)
        del x
    pd.DataFrame(rows).to_csv(f"{out}/summary.csv",index=False);print(pd.DataFrame(rows))

if __name__=="__main__":main()
