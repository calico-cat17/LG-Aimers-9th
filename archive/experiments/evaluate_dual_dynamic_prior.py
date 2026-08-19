"""Row-local 2025 environment estimator from pitcher and batter season deltas."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from evaluate_crossfit_psych_profiles import load_predictions, skill
from evaluate_psych_residual_on_adaptive import fit_adaptive
from src.season_delta_features import build_snapshots, attach_season_delta
from src.season_history_v3 import build_entity_snapshots, attach_entity_season


raw = pd.read_csv("data/train.csv", low_memory=False)
loaded = {y: load_predictions(raw, y) for y in (2022, 2023, 2024)}
r22,y22,g22,o22=loaded[2022];r23,y23,g23,o23=loaded[2023];r24,y24,g24,o24=loaded[2024]
gate=fit_adaptive(g22,y22,o22,2340023)
base={2023:np.clip(o23+gate.predict(g23),1e-6,1-1e-6),2024:np.load("old/experiments/adaptive_gate_2024.npz")["p"]}
rows={2023:r23,2024:r24};ys={2023:y23,2024:y24}


def season_signal(year: int, alpha: float):
    r=rows[year].reset_index(drop=True);hist=raw.loc[raw.season.lt(year)];prior=float(hist.control_success.mean())
    pdlt=attach_season_delta(pd.DataFrame(index=r.index),r,build_snapshots(hist),prior)
    bs=build_entity_snapshots(hist,"batter_id","asof_batter_n",["asof_batter_success_rate"],"control_success")
    bdlt=attach_entity_season(pd.DataFrame(index=r.index),r,bs,"batter_id","asof_batter_n",["asof_batter_success_rate"],"batter_season",prior)
    def one(n_total,career,sn,sraw):
        n_total=pd.to_numeric(n_total,errors="coerce").fillna(0).to_numpy(float);career=pd.to_numeric(career,errors="coerce").fillna(prior).to_numpy(float);sn=np.asarray(sn,float);sraw=np.asarray(sraw,float)
        pre_n=np.maximum(n_total-sn,0);pre_count=np.maximum(n_total*career-sn*sraw,0);pre=(pre_count+prior*80)/(pre_n+80)
        season=(sn*sraw+pre*alpha)/(sn+alpha);rel=sn/(sn+alpha);return season-pre,rel,pre,season
    pg,pr,pp,ps=one(r.asof_pitcher_n,r.asof_pitcher_success_rate,pdlt.season_pitcher_n,pdlt.season_success_rate_raw)
    bg,br,bp,bsx=one(r.asof_batter_n,r.asof_batter_success_rate,bdlt.batter_season_n,bdlt.batter_season_success_rate_raw)
    return pg,pr,bg,br


records=[]
for alpha in (20.,40.,80.,120.,200.,400.):
 sig={y:season_signal(y,alpha) for y in rows}
 for batter_weight in (.25,.5,.75,1.,1.5,2.,3.):
  for agreement_floor in (0.,.25,.5,1.):
   corr={}
   for y,(pg,pr,bg,br) in sig.items():
    # Independent pitcher and batter views of the same league environment.
    den=pr+batter_weight*br+1e-8;mean=(pr*pg+batter_weight*br*bg)/den
    agreement=np.exp(-np.abs(pg-bg)/.04);confidence=(agreement_floor+agreement)/(agreement_floor+1)
    corr[y]=mean*confidence*np.sqrt(np.clip(den/(1+batter_weight),0,1))
   for scale in (.1,.2,.35,.5,.75,1.,1.25,1.5,2.):
    pred={y:np.clip(base[y]+scale*corr[y],1e-6,1-1e-6) for y in rows}
    gains={y:skill(ys[y],pred[y])-skill(ys[y],base[y]) for y in rows}
    records.append((min(gains.values()),sum(gains.values())/2,alpha,batter_weight,agreement_floor,scale,pred,gains,corr))
best=max(records,key=lambda z:(z[0],z[1]));worst,avg,alpha,bw,floor,scale,pred,gains,corr=best
print("best",dict(worst_gain=worst,mean_gain=avg,alpha=alpha,batter_weight=bw,agreement_floor=floor,scale=scale,gains=gains,scores={y:skill(ys[y],pred[y]) for y in rows}),flush=True)
np.savez_compressed("evaluation/dual_dynamic_prior.npz",y23=y23,p23=pred[2023],base23=base[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=pred[2024],base24=base[2024],pitcher24=r24.pitcher_id.to_numpy(),corr23=corr[2023],corr24=corr[2024],alpha=alpha,batter_weight=bw,agreement_floor=floor,scale=scale)
json.dump(dict(alpha=alpha,batter_weight=bw,agreement_floor=floor,scale=scale),open("evaluation/dual_dynamic_prior_config.json","w"),indent=2)
