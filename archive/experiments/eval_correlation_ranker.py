"""Test a ranking objective aimed at rho, the only unsaturated leaderboard axis."""
from __future__ import annotations
import os,time
import numpy as np,pandas as pd
from catboost import CatBoostRanker,Pool
from src.catboost_features import CAT_COLS,build_catboost_features
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive

def ceiling(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);v=y.mean()*(1-y.mean());c=np.mean((p-p.mean())*(y-y.mean()));return 1e5*c*c/(p.var()*v)

target=int(os.environ.get('RANK_TARGET','2024'));raw=pd.read_csv('data/train.csv',low_memory=False);hist=raw[raw.season<target];prior=float(hist.control_success.mean())
frame=raw[(raw.season<=target)&(raw.season>=target-2)].reset_index(drop=True);x=build_catboost_features(frame,prior);y=frame.control_success.to_numpy(float);season=frame.season.to_numpy();fit=season<target;te=season==target
# Each season is a query, so the objective learns within-season ordering and is
# invariant to the large yearly intercept drift already solved by probes.
pid_code=pd.factorize(frame.pitcher_id.astype(str),sort=True)[0];query=(season.astype(np.int64)*10000+pid_code);fi=np.flatnonzero(fit);order=fi[np.argsort(query[fi],kind='stable')]
pool=Pool(x.iloc[order],y[order],group_id=query[order],cat_features=CAT_COLS)
m=CatBoostRanker(iterations=60,depth=5,learning_rate=.05,loss_function='YetiRankPairwise',l2_leaf_reg=30,random_strength=.3,bootstrap_type='Bernoulli',subsample=.8,random_seed=2910000+target,thread_count=6,allow_writing_files=False,verbose=False)
t=time.time();m.fit(pool);rank=m.predict(x[te]);yt=y[te]
# Put arbitrary rank scores on the incumbent mean/scale before blending; affine
# ceiling itself is invariant to this transform.
rank=(rank-rank.mean())/(rank.std()+1e-9)
# Use the identical strict OOF gate lineage for every target year.
strict=np.load('evaluation/season_state_crossfit.npz')
base=strict[f'base{str(target)[-2:]}'].astype(float)
rank=base.mean()+base.std()*rank
best=max((ceiling(yt,(1-a)*base+a*rank),a) for a in np.linspace(0,1,41))
print('target',target,'seconds',round(time.time()-t,1),'base ceiling',ceiling(yt,base),'rank ceiling',ceiling(yt,rank),'best blend',best,'raw base',skill(yt,base),flush=True)
np.savez_compressed(f'evaluation/correlation_ranker_{target}.npz',y=yt,base=base,rank=rank,best_weight=best[1])
