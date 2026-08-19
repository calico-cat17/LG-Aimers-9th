import numpy as np,pandas as pd
from src.season_delta_features import build_snapshots,attach_season_delta
R='old/experiments';raw=pd.read_csv('data/train.csv',low_memory=False);C=np.array([.93505266,-.00520129,.01091677,-.02528331]);I=.0300329767
def load(yr):
 mask=raw.season.eq(yr);r=raw.loc[mask].reset_index(drop=True);hist=raw.loc[raw.season<yr];prior=hist.control_success.mean();snap=build_snapshots(hist);d=attach_season_delta(pd.DataFrame(index=np.arange(len(r))),r,snap,prior)
 if yr<2024:y=np.load(f'{R}/result_dirs/v2_multisplit/{yr}.npz')['y'];a=np.load(f'{R}/result_dirs/v2_multisplit/{yr}.npz')['p'];b=np.load(f'{R}/result_dirs/v3_multisplit/{yr}.npz')['p'];c=np.load(f'{R}/result_dirs/v3_multisplit/{yr}_d30.npz')['p'];Q=np.load(f'{R}/result_dirs/v3_subtype_oof/{yr}.npz')['risks']
 else:z=np.load(f'{R}/v3_subtypes_2024.npz');y=z['y'];Q=z['risks'];a=np.load(f'{R}/result_dirs/v2_multisplit/2024.npz')['p'];b=np.load(f'{R}/model_v3_2024.npz')['p'];c=np.load(f'{R}/model_v3_decay_30.npz')['p']
 main=.27358084*a+.26512224*b+.46129691*c;p=np.clip(I+np.c_[main,Q]@C,1e-6,1-1e-6)
 n=r.asof_pitcher_n.fillna(0).clip(lower=0);career=r.asof_pitcher_success_rate.fillna(prior);recent=r[['asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate']].apply(pd.to_numeric,errors='coerce');std=recent.std(1).fillna(.15).clip(0,.5);strength=(55+220*std+40/(1+np.log1p(n))).clip(50,180);career_base=(career*n+prior*strength)/(n+strength);sn=d.season_pitcher_n.clip(lower=0);season=(d.season_success_rate_raw*sn+prior*30)/(sn+30);rel=sn/(sn+80);gap=(season-career_base).to_numpy();rg=(recent.mean(1).fillna(prior)-career_base).to_numpy()
 X=np.c_[gap,gap*rel.to_numpy(),gap*np.sqrt(rel.to_numpy()),rg,rg/(1+5*std.to_numpy()),gap*rg,np.log1p(sn.to_numpy())*gap]
 return y,p,np.nan_to_num(X)
parts={q:load(q) for q in (2022,2023,2024)}
def fit(yrs,a):
 X=np.concatenate([parts[q][2] for q in yrs]);e=np.concatenate([parts[q][0]-parts[q][1] for q in yrs]);mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1;Z=(X-mu)/sd;c=np.linalg.solve(Z.T@Z+a*np.eye(Z.shape[1]),Z.T@e);return mu,sd,c
def sc(y,p):r=y.mean();return 1e5*(1-np.mean((y-p)**2)/(r*(1-r)))
for a in [1,10,100,1000,10000]:
 for yr,tr in [(2023,[2022]),(2024,[2022,2023])]:
  mu,sd,c=fit(tr,a);y,p,X=parts[yr];corr=(X-mu)/sd@c;best=max((sc(y,np.clip(p+s*corr,1e-6,1-1e-6)),s) for s in [.25,.5,.75,1]);print(a,yr,'base',sc(y,p),'best',best,'coef',c,flush=True)
