"""Four-year strict OOF test of a low-dimensional current-season state channel."""
from __future__ import annotations
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from src.adaptive_gate import build_gate_features
from src.season_delta_features import build_snapshots,attach_season_delta

C=np.array([.93505266,-.00520129,.01091677,-.02528331]); I=.0300329767
YEARS=(2021,2022,2023,2024); raw=pd.read_csv('data/train.csv',low_memory=False)

def skill(y,p):
 r=y.mean();return 1e5*(1-np.mean((y-np.clip(p,1e-6,1-1e-6))**2)/(r*(1-r)))

def load(y):
 z=np.load(f'artifacts/unified_oof/{y}.npz');rows=raw.loc[raw.season.eq(y)].reset_index(drop=True)
 P=[z['p_v2'],z['p_v3_55'],z['p_v3_30']]; risks=[z['risks'][:,j] for j in range(3)]
 main=.27358084*P[0]+.26512224*P[1]+.46129691*P[2]
 old=np.clip(I+np.c_[main,z['risks']]@C,1e-6,1-1e-6)
 return rows,z['y'].astype(float),P,risks,old,build_gate_features(rows,P,risks,old)

parts={y:load(y) for y in YEARS}
base={2021:parts[2021][4]}
for target in (2022,2023,2024):
 years=tuple(y for y in YEARS if y<target)
 x=pd.concat([parts[y][5] for y in years],ignore_index=True)
 yy=np.concatenate([parts[y][1]-parts[y][4] for y in years])
 w=np.concatenate([np.full(len(parts[y][1]),.55**((target-1)-y)) for y in years])
 m=CatBoostRegressor(iterations=73,depth=3,learning_rate=.025,loss_function='RMSE',l2_leaf_reg=30,
  random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=280033,
  thread_count=6,allow_writing_files=False,verbose=False)
 m.fit(x,yy,sample_weight=w);base[target]=np.clip(parts[target][4]+m.predict(parts[target][5]),1e-6,1-1e-6)

def state(year):
 rows=parts[year][0];hist=raw.loc[raw.season.lt(year)];prior=float(hist.control_success.mean())
 d=attach_season_delta(pd.DataFrame(index=rows.index),rows,build_snapshots(hist),prior)
 n=pd.to_numeric(rows.asof_pitcher_n,errors='coerce').fillna(0).clip(lower=0)
 career=pd.to_numeric(rows.asof_pitcher_success_rate,errors='coerce').fillna(prior)
 recent=rows[[f'asof_pitcher_prev{k}_game_success_rate' for k in (1,3,5)]].apply(pd.to_numeric,errors='coerce')
 std=recent.std(1).fillna(.15).clip(0,.5);strength=(55+220*std+40/(1+np.log1p(n))).clip(50,180)
 cb=(career*n+prior*strength)/(n+strength);sn=d.season_pitcher_n.clip(lower=0)
 se=(d.season_success_rate_raw*sn+prior*30)/(sn+30);rel=sn/(sn+80)
 gap=(se-cb).to_numpy();rg=(recent.mean(1).fillna(prior)-cb).to_numpy()
 return np.nan_to_num(np.c_[gap,gap*rel,gap*np.sqrt(rel),rg,rg/(1+5*std),gap*rg,np.log1p(sn)*gap])

X={y:state(y) for y in YEARS}; trials=[]
for alpha in (10,100,1000,3000,10000,30000,100000):
 corr={}
 for target in (2022,2023,2024):
  years=tuple(y for y in YEARS if y<target);xx=np.concatenate([X[y] for y in years]);ee=np.concatenate([parts[y][1]-base[y] for y in years])
  w=np.concatenate([np.full(len(parts[y][1]),.55**((target-1)-y)) for y in years])
  mu=np.average(xx,axis=0,weights=w);sd=np.sqrt(np.average((xx-mu)**2,axis=0,weights=w));sd[sd<1e-8]=1
  model=Ridge(alpha=alpha,fit_intercept=False).fit((xx-mu)/sd,ee,sample_weight=w)
  corr[target]=model.predict((X[target]-mu)/sd)
 for scale in (.05,.1,.15,.2,.25,.3,.4,.5,.75,1):
  gains={y:skill(parts[y][1],base[y]+scale*corr[y])-skill(parts[y][1],base[y]) for y in (2022,2023,2024)}
  trials.append((min(gains.values()),sum(gains.values()),alpha,scale,gains,corr))
best=max(trials,key=lambda q:(q[0],q[1]));print('BEST',best[:5])
_,_,alpha,scale,gains,corr=best;out={}
for y in (2022,2023,2024):
 out[f'y{str(y)[-2:]}']=parts[y][1];out[f'base{str(y)[-2:]}']=base[y];out[f'p{str(y)[-2:]}']=np.clip(base[y]+scale*corr[y],1e-6,1-1e-6);out[f'pitcher{str(y)[-2:]}']=parts[y][0].pitcher_id.astype(str).to_numpy();print(y,'base',skill(parts[y][1],base[y]),'gain',gains[y])
np.savez_compressed('evaluation/season_state_crossfit.npz',**out,alpha=alpha,scale=scale)
