"""Soft pitch-family mixture-of-experts on strict 2021-2024 unified OOF."""
from __future__ import annotations
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.adaptive_gate import build_gate_features

C=np.array([.93505266,-.00520129,.01091677,-.02528331]);I=.0300329767
raw=pd.read_csv('data/train.csv',low_memory=False); latent=pd.read_csv('artifacts/latent_pitch_context.csv',dtype={'pitcher_id':str})
YEARS=(2021,2022,2023,2024)
def skill(y,p):r=y.mean();return 1e5*(1-np.mean((y-np.clip(p,1e-6,1-1e-6))**2)/(r*(1-r)))
def probs(rows):
 r=rows.reset_index(drop=True);hand=r.batter_hand.map({1:'Left',2:'Right'}).fillna(r.batter_hand.astype(str))
 k=pd.DataFrame({'pitcher_id':r.pitcher_id.astype(str),'season':r.season.astype(int),'balls_before':r.balls_before.astype(int),'strikes_before':r.strikes_before.astype(int),'batter_hand':hand})
 z=k.merge(latent,on=['pitcher_id','season','balls_before','strikes_before','batter_hand'],how='left',sort=False)
 fallback=np.c_[pd.to_numeric(r.asof_pitcher_fastball_rate,errors='coerce'),pd.to_numeric(r.asof_pitcher_breaking_rate,errors='coerce'),pd.to_numeric(r.asof_pitcher_offspeed_rate,errors='coerce')]
 P=z[['latent_fastball_prob','latent_breaking_prob','latent_offspeed_prob']].to_numpy(float);P=np.where(np.isfinite(P),P,fallback);P=np.nan_to_num(P,nan=1/3);P=np.clip(P,.01,1);P/=P.sum(1,keepdims=True);return P
def load(y):
 z=np.load(f'artifacts/unified_oof/{y}.npz');r=raw.loc[raw.season.eq(y)].reset_index(drop=True);ps=[z['p_v2'],z['p_v3_55'],z['p_v3_30']];rs=[z['risks'][:,j] for j in range(3)];main=.27358084*ps[0]+.26512224*ps[1]+.46129691*ps[2];old=np.clip(I+np.c_[main,z['risks']]@C,1e-6,1-1e-6);gx=build_gate_features(r,ps,rs,old);pr=probs(r);gx=gx.copy();gx[['mix_fast','mix_break','mix_off']]=pr;return r,z['y'].astype(float),old,gx,pr
parts={y:load(y) for y in YEARS};base={2021:parts[2021][2]}
for target in (2022,2023,2024):
 years=tuple(y for y in YEARS if y<target);x=pd.concat([parts[y][3] for y in years],ignore_index=True);e=np.concatenate([parts[y][1]-parts[y][2] for y in years]);w=np.concatenate([np.full(len(parts[y][1]),.55**((target-1)-y)) for y in years]);m=CatBoostRegressor(iterations=73,depth=3,learning_rate=.025,loss_function='RMSE',l2_leaf_reg=30,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=280033,thread_count=6,allow_writing_files=False,verbose=False);m.fit(x,e,sample_weight=w);base[target]=np.clip(parts[target][2]+m.predict(parts[target][3]),1e-6,1-1e-6)
corr={}
for target in (2022,2023,2024):
 years=tuple(y for y in YEARS if y<target);x=pd.concat([parts[y][3] for y in years],ignore_index=True);e=np.concatenate([parts[y][1]-base[y] for y in years]);dec=np.concatenate([np.full(len(parts[y][1]),.55**((target-1)-y)) for y in years]);pt=np.concatenate([parts[y][4] for y in years]);members=[]
 for j in range(3):
  m=CatBoostRegressor(iterations=180,depth=5,learning_rate=.025,loss_function='RMSE',l2_leaf_reg=80,random_strength=.3,bootstrap_type='Bernoulli',subsample=.8,random_seed=880000+target*3+j,thread_count=6,allow_writing_files=False,verbose=False);m.fit(x,e,sample_weight=dec*(.15+.85*pt[:,j]));members.append(m.predict(parts[target][3]))
 corr[target]=np.sum(np.column_stack(members)*parts[target][4],axis=1)
best=None
for scale in (.02,.05,.08,.1,.15,.2,.3,.4,.5,.75,1):
 gains={y:skill(parts[y][1],base[y]+scale*corr[y])-skill(parts[y][1],base[y]) for y in (2022,2023,2024)};row=(min(gains.values()),sum(gains.values()),scale,gains);best=row if best is None or row[:2]>best[:2] else best
print('BEST',best)
out={};scale=best[2]
for y in (2022,2023,2024):out[f'y{str(y)[-2:]}']=parts[y][1];out[f'base{str(y)[-2:]}']=base[y];out[f'p{str(y)[-2:]}']=np.clip(base[y]+scale*corr[y],1e-6,1-1e-6);out[f'pitcher{str(y)[-2:]}']=parts[y][0].pitcher_id.astype(str).to_numpy()
np.savez_compressed('evaluation/pitch_mixture_crossfit.npz',**out,scale=scale)
