"""Evaluate the original stack/gate after eliminating OOF recipe mismatch."""
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from pathlib import Path
from src.adaptive_gate import build_gate_features
from evaluate_crossfit_psych_profiles import skill

P=Path('artifacts/unified_oof');raw=pd.read_csv('data/train.csv',low_memory=False);C=np.array([.93505266,-.00520129,.01091677,-.02528331]);I=.0300329767;parts={}
for yr in (2022,2023,2024):
 z=np.load(P/f'{yr}.npz');r=raw.loc[raw.season.eq(yr)].reset_index(drop=True);pred=[z['p_v2'],z['p_v3_55'],z['p_v3_30']];risks=[z['risks'][:,j] for j in range(3)];main=.27358084*pred[0]+.26512224*pred[1]+.46129691*pred[2];old=np.clip(I+np.column_stack([main]+risks)@C,1e-6,1-1e-6);parts[yr]=(r,z['y'],build_gate_features(r,pred,risks,old),old,z['pitcher_id'])
def fit(years,seed):
 x=pd.concat([parts[y][2] for y in years],ignore_index=True);target=np.concatenate([parts[y][1]-parts[y][3] for y in years]);w=np.concatenate([np.full(len(parts[y][1]),.55**(max(years)-y)) for y in years]);m=CatBoostRegressor(iterations=73,depth=3,learning_rate=.025,loss_function='RMSE',l2_leaf_reg=30,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=seed,thread_count=6,allow_writing_files=False,verbose=False);m.fit(x,target,sample_weight=w);return m
m23=fit((2022,),280033);m24=fit((2022,2023),280033);p23=np.clip(parts[2023][3]+m23.predict(parts[2023][2]),1e-6,1-1e-6);p24=np.clip(parts[2024][3]+m24.predict(parts[2024][2]),1e-6,1-1e-6);print('unified scores',skill(parts[2023][1],p23),skill(parts[2024][1],p24))
np.savez_compressed('evaluation/unified_adaptive_gate.npz',y23=parts[2023][1],p23=p23,base23=parts[2023][3],pitcher23=parts[2023][4],y24=parts[2024][1],p24=p24,base24=parts[2024][3],pitcher24=parts[2024][4])
