import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.adaptive_gate import build_gate_features
raw=pd.read_csv('data/train.csv',low_memory=False);parts=[]
C=np.array([.93505266,-.00520129,.01091677,-.02528331]);I=.0300329767
for yr in (2022,2023,2024):
 r=raw.loc[raw.season.eq(yr)].reset_index(drop=True)
 if yr<2024:
  y=np.load(f'v2_multisplit/{yr}.npz')['y'];p=[np.load(f'v2_multisplit/{yr}.npz')['p'],np.load(f'v3_multisplit/{yr}.npz')['p'],np.load(f'v3_multisplit/{yr}_d30.npz')['p']];R=np.load(f'v3_subtype_oof/{yr}.npz')['risks']
 else:
  z=np.load('v3_subtypes_2024.npz');y=z['y'];R=z['risks'];p=[np.load('v2_multisplit/2024.npz')['p'],np.load('model_v3_2024.npz')['p'],np.load('model_v3_decay_30.npz')['p']]
 main=.27358084*p[0]+.26512224*p[1]+.46129691*p[2];old=np.clip(I+np.c_[main,R]@C,1e-6,1-1e-6)
 parts.append((yr,y,build_gate_features(r,p,[R[:,i] for i in range(3)],old),old))
X=pd.concat([q[2] for q in parts],ignore_index=True);y=np.concatenate([q[1] for q in parts]);old=np.concatenate([q[3] for q in parts]);years=np.concatenate([np.full(len(q[1]),q[0]) for q in parts]);w=np.power(.55,2024-years)
m=CatBoostRegressor(iterations=73,depth=3,learning_rate=.025,loss_function='RMSE',l2_leaf_reg=30,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=280033,thread_count=6,allow_writing_files=False,verbose=25)
m.fit(X,y-old,sample_weight=w);m.save_model('model_hierarchical_stack/adaptive_gate.cbm')
print('saved adaptive gate',X.shape)
