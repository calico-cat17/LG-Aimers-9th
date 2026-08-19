import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.adaptive_gate import build_gate_features
R='old/experiments';raw=pd.read_csv('data/train.csv',low_memory=False);C=np.array([.93505266,-.00520129,.01091677,-.02528331]);I=.0300329767
def load(yr):
 r=raw.loc[raw.season.eq(yr)].reset_index(drop=True)
 if yr<2024:y=np.load(f'{R}/result_dirs/v2_multisplit/{yr}.npz')['y'];p=[np.load(f'{R}/result_dirs/v2_multisplit/{yr}.npz')['p'],np.load(f'{R}/result_dirs/v3_multisplit/{yr}.npz')['p'],np.load(f'{R}/result_dirs/v3_multisplit/{yr}_d30.npz')['p']];Q=np.load(f'{R}/result_dirs/v3_subtype_oof/{yr}.npz')['risks']
 else:z=np.load(f'{R}/v3_subtypes_2024.npz');y=z['y'];Q=z['risks'];p=[np.load(f'{R}/result_dirs/v2_multisplit/2024.npz')['p'],np.load(f'{R}/model_v3_2024.npz')['p'],np.load(f'{R}/model_v3_decay_30.npz')['p']]
 main=.27358084*p[0]+.26512224*p[1]+.46129691*p[2];old=np.clip(I+np.c_[main,Q]@C,1e-6,1-1e-6);return r,y,p,Q,old
a=load(2022);b=load(2023);v=load(2024);X=pd.concat([build_gate_features(a[0],a[2],[a[3][:,i] for i in range(3)],a[4]),build_gate_features(b[0],b[2],[b[3][:,i] for i in range(3)],b[4])],ignore_index=True);yt=np.r_[a[1],b[1]];ot=np.r_[a[4],b[4]];years=np.r_[np.full(len(a[1]),2022),np.full(len(b[1]),2023)];w=np.power(.10,2023-years)
gate=CatBoostRegressor(iterations=157,depth=3,learning_rate=.02,loss_function='RMSE',l2_leaf_reg=30,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=330010,thread_count=6,allow_writing_files=False,verbose=False);gate.fit(X,yt-ot,sample_weight=w)
seed=np.column_stack([np.load(f'{R}/model_v3_decay_30.npz')['p'],np.load(f'{R}/model_v3_seed_260811.npz')['p'],np.load(f'{R}/model_v3_seed_3407.npz')['p']]);seed_ens=seed@np.array([.3,.6,.1]);temporal=np.load('temporal_catboost_False.npz')['p'];special=np.load('game_type_experts_2024.npz')['p'];mask=v[0].game_type.eq('F').to_numpy()
def sc(p):r=v[1].mean();return 1e5*(1-np.mean((v[1]-p)**2)/(r*(1-r)))
for tw in [0,.1,.2,.3,.4]:
 ens=(1-tw)*seed_ens+tw*temporal
 for fw in [0,.5,1]:
  p30=ens.copy();p30[mask]=(1-fw)*ens[mask]+fw*special[mask];P=[v[2][0],v[2][1],p30];main=.27358084*P[0]+.26512224*P[1]+.46129691*P[2];old=np.clip(I+np.c_[main,v[3]]@C,1e-6,1-1e-6);gx=build_gate_features(v[0],P,[v[3][:,i] for i in range(3)],old);corr=gate.predict(gx)
  print('Tweight',tw,'Fweight',fw,'old',sc(old),flush=True)
  for gs in [.25,.5,.75]:print(' gate',gs,sc(np.clip(old+gs*corr,1e-6,1-1e-6)),flush=True)
  if tw==0 and fw==1:
   np.savez_compressed('expanded_blend_2024.npz',y=v[1],p=np.clip(old+.5*corr,1e-6,1-1e-6))
