"""Strict one-season-only experts: select on 2022->2023, audit on 2023->2024."""
import gc,time
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive

raw=pd.read_csv('data/train.csv',low_memory=False);Y=raw.control_success.to_numpy(np.float32);S=raw.season.to_numpy();loaded={y:load_predictions(raw,y) for y in (2022,2023,2024)};r22,y22,g22,h22=loaded[2022];r23,y23,g23,h23=loaded[2023];r24,y24,g24,h24=loaded[2024];gm=fit_adaptive(g22,y22,h22,2020023);adaptive={2023:np.clip(h23+gm.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']};rows={2023:r23,2024:r24};ys={2023:y23,2024:y24}
configs=[('res150',False,150),('res300',False,300),('direct150',True,150),('direct300',True,300)];preds={k:{} for k,_,_ in configs}
for target in (2023,2024):
 train_year=target-1;hist=raw.loc[S<target];prior=float(Y[S==train_year].mean());ps=build_snapshots(hist);bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,base=build_v3_features(raw,prior,ps,bs,ms,'model_hierarchical_stack/trackman_prior_features.csv');tr=S==train_year;va=S==target
 for name,direct,it in configs:
  m=CatBoostRegressor(iterations=it,depth=6,learning_rate=.03,loss_function='RMSE',l2_leaf_reg=30,random_strength=.3,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=810000+it+(1000 if direct else 0),thread_count=6,allow_writing_files=False,verbose=False);target_y=Y if direct else Y-base;t=time.time();m.fit(x.loc[tr],target_y[tr],cat_features=CAT_V2);p=m.predict(x.loc[va]);preds[name][target]=np.clip(p if direct else base[va]+p,1e-6,1-1e-6);print(target,name,it,skill(ys[target],preds[name][target]),round(time.time()-t,1),flush=True);del m
 del x,base;gc.collect()

# Select architecture and a conservative blend only on 2023.
choices=[]
for name,_,_ in configs:
 for w in (0,.05,.1,.15,.2,.3,.4,.5,.65,.8,1):
  p=(1-w)*adaptive[2023]+w*preds[name][2023];choices.append((skill(y23,p),name,w))
selected=max(choices);_,name,w=selected;p23=np.clip((1-w)*adaptive[2023]+w*preds[name][2023],1e-6,1-1e-6);p24=np.clip((1-w)*adaptive[2024]+w*preds[name][2024],1e-6,1-1e-6);print('SELECTED ON 2023',selected,'AUDIT 2024',skill(y24,p24),'corr24',np.corrcoef(adaptive[2024]-y24,preds[name][2024]-y24)[0,1])
np.savez_compressed('evaluation/one_season_regime.npz',y23=y23,p23=p23,base23=adaptive[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=p24,base24=adaptive[2024],pitcher24=r24.pitcher_id.to_numpy(),expert23=preds[name][2023],expert24=preds[name][2024],model=name,weight=w)
