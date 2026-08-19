"""Train validated diversity members for the capacity-expanded candidate."""
import shutil
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots

ROOT=Path('.');OUT=Path('model_expanded');OUT.mkdir(exist_ok=True)
raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);s=raw.season.to_numpy();prior=float(y.mean());ps=build_snapshots(raw);bs=build_entity_snapshots(raw,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(raw,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,base=build_v3_features(raw,prior,ps,bs,ms,'model/trackman_prior_features.csv');target=y-base;w=np.power(.30,2024-s)
for seed,weight in [(260811,.60),(3407,.10)]:
 m=CatBoostRegressor(iterations=199,depth=8,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=12,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=seed,thread_count=6,allow_writing_files=False,verbose=50);m.fit(x,target,sample_weight=w,cat_features=CAT_V2);m.save_model(OUT/f'v3_decay30_seed{seed}.cbm')
# F-only specialist; R rows continue to use the original ensemble.
f=raw.game_type.eq('F').to_numpy();m=CatBoostRegressor(iterations=123,depth=7,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=15,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=360001,thread_count=6,allow_writing_files=False,verbose=50);m.fit(x.loc[f],target[f],sample_weight=w[f],cat_features=CAT_V2);m.save_model(OUT/'game_type_F.cbm')
print('saved expanded members')
