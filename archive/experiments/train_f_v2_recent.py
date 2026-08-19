from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v2_features
from src.season_delta_features import build_snapshots
raw=pd.read_csv('data/train.csv',low_memory=False);y=raw.control_success.to_numpy(np.float32);prior=float(y.mean());snap=build_snapshots(raw);x,b=build_v2_features(raw,prior,snap,'model/trackman_prior_features.csv');mask=raw.game_type.eq('F').to_numpy()&raw.season.eq(2024).to_numpy();out=Path('f_v2_recent_models');out.mkdir(exist_ok=True)
for j in range(4):
 m=CatBoostRegressor(iterations=140,depth=8,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=20,random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=978000+j,thread_count=6,allow_writing_files=False,verbose=False);m.fit(x.loc[mask],(y-b)[mask],cat_features=CAT_V2);m.save_model(out/f'f_v2_recent_{j}.cbm');print(j,flush=True)
