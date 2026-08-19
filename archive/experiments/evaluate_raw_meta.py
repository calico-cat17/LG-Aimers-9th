import numpy as np,pandas as pd
R='old/experiments';raw=pd.read_csv('data/train.csv',low_memory=False);C=np.array([.93505266,-.00520129,.01091677,-.02528331]);I=.0300329767
cols=['asof_pitcher_success_rate','asof_pitcher_reverse_rate','asof_pitcher_middle_rate','asof_pitcher_ball_rate','asof_pitcher_strike_rate','asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate','asof_pitcher_prev1_game_middle_rate','asof_pitcher_prev3_game_middle_rate','asof_pitcher_prev5_game_middle_rate','asof_batter_success_rate','asof_batter_middle_rate','asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']
def load(yr):
 r=raw.loc[raw.season.eq(yr)].reset_index(drop=True)
 if yr<2024:y=np.load(f'{R}/result_dirs/v2_multisplit/{yr}.npz')['y'];a=np.load(f'{R}/result_dirs/v2_multisplit/{yr}.npz')['p'];b=np.load(f'{R}/result_dirs/v3_multisplit/{yr}.npz')['p'];c=np.load(f'{R}/result_dirs/v3_multisplit/{yr}_d30.npz')['p'];Q=np.load(f'{R}/result_dirs/v3_subtype_oof/{yr}.npz')['risks']
 else:z=np.load(f'{R}/v3_subtypes_2024.npz');y=z['y'];Q=z['risks'];a=np.load(f'{R}/result_dirs/v2_multisplit/2024.npz')['p'];b=np.load(f'{R}/model_v3_2024.npz')['p'];c=np.load(f'{R}/model_v3_decay_30.npz')['p']
 main=.27358084*a+.26512224*b+.46129691*c;old=I+np.c_[main,Q]@C
 X=r[cols].apply(pd.to_numeric,errors='coerce');recent=X[[q for q in cols if 'game_success' in q]];middle=X[[q for q in cols if 'game_middle' in q]]
 X['recent_mean']=recent.mean(1);X['recent_std']=recent.std(1);X['recent_slope']=recent.iloc[:,0]-recent.iloc[:,-1];X['middle_mean']=middle.mean(1);X['failure_sum']=X.asof_pitcher_middle_rate+X.asof_pitcher_reverse_rate;X['old']=old;X['log_pitcher_n']=np.log1p(r.asof_pitcher_n.clip(lower=0));X['log_batter_n']=np.log1p(r.asof_batter_n.clip(lower=0));X['count_pressure']=r.balls_before-r.strikes_before;X['li']=r.li;X['inning']=r.inning
 return y,X.replace([np.inf,-np.inf],np.nan),old
parts={q:load(q) for q in (2022,2023,2024)}
def fit(train,a):
 X=pd.concat([parts[q][1] for q in train],ignore_index=True);y=np.concatenate([parts[q][0] for q in train]);med=X.median();X=X.fillna(med).to_numpy();mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1;Z=(X-mu)/sd;A=np.c_[np.ones(len(Z)),Z];pen=np.eye(A.shape[1])*a;pen[0,0]=0;coef=np.linalg.solve(A.T@A+pen,A.T@y);return med,mu,sd,coef
def pred(model,yr):med,mu,sd,c=model;X=parts[yr][1].fillna(med).to_numpy();return np.clip(np.c_[np.ones(len(X)),(X-mu)/sd]@c,1e-6,1-1e-6)
def sc(y,p):r=y.mean();return 1e5*(1-np.mean((y-p)**2)/(r*(1-r)))
for a in [1,10,100,1000,10000]:
 for yr,tr in [(2023,[2022]),(2024,[2022,2023])]:print(a,yr,'base',sc(parts[yr][0],parts[yr][2]),'meta',sc(parts[yr][0],pred(fit(tr,a),yr)),flush=True)
