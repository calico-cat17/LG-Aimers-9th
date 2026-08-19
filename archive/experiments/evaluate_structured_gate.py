"""Categorical temporal residual gate with confidence and psychology contexts."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from src.adaptive_gate import build_gate_features

R=Path(__file__).resolve().parent; E=R/'old/experiments'; D=E/'result_dirs'
raw=pd.read_csv(R/'data/train.csv',low_memory=False)
C=np.array([.93505266,-.00520129,.01091677,-.02528331]); I=.0300329767

def score(y,p):return 1e5*(1-np.mean((y-np.clip(p,1e-5,1-1e-5))**2)/(y.mean()*(1-y.mean())))

def load(yr):
 r=raw.loc[raw.season.eq(yr)].reset_index(drop=True)
 if yr<2024:
  z2=np.load(D/f'v2_multisplit/{yr}.npz'); z55=np.load(D/f'v3_multisplit/{yr}.npz');z30=np.load(D/f'v3_multisplit/{yr}_d30.npz');zs=np.load(D/f'v3_subtype_oof/{yr}.npz')
  y=z2['y'];pred=[z2['p'],z55['p'],z30['p']];risks=[zs['risks'][:,j] for j in range(3)]
 else:
  z2=np.load(D/'v2_multisplit/2024.npz');z55=np.load(E/'model_v3_2024.npz');z30=np.load(E/'model_v3_decay_30.npz');zs=np.load(E/'v3_subtypes_2024.npz')
  y=z2['y'];pred=[z2['p'],z55['p'],z30['p']];risks=[zs['risks'][:,j] for j in range(3)]
 main=.27358084*pred[0]+.26512224*pred[1]+.46129691*pred[2]
 old=np.clip(I+np.column_stack([main]+risks)@C,1e-6,1-1e-6)
 x=build_gate_features(r,pred,risks,old)
 # Low-cardinality categorical regimes; raw player IDs are intentionally excluded.
 inning=pd.cut(r.inning,[-1,3,6,9,99],labels=['early','middle','late','extra']).astype(str)
 scorebin=pd.cut(r.score_diff_pitcher_team,[-99,-4,-2,1,3,99],labels=['far_behind','behind','close','ahead','far_ahead']).astype(str)
 libin=pd.cut(r.li,[-1,.75,1.5,3,999],labels=['low','normal','high','extreme']).astype(str)
 x['hand']=r.pitcher_hand.astype(str)+'|'+r.batter_hand.astype(str)
 x['count']=r.balls_before.astype(str)+'-'+r.strikes_before.astype(str)
 x['base']=r.base_state.astype(str)
 x['game_type']=r.game_type.astype(str)
 x['inning_score']=inning+'|'+scorebin
 x['pressure']=libin+'|'+r.num_runners_on.astype(str)+'|'+x['count']
 recent=r[['asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate']]
 direction=np.select([recent.iloc[:,0]>recent.iloc[:,2]+.03,recent.iloc[:,0]<recent.iloc[:,2]-.03],['hot','cold'],default='flat')
 x['form_pressure']=pd.Series(direction).astype(str)+'|'+libin
 cats=['hand','count','base','game_type','inning_score','pressure','form_pressure']
 for c in cats:x[c]=x[c].fillna('__MISSING__').astype(str)
 return r,y.astype(float),x,old,cats

parts=[load(y) for y in (2022,2023,2024)]
xt=pd.concat([parts[0][2],parts[1][2]],ignore_index=True);yt=np.r_[parts[0][1],parts[1][1]];ot=np.r_[parts[0][3],parts[1][3]]
xv=parts[2][2];yv=parts[2][1];ov=parts[2][3];cats=parts[0][4];years=np.r_[np.full(len(parts[0][1]),2022),np.full(len(parts[1][1]),2023)]
print('base',score(yv,ov),flush=True)
best=None
for depth,l2,rs in [(3,50,.25),(4,50,.25),(4,150,.15),(5,150,.15),(5,400,.1)]:
 m=CatBoostRegressor(iterations=800,depth=depth,learning_rate=.02,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=l2,random_strength=rs,bootstrap_type='Bernoulli',subsample=.8,random_seed=510000+depth+l2,thread_count=6,allow_writing_files=False,verbose=False)
 w=np.where(years==2023,1,.55);t=time.time();m.fit(xt,yt-ot,sample_weight=w,cat_features=cats,eval_set=(xv,yv-ov),early_stopping_rounds=130,use_best_model=True)
 corr=m.predict(xv)
 vals=[]
 for scale in (.25,.5,.75,1):vals.append((score(yv,ov+scale*corr),scale))
 row=(max(vals),depth,l2,m.get_best_iteration()+1,time.time()-t);print(row,flush=True)
 if best is None or row[0][0]>best[0][0]:best=row+(m,corr)
print('BEST',best[:5],flush=True)
np.savez_compressed('structured_gate_2024.npz',y=yv,p=np.clip(ov+best[0][1]*best[-1],1e-6,1-1e-6),correction=best[-1])
best[-2].save_model('structured_gate_validation.cbm')
