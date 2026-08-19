"""Fast rho-oriented LambdaRank channel with pitcher-season query groups."""
import numpy as np,pandas as pd
from lightgbm import LGBMRanker
raw=pd.read_csv('data/train.csv',low_memory=False)
strict=np.load('evaluation/season_state_crossfit.npz')
def ceiling(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);v=y.mean()*(1-y.mean());c=np.mean((p-p.mean())*(y-y.mean()));return 1e5*c*c/(p.var()*v)
def frame(d):
 x=d.select_dtypes(include=[np.number,'bool']).drop(columns=['control_success','season','pitcher_id','batter_id','pitcher_team_id','batter_team_id'],errors='ignore').copy()
 return x.replace([np.inf,-np.inf],np.nan).fillna(-9).astype('float32')
results={}
for target in (2022,2023,2024):
 tr=raw[(raw.season<target)&(raw.season>=target-2)].copy();va=raw[raw.season==target].copy();tr['_q']=tr.season.astype(str)+'|'+tr.pitcher_id.astype(str);tr=tr.sort_values('_q');groups=tr.groupby('_q',sort=False).size().to_numpy();x=frame(tr);xv=frame(va).reindex(columns=x.columns,fill_value=-9);y=tr.control_success.to_numpy();yv=va.control_success.to_numpy();base=strict[f'base{str(target)[-2:]}']
 m=LGBMRanker(objective='lambdarank',n_estimators=250,learning_rate=.03,num_leaves=31,min_child_samples=300,colsample_bytree=.7,reg_lambda=30,verbosity=-1,n_jobs=6,random_state=930000+target)
 m.fit(x,y,group=groups);rank=m.predict(xv);rank=(rank-rank.mean())/(rank.std()+1e-9);rank=base.mean()+base.std()*rank
 best=max((ceiling(yv,(1-a)*base+a*rank),a) for a in np.linspace(0,1,41));results[target]=(ceiling(yv,base),ceiling(yv,rank),best);print(target,results[target],flush=True)
print('all',results)
