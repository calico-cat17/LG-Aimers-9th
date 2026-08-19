"""Nonlinear reliability gate trained on 2023 cross-season channels, tested on 2024."""
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
R=Path(__file__).resolve().parent;z=np.load(R/'group_channels_crossseason.npz');raw=pd.read_csv(R/'data/train.csv',low_memory=False);r23=raw.loc[raw.season.eq(2023)].reset_index(drop=True);r24=raw.loc[raw.season.eq(2024)].reset_index(drop=True);names=[str(x) for x in z['names']]
def feat(r,C,p):
 x=pd.DataFrame(C,columns=['corr_'+n for n in names]);
 for j,n in enumerate(names):x['abs_'+n]=np.abs(C[:,j])
 x['p']=p;x['log_pitcher_n']=np.log1p(r.asof_pitcher_n.clip(lower=0));x['log_batter_n']=np.log1p(r.asof_batter_n.clip(lower=0));x['li']=r.li.fillna(0);x['month']=r.game_month;x['runners']=r.num_runners_on;x['balls']=r.balls_before;x['strikes']=r.strikes_before
 rec=r[['asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate']].apply(pd.to_numeric,errors='coerce');x['recent_gap']=rec.mean(1)-r.asof_pitcher_success_rate;x['recent_std']=rec.std(1)
 return x.replace([np.inf,-np.inf],np.nan)
X23=feat(r23,z['x23'],z['p23']);X24=feat(r24,z['x24'],z['p24']);y23=z['y23'];y24=z['y24'];base24=np.load(R/'expanded_blend_2024.npz')['p'];pair=np.load(R/'expanded_crossseason_pair_2024.npz')['p'];month=r24.game_month.to_numpy()
def sc(y,p):return 1e5*(1-np.mean((y-np.clip(p,1e-5,1-1e-5))**2)/(y.mean()*(1-y.mean())))
for depth,l2,it in [(2,100,100),(2,300,150),(3,300,100),(3,1000,150)]:
 m=CatBoostRegressor(iterations=it,depth=depth,learning_rate=.02,loss_function='RMSE',l2_leaf_reg=l2,random_strength=.15,bootstrap_type='Bernoulli',subsample=.8,random_seed=970000+depth+l2,thread_count=6,allow_writing_files=False,verbose=False);m.fit(X23,y23-z['p23']);c=m.predict(X24)
 print('\n',depth,l2,it,'raw correction',sc(y24,z['p24']+c))
 for anchor_name,anchor in [('expanded',base24),('pair',pair)]:
  for scale in (.1,.2,.35,.5,.75,1):
   p=anchor+scale*c;vals=[sc(y24[k],p[k]) for k in [month<=5,(month>=6)&(month<=7),month>=8,np.ones(len(y24),bool)]];print(anchor_name,scale,vals)
