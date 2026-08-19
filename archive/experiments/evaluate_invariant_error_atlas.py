"""Low-cardinality, forward-only error atlas on top of Adaptive Gate."""
import numpy as np,pandas as pd
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive

raw=pd.read_csv('data/train.csv',low_memory=False);q={y:load_predictions(raw,y) for y in (2022,2023,2024)}
r22,y22,x22,h22=q[2022];r23,y23,x23,h23=q[2023];r24,y24,x24,h24=q[2024];g23=fit_adaptive(x22,y22,h22,2020023)
p={2022:h22,2023:np.clip(h23+g23.predict(x23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']};rows={2022:r22,2023:r23,2024:r24};ys={2022:y22,2023:y23,2024:y24}
def cats(r,pred):
 x=pd.DataFrame(index=r.index);x['pred']=pd.Series(pd.cut(pred,np.linspace(.2,.8,25),labels=False,include_lowest=True),index=r.index).fillna(-1).astype(str);x['pn']=pd.cut(np.log1p(r.asof_pitcher_n.fillna(0)),[-1,2,4,6,8,99],labels=False).fillna(-1).astype(str);x['bn']=pd.cut(np.log1p(r.asof_batter_n.fillna(0)),[-1,2,4,6,8,99],labels=False).fillna(-1).astype(str);x['count']=r.balls_before.astype(str)+'-'+r.strikes_before.astype(str);x['li']=pd.cut(r.li.fillna(0),[-1,.75,1.5,3,999],labels=False).astype(str);x['inning']=pd.cut(r.inning,[-1,3,6,9,99],labels=False).astype(str);rec=r[['asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate']].apply(pd.to_numeric,errors='coerce');x['vol']=pd.cut(rec.std(1).fillna(.15),[-1,.04,.1,.2,99],labels=False).astype(str);x['hand']=r.pitcher_hand.astype(str)+'-'+r.batter_hand.astype(str);x['runners']=r.num_runners_on.astype(str);x['home']=r.top_bottom.astype(str)
 for a,b in [('pred','pn'),('pred','count'),('pred','li'),('pred','vol'),('count','li'),('inning','li'),('pn','vol'),('hand','count')]:x[a+'__'+b]=x[a]+'|'+x[b]
 return x
C={y:cats(rows[y],p[y]) for y in rows};keys=list(C[2022].columns)
def channels(train_years,valid,alpha):
 out=[]
 for key in keys:
  sums=[];counts=[]
  for y in train_years:
   t=pd.DataFrame({'k':C[y][key],'e':ys[y]-p[y]}).groupby('k').e.agg(['sum','count']);sums.append(t['sum']);counts.append(t['count'])
  S=pd.concat(sums,axis=1).fillna(0);N=pd.concat(counts,axis=1).fillna(0);rate=S.sum(1)/(N.sum(1)+alpha)
  # Only retain categories whose observed yearly residual signs agree.
  yearly=S/(N+alpha/len(train_years));agree=np.abs(np.sign(yearly).mean(1));rate*=agree
  out.append(C[valid][key].map(rate).fillna(0).to_numpy())
 return np.column_stack(out)

best=None
for alpha in (300,1000,3000,10000):
 c23=channels((2022,),2023,alpha);c24=channels((2022,2023),2024,alpha)
 # Equal-weight robust average; clipping prevents rare bins dominating.
 a23=np.clip(np.median(c23,axis=1),-.03,.03);a24=np.clip(np.median(c24,axis=1),-.03,.03)
 for scale in (.1,.2,.35,.5,.75,1):
  s23=skill(y23,np.clip(p[2023]+scale*a23,1e-6,1-1e-6));s24=skill(y24,np.clip(p[2024]+scale*a24,1e-6,1-1e-6));row=(min(s23-skill(y23,p[2023]),s24-skill(y24,p[2024])),s23+s24,alpha,scale,s23,s24,a23,a24)
  if best is None or row[:2]>best[:2]:best=row
print('best',best[:6]);_,_,alpha,scale,s23,s24,a23,a24=best
np.savez_compressed('evaluation/invariant_error_atlas.npz',y23=y23,p23=np.clip(p[2023]+scale*a23,1e-6,1-1e-6),base23=p[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=np.clip(p[2024]+scale*a24,1e-6,1-1e-6),base24=p[2024],pitcher24=r24.pitcher_id.to_numpy(),alpha=alpha,scale=scale,corr23=a23,corr24=a24)
