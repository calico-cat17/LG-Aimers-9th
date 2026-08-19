"""Strict forward cohort-prior mixture restricted to cold-start pitchers."""
import numpy as np,pandas as pd
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive

def cohort_prior(history,rows,alpha):
 h=history.copy();r=rows.copy();global_rate=h.control_success.mean()
 for d in (h,r):
  d['_count']=d.balls_before.astype(str)+'-'+d.strikes_before.astype(str)
  d['_inning']=pd.cut(d.inning,[-1,3,6,9,99],labels=False).fillna(0).astype(int)
 keys=['pitcher_hand','pitcher_team_id','batter_hand','_count','_inning','base_state']
 # Back off successively; each level is trained on history only.
 pred=np.full(len(r),global_rate);remaining=np.ones(len(r),bool)
 for use in (keys,keys[:-1],['pitcher_hand','batter_hand','_count','_inning'],['pitcher_hand','batter_hand','_count'],['pitcher_hand','batter_hand']):
  agg=h.groupby(use).control_success.agg(['sum','count']);rate=(agg['sum']+alpha*global_rate)/(agg['count']+alpha)
  mi=pd.MultiIndex.from_frame(r[use]);v=rate.reindex(mi).to_numpy();n=agg['count'].reindex(mi).fillna(0).to_numpy();ok=remaining&np.isfinite(v)&(n>=20);pred[ok]=v[ok];remaining[ok]=False
 return pred

raw=pd.read_csv('data/train.csv',low_memory=False);parts={y:load_predictions(raw,y) for y in (2022,2023,2024)}
r22,y22,x22,b22=parts[2022];r23,y23,x23,b23=parts[2023];r24,y24,x24,b24=parts[2024];g23=fit_adaptive(x22,y22,b22,1020023);base={2023:np.clip(b23+g23.predict(x23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']};ys={2023:y23,2024:y24};rows={2023:r23,2024:r24}
print('base',*[skill(ys[y],base[y]) for y in (2023,2024)])
for alpha in (100,300,1000,3000,10000):
 cohort={y:cohort_prior(raw.loc[raw.season.lt(y)],rows[y],alpha) for y in rows}
 for threshold in (0,5,20,50,100,300):
  for scale in (.05,.1,.2,.3,.5,.75,1):
   scores=[];counts=[]
   for y in (2023,2024):
    mask=rows[y].asof_pitcher_n.fillna(0).le(threshold).to_numpy();p=base[y].copy();p[mask]=(1-scale)*p[mask]+scale*cohort[y][mask];scores.append(skill(ys[y],p));counts.append(mask.sum())
   gains=np.array(scores)-np.array([skill(ys[y],base[y]) for y in (2023,2024)])
   if gains.min()>0 and (scale in (.1,.3,.5,1)):print(alpha,threshold,scale,'counts',counts,'scores',[round(q,4) for q in scores],'gains',[round(q,4) for q in gains],flush=True)
