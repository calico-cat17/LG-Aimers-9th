"""Forward residual tests for role, batter-induction, and team-system profiles."""
import os
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive,standardize

def shrink(sum_,n,prior,alpha):return (sum_+alpha*prior)/(n+alpha)

def entity_profile(history,rows,key,alpha,prefix):
 h=history.copy();global_rate=h.control_success.mean();agg=h.groupby(key).control_success.agg(['sum','count']);rate=shrink(agg['sum'],agg['count'],global_rate,alpha)
 out=pd.DataFrame({key:rows[key].astype(str) if rows[key].dtype=='object' else rows[key]}).merge(pd.DataFrame({key:agg.index,f'{prefix}_rate':rate.values,f'{prefix}_n':agg['count'].values}),on=key,how='left',sort=False).drop(columns=key)
 out[f'{prefix}_effect']=out[f'{prefix}_rate'].fillna(global_rate)-global_rate;out[f'{prefix}_rel']=out[f'{prefix}_n'].fillna(0)/(out[f'{prefix}_n'].fillna(0)+alpha);out[f'{prefix}_logn']=np.log1p(out[f'{prefix}_n'].fillna(0));return out

def role_features(history,rows):
 h=history.copy();h['_late']=(h.inning>=7).astype(float);h['_high']=(h.li>=1.5).astype(float);h['_early']=(h.inning<=3).astype(float);h['_close']=(h.score_diff_pitcher_team.abs()<=1).astype(float)
 career=h.groupby('pitcher_id')[['_late','_high','_early','_close']].mean();last_year=int(history.season.max());recent=h.loc[h.season.eq(last_year)].groupby('pitcher_id')[['_late','_high','_early','_close']].agg(['mean','count'])
 ids=rows.pitcher_id;out=pd.DataFrame(index=rows.index)
 for c in ('_late','_high','_early','_close'):
  ca=career[c].reindex(ids).to_numpy();rr=recent[(c,'mean')].reindex(ids).to_numpy();n=recent[(c,'count')].reindex(ids).fillna(0).to_numpy();rel=n/(n+100)
  out[f'role{c}_career']=ca;out[f'role{c}_recent']=rr;out[f'role{c}_shift']=(rr-ca)*rel;out[f'role{c}_rel']=rel
 inn=pd.to_numeric(rows.inning,errors='coerce').fillna(0).to_numpy();li=pd.to_numeric(rows.li,errors='coerce').fillna(0).to_numpy();close=(rows.score_diff_pitcher_team.abs()<=1).to_numpy()
 out['role_late_active']=out.role_late_shift.to_numpy()*(inn>=7);out['role_high_active']=out.role_high_shift.to_numpy()*(li>=1.5);out['role_close_active']=out.role_close_shift.to_numpy()*close
 return out

def batter_features(history,rows):
 out=entity_profile(history,rows,'batter_id',300,'batter')
 for label,mask_h,mask_r in [('threeball',history.balls_before.eq(3),rows.balls_before.eq(3)),('twostrike',history.strikes_before.eq(2),rows.strikes_before.eq(2)),('risp',(history.runner_on_2b.eq(1)|history.runner_on_3b.eq(1)),(rows.runner_on_2b.eq(1)|rows.runner_on_3b.eq(1)))]:
  sub=history.loc[mask_h];g=sub.control_success.mean();agg=sub.groupby('batter_id').control_success.agg(['sum','count']);rate=shrink(agg['sum'],agg['count'],g,300);v=pd.Series(rate,index=agg.index).reindex(rows.batter_id).fillna(g).to_numpy()-g;n=pd.Series(agg['count'],index=agg.index).reindex(rows.batter_id).fillna(0).to_numpy();out[f'batter_{label}_active']=v*(n/(n+300))*mask_r.to_numpy()
 return out

def team_features(history,rows):
 pieces=[]
 for key,prefix in [('pitcher_team_id','pitchteam'),('batter_team_id','batteam')]:pieces.append(entity_profile(history,rows,key,1000,prefix))
 # Stable team failure profile proxies supplied before each pitch.
 out=pd.concat(pieces,axis=1);out['team_match_effect']=out.pitchteam_effect+out.batteam_effect
 contexts=[('highli',history.li.ge(1.5),rows.li.ge(1.5)),('late',history.inning.ge(7),rows.inning.ge(7)),('threeball',history.balls_before.eq(3),rows.balls_before.eq(3)),('twostrike',history.strikes_before.eq(2),rows.strikes_before.eq(2)),('risp',(history.runner_on_2b.eq(1)|history.runner_on_3b.eq(1)),(rows.runner_on_2b.eq(1)|rows.runner_on_3b.eq(1)))]
 for key,prefix in [('pitcher_team_id','pitchteam'),('batter_team_id','batteam')]:
  for label,mh,mr in contexts:
   sub=history.loc[mh];prior=sub.control_success.mean();agg=sub.groupby(key).control_success.agg(['sum','count']);rate=shrink(agg['sum'],agg['count'],prior,1000);effect=pd.Series(rate-prior,index=agg.index).reindex(rows[key]).fillna(0).to_numpy();n=pd.Series(agg['count'],index=agg.index).reindex(rows[key]).fillna(0).to_numpy();out[f'{prefix}_{label}_active']=effect*(n/(n+1000))*mr.to_numpy()
 # Previous-season system shift, strongly shrunk toward the multi-year team effect.
 last=history.loc[history.season.eq(history.season.max())]
 for key,prefix in [('pitcher_team_id','pitchteam'),('batter_team_id','batteam')]:
  prior=last.control_success.mean();agg=last.groupby(key).control_success.agg(['sum','count']);rate=shrink(agg['sum'],agg['count'],prior,1500);out[f'{prefix}_last_effect']=pd.Series(rate-prior,index=agg.index).reindex(rows[key]).fillna(0).to_numpy()
 return out

raw=pd.read_csv('data/train.csv',low_memory=False);parts={y:load_predictions(raw,y) for y in (2022,2023,2024)}
r22,y22,x22,b22=parts[2022];r23,y23,x23,b23=parts[2023];r24,y24,x24,b24=parts[2024];g23=fit_adaptive(x22,y22,b22,920023);base={2022:b22,2023:np.clip(b23+g23.predict(x23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']};ys={2022:y22,2023:y23,2024:y24};rows={2022:r22,2023:r23,2024:r24}
channels={};builders={'role':role_features,'batter':batter_features,'team':team_features}
for name,fn in builders.items():channels[name]={y:fn(raw.loc[raw.season.lt(y)],rows[y]) for y in rows}
channels['all']={y:pd.concat([channels[k][y] for k in builders],axis=1) for y in rows};scales=(.02,.05,.08,.1,.15,.2,.25,.3,.4)
for name,data in channels.items():
 for ridge in (100,300,1000,3000,10000,30000,100000):
  z22,z23=standardize(data[2022],data[2023]);m23=Ridge(alpha=ridge,fit_intercept=False).fit(z22,y22-base[2022]);c23=m23.predict(z23)
  tr=pd.concat([data[2022],data[2023]],ignore_index=True);ztr,z24=standardize(tr,data[2024]);m24=Ridge(alpha=ridge,fit_intercept=False).fit(ztr,np.r_[y22-base[2022],y23-base[2023]],sample_weight=np.r_[np.full(len(y22),.55),np.ones(len(y23))]);c24=m24.predict(z24)
  s23=np.array([skill(y23,np.clip(base[2023]+s*c23,1e-6,1-1e-6)) for s in scales]);s24=np.array([skill(y24,np.clip(base[2024]+s*c24,1e-6,1-1e-6)) for s in scales]);gain=np.minimum(s23-skill(y23,base[2023]),s24-skill(y24,base[2024]));j=gain.argmax()
  if gain[j]>0:print(name,ridge,scales[j],round(s23[j],4),round(s24[j],4),'gains',round(s23[j]-skill(y23,base[2023]),4),round(s24[j]-skill(y24,base[2024]),4),flush=True)
  if name=='team' and ridge==3000:
   np.savez_compressed('team_system_forward.npz',y23=y23,base23=base[2023],p23=np.clip(base[2023]+.15*c23,1e-6,1-1e-6),pitcher23=r23.pitcher_id.to_numpy(),y24=y24,base24=base[2024],p24=np.clip(base[2024]+.15*c24,1e-6,1-1e-6),pitcher24=r24.pitcher_id.to_numpy())
