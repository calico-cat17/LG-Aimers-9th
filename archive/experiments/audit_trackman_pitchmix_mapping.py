"""Audit pitcher identity recovery using cumulative pitch-mix fingerprints."""
import numpy as np,pandas as pd
tr=pd.read_csv('data/train.csv',usecols=['pitcher_id','pitcher_hand','season','asof_pitcher_pitchmix_n','asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'])
tm=pd.read_csv('data/trackman_history.csv',usecols=['pitcher_trackman_id','pitcher_hand','pitch_type_group'])
# Last official pre-pitch state approximates the full history, with at most one omitted pitch.
last=tr.sort_values(['pitcher_id','season','asof_pitcher_pitchmix_n']).groupby('pitcher_id').tail(1).set_index('pitcher_id')
ct=pd.crosstab(tm.pitcher_trackman_id,tm.pitch_type_group);den=ct.sum(1);rates=pd.DataFrame(index=ct.index)
for c in ['fastball','breaking','offspeed']:rates[c]=ct.get(c,0)/den
hand=tm.groupby('pitcher_trackman_id').pitcher_hand.first().map({'Left':1,'Right':2})
M=last[['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']].to_numpy(float);T=rates[['fastball','breaking','offspeed']].to_numpy(float)
# Composition distance plus a weak log-count ratio. Ratios are robust to different log coverage.
dist=np.sqrt(((M[:,None,:]-T[None,:,:])**2).sum(2));mn=last.asof_pitcher_pitchmix_n.to_numpy(float);tn=den.reindex(rates.index).to_numpy(float);dist+=.05*np.abs(np.log1p(mn[:,None])-np.log1p(tn[None,:]))
mh=last.pitcher_hand.to_numpy();th=hand.reindex(rates.index).to_numpy();dist[mh[:,None]!=th[None,:]]=999
j=dist.argmin(1);order=np.sort(dist,axis=1);out=pd.DataFrame({'pitcher_id':last.index,'mix_best_id':rates.index[j],'mix_distance':dist[np.arange(len(last)),j],'mix_margin':order[:,1]-order[:,0]})
strict=pd.read_csv('artifacts/pitcher_trackman_mapping.csv');q=out.merge(strict,on='pitcher_id');q['agree']=q.mix_best_id.eq(q.pitcher_trackman_id)
print('strict agreement',q.agree.mean(),q.groupby(pd.cut(q.mix_distance,[0,.02,.05,.1,.2,.5,999])).agree.agg(['mean','count']))
print('distance agree',q.loc[q.agree,'mix_distance'].describe(),'disagree',q.loc[~q.agree,'mix_distance'].describe())
coarse=pd.read_csv('artifacts/mapping_audit_coarse.csv');both=coarse.merge(out,on='pitcher_id');both['agree']=both.pitcher_trackman_id.eq(both.mix_best_id);print('coarse/mix agreement',both.agree.mean(),both.groupby(pd.cut(both.similarity,[0,.5,.7,.8,.9,1])).agree.agg(['mean','count']))
both.to_csv('artifacts/mapping_audit_pitchmix.csv',index=False)
