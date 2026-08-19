"""Strict prior-season empirical-Bayes group features."""
from __future__ import annotations
import numpy as np,pandas as pd

def build_temporal_pitcher_hand(train:pd.DataFrame, target='control_success', alpha=100.0):
    """Return row features and a final lookup for next-season inference."""
    d=train[['season','pitcher_id','batter_hand',target]].copy();d['pitcher_id']=d.pitcher_id.astype(str);d['batter_hand']=d.batter_hand.astype(str)
    feats=np.zeros((len(d),3),np.float32);global_prior=float(d[target].mean())
    for yr in sorted(d.season.unique()):
        va=d.season.eq(yr);hist=d.season.lt(yr)
        if hist.any():
            t=d.loc[hist].groupby(['pitcher_id','batter_hand'])[target].agg(['sum','count']);keys=pd.MultiIndex.from_arrays([d.loc[va,'pitcher_id'],d.loc[va,'batter_hand']]);n=t['count'].reindex(keys).fillna(0).to_numpy();sm=t['sum'].reindex(keys).fillna(0).to_numpy();rate=(sm+alpha*global_prior)/(n+alpha)
        else:n=np.zeros(va.sum());rate=np.full(va.sum(),global_prior)
        feats[va,0]=rate;feats[va,1]=np.log1p(n);feats[va,2]=n/(n+alpha)
    final=d.groupby(['pitcher_id','batter_hand'])[target].agg(['sum','count']);final['rate']=(final['sum']+alpha*global_prior)/(final['count']+alpha)
    return feats,final,global_prior

def attach_temporal_pitcher_hand(x:pd.DataFrame,raw:pd.DataFrame,features:np.ndarray):
    out=x.copy();out['prior_pitcher_hand_success']=features[:,0];out['prior_pitcher_hand_log_n']=features[:,1];out['prior_pitcher_hand_reliability']=features[:,2];career=pd.to_numeric(raw.asof_pitcher_success_rate,errors='coerce').to_numpy(float);career=np.where(np.isfinite(career),career,features[:,0]);out['prior_pitcher_hand_gap']=features[:,0]-career;return out
