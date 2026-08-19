"""Row-local latent game-state proxies from rational recent-game rates.

Recent rates are ratios of integer event counts.  Jointly matching success and
middle rates recovers a denominator (or a divisor of it), which exposes how
much evidence each recent-game summary actually contains.  No other test row is
read.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _joint_denominator(a, b, max_n, tol=8e-5):
    a=np.asarray(a,float);b=np.asarray(b,float);out=np.zeros(len(a),np.float32)
    valid=np.isfinite(a)&np.isfinite(b);idx=np.flatnonzero(valid)
    # Chunk to keep the n-grid temporary small.
    grid=np.arange(1,max_n+1,dtype=np.float32)
    for st in range(0,len(idx),50000):
        ii=idx[st:st+50000];aa=a[ii,None];bb=b[ii,None]
        err=np.maximum(np.abs(aa*grid-np.rint(aa*grid)),np.abs(bb*grid-np.rint(bb*grid)))
        ok=err<=tol;first=np.argmax(ok,axis=1)+1;has=ok.any(axis=1)
        out[ii]=np.where(has,first,grid[np.argmin(err,axis=1)])
    return out


def build_rational_game_features(raw: pd.DataFrame) -> pd.DataFrame:
    r=raw.reset_index(drop=True);x=pd.DataFrame(index=r.index)
    den={}
    for k,mx in ((1,150),(3,400),(5,650)):
        s=pd.to_numeric(r[f'asof_pitcher_prev{k}_game_success_rate'],errors='coerce').to_numpy(float)
        m=pd.to_numeric(r[f'asof_pitcher_prev{k}_game_middle_rate'],errors='coerce').to_numpy(float)
        d=_joint_denominator(s,m,mx);den[k]=d
        x[f'prev{k}_joint_den']=d;x[f'prev{k}_success_count_proxy']=np.rint(np.nan_to_num(s)*d)
        x[f'prev{k}_middle_count_proxy']=np.rint(np.nan_to_num(m)*d)
        x[f'prev{k}_rate_quant_error']=np.maximum(np.abs(np.nan_to_num(s)*d-np.rint(np.nan_to_num(s)*d)),np.abs(np.nan_to_num(m)*d-np.rint(np.nan_to_num(m)*d)))
    x['prev3_increment_den']=np.maximum(den[3]-den[1],0);x['prev5_increment_den']=np.maximum(den[5]-den[3],0)
    x['recent_sample_consistent']=((den[5]>=den[3])&(den[3]>=den[1])&(den[1]>0)).astype('int8')
    x['starter_proxy']=(den[1]>=45).astype('int8');x['reliever_proxy']=((den[1]>0)&(den[1]<=25)).astype('int8')
    inning=pd.to_numeric(r.inning,errors='coerce').fillna(1).to_numpy(float);n=pd.to_numeric(r.asof_pitcher_n,errors='coerce').fillna(0).to_numpy(float)
    x['starter_late']=x.starter_proxy*(inning>=6);x['starter_early']=x.starter_proxy*(inning<=3);x['den1_x_inning']=den[1]*inning
    x['career_phase_by_game']=np.log1p(n)/(1+np.log1p(den[5]))
    for k in (1,3,5):x[f'prev{k}_reliability_proxy']=den[k]/(den[k]+40*k)
    return x.replace([np.inf,-np.inf],np.nan).fillna(0).astype('float32')
