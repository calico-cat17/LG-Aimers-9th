"""Per-season forward predictions from both architectures, then their blend."""
from __future__ import annotations
import numpy as np, pandas as pd, torch, time
from torch import nn
from catboost import CatBoostRegressor, Pool
from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
from src.hierarchical_state import HierarchicalState, build_situations, build_state_inputs
torch.manual_seed(0); torch.set_num_threads(6)
tr=pd.read_csv('data/train.csv',low_memory=False)
CONTEXT=['balls_before','strikes_before','outs_before','inning','num_runners_on','li',
 'score_diff_pitcher_team','home_win_expectancy','asof_pitcher_middle_rate','asof_pitcher_reverse_rate',
 'asof_pitcher_ball_rate','asof_batter_success_rate','asof_batter_middle_rate',
 'asof_pitcher_fastball_rate','asof_pitcher_breaking_rate']
def sstate(frame,snap,prior):
    k=pd.MultiIndex.from_arrays([frame.pitcher_id.astype(str),frame.season.astype(int)]);s=snap.set_index(['pitcher_id','season'])
    pn=s.snapshot_n.reindex(k).fillna(0).to_numpy(float);pc=s['snapshot_asof_pitcher_success_rate_count'].reindex(k).fillna(0).to_numpy(float)
    n=pd.to_numeric(frame.asof_pitcher_n,errors='coerce').fillna(0).to_numpy(float)
    r=pd.to_numeric(frame.asof_pitcher_success_rate,errors='coerce').fillna(prior).to_numpy(float)
    sn=np.maximum(n-pn,0);sc=np.maximum(n*r-pc,0)
    return np.divide(sc,sn,out=np.full(len(frame),prior),where=sn>0),sn
res={}
for t in (2022,2023,2024):
    hist=tr[tr.season<t];prior=float(hist.control_success.mean());snap=build_snapshots(hist)
    bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success')
    ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'])
    frame=tr[tr.season<=t].reset_index(drop=True)
    y=frame.control_success.to_numpy(float);season=frame.season.to_numpy();te=season==t;fit=~te
    x,base=build_v3_features(frame,prior,snap,bs,ms,'model/trackman_prior_features.csv')
    w=np.power(.55,t-season[fit])
    cb=CatBoostRegressor(iterations=220,depth=8,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=12,
      random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,
      random_seed=260803,thread_count=6,allow_writing_files=False,verbose=0)
    cb.fit(Pool(x[fit],(y-base)[fit],weight=w,cat_features=CAT_V2))
    p_cb=np.clip(base[te]+cb.predict(x[te]),1e-5,1-1e-5)
    rate,sn=sstate(frame,snap,prior)
    S=build_state_inputs(frame,rate,sn,prior);Z=build_situations(frame)
    C=frame[CONTEXT].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(np.float32);C=(C-C.mean(0))/(C.std(0)+1e-6)
    pid=pd.factorize(frame.pitcher_id.astype(str))[0];bid=pd.factorize(frame.batter_id.astype(str))[0]
    T=lambda a,d=torch.float32: torch.as_tensor(a,dtype=d)
    St,Zt,Ct,Pt,Bt,Yt=T(S),T(Z),T(C),T(pid,torch.long),T(bid,torch.long),T(y.astype(np.float32))
    fi=np.flatnonzero(fit);tei=np.flatnonzero(te);wt=T(np.power(.55,t-season[fi]).astype(np.float32))
    mdl=HierarchicalState(pid.max()+1,bid.max()+1,C.shape[1])
    with torch.no_grad():
        p0=float(np.average(y[fi],weights=np.power(.55,t-season[fi])));mdl.season_level.fill_(np.log(p0/(1-p0)))
    emb=list(mdl.slopes.parameters())+list(mdl.batter.parameters());ids={id(q) for q in emb}
    opt=torch.optim.Adam([{'params':[q for q in mdl.parameters() if id(q) not in ids],'lr':3e-3},{'params':emb,'lr':.05}])
    lf=nn.BCEWithLogitsLoss(reduction='none')
    for ep in range(30):
        pm=np.random.permutation(len(fi))
        for st in range(0,len(pm),16384):
            j=pm[st:st+16384];ii=T(fi[j],torch.long)
            loss=(lf(mdl(St[ii],Zt[ii],Ct[ii],Pt[ii],Bt[ii]),Yt[ii])*wt[j]).mean()+mdl.penalty(Pt[ii],Bt[ii],.30,.05)
            opt.zero_grad();loss.backward();opt.step()
    mdl.eval()
    with torch.no_grad():
        ii=T(tei,torch.long);p_hs=torch.sigmoid(mdl(St[ii],Zt[ii],Ct[ii],Pt[ii],Bt[ii])).numpy().astype(float)
    # The learned logit-scale level replaces the hand-written shrinkage formula,
    # and the tree now only has to explain what is left over it.
    with torch.no_grad():
        allp=torch.sigmoid(mdl(St,Zt,Ct,Pt,Bt)).numpy().astype(float)
    cb2=CatBoostRegressor(iterations=220,depth=8,learning_rate=.035,loss_function='RMSE',l2_leaf_reg=12,
      random_strength=.35,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,
      random_seed=260803,thread_count=6,allow_writing_files=False,verbose=0)
    cb2.fit(Pool(x[fit],(y-allp)[fit],weight=w,cat_features=CAT_V2))
    p_hy=np.clip(allp[te]+cb2.predict(x[te]),1e-5,1-1e-5)
    res[t]=(y[te],p_cb,p_hs,p_hy)
    s=lambda q:1e5*(1-((q-y[te])**2).mean()/(y[te].mean()*(1-y[te].mean())))
    print(f'{t}: catboost={s(p_cb):8.1f}  hstate={s(p_hs):8.1f}  HYBRID(hstate base + tree)={s(p_hy):8.1f}',flush=True)
sc=lambda y,p:1e5*(1-((np.clip(p,1e-5,1-1e-5)-y)**2).mean()/(y.mean()*(1-y.mean())))
print(f'\n{"w(catboost)":>12}' + ''.join(f'{t:>10}' for t in res) + f'{"worst":>10}{"mean":>10}')
best=None
for w in np.arange(0,1.01,.1):
    s={t:sc(res[t][0],w*res[t][1]+(1-w)*res[t][3]) for t in res}
    mn=min(s.values())
    if best is None or mn>best[1]: best=(w,mn)
    print(f'{w:>12.1f}'+''.join(f'{s[t]:>10.1f}' for t in res)+f'{mn:>10.1f}{np.mean(list(s.values())):>10.1f}')
print(f'\nbest worst-year weight w={best[0]:.1f}  worst={best[1]:.1f}')
np.savez_compressed('/tmp/blend_years.npz',**{f'{k}_{n}':v for k,(a,b,c,d) in res.items() for n,v in (('y',a),('cb',b),('hs',c),('hy',d))})
