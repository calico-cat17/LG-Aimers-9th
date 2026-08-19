"""Recover pitcher IDs with high-dimensional pitch-context fingerprints."""
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix

main_cols=['pitcher_id','season','game_month','game_dayofweek','inning','top_bottom','balls_before','strikes_before','outs_before','pitcher_hand','batter_hand']
tm_cols=['pitcher_trackman_id']+main_cols[1:]
a=pd.read_csv('data/train.csv',usecols=main_cols);b=pd.read_csv('data/trackman_history.csv',usecols=tm_cols)
b['pitcher_hand']=b.pitcher_hand.map({'Left':1,'Right':2});b['batter_hand']=b.batter_hand.map({'Left':1,'Right':2})

def mapping(keys,label):
 ga=a.groupby(['pitcher_id']+keys,observed=True).size().rename('n').reset_index();gb=b.groupby(['pitcher_trackman_id']+keys,observed=True).size().rename('n').reset_index()
 allkeys=pd.concat([ga[keys],gb[keys]],ignore_index=True).drop_duplicates().reset_index(drop=True);allkeys['_k']=np.arange(len(allkeys))
 ga=ga.merge(allkeys,on=keys);gb=gb.merge(allkeys,on=keys)
 ai=pd.Index(sorted(ga.pitcher_id.unique()));bi=pd.Index(sorted(gb.pitcher_trackman_id.unique()))
 A=csr_matrix((ga.n,(ai.get_indexer(ga.pitcher_id),ga._k)),shape=(len(ai),len(allkeys)),dtype=np.float64)
 B=csr_matrix((gb.n,(bi.get_indexer(gb.pitcher_trackman_id),gb._k)),shape=(len(bi),len(allkeys)),dtype=np.float64)
 an=np.sqrt(A.multiply(A).sum(1)).A1;bn=np.sqrt(B.multiply(B).sum(1)).A1;sim=(A@B.T).toarray()/np.maximum(an[:,None]*bn[None,:],1e-12)
 ah=a.groupby('pitcher_id').pitcher_hand.first().reindex(ai).to_numpy();bh=b.groupby('pitcher_trackman_id').pitcher_hand.first().reindex(bi).to_numpy();sim[ah[:,None]!=bh[None,:]]=-1
 j=sim.argmax(1);rev=sim.argmax(0);ss=np.sort(sim,axis=1);conf=sim[np.arange(len(ai)),j];margin=ss[:,-1]-ss[:,-2];mutual=rev[j]==np.arange(len(ai))
 out=pd.DataFrame({'pitcher_id':ai,'pitcher_trackman_id':bi[j],'similarity':conf,'margin':margin,'mutual':mutual})
 print('\n',label,'keys',len(allkeys),'mutual',mutual.sum())
 for confmin,marginmin in [(0,0),(.3,.03),(.5,.05),(.7,.05),(.8,.03),(.9,.03)]:
  ok=mutual&(conf>=confmin)&(margin>=marginmin);print(confmin,marginmin,'mapped',ok.sum(),'rowcov',a.pitcher_id.isin(out.loc[ok,'pitcher_id']).mean(),'median',np.median(conf[ok]) if ok.any() else 0)
 out.to_csv(f'artifacts/mapping_audit_{label}.csv',index=False)
 return out

mapping(['season','game_month','game_dayofweek'],'coarse')
mapping(['season','game_month','game_dayofweek','inning','top_bottom'],'game_state')
mapping(['season','game_month','game_dayofweek','inning','top_bottom','balls_before','strikes_before','outs_before','batter_hand'],'pitch_context')
