"""Forward-within-season residual tables: early->mid selection, <=mid->late audit."""
import numpy as np,pandas as pd
from pathlib import Path
R=Path(__file__).resolve().parent;raw=pd.read_csv(R/'data/train.csv',low_memory=False);d=raw.loc[raw.season.eq(2024)].reset_index(drop=True);z=np.load(R/'expanded_blend_2024.npz');y=z['y'].astype(float);p=z['p'].astype(float);month=d.game_month.to_numpy();early=month<=5;mid=(month>=6)&(month<=7);late=month>=8
def sc(mask,pred):q=y[mask];return 1e5*(1-np.mean((q-np.clip(pred[mask],1e-5,1-1e-5))**2)/(q.mean()*(1-q.mean())))
def contexts(r):
 x=pd.DataFrame(index=r.index);x['pitcher']=r.pitcher_id.astype(str);x['batter']=r.batter_id.astype(str);x['pitcher_team']=r.pitcher_team_id.astype(str);x['batter_team']=r.batter_team_id.astype(str);x['hand']=r.pitcher_hand.astype(str)+'|'+r.batter_hand.astype(str);x['count']=r.balls_before.astype(str)+'-'+r.strikes_before.astype(str);x['base']=r.base_state.astype(str);x['game_type']=r.game_type.astype(str)
 x['inning']=pd.cut(r.inning,[-1,3,6,9,99],labels=False).fillna(-1).astype(str);x['score']=pd.cut(r.score_diff_pitcher_team,[-99,-4,-2,1,3,99],labels=False).fillna(-1).astype(str);x['li']=pd.cut(r.li,[-1,.75,1.5,3,999],labels=False).fillna(-1).astype(str)
 x['pressure']=x.li+'|'+r.num_runners_on.astype(str)+'|'+x['count'];x['pitcher_count']=x.pitcher+'|'+x['count'];x['pitcher_hand']=x.pitcher+'|'+r.batter_hand.astype(str);x['team_score']=x.pitcher_team+'|'+x.score
 return x
x=contexts(d);res=y-p
def correction(train_mask,test_mask,col,alpha):
 t=pd.DataFrame({'k':x.loc[train_mask,col],'r':res[train_mask]}).groupby('k').r.agg(['mean','count']);v=t['mean']*t['count']/(t['count']+alpha);out=np.zeros(len(y));out[test_mask]=x.loc[test_mask,col].map(v).fillna(0);return out
choices=[]
for col in x:
 for alpha in (20,50,100,300,1000,3000,10000):
  c=correction(early,mid,col,alpha)
  for scale in (.1,.2,.35,.5,.75,1):choices.append((sc(mid,p+scale*c)-sc(mid,p),col,alpha,scale))
print('base',sc(early,p),sc(mid,p),sc(late,p));print('mid selected top')
for q in sorted(choices,reverse=True)[:20]:print(q)
print('strict late audit')
for q in sorted(choices,reverse=True)[:20]:
 _,col,alpha,scale=q;c=correction(early|mid,late,col,alpha);print(col,alpha,scale,'late delta',sc(late,p+scale*c)-sc(late,p),'all proxy',sc(late,p+scale*c))

stable=[]
for mid_delta,col,alpha,scale in choices:
 c=correction(early|mid,late,col,alpha);late_delta=sc(late,p+scale*c)-sc(late,p)
 stable.append((min(mid_delta,late_delta),(mid_delta+late_delta)/2,mid_delta,late_delta,col,alpha,scale))
print('stable configurations')
for q in sorted(stable,reverse=True)[:30]:print(q)

# Pre-registered conservative pair: strongest transferable pitcher-hand channel plus
# a separately shrunk pitcher-count channel. Both tables use only earlier months.
print('independent stable channel combinations')
for ah,sh in [(100,.1),(100,.2),(300,.1)]:
 for ac,scalec in [(300,.05),(300,.1),(1000,.1),(3000,.1)]:
  cm=correction(early,mid,'pitcher_hand',ah);ccm=correction(early,mid,'pitcher_count',ac);cl=correction(early|mid,late,'pitcher_hand',ah);ccl=correction(early|mid,late,'pitcher_count',ac)
  dm=sc(mid,p+sh*cm+scalec*ccm)-sc(mid,p);dl=sc(late,p+sh*cl+scalec*ccl)-sc(late,p)
  print(ah,sh,ac,scalec,'mid',dm,'late',dl,'min',min(dm,dl))

out=p.copy();out[mid]=p[mid]+.2*correction(early,mid,'pitcher_hand',100)[mid]+.1*correction(early,mid,'pitcher_count',300)[mid];out[late]=p[late]+.2*correction(early|mid,late,'pitcher_hand',100)[late]+.1*correction(early|mid,late,'pitcher_count',300)[late]
print('forward OOF total',sc(np.ones(len(y),bool),out))
np.savez_compressed('forward_context_2024.npz',y=y,p=out,month=month)
