"""Nested forward calibration: gate(2022)->2023 residual table->2024 adaptive prediction."""
import numpy as np,pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
from src.adaptive_gate import build_gate_features

R=Path(__file__).resolve().parent;E=R/'old/experiments';D=E/'result_dirs';raw=pd.read_csv(R/'data/train.csv',low_memory=False)
C=np.array([.93505266,-.00520129,.01091677,-.02528331]);I=.0300329767
def sc(y,p):return 1e5*(1-np.mean((y-np.clip(p,1e-5,1-1e-5))**2)/(y.mean()*(1-y.mean())))
def frame(yr):
 r=raw.loc[raw.season.eq(yr)].reset_index(drop=True)
 if yr<2024:
  a=np.load(D/f'v2_multisplit/{yr}.npz');b=np.load(D/f'v3_multisplit/{yr}.npz');c=np.load(D/f'v3_multisplit/{yr}_d30.npz');q=np.load(D/f'v3_subtype_oof/{yr}.npz');y=a['y'];pred=[a['p'],b['p'],c['p']];risks=[q['risks'][:,j] for j in range(3)]
 else:
  a=np.load(D/'v2_multisplit/2024.npz');b=np.load(E/'model_v3_2024.npz');c=np.load(E/'model_v3_decay_30.npz');q=np.load(E/'v3_subtypes_2024.npz');y=a['y'];pred=[a['p'],b['p'],c['p']];risks=[q['risks'][:,j] for j in range(3)]
 main=.27358084*pred[0]+.26512224*pred[1]+.46129691*pred[2];old=np.clip(I+np.column_stack([main]+risks)@C,1e-6,1-1e-6)
 return r,y.astype(float),build_gate_features(r,pred,risks,old),old
r22,y22,x22,o22=frame(2022);r23,y23,x23,o23=frame(2023);r24,y24,x24,o24=frame(2024)
m=CatBoostRegressor(iterations=150,depth=3,learning_rate=.02,loss_function='RMSE',l2_leaf_reg=30,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=710001,thread_count=6,allow_writing_files=False,verbose=False)
m.fit(x22,y22-o22);p23=np.clip(o23+m.predict(x23),1e-6,1-1e-6);p24=np.load(E/'adaptive_gate_2024.npz')['p'].astype(float)
print('nested 2023 old/gate',sc(y23,o23),sc(y23,p23),'2024 adaptive',sc(y24,p24))

def ctx(r,p):
 x=pd.DataFrame(index=r.index);x['hand']=r.pitcher_hand.astype(str)+'|'+r.batter_hand.astype(str);x['count']=r.balls_before.astype(str)+'-'+r.strikes_before.astype(str);x['base']=r.base_state.astype(str)
 x['pitcher']=r.pitcher_id.astype(str);x['batter']=r.batter_id.astype(str)
 x['month']=r.game_month.astype(str);x['dayofweek']=r.game_dayofweek.astype(str);x['top_bottom']=r.top_bottom.astype(str);x['team']=r.pitcher_team_id.astype(str)
 x['inning']=pd.cut(r.inning,[-1,3,6,9,99],labels=False).fillna(-1).astype(str);x['score']=pd.cut(r.score_diff_pitcher_team,[-99,-4,-2,1,3,99],labels=False).fillna(-1).astype(str);x['li']=pd.cut(r.li,[-1,.75,1.5,3,999],labels=False).fillna(-1).astype(str)
 x['pressure']=x['li']+'|'+r.num_runners_on.astype(str)+'|'+x['count'];x['pred_bin']=pd.Series(pd.cut(p,np.linspace(0,1,21),labels=False,include_lowest=True),index=r.index).fillna(-1).astype(str)
 x['hand_pred']=x.hand+'|'+x.pred_bin;x['count_pred']=x['count']+'|'+x.pred_bin
 x['month_hand']=x['month']+'|'+x['hand'];x['month_count']=x['month']+'|'+x['count'];x['team_score']=x['team']+'|'+x['score']
 x['pitcher_batterhand']=x['pitcher']+'|'+r.batter_hand.astype(str);x['pitcher_count']=x['pitcher']+'|'+x['count'];x['pitcher_base']=x['pitcher']+'|'+x['base'];x['batter_pitcherhand']=x['batter']+'|'+r.pitcher_hand.astype(str)
 return x
c23,c24=ctx(r23,p23),ctx(r24,p24);res=y23-p23
rows=[]
for col in c23:
 for alpha in (100,300,1000,3000,10000):
  t=pd.DataFrame({'k':c23[col],'r':res}).groupby('k').r.agg(['mean','count']);v=t['mean']*t['count']/(t['count']+alpha);corr=c24[col].map(v).fillna(0).to_numpy()
  for scale in (.1,.2,.35,.5,.75,1):rows.append((sc(y24,p24+scale*corr),col,alpha,scale))
print('best nested corrections')
for z in sorted(rows,reverse=True)[:20]:print(z)
best=sorted(rows,reverse=True)[0];_,col,alpha,scale=best
t=pd.DataFrame({'k':c23[col],'r':res}).groupby('k').r.agg(['mean','count']);v=t['mean']*t['count']/(t['count']+alpha);corr=c24[col].map(v).fillna(0).to_numpy()
expanded=np.load(R/'expanded_blend_2024.npz')['p'].astype(float)
print('expanded nested correction',sc(y24,expanded),sc(y24,expanded+scale*corr))
np.savez_compressed('nested_hand_correction_2024.npz',y=y24,correction=corr,scale=scale,p_adaptive=np.clip(p24+scale*corr,1e-6,1-1e-6),p_expanded=np.clip(expanded+scale*corr,1e-6,1-1e-6))

# Smooth global affine calibration, heavily shrunk toward identity.
A=np.c_[np.ones(len(p23)),p23];coef=np.linalg.solve(A.T@A+np.diag([1000,1000]),A.T@y23)
for scale in (.1,.2,.35,.5,1):
 calibrated=p24+scale*((coef[0]+coef[1]*p24)-p24);print('affine',scale,sc(y24,calibrated),coef)

# Reconstruct an expanded-like 2023 seed ensemble, then transfer its residual table
# to the actual expanded 2024 prediction. This is the closest season-boundary proxy.
z30=np.load(D/'v3_multisplit/2023_d30.npz')['p'];zs=np.load(D/'v3_multisplit/2023_seed260811.npz')['p'];z55=np.load(D/'v3_multisplit/2023.npz')['p'];seed23=.3*z30+.6*zs+.1*z55
a23=np.load(D/'v2_multisplit/2023.npz')['p'];q23=np.load(D/'v3_subtype_oof/2023.npz')['risks'];P23=[a23,z55,seed23];main23=.27358084*P23[0]+.26512224*P23[1]+.46129691*P23[2];old23=np.clip(I+np.column_stack([main23]+[q23[:,j] for j in range(3)])@C,1e-6,1-1e-6);gx23=build_gate_features(r23,P23,[q23[:,j] for j in range(3)],old23);pexp23=np.clip(old23+.5*m.predict(gx23),1e-6,1-1e-6)
expanded=np.load(R/'expanded_blend_2024.npz')['p'].astype(float);cx23,cx24=ctx(r23,pexp23),ctx(r24,expanded);rr=y23-pexp23;rows=[]
for col in cx23:
 for alpha in (50,100,300,1000,3000,10000):
  tt=pd.DataFrame({'k':cx23[col],'r':rr}).groupby('k').r.agg(['mean','count']);vv=tt['mean']*tt['count']/(tt['count']+alpha);co=cx24[col].map(vv).fillna(0).to_numpy()
  for scale in (.05,.1,.2,.35,.5):rows.append((sc(y24,expanded+scale*co),col,alpha,scale))
print('expanded-like cross-season top',sc(y23,pexp23),sc(y24,expanded))
for row in sorted(rows,reverse=True)[:20]:print(row)
tt=pd.DataFrame({'k':cx23['pitcher_batterhand'],'r':rr}).groupby('k').r.agg(['mean','count']);vv=tt['mean']*tt['count']/(tt['count']+100);cross_corr=cx24['pitcher_batterhand'].map(vv).fillna(0).to_numpy();corrected=np.clip(expanded+.2*cross_corr,1e-6,1-1e-6)
np.savez_compressed('expanded_crossseason_2024.npz',y=y24,p=corrected,correction=cross_corr)
print('saved expanded crossseason',sc(y24,corrected))

def exp_corr(col,alpha):
 ttt=pd.DataFrame({'k':cx23[col],'r':rr}).groupby('k').r.agg(['mean','count']);val=ttt['mean']*ttt['count']/(ttt['count']+alpha);return cx24[col].map(val).fillna(0).to_numpy()
low_hand=exp_corr('hand',1000);batter_hand=exp_corr('batter_pitcherhand',1000);pitch_count=exp_corr('pitcher_count',1000)
batter_hand_100=exp_corr('batter_pitcherhand',100);pitch_count_100=exp_corr('pitcher_count',100)
print('cross-season correction combinations')
for lh in (0,.05,.1,.2):
 for bh in (0,.05,.1,.2):
  for pc in (0,.05,.1):
   pp=np.clip(expanded+.2*cross_corr+lh*low_hand+bh*batter_hand+pc*pitch_count,1e-6,1-1e-6);print(lh,bh,pc,sc(y24,pp))
for bh in (.05,.1,.2,.35):
 for pc in (0,.05,.1,.2):
  pp=np.clip(expanded+.2*cross_corr+bh*batter_hand_100+pc*pitch_count_100,1e-6,1-1e-6);print('alpha100',bh,pc,sc(y24,pp))
pair=np.clip(expanded+.2*cross_corr+.1*batter_hand_100,1e-6,1-1e-6);np.savez_compressed('expanded_crossseason_pair_2024.npz',y=y24,p=pair,pitcher_batterhand=cross_corr,batter_pitcherhand=batter_hand_100);print('saved pair',sc(y24,pair))

def weighted_exp_corr(col,alpha,decay):
 ww=np.power(decay,r23.game_month.max()-r23.game_month.to_numpy());tmp=pd.DataFrame({'k':cx23[col],'wr':ww*rr,'w':ww});tt=tmp.groupby('k').agg({'wr':'sum','w':'sum'});val=tt.wr/(tt.w+alpha);return cx24[col].map(val).fillna(0).to_numpy()
recency=[]
for decay in (.5,.7,.85,.95,1.0):
 for alpha in (20,50,100,300):
  cp=weighted_exp_corr('pitcher_batterhand',alpha,decay);cb=weighted_exp_corr('batter_pitcherhand',alpha,decay)
  for sp in (.1,.2,.3,.35):
   for sb in (0,.05,.1,.2):recency.append((sc(y24,expanded+sp*cp+sb*cb),decay,alpha,sp,sb))
print('recency group top')
for row in sorted(recency,reverse=True)[:20]:print(row)
cp=weighted_exp_corr('pitcher_batterhand',300,.85);cb=weighted_exp_corr('batter_pitcherhand',300,.85);recency_p=np.clip(expanded+.35*cp+.2*cb,1e-6,1-1e-6);np.savez_compressed('expanded_recency_groups_2024.npz',y=y24,p=recency_p,pitcher_batterhand=cp,batter_pitcherhand=cb);print('saved recency groups',sc(y24,recency_p))
cp_mm=weighted_exp_corr('pitcher_batterhand',100,.85);cb_mm=weighted_exp_corr('batter_pitcherhand',100,.85);minimax_p=np.clip(expanded+.30*cp_mm+.10*cb_mm,1e-6,1-1e-6);np.savez_compressed('expanded_minimax_groups_2024.npz',y=y24,p=minimax_p,pitcher_batterhand=cp_mm,batter_pitcherhand=cb_mm);print('saved minimax groups',sc(y24,minimax_p))
