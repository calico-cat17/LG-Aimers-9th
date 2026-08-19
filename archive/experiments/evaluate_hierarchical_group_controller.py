"""Strict two-stage cross-season controller for stable context residual tables."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import lsq_linear
R=Path(__file__).resolve().parent;E=R/'old/experiments';D=E/'result_dirs';raw=pd.read_csv(R/'data/train.csv',low_memory=False);C=np.array([.93505266,-.00520129,.01091677,-.02528331]);I=.0300329767
def score(y,p):return 1e5*(1-np.mean((y-np.clip(p,1e-5,1-1e-5))**2)/(y.mean()*(1-y.mean())))
def pred(yr):
 r=raw.loc[raw.season.eq(yr)].reset_index(drop=True)
 if yr<2024:
  a=np.load(D/f'v2_multisplit/{yr}.npz');v55=np.load(D/f'v3_multisplit/{yr}.npz')['p'];d30=np.load(D/f'v3_multisplit/{yr}_d30.npz')['p'];seed=np.load(D/f'v3_multisplit/{yr}_seed260811.npz')['p'];q=np.load(D/f'v3_subtype_oof/{yr}.npz')['risks'];y=a['y'];third=.3*d30+.6*seed+.1*v55
 else:
  a=np.load(D/'v2_multisplit/2024.npz');v55=np.load(E/'model_v3_2024.npz')['p'];d30=np.load(E/'model_v3_decay_30.npz')['p'];seed=np.load(E/'model_v3_seed_260811.npz')['p'];q=np.load(E/'v3_subtypes_2024.npz')['risks'];y=a['y'];third=.3*d30+.6*seed+.1*v55
 main=.27358084*a['p']+.26512224*v55+.46129691*third;p=np.clip(I+np.column_stack([main,q])@C,1e-6,1-1e-6);return r,y.astype(float),p
def ctx(r):
 x=pd.DataFrame(index=r.index);x['count']=r.balls_before.astype(str)+'-'+r.strikes_before.astype(str);x['base']=r.base_state.astype(str);x['runner']=r.num_runners_on.astype(str);x['hand']=r.pitcher_hand.astype(str)+'|'+r.batter_hand.astype(str);x['pitcher']=r.pitcher_id.astype(str);x['batter']=r.batter_id.astype(str);x['team']=r.pitcher_team_id.astype(str);x['pitcher_batterhand']=x.pitcher+'|'+r.batter_hand.astype(str);x['pitcher_count']=x.pitcher+'|'+x['count'];x['pitcher_base']=x.pitcher+'|'+x.base;x['batter_pitcherhand']=x.batter+'|'+r.pitcher_hand.astype(str);x['count_base']=x['count']+'|'+x.base;x['month_hand']=r.game_month.astype(str)+'|'+x.hand;x['phase']=np.select([r.game_month<=5,r.game_month<=7],['early','middle'],default='late');x['pitcher_batterhand_phase']=x.pitcher_batterhand+'|'+x.phase;x['batter_pitcherhand_phase']=x.batter_pitcherhand+'|'+x.phase;x['pitcher_phase']=x.pitcher+'|'+x.phase;x['team_count']=x.team+'|'+x['count'];x['team_base']=x.team+'|'+x.base;x['team_batterhand']=x.team+'|'+r.batter_hand.astype(str);scorebin=pd.cut(r.score_diff_pitcher_team,[-99,-4,-2,1,3,99],labels=False).fillna(-1).astype(str);x['pitcher_score']=x.pitcher+'|'+scorebin;return x
groups=['hand','count','base','runner','count_base','pitcher','batter','pitcher_batterhand','pitcher_count','pitcher_base','batter_pitcherhand','month_hand','pitcher_batterhand_phase','batter_pitcherhand_phase','pitcher_phase','team_count','team_base','team_batterhand','pitcher_score']
def channels(rs,ys,ps,rd,alpha):
 a=ctx(rs);b=ctx(rd);res=ys-ps;out=[]
 for col in groups:
  t=pd.DataFrame({'k':a[col],'r':res}).groupby('k').r.agg(['mean','count']);v=t['mean']*t['count']/(t['count']+alpha);out.append(b[col].map(v).fillna(0).to_numpy())
 return np.column_stack(out)
r22,y22,p22=pred(2022);r23,y23,p23=pred(2023);r24,y24,p24=pred(2024)
expanded_ref=np.load(R/'expanded_blend_2024.npz')['p'].astype(float);months=r24.game_month.to_numpy();late_mask=months>=8
print('bases',score(y22,p22),score(y23,p23),score(y24,p24))
for alpha in (50,100,300,1000):
 X23=channels(r22,y22,p22,r23,alpha);X24=channels(r23,y23,p23,r24,alpha)
 for ridge in (0,10,100,1000,10000):
  A=np.vstack([X23,np.sqrt(ridge)*np.eye(len(groups))]);b=np.r_[y23-p23,np.zeros(len(groups))];fit=lsq_linear(A,b,bounds=(0,1),lsmr_tol='auto',max_iter=200);w=fit.x;v23=score(y23,p23+X23@w);v24=score(y24,p24+X24@w)
  ep=np.clip(expanded_ref+X24@w,1e-6,1-1e-6);late_delta=score(y24[late_mask],ep[late_mask])-score(y24[late_mask],expanded_ref[late_mask]);print(alpha,ridge,'fit23',v23,'untouched24',v24,'delta24',v24-score(y24,p24),'expanded',score(y24,ep),'late_delta',late_delta,'weights',[(g,round(q,3)) for g,q in zip(groups,w) if q>.01],flush=True)

# Fixed choice based on the strict 2023 fit/regularization curve, applied to the stronger stack.
alpha=100;ridge=100;X23=channels(r22,y22,p22,r23,alpha);X24=channels(r23,y23,p23,r24,alpha);A=np.vstack([X23,np.sqrt(ridge)*np.eye(len(groups))]);b=np.r_[y23-p23,np.zeros(len(groups))];w=lsq_linear(A,b,bounds=(0,1),lsmr_tol='auto',max_iter=200).x
expanded=np.load(R/'expanded_blend_2024.npz')['p'].astype(float);corrected=np.clip(expanded+X24@w,1e-6,1-1e-6);print('expanded fixed controller',score(y24,expanded),score(y24,corrected));np.savez_compressed('expanded_group_controller_2024.npz',y=y24,p=corrected,corrections=X24,weights=w,names=np.array(groups))
np.savez_compressed('group_channels_crossseason.npz',y23=y23,p23=p23,x23=X23,y24=y24,p24=p24,x24=X24,names=np.array(groups))

def weighted_channel(rs,ys,ps,rd,col,alpha,decay):
 a=ctx(rs);b=ctx(rd);ww=np.power(decay,rs.game_month.max()-rs.game_month.to_numpy());tmp=pd.DataFrame({'k':a[col],'wr':ww*(ys-ps),'w':ww});t=tmp.groupby('k').agg({'wr':'sum','w':'sum'});v=t.wr/(t.w+alpha);return b[col].map(v).fillna(0).to_numpy()
for src in [(r22,y22,p22,r23,y23,p23,'22to23'),(r23,y23,p23,r24,y24,p24,'23to24')]:
 rs,ys,ps,rd,yd,p_dest,label=src;cp=weighted_channel(rs,ys,ps,rd,'pitcher_batterhand',300,.85);cb=weighted_channel(rs,ys,ps,rd,'batter_pitcherhand',300,.85);print('recency fixed',label,score(yd,p_dest),score(yd,p_dest+.35*cp+.2*cb))
rec23=.35*weighted_channel(r22,y22,p22,r23,'pitcher_batterhand',300,.85)+.2*weighted_channel(r22,y22,p22,r23,'batter_pitcherhand',300,.85);rec24=.35*weighted_channel(r23,y23,p23,r24,'pitcher_batterhand',300,.85)+.2*weighted_channel(r23,y23,p23,r24,'batter_pitcherhand',300,.85);alt23=.1*channels(r22,y22,p22,r23,100)[:,groups.index('pitcher_batterhand')];alt24=.1*channels(r23,y23,p23,r24,100)[:,groups.index('pitcher_batterhand')];np.savez_compressed('late_replacement_channels.npz',y23=y23,p23=p23,month23=r23.game_month.to_numpy(),rec23=rec23,alt23=alt23,y24=y24,p24=p24,month24=r24.game_month.to_numpy(),rec24=rec24,alt24=alt24)

# Hyperparameters selected only on 2022->2023, followed by a single untouched 2024 audit.
selection=[]
stable_configs=[]
for decay in (.5,.7,.85,.95,1):
 for alpha in (20,50,100,300,1000):
  cp23=weighted_channel(r22,y22,p22,r23,'pitcher_batterhand',alpha,decay);cb23=weighted_channel(r22,y22,p22,r23,'batter_pitcherhand',alpha,decay)
  cp24=weighted_channel(r23,y23,p23,r24,'pitcher_batterhand',alpha,decay);cb24=weighted_channel(r23,y23,p23,r24,'batter_pitcherhand',alpha,decay)
  for sp in (.05,.1,.2,.3,.35,.5):
   for sb in (0,.05,.1,.2,.35):
    s23=score(y23,p23+sp*cp23+sb*cb23);s24=score(y24,p24+sp*cp24+sb*cb24);selection.append((s23,decay,alpha,sp,sb));stable_configs.append((min(s23-score(y23,p23),s24-score(y24,p24)),(s23+s24)/2,s23,s24,score(y24,expanded_ref+sp*cp24+sb*cb24),decay,alpha,sp,sb))
print('selected solely 2023')
for row in sorted(selection,reverse=True)[:10]:
 _,decay,alpha,sp,sb=row;cp24=weighted_channel(r23,y23,p23,r24,'pitcher_batterhand',alpha,decay);cb24=weighted_channel(r23,y23,p23,r24,'batter_pitcherhand',alpha,decay);print(row,'untouched24 base',score(y24,p24+sp*cp24+sb*cb24),'expanded',score(y24,expanded_ref+sp*cp24+sb*cb24))
print('minimax both transitions')
for row in sorted(stable_configs,reverse=True)[:15]:print(row)
cp_mm=weighted_channel(r23,y23,p23,r24,'pitcher_batterhand',100,.85);cb_mm=weighted_channel(r23,y23,p23,r24,'batter_pitcherhand',100,.85);expanded_mm=np.clip(expanded_ref+.30*cp_mm+.10*cb_mm,1e-6,1-1e-6);np.savez_compressed('expanded_minimax_controller_2024.npz',y=y24,p=expanded_mm,pitcher_batterhand=cp_mm,batter_pitcherhand=cb_mm);print('saved consistent minimax',score(y24,expanded_mm))
