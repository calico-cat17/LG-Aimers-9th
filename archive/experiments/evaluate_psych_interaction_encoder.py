"""Forward evaluation of separate psychological-state and situation encoders."""
import copy,random
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,TensorDataset
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive
from src.context_adjusted_psych import attach_context_adjusted_psych
from src.label_recovery import recover_failure_labels
from src.psych_film_expert import PsychFailureController

torch.set_num_threads(6);seed=260814;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
raw=pd.read_csv('data/train.csv',low_memory=False);labels,label_mask=recover_failure_labels(raw);parts={y:load_predictions(raw,y) for y in (2022,2023,2024)}
r22,y22,g22,b22=parts[2022];r23,y23,g23,b23=parts[2023];r24,y24,g24,b24=parts[2024];gate23=fit_adaptive(g22,y22,b22,2020023);base={2022:b22,2023:np.clip(b23+gate23.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']};rows={2022:r22,2023:r23,2024:r24};ys={2022:y22,2023:y23,2024:y24}
def num(d,c,fill=0):return pd.to_numeric(d[c],errors='coerce').fillna(fill).to_numpy(np.float32)
def make(year):
 d=rows[year];psych=attach_context_adjusted_psych(d,raw.loc[raw.season.lt(year)],400.,100.)
 effects=[c for c in psych if c.endswith(('stable_effect','active_stable_effect'))];rels=[c for c in psych if c.endswith('active_reliability')];career=num(d,'asof_pitcher_success_rate',.52)
 succ=np.column_stack([num(d,f'asof_pitcher_prev{k}_game_success_rate',.52) for k in (1,3,5)])
 middle=np.column_stack([num(d,f'asof_pitcher_prev{k}_game_middle_rate',0) for k in (1,3,5)])
 state=np.column_stack([psych[effects].to_numpy(np.float32),succ,middle,succ[:,0]-succ[:,2],succ.mean(1)-career,succ.std(1),middle[:,0]-middle[:,2],middle.std(1)])
 reliability=np.clip(psych[rels].max(1).to_numpy(np.float32),0,1);balls=num(d,'balls_before');strikes=num(d,'strikes_before');outs=num(d,'outs_before');inning=num(d,'inning');li=np.clip(num(d,'li'),0,10);runners=num(d,'num_runners_on');score=num(d,'score_diff_pitcher_team');risp=((num(d,'runner_on_2b')>0)|(num(d,'runner_on_3b')>0)).astype(np.float32)
 situation=np.column_stack([balls,strikes,outs,inning,li,runners,score,risp,(balls==3),(strikes==2),(inning>=7),(abs(score)<=1),num(d,'home_win_expectancy',.5),num(d,'away_win_expectancy',.5),(d.top_bottom.astype(str).to_numpy()=='B').astype(np.float32),(num(d,'pitcher_hand')==num(d,'batter_hand')).astype(np.float32)])
 pressure=np.clip((np.log1p(li)/np.log(11))*(.25+.2*runners+.2*risp+.15*(balls==3)+.1*(outs==2)+.1*(inning>=7)),0,1).astype(np.float32)
 bx=np.column_stack([base[year],career,num(d,'asof_pitcher_n'),num(d,'asof_batter_success_rate',.52),num(d,'asof_pitcher_fastball_rate'),num(d,'asof_pitcher_breaking_rate'),num(d,'asof_pitcher_offspeed_rate')]).astype(np.float32);sel=raw.season.eq(year).to_numpy()
 return bx,state.astype(np.float32),situation.astype(np.float32),reliability,pressure,labels[sel],label_mask[sel]
data={y:make(y) for y in rows}
def fit_predict(train_years,valid_year):
 a=[np.concatenate([data[y][j] for y in train_years]) for j in range(7)];bx,px,sx,rel,pressure,lab,lmask=a;target=np.concatenate([ys[y]-base[y] for y in train_years]).astype(np.float32);means=[bx.mean(0),px.mean(0),sx.mean(0)];stds=[bx.std(0)+1e-4,px.std(0)+1e-4,sx.std(0)+1e-4];bx=np.nan_to_num((bx-means[0])/stds[0]);px=np.nan_to_num((px-means[1])/stds[1]);sx=np.nan_to_num((sx-means[2])/stds[2]);vb,vp,vs,vr,vpr,_,_=data[valid_year];vb=np.nan_to_num((vb-means[0])/stds[0]);vp=np.nan_to_num((vp-means[1])/stds[1]);vs=np.nan_to_num((vs-means[2])/stds[2]);ds=TensorDataset(*map(torch.from_numpy,(bx,px,sx,rel,pressure,target,lab,lmask)));loader=DataLoader(ds,batch_size=4096,shuffle=True,num_workers=0);m=PsychFailureController(bx.shape[1],px.shape[1],sx.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-2);bestloss=1e9;best=None
 for epoch in range(10):
  m.train();total=0
  for b,p,s,r,pr,y,l,lm in loader:
   opt.zero_grad();corr,aux,g,pe,se=m(b,p,s,r,pr);primary=((corr-y)**2).mean();auxloss=torch.nn.functional.binary_cross_entropy_with_logits(aux,l,reduction='none').mean(1);auxloss=(auxloss*lm).sum()/lm.sum().clamp_min(1);loss=primary+.025*auxloss+.02*(corr**2).mean()+.002*((pe.mean(0)**2).mean()+(se.mean(0)**2).mean());loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();total+=loss.item()*len(y)
  avg=total/len(ds);print('valid',valid_year,'epoch',epoch+1,'loss',avg,flush=True)
  if avg<bestloss:bestloss=avg;best=copy.deepcopy(m.state_dict())
 m.load_state_dict(best);m.eval();cs=[];gs=[]
 with torch.no_grad():
  for st in range(0,len(vr),8192):c,_,g,_,_=m(torch.from_numpy(vb[st:st+8192]),torch.from_numpy(vp[st:st+8192]),torch.from_numpy(vs[st:st+8192]),torch.from_numpy(vr[st:st+8192]),torch.from_numpy(vpr[st:st+8192]));cs.append(c.numpy());gs.append(g.numpy())
 return np.concatenate(cs),np.concatenate(gs)
c23,g23=fit_predict((2022,),2023);c24,g24=fit_predict((2022,2023),2024);scales=(.10,.15,.25,.4,.55,.7,1.0);scores=[(s,skill(y23,np.clip(base[2023]+s*c23,1e-6,1-1e-6)),skill(y24,np.clip(base[2024]+s*c24,1e-6,1-1e-6))) for s in scales];best=max(scores,key=lambda q:min(q[1]-skill(y23,base[2023]),q[2]-skill(y24,base[2024])));s=best[0];print('scales',scores,'best',best,'gate',g23.mean(),g24.mean());np.savez_compressed('evaluation/psych_failure_controller.npz',y23=y23,p23=np.clip(base[2023]+s*c23,1e-6,1-1e-6),base23=base[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=np.clip(base[2024]+s*c24,1e-6,1-1e-6),base24=base[2024],pitcher24=r24.pitcher_id.to_numpy(),scale=s,gate23=g23,gate24=g24,corr23=c23,corr24=c24)
