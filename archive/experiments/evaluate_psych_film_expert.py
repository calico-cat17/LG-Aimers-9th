"""Forward evaluation of context-adjusted psychological FiLM modulation."""
import copy,random
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,TensorDataset
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive
from src.context_adjusted_psych import attach_context_adjusted_psych
from src.psych_film_expert import PsychRegimeFiLMExpert

torch.set_num_threads(6);SEED=260813;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
raw=pd.read_csv('data/train.csv',low_memory=False);parts={y:load_predictions(raw,y) for y in (2022,2023,2024)};r22,y22,g22,b22=parts[2022];r23,y23,g23,b23=parts[2023];r24,y24,g24,b24=parts[2024];gate23=fit_adaptive(g22,y22,b22,2020023);base={2022:b22,2023:np.clip(b23+gate23.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']};rows={2022:r22,2023:r23,2024:r24};ys={2022:y22,2023:y23,2024:y24}

def num(d,c,fill=0):return pd.to_numeric(d[c],errors='coerce').fillna(fill).to_numpy(np.float32)
def make(year):
 d=rows[year];hist=raw.loc[raw.season.lt(year)];psych=attach_context_adjusted_psych(d,hist,400.,100.);effect=[c for c in psych if c.endswith(('stable_effect','active_stable_effect'))];relcols=[c for c in psych if c.endswith('active_reliability')];psych_x=psych[effect].to_numpy(np.float32);psych_rel=np.clip(psych[relcols].max(1).to_numpy(np.float32),0,1);balls=num(d,'balls_before');strikes=num(d,'strikes_before');outs=num(d,'outs_before');inning=num(d,'inning');li=np.clip(num(d,'li'),0,10);runners=num(d,'num_runners_on');score=num(d,'score_diff_pitcher_team');risp=((num(d,'runner_on_2b')>0)|(num(d,'runner_on_3b')>0)).astype(np.float32);recent=np.c_[num(d,'asof_pitcher_prev1_game_success_rate',.52),num(d,'asof_pitcher_prev3_game_success_rate',.52),num(d,'asof_pitcher_prev5_game_success_rate',.52)];base_x=np.c_[base[year],num(d,'asof_pitcher_success_rate',.52),recent,recent.mean(1)-num(d,'asof_pitcher_success_rate',.52),num(d,'asof_batter_success_rate',.52),num(d,'asof_pitcher_fastball_rate'),num(d,'asof_pitcher_breaking_rate'),num(d,'asof_pitcher_offspeed_rate'),(num(d,'pitcher_hand')==num(d,'batter_hand')).astype(np.float32)];pressure_x=np.c_[balls,strikes,outs,inning,li,runners,score,risp,(balls==3),(strikes==2),(inning>=7),(abs(score)<=1)];prior=np.clip((np.log1p(li)/np.log(11))*(.30+.25*runners+.20*risp+.15*(balls==3)+.10*(inning>=7)),0,1).astype(np.float32);return base_x.astype(np.float32),psych_x,pressure_x.astype(np.float32),psych_rel,prior

data={y:make(y) for y in rows}
def fit_predict(train_years,valid_year):
 arrays=[np.concatenate([data[y][j] for y in train_years]) for j in range(5)];target=np.concatenate([ys[y]-base[y] for y in train_years]).astype(np.float32);bx,px,qx,rel,prior=arrays;means=[bx.mean(0),px.mean(0),qx.mean(0)];stds=[bx.std(0)+1e-4,px.std(0)+1e-4,qx.std(0)+1e-4];bx=np.nan_to_num((bx-means[0])/stds[0]);px=np.nan_to_num((px-means[1])/stds[1]);qx=np.nan_to_num((qx-means[2])/stds[2]);vb,vp,vq,vrel,vprior=data[valid_year];vb=np.nan_to_num((vb-means[0])/stds[0]);vp=np.nan_to_num((vp-means[1])/stds[1]);vq=np.nan_to_num((vq-means[2])/stds[2]);ds=TensorDataset(torch.from_numpy(bx),torch.from_numpy(px),torch.from_numpy(qx),torch.from_numpy(rel),torch.from_numpy(prior),torch.from_numpy(target));loader=DataLoader(ds,batch_size=4096,shuffle=True,num_workers=0);model=PsychRegimeFiLMExpert(bx.shape[1],px.shape[1],qx.shape[1],width=128,n_regimes=9,n_heads=4,dropout=.15);opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=5e-3);bestloss=1e9;best=None
 for epoch in range(8):
  model.train();total=0
  for b,p,q,r,pr,y in loader:
   opt.zero_grad();corr,delta,gate,regime,disagree,members=model(b,p,q,r,pr);primary=((corr-y)**2).mean();low=((gate*(pr<.15))**2).mean();size=(delta**2).mean();balance=((regime.mean(0)-1/9)**2).mean();member_loss=((members-y[:,None])**2).mean();loss=primary+.10*low+.03*size+.02*balance+.15*member_loss;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();total+=loss.item()*len(y)
  avg=total/len(ds);print('valid',valid_year,'epoch',epoch+1,'loss',avg,flush=True)
  if avg<bestloss:bestloss=avg;best=copy.deepcopy(model.state_dict())
 model.load_state_dict(best);model.eval();corr=[];gates=[];regimes=[];disagreements=[]
 with torch.no_grad():
  for st in range(0,len(vrel),8192):
   c,_,g,rg,dg,_=model(torch.from_numpy(vb[st:st+8192]),torch.from_numpy(vp[st:st+8192]),torch.from_numpy(vq[st:st+8192]),torch.from_numpy(vrel[st:st+8192]),torch.from_numpy(vprior[st:st+8192]));corr.append(c.numpy());gates.append(g.numpy());regimes.append(rg.numpy());disagreements.append(dg.numpy())
 return np.concatenate(corr),np.concatenate(gates),np.concatenate(regimes),np.concatenate(disagreements)

c23,gate23,reg23,dis23=fit_predict((2022,),2023);c24,gate24,reg24,dis24=fit_predict((2022,2023),2024);scales=(.25,.5,.75,1,1.25,1.5);scores=[(s,skill(y23,np.clip(base[2023]+s*c23,1e-6,1-1e-6)),skill(y24,np.clip(base[2024]+s*c24,1e-6,1-1e-6))) for s in scales];best=max(scores,key=lambda q:min(q[1]-skill(y23,base[2023]),q[2]-skill(y24,base[2024])));s=best[0];print('scales',scores,'best',best,'gate means',gate23.mean(),gate24.mean(),'regimes',reg23.mean(0),reg24.mean(0),'disagreement',dis23.mean(),dis24.mean());np.savez_compressed('evaluation/psych_regime_film.npz',y23=y23,p23=np.clip(base[2023]+s*c23,1e-6,1-1e-6),base23=base[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=np.clip(base[2024]+s*c24,1e-6,1-1e-6),base24=base[2024],pitcher24=r24.pitcher_id.to_numpy(),scale=s,gate23=gate23,gate24=gate24,corr23=c23,corr24=c24,regime23=reg23,regime24=reg24,disagreement23=dis23,disagreement24=dis24)
