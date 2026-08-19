"""Independent semantic-token Transformer and conservative Adaptive ensemble."""
import copy,random
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,TensorDataset
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive
from src.context_token_transformer import ContextTokenTransformer

torch.set_num_threads(6);SEED=260812;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
raw=pd.read_csv('data/train.csv',low_memory=False);parts={y:load_predictions(raw,y) for y in (2022,2023,2024)};r22,y22,g22,b22=parts[2022];r23,y23,g23,b23=parts[2023];r24,y24,g24,b24=parts[2024];gate23=fit_adaptive(g22,y22,b22,1920023);base={2022:b22,2023:np.clip(b23+gate23.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']};rows={2022:r22,2023:r23,2024:r24};ys={2022:y22,2023:y23,2024:y24}

def num(d,c,fill=0):return pd.to_numeric(d[c],errors='coerce').fillna(fill).to_numpy(np.float32)
def make(d):
 n=num(d,'asof_pitcher_n');bn=num(d,'asof_batter_n');mn=num(d,'asof_pitcher_pitchmix_n');rec=np.c_[num(d,'asof_pitcher_prev1_game_success_rate',.52),num(d,'asof_pitcher_prev3_game_success_rate',.52),num(d,'asof_pitcher_prev5_game_success_rate',.52)];ph=num(d,'pitcher_hand');bh=num(d,'batter_hand');risp=((num(d,'runner_on_2b')>0)|(num(d,'runner_on_3b')>0)).astype(np.float32)
 tokens=[np.c_[num(d,'asof_pitcher_success_rate',.52),num(d,'asof_pitcher_middle_rate'),num(d,'asof_pitcher_reverse_rate'),np.log1p(n)],np.c_[rec,rec.mean(1)-num(d,'asof_pitcher_success_rate',.52),rec.std(1)],np.c_[num(d,'balls_before'),num(d,'strikes_before'),num(d,'outs_before'),num(d,'inning')],np.c_[num(d,'num_runners_on'),risp,num(d,'li'),num(d,'score_diff_pitcher_team'),num(d,'home_win_expectancy')/100,num(d,'away_win_expectancy')/100],np.c_[(ph==bh),ph==1,bh==1,num(d,'asof_batter_success_rate',.52),np.log1p(bn)],np.c_[num(d,'asof_pitcher_fastball_rate'),num(d,'asof_pitcher_breaking_rate'),num(d,'asof_pitcher_offspeed_rate'),np.log1p(mn)]]
 rel=np.c_[n/(n+100),n/(n+100),np.ones(len(d)),np.ones(len(d)),bn/(bn+100),mn/(mn+100)].astype(np.float32);return tokens,rel

data={y:make(rows[y]) for y in rows}
def fit_predict(train_years,valid_year):
 tt=[np.concatenate([data[y][0][j] for y in train_years]) for j in range(6)];rr=np.concatenate([data[y][1] for y in train_years]);target=np.concatenate([ys[y] for y in train_years]).astype(np.float32);means=[a.mean(0) for a in tt];stds=[a.std(0)+1e-4 for a in tt];tt=[np.nan_to_num((a-m)/s) for a,m,s in zip(tt,means,stds)];vt=[np.nan_to_num((a-m)/s) for a,m,s in zip(data[valid_year][0],means,stds)];vr=data[valid_year][1]
 # Deterministic cap keeps CPU experimentation quick while retaining all token regimes.
 rng=np.random.default_rng(SEED+valid_year);idx=rng.choice(len(target),min(300000,len(target)),replace=False);ds=TensorDataset(*[torch.from_numpy(a[idx]) for a in tt],torch.from_numpy(rr[idx]),torch.from_numpy(target[idx]));loader=DataLoader(ds,batch_size=2048,shuffle=True,num_workers=0);model=ContextTokenTransformer([a.shape[1] for a in tt],direct_probability=True);opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-3);best=None;bestloss=1e9
 for epoch in range(4):
  model.train();total=0
  for batch in loader:
   tok=list(batch[:6]);rel=batch[6];y=batch[7];opt.zero_grad();pred=model(tok,rel);loss=((pred-y)**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();total+=loss.item()*len(y)
  avg=total/len(ds);print('valid',valid_year,'epoch',epoch+1,'loss',avg,flush=True)
  if avg<bestloss:bestloss=avg;best=copy.deepcopy(model.state_dict())
 model.load_state_dict(best);model.eval();out=[]
 with torch.no_grad():
  for start in range(0,len(vr),4096):out.append(model([torch.from_numpy(a[start:start+4096]) for a in vt],torch.from_numpy(vr[start:start+4096])).numpy())
 return np.concatenate(out)

t23=fit_predict((2022,),2023);t24=fit_predict((2022,2023),2024);weights=(0,.05,.1,.15,.2,.3,.4,.5,1);scores=[]
for w in weights:scores.append((w,skill(y23,np.clip((1-w)*base[2023]+w*t23,1e-6,1-1e-6)),skill(y24,np.clip((1-w)*base[2024]+w*t24,1e-6,1-1e-6))))
best=max(scores,key=lambda q:min(q[1]-skill(y23,base[2023]),q[2]-skill(y24,base[2024])));print('transformer standalone',skill(y23,t23),skill(y24,t24));print('blend weights',scores,'best',best);w=best[0];np.savez_compressed('evaluation/context_token_independent.npz',y23=y23,p23=np.clip((1-w)*base[2023]+w*t23,1e-6,1-1e-6),base23=base[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=np.clip((1-w)*base[2024]+w*t24,1e-6,1-1e-6),base24=base[2024],pitcher24=r24.pitcher_id.to_numpy(),weight=w,transformer23=t23,transformer24=t24)
