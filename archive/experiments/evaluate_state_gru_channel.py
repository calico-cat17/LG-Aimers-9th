"""Independent row-local state-sequence GRU residual channel."""
from __future__ import annotations
import copy,random
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive
from src.season_delta_features import build_snapshots,attach_season_delta
torch.set_num_threads(6);SEED=978001;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
raw=pd.read_csv('data/train.csv',low_memory=False);z={y:load_predictions(raw,y) for y in (2022,2023,2024)};r22,y22,g22,o22=z[2022];r23,y23,g23,o23=z[2023];r24,y24,_,_=z[2024];gg=fit_adaptive(g22,y22,o22,520023);base={2022:o22,2023:np.clip(o23+gg.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p'].astype(float)};rows={2022:r22,2023:r23,2024:r24};ys={2022:y22,2023:y23,2024:y24}
def num(d,c,fill=0):return pd.to_numeric(d[c],errors='coerce').fillna(fill).to_numpy(np.float32)
def make(year):
 d=rows[year];hist=raw[raw.season<year];prior=float(hist.control_success.mean());sd=attach_season_delta(pd.DataFrame(index=np.arange(len(d))),d,build_snapshots(hist),prior)
 def tok(success,middle,reverse,ball,strike,rel):return np.c_[success,middle,reverse,ball,strike,rel].astype(np.float32)
 n=num(d,'asof_pitcher_n');career=tok(num(d,'asof_pitcher_success_rate',prior),num(d,'asof_pitcher_middle_rate'),num(d,'asof_pitcher_reverse_rate'),num(d,'asof_pitcher_ball_rate'),num(d,'asof_pitcher_strike_rate'),n/(n+100))
 sn=sd.season_pitcher_n.to_numpy(np.float32);season=tok(sd.season_success_rate.to_numpy(),sd.season_middle_rate.to_numpy(),sd.season_reverse_rate.to_numpy(),sd.season_ball_rate.to_numpy(),sd.season_strike_rate.to_numpy(),sn/(sn+80))
 seq=[career,season]
 for k in (5,3,1):
  suc=num(d,f'asof_pitcher_prev{k}_game_success_rate',prior);mid=num(d,f'asof_pitcher_prev{k}_game_middle_rate');seq.append(tok(suc,mid,np.zeros(len(d)),np.zeros(len(d)),np.zeros(len(d)),np.full(len(d),k/5)))
 current=tok(np.full(len(d),prior),num(d,'li'),num(d,'balls_before')/3,num(d,'strikes_before')/2,num(d,'num_runners_on')/3,np.ones(len(d)));seq.append(current)
 static=np.c_[num(d,'inning')/12,num(d,'outs_before')/2,num(d,'score_diff_pitcher_team')/10,num(d,'home_win_expectancy')/100,num(d,'asof_batter_success_rate',prior),np.log1p(num(d,'asof_batter_n'))/10,num(d,'asof_pitcher_fastball_rate'),num(d,'asof_pitcher_breaking_rate'),num(d,'asof_pitcher_offspeed_rate'),(d.game_type.astype(str)=='F').to_numpy(np.float32),(d.pitcher_hand.astype(str)==d.batter_hand.astype(str)).to_numpy(np.float32),base[year]]
 return np.stack(seq,1),static.astype(np.float32)
data={y:make(y) for y in rows}
class Model(nn.Module):
 def __init__(self):super().__init__();self.gru=nn.GRU(6,24,2,batch_first=True,dropout=.15);self.head=nn.Sequential(nn.Linear(24+12,48),nn.GELU(),nn.Dropout(.15),nn.Linear(48,1))
 def forward(self,x,s):h=self.gru(x)[0][:,-1];return .04*torch.tanh(self.head(torch.cat([h,s],1)).squeeze(1))
def run(train_years,valid):
 X=np.concatenate([data[y][0] for y in train_years]);S=np.concatenate([data[y][1] for y in train_years]);T=np.concatenate([ys[y]-base[y] for y in train_years]).astype(np.float32);mu=X.mean((0,1));sd=X.std((0,1))+1e-4;sm=S.mean(0);ss=S.std(0)+1e-4;X=np.nan_to_num((X-mu)/sd);S=np.nan_to_num((S-sm)/ss);VX=np.nan_to_num((data[valid][0]-mu)/sd);VS=np.nan_to_num((data[valid][1]-sm)/ss)
 ds=TensorDataset(torch.from_numpy(X),torch.from_numpy(S),torch.from_numpy(T));loader=DataLoader(ds,batch_size=4096,shuffle=True);m=Model();opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=3e-3);best=None;bl=1e9
 for ep in range(6):
  m.train();tot=0
  for x,s,t in loader:opt.zero_grad();p=m(x,s);loss=((p-t)**2).mean();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();tot+=loss.item()*len(t)
  av=tot/len(ds);print(valid,'epoch',ep+1,av,flush=True)
  if av<bl:bl=av;best=copy.deepcopy(m.state_dict())
 m.load_state_dict(best);m.eval();out=[]
 with torch.no_grad():
  for st in range(0,len(VX),8192):out.append(m(torch.from_numpy(VX[st:st+8192]),torch.from_numpy(VS[st:st+8192])).numpy())
 return np.concatenate(out)
c23=run((2022,),2023);c24=run((2022,2023),2024);print('bases',skill(y23,base[2023]),skill(y24,base[2024]));best=(-1,None)
for a in (0,.1,.2,.3,.4,.5,.75,1):
 p23=np.clip(base[2023]+a*c23,1e-6,1-1e-6);p24=np.clip(base[2024]+a*c24,1e-6,1-1e-6);g23s=skill(y23,p23)-skill(y23,base[2023]);g24s=skill(y24,p24)-skill(y24,base[2024]);print(a,g23s,g24s,flush=True)
 if min(g23s,g24s)>best[0]:best=(min(g23s,g24s),(a,p23,p24,g23s,g24s))
print('BEST',best[0],best[1][0],best[1][3:]);np.savez_compressed('evaluation/state_gru_channel.npz',y23=y23,p23=best[1][1],base23=base[2023],y24=y24,p24=best[1][2],base24=base[2024],weight=best[1][0])
