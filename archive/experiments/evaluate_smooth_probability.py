"""Low-variance smooth probability expert with strict forward-season tests."""
from pathlib import Path
import numpy as np,pandas as pd

R=Path(__file__).resolve().parent
d=pd.read_csv(R/'data/train.csv',low_memory=False);y=d.control_success.to_numpy(float);season=d.season.to_numpy(int)

num=['game_month','inning','balls_before','strikes_before','outs_before','run_total_before','score_diff_pitcher_team','num_runners_on','home_win_expectancy','away_win_expectancy','li','asof_pitcher_n','asof_pitcher_success_rate','asof_pitcher_reverse_rate','asof_pitcher_middle_rate','asof_pitcher_ball_rate','asof_pitcher_strike_rate','asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate','asof_pitcher_prev1_game_middle_rate','asof_pitcher_prev3_game_middle_rate','asof_pitcher_prev5_game_middle_rate','asof_batter_n','asof_batter_success_rate','asof_batter_middle_rate','asof_pitcher_pitchmix_n','asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']
cat=['top_bottom','game_type','base_state','pitcher_hand','batter_hand']

def matrix(frame,stats=None):
 a=frame[num].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
 if stats is None:
  med=a.median().fillna(0);mu=a.fillna(med).mean();sd=a.fillna(med).std().replace(0,1).fillna(1)
 else:med,mu,sd=stats
 z=((a.fillna(med)-mu)/sd).to_numpy(np.float32)
 # Smooth nonlinear basis and domain interactions.
 z=np.c_[z,np.clip(z,-3,3)**2]
 extra=np.c_[
  frame.balls_before-frame.strikes_before,
  frame.num_runners_on*frame.li.fillna(0),
  frame.score_diff_pitcher_team*frame.li.fillna(0),
  frame.asof_pitcher_prev1_game_success_rate-frame.asof_pitcher_prev5_game_success_rate,
  frame.asof_pitcher_success_rate-frame.asof_batter_success_rate,
  np.log1p(frame.asof_pitcher_n.clip(lower=0)),np.log1p(frame.asof_batter_n.clip(lower=0))]
 extra=np.nan_to_num(extra.astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
 cats=pd.get_dummies(frame[cat].astype(str),dtype=np.float32)
 return np.c_[np.ones(len(frame)),z,extra,cats.to_numpy()],(med,mu,sd),list(cats.columns)

def fit_predict(vy,decay,alpha):
 tr=season<vy;va=season==vy;xt,stats,cn=matrix(d.loc[tr]);xv,_,cnv=matrix(d.loc[va],stats)
 # get_dummies schema is stable in this data; guard any rare missing level.
 if cn!=cnv: raise RuntimeError('category schema changed')
 w=np.power(decay,(vy-1)-season[tr]);sw=np.sqrt(w)[:,None];A=xt*sw;b=y[tr]*sw[:,0]
 reg=np.eye(A.shape[1])*alpha;reg[0,0]=0;coef=np.linalg.solve(A.T@A+reg,A.T@b);return y[va],np.clip(xv@coef,1e-5,1-1e-5)

def sc(y,p):return 1e5*(1-np.mean((y-p)**2)/(y.mean()*(1-y.mean())))
for decay in (.3,.55,1):
 for alpha in (100,1000):
  vals=[]
  for yr in (2022,2023,2024):
   yy,p=fit_predict(yr,decay,alpha);vals.append(sc(yy,p))
  print(decay,alpha,[round(v,2) for v in vals],round(min(vals),2),flush=True)
