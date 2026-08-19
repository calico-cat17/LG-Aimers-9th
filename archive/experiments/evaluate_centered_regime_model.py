"""Learn season-centered individual effects, then add a forward league forecast."""
import gc,time
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from src.preprocessing_v2 import CAT_V2,build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive

raw=pd.read_csv('data/train.csv',low_memory=False);Y=raw.control_success.to_numpy(np.float32);S=raw.season.to_numpy();means=raw.groupby('season').control_success.mean();centered=Y-pd.Series(S).map(means).to_numpy();loaded={y:load_predictions(raw,y) for y in (2022,2023,2024)};r22,y22,g22,h22=loaded[2022];r23,y23,g23,h23=loaded[2023];r24,y24,g24,h24=loaded[2024];gm=fit_adaptive(g22,y22,h22,2020023);adaptive={2023:np.clip(h23+gm.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']};rows={2023:r23,2024:r24};ys={2023:y23,2024:y24};expert={}
def forecast(target):
 s=means.loc[means.index<target].iloc[-5:];return float(np.polyval(np.polyfit(s.index.to_numpy(float),s.to_numpy(float),1),target))
for target in (2023,2024):
 tr=S<target;va=S==target;hist=raw.loc[tr];prior=float(Y[tr].mean());ps=build_snapshots(hist);bs=build_entity_snapshots(hist,'batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'],'control_success');ms=build_entity_snapshots(hist,'pitcher_id','asof_pitcher_pitchmix_n',['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']);x,_=build_v3_features(raw,prior,ps,bs,ms,'model_hierarchical_stack/trackman_prior_features.csv');m=CatBoostRegressor(iterations=300,depth=7,learning_rate=.03,loss_function='RMSE',l2_leaf_reg=30,random_strength=.3,bootstrap_type='Bernoulli',subsample=.85,one_hot_max_size=16,random_seed=830000,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(.55,(target-1)-S[tr]);t=time.time();m.fit(x.loc[tr],centered[tr],sample_weight=w,cat_features=CAT_V2);expert[target]=np.clip(forecast(target)+m.predict(x.loc[va]),1e-6,1-1e-6);print(target,'forecast',forecast(target),'expert',skill(ys[target],expert[target]),'mean',expert[target].mean(),'sec',round(time.time()-t,1),flush=True);del x,m;gc.collect()
choices=[]
for w in (0,.025,.05,.075,.1,.15,.2,.3,.4,.5,.65,.8,1):choices.append((skill(y23,(1-w)*adaptive[2023]+w*expert[2023]),w))
_,w=max(choices);p23=np.clip((1-w)*adaptive[2023]+w*expert[2023],1e-6,1-1e-6);p24=np.clip((1-w)*adaptive[2024]+w*expert[2024],1e-6,1-1e-6);print('selected weight',w,'scores',skill(y23,p23),skill(y24,p24),'corr24',np.corrcoef(adaptive[2024]-y24,expert[2024]-y24)[0,1])
np.savez_compressed('evaluation/centered_regime_model.npz',y23=y23,p23=p23,base23=adaptive[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=p24,base24=adaptive[2024],pitcher24=r24.pitcher_id.to_numpy(),expert23=expert[2023],expert24=expert[2024],weight=w,forecast2025=forecast(2025))
