"""Forward-only adaptive meta learner with pressure and performance-state features."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT='old/experiments'
C=np.array([.93505266,-.00520129,.01091677,-.02528331]);I=.0300329767
raw=pd.read_csv('data/train.csv',low_memory=False)

def load_year(yr):
 r=raw.loc[raw.season.eq(yr)].reset_index(drop=True)
 if yr<2024:
  y=np.load(f'{ROOT}/result_dirs/v2_multisplit/{yr}.npz')['y'];p2=np.load(f'{ROOT}/result_dirs/v2_multisplit/{yr}.npz')['p'];p55=np.load(f'{ROOT}/result_dirs/v3_multisplit/{yr}.npz')['p'];p30=np.load(f'{ROOT}/result_dirs/v3_multisplit/{yr}_d30.npz')['p'];R=np.load(f'{ROOT}/result_dirs/v3_subtype_oof/{yr}.npz')['risks']
 else:
  z=np.load(f'{ROOT}/v3_subtypes_2024.npz');y=z['y'];R=z['risks'];p2=np.load(f'{ROOT}/result_dirs/v2_multisplit/2024.npz')['p'];p55=np.load(f'{ROOT}/model_v3_2024.npz')['p'];p30=np.load(f'{ROOT}/model_v3_decay_30.npz')['p']
 P=[p2,p55,p30];main=.27358084*p2+.26512224*p55+.46129691*p30;old=np.clip(I+np.c_[main,R]@C,1e-6,1-1e-6)
 x=pd.DataFrame(np.c_[p2,p55,p30,R],columns=['p2','p55','p30','middle','wild','reverse'])
 x['model_std']=np.std(np.c_[p2,p55,p30],axis=1);x['model_range']=np.ptp(np.c_[p2,p55,p30],axis=1);x['old']=old
 score=pd.to_numeric(r.score_diff_pitcher_team,errors='coerce').fillna(0);li=pd.to_numeric(r.li,errors='coerce').fillna(0).clip(0,10);runners=r.num_runners_on.fillna(0);balls=r.balls_before.fillna(0);strikes=r.strikes_before.fillna(0);inning=r.inning.fillna(0)
 top=r.top_bottom.astype(str).eq('T');pwe=np.where(top,r.home_win_expectancy,r.away_win_expectancy)/100.0
 recent=r[['asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate']].apply(pd.to_numeric,errors='coerce');career=pd.to_numeric(r.asof_pitcher_success_rate,errors='coerce');form=recent.mean(axis=1).fillna(career)-career;vol=recent.std(axis=1).fillna(.15)
 x['log_pitcher_n']=np.log1p(r.asof_pitcher_n.clip(lower=0));x['log_batter_n']=np.log1p(r.asof_batter_n.clip(lower=0));x['li']=li;x['inning']=inning;x['balls']=balls;x['strikes']=strikes;x['runners']=runners;x['form']=form;x['volatility']=vol
 x['trailing']=(score<0).astype(int);x['tied']=score.eq(0).astype(int);x['leading']=(score>0).astype(int);x['score_abs']=score.abs();x['pitcher_win_expectancy']=pwe
 x['count_stress']=(balls+1)/(strikes+1);x['traffic_stress']=(runners+1)*li;x['overall_pressure']=x.count_stress*x.traffic_stress
 x['protect_lead']=x.leading*li*(1+runners);x['deficit_pressure']=x.trailing*li*(1+runners);x['tie_pressure']=x.tied*li*(1+runners)
 x['cold_under_pressure']=np.maximum(-form,0)*x.overall_pressure;x['hot_under_pressure']=np.maximum(form,0)*x.overall_pressure;x['volatile_pressure']=vol*x.overall_pressure
 x['late_pressure']=(inning>=7).astype(int)*x.overall_pressure;x['expectancy_tension']=4*pwe*(1-pwe)*li
 x['expectancy_score_gap']=pwe-1/(1+np.exp(-score/2));x['season_month']=r.game_month
 return r,y,x.replace([np.inf,-np.inf],np.nan),old

parts={yr:load_year(yr) for yr in (2022,2023,2024)}
def score(y,p):q=y.mean();return 1e5*(1-np.mean((y-p)**2)/(q*(1-q)))
for valid_year,train_years in [(2023,[2022]),(2024,[2022,2023])]:
 X=pd.concat([parts[q][2] for q in train_years],ignore_index=True);y=np.concatenate([parts[q][1] for q in train_years]);old=np.concatenate([parts[q][3] for q in train_years]);years=np.concatenate([np.full(len(parts[q][1]),q) for q in train_years]);rv,yv,xv,ov=parts[valid_year]
 for depth,l2 in [(2,30),(3,30),(3,100),(4,100)]:
  m=CatBoostRegressor(iterations=500,depth=depth,learning_rate=.02,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=l2,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=320000+valid_year+depth+l2,thread_count=6,allow_writing_files=False,verbose=False);w=np.power(.55,(valid_year-1)-years);m.fit(X,y-old,sample_weight=w,eval_set=(xv,yv-ov),early_stopping_rounds=100,use_best_model=True);corr=m.predict(xv);vals=[score(yv,np.clip(ov+s*corr,1e-6,1-1e-6)) for s in (.25,.5,.75,1)];print(valid_year,depth,l2,m.get_best_iteration()+1,'base',score(yv,ov),'scales',vals,flush=True)
