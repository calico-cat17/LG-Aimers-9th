"""Forward-evaluate domain context and a dedicated outs/runners expert."""
import os
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive
from src.context_pressure_features import build_context_pressure_features,CAT_CONTEXT

raw=pd.read_csv('data/train.csv',low_memory=False);parts={y:load_predictions(raw,y) for y in (2022,2023,2024)}
r22,y22,g22,b22=parts[2022];r23,y23,g23,b23=parts[2023];r24,y24,g24,b24=parts[2024];gate23=fit_adaptive(g22,y22,b22,1220023);base={2022:b22,2023:np.clip(b23+gate23.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']};rows={2022:r22,2023:r23,2024:r24};ys={2022:y22,2023:y23,2024:y24};allx={y:build_context_pressure_features(rows[y]) for y in rows}
blocks={'home':['pitcher_is_home','pitcher_win_expectancy','expectancy_tension','home_pressure','away_pressure','home_count_context'],'form':[c for c in allx[2022] if c.startswith(('recent_','bad_','good_','form_'))],'platoon':[c for c in allx[2022] if 'platoon' in c or c in ('same_hand_matchup','opposite_hand_matchup','left_pitcher','rare_left_opposite')],'pitchmix':[c for c in allx[2022] if c.startswith(('fastball','breaking','offspeed','pitchmix'))],'fatigue':[c for c in allx[2022] if c.startswith('fatigue')],'outs_runners':[c for c in allx[2022] if any(q in c for q in ('out','runner','risp','loaded','crisis','compound'))]};blocks['platoon_pitchmix']=list(dict.fromkeys(blocks['platoon']+blocks['pitchmix']));blocks['all']=list(allx[2022].columns)
only=os.environ.get('CONTEXT_BLOCK');blocks={k:v for k,v in blocks.items() if not only or k==only}
scales=(.05,.1,.15,.2,.25,.3,.4,.5,1.0)
for tag,cols in blocks.items():
 cats=[c for c in cols if c in CAT_CONTEXT]
 train23=allx[2022][cols];valid23=allx[2023][cols];m23=CatBoostRegressor(iterations=180,depth=3,learning_rate=.02,loss_function='RMSE',l2_leaf_reg=100,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=130000+len(cols),thread_count=6,allow_writing_files=False,verbose=False);m23.fit(train23,y22-base[2022],cat_features=cats);c23=m23.predict(valid23)
 train24=pd.concat([allx[2022][cols],allx[2023][cols]],ignore_index=True);target=np.r_[y22-base[2022],y23-base[2023]];w=np.r_[np.full(len(y22),.55),np.ones(len(y23))];m24=CatBoostRegressor(iterations=180,depth=3,learning_rate=.02,loss_function='RMSE',l2_leaf_reg=100,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=140000+len(cols),thread_count=6,allow_writing_files=False,verbose=False);m24.fit(train24,target,sample_weight=w,cat_features=cats);c24=m24.predict(allx[2024][cols])
 s23=np.array([skill(y23,np.clip(base[2023]+s*c23,1e-6,1-1e-6)) for s in scales]);s24=np.array([skill(y24,np.clip(base[2024]+s*c24,1e-6,1-1e-6)) for s in scales]);gain=np.minimum(s23-skill(y23,base[2023]),s24-skill(y24,base[2024]));j=int(gain.argmax());print(tag,'scale',scales[j],'scores',round(s23[j],4),round(s24[j],4),'gains',round(s23[j]-skill(y23,base[2023]),4),round(s24[j]-skill(y24,base[2024]),4),flush=True)
 np.savez_compressed(f'evaluation/context_{tag}.npz',y23=y23,p23=np.clip(base[2023]+scales[j]*c23,1e-6,1-1e-6),base23=base[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=np.clip(base[2024]+scales[j]*c24,1e-6,1-1e-6),base24=base[2024],pitcher24=r24.pitcher_id.to_numpy(),scale=scales[j])
