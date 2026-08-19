"""Proxy adversarial density weighting and year-group DRO for the adaptive gate."""
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from evaluate_crossfit_psych_profiles import load_predictions,skill

raw=pd.read_csv('data/train.csv',low_memory=False);parts={y:load_predictions(raw,y) for y in (2022,2023,2024)}
rows={y:parts[y][0] for y in parts};ys={y:parts[y][1] for y in parts};X={y:parts[y][2].reset_index(drop=True) for y in parts};base={y:parts[y][3] for y in parts}
cols=list(X[2022].columns);fill=pd.concat([X[2022],X[2023]],ignore_index=True).median().fillna(0)
def clean(x):return x[cols].replace([np.inf,-np.inf],np.nan).fillna(fill).to_numpy(np.float32)
Z={y:clean(X[y]) for y in X}
def domain_weights(old_year,new_year):
 a,b=Z[old_year],Z[new_year];rng=np.random.default_rng(260814+new_year);n=min(len(a),len(b),180000);ia=rng.choice(len(a),n,False);ib=rng.choice(len(b),n,False);xx=np.r_[a[ia],b[ib]];yy=np.r_[np.zeros(n),np.ones(n)]
 order=rng.permutation(len(yy));cut=int(len(yy)*.8);tr,va=order[:cut],order[cut:];m=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,learning_rate=.06,l2_regularization=10,random_state=260814).fit(xx[tr],yy[tr]);auc=roc_auc_score(yy[va],m.predict_proba(xx[va])[:,1]);prob=np.clip(m.predict_proba(a)[:,1],.05,.95);odds=prob/(1-prob);odds=np.clip(odds,.5,2);odds/=odds.mean();return odds,auc
def gate(train_x,target,weight,seed):
 m=CatBoostRegressor(iterations=73,depth=3,learning_rate=.025,loss_function='RMSE',l2_leaf_reg=30,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=seed,thread_count=6,allow_writing_files=False,verbose=False);m.fit(train_x,target,sample_weight=weight);return m

# 2022 -> 2023: transductive proxy audit of whether density ratios contain signal.
w22,auc23=domain_weights(2022,2023);m23=gate(X[2022],ys[2022]-base[2022],w22,610023);p23=np.clip(base[2023]+m23.predict(X[2023]),1e-6,1-1e-6)
plain23=gate(X[2022],ys[2022]-base[2022],np.ones(len(X[2022])),2020023);adaptive23=np.clip(base[2023]+plain23.predict(X[2023]),1e-6,1-1e-6)

# 2022+2023 -> 2024: only the 2022-vs-2023 shift direction is used. No 2024
# labels/features enter the weighting rule, so this is the production-relevant audit.
w_old,auc24=domain_weights(2022,2023);w23=np.ones(len(X[2023]));xtr=pd.concat([X[2022],X[2023]],ignore_index=True);target=np.r_[ys[2022]-base[2022],ys[2023]-base[2023]]
base_w=np.r_[.55*w_old,w23]

# Approximate Group DRO: repeatedly raise the weight of the worse training year.
year_mult=np.ones(2);model=None
for step in range(4):
 w=base_w*np.r_[np.full(len(X[2022]),year_mult[0]),np.full(len(X[2023]),year_mult[1])];model=gate(xtr,target,w,620024+step);pred=model.predict(xtr);loss=[np.mean((pred[:len(X[2022])]-(ys[2022]-base[2022]))**2),np.mean((pred[len(X[2022]):]-(ys[2023]-base[2023]))**2)];worst=int(np.argmax(loss));year_mult[worst]*=1.20;year_mult/=year_mult.mean()
p24=np.clip(base[2024]+model.predict(X[2024]),1e-6,1-1e-6)
adaptive24=np.load('old/experiments/adaptive_gate_2024.npz')['p']
print('domain_auc',auc23,auc24,'year_mult',year_mult,'scores',skill(ys[2023],p23),skill(ys[2024],p24))
np.savez_compressed('evaluation/temporal_shift_dro.npz',y23=ys[2023],p23=p23,base23=adaptive23,pitcher23=rows[2023].pitcher_id.to_numpy(),y24=ys[2024],p24=p24,base24=adaptive24,pitcher24=rows[2024].pitcher_id.to_numpy(),domain_auc23=auc23,domain_auc24=auc24,year_mult=year_mult)
