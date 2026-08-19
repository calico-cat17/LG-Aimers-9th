"""Remove gate features that reveal year identity and audit forward transfer."""
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
from evaluate_crossfit_psych_profiles import load_predictions,skill

raw=pd.read_csv('data/train.csv',low_memory=False);q={y:load_predictions(raw,y) for y in (2022,2023,2024)};r={y:q[y][0] for y in q};Y={y:q[y][1] for y in q};X={y:q[y][2] for y in q};B={y:q[y][3] for y in q}
def drift(col):
 a=pd.to_numeric(X[2022][col],errors='coerce').replace([np.inf,-np.inf],np.nan);b=pd.to_numeric(X[2023][col],errors='coerce').replace([np.inf,-np.inf],np.nan);v=pd.concat([a,b]).median();aa=a.fillna(v);bb=b.fillna(v);z=np.r_[aa,bb];lab=np.r_[np.zeros(len(aa)),np.ones(len(bb))];auc=roc_auc_score(lab,z);return max(auc,1-auc)
ranking=sorted([(drift(c),c) for c in X[2022]],reverse=True);print('drift ranking',ranking)
def fit(tx,ty,tb,cols,weights,seed):
 m=CatBoostRegressor(iterations=73,depth=3,learning_rate=.025,loss_function='RMSE',l2_leaf_reg=30,random_strength=.2,bootstrap_type='Bernoulli',subsample=.8,random_seed=seed,thread_count=6,allow_writing_files=False,verbose=False);m.fit(tx[cols],ty-tb,sample_weight=weights);return m
allcols=list(X[2022]);variants={'all':allcols}
for k in (1,2,3,4,5,6,8):variants['drop'+str(k)]=[c for c in allcols if c not in {q[1] for q in ranking[:k]}]
best=None
for name,cols in variants.items():
 m23=fit(X[2022],Y[2022],B[2022],cols,np.ones(len(Y[2022])),730023+len(cols));p23=np.clip(B[2023]+m23.predict(X[2023][cols]),1e-6,1-1e-6)
 tx=pd.concat([X[2022],X[2023]],ignore_index=True);ty=np.r_[Y[2022],Y[2023]];tb=np.r_[B[2022],B[2023]];w=np.r_[np.full(len(Y[2022]),.55),np.ones(len(Y[2023]))];m24=fit(tx,ty,tb,cols,w,740024+len(cols));p24=np.clip(B[2024]+m24.predict(X[2024][cols]),1e-6,1-1e-6);s23=skill(Y[2023],p23);s24=skill(Y[2024],p24);row=(min(s23-437.01803981,s24-912.72488676),s23+s24,name,cols,p23,p24);print(name,s23,s24,row[0]);best=row if best is None or row[:2]>best[:2] else best
_,_,name,cols,p23,p24=best;print('best',name,cols);np.savez_compressed('evaluation/year_neutral_gate.npz',y23=Y[2023],p23=p23,base23=np.clip(B[2023]+fit(X[2022],Y[2022],B[2022],allcols,np.ones(len(Y[2022])),2020023).predict(X[2023]),1e-6,1-1e-6),pitcher23=r[2023].pitcher_id.to_numpy(),y24=Y[2024],p24=p24,base24=np.load('old/experiments/adaptive_gate_2024.npz')['p'],pitcher24=r[2024].pitcher_id.to_numpy(),variant=name,columns=np.array(cols))
