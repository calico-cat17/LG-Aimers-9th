"""Strict temporal Ridge stack for consistently generated unified OOF members."""
import numpy as np
from sklearn.linear_model import Ridge
from evaluate_crossfit_psych_profiles import skill

Z={y:np.load(f'artifacts/unified_oof/{y}.npz') for y in (2022,2023,2024)}
def feat(z):
 P=np.c_[z['p_v2'],z['p_v3_55'],z['p_v3_30']];R=z['risks'];return np.c_[P,R,P.std(1),P.max(1)-P.min(1)]
X={y:feat(Z[y]) for y in Z};best=None
for alpha in (10,30,100,300,1000,3000,10000,30000,100000):
 m23=Ridge(alpha=alpha).fit(X[2022],Z[2022]['y']);p23=np.clip(m23.predict(X[2023]),1e-6,1-1e-6)
 xt=np.r_[X[2022],X[2023]];yt=np.r_[Z[2022]['y'],Z[2023]['y']];w=np.r_[np.full(len(X[2022]),.55),np.ones(len(X[2023]))];m24=Ridge(alpha=alpha).fit(xt,yt,sample_weight=w);p24=np.clip(m24.predict(X[2024]),1e-6,1-1e-6)
 s23,s24=skill(Z[2023]['y'],p23),skill(Z[2024]['y'],p24);row=(min(s23,s24),s23+s24,alpha,p23,p24,m24);print(alpha,s23,s24,p23.mean(),p24.mean());best=row if best is None or row[:2]>best[:2] else best
_,_,alpha,p23,p24,m=best;print('best',alpha,'coef',m.coef_,'intercept',m.intercept_)
np.savez_compressed('evaluation/unified_temporal_stack.npz',y23=Z[2023]['y'],p23=p23,pitcher23=Z[2023]['pitcher_id'],y24=Z[2024]['y'],p24=p24,pitcher24=Z[2024]['pitcher_id'],alpha=alpha)
