"""Diagnostic coefficient stability for team-system features across seasons."""
import numpy as np
from sklearn.linear_model import Ridge
import evaluate_stable_entity_profiles as e

coefs=[];names=e.channels['team'][2022].columns.tolist()
for year in (2022,2023,2024):
 x=e.channels['team'][year].astype(float);mean=x.mean();std=x.std().replace(0,1).fillna(1);z=((x-mean)/std).fillna(0)
 m=Ridge(alpha=3000,fit_intercept=False).fit(z,e.ys[year]-e.base[year]);coefs.append(m.coef_)
C=np.vstack(coefs);sign=np.sign(C);stable=(np.abs(C).min(0)>1e-7)&((sign==sign[0]).all(0))
print('features',len(names),'stable_all_three',stable.sum())
for j in np.argsort(np.mean(np.abs(C),0))[::-1]:
 print(names[j],'stable' if stable[j] else 'unstable',*[round(v,7) for v in C[:,j]])
