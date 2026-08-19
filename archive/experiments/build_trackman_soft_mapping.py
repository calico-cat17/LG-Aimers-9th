"""Shrink uncertain recovered Trackman identities toward season population values."""
import numpy as np,pandas as pd
p=pd.read_csv('artifacts/trackman_prior_cov96.csv',dtype={'pitcher_id':str});m=pd.read_csv('artifacts/mapping_audit_coarse.csv',dtype={'pitcher_id':str})[['pitcher_id','margin']];p=p.merge(m,on='pitcher_id',how='left')
strict=(p.tm_mapping_similarity>=.9)&(p.margin>=.03)
q=((p.tm_mapping_similarity-.5)/.4).clip(0,1)*(p.margin/.15).clip(0,1);q=np.where(strict,1,q*.35)
features=[c for c in p if c.startswith('tm_') and c not in ['tm_mapping_similarity','tm_n']]
for c in features:
 mean=p.groupby('season')[c].transform('mean');p[c]=mean+q*(p[c]-mean)
p['tm_n']=p.tm_n*q
p.drop(columns='margin').to_csv('artifacts/trackman_prior_soft.csv',index=False)
print('rows',len(p),'strict',strict.sum(),'soft q',pd.Series(q[~strict]).describe().to_dict())
