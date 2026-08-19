"""Uncertainty and worst-case audit for forward candidate predictions."""
import numpy as np

z=np.load('team_system_forward.npz');rng=np.random.default_rng(260812)
def audit(year):
 suffix=str(year)[-2:];y=z[f'y{suffix}'];a=z[f'base{suffix}'];p=z[f'p{suffix}'];pid=z[f'pitcher{suffix}']
 loss_a=(a-y)**2;loss_p=(p-y)**2;delta=loss_a-loss_p
 se=delta.std(ddof=1)/np.sqrt(len(delta));unique=np.unique(pid);means=[]
 # Cluster bootstrap preserves within-pitcher dependence and varying pitcher size.
 groups={q:delta[pid==q] for q in unique}
 for _ in range(1000):
  sampled=rng.choice(unique,len(unique),replace=True);means.append(np.concatenate([groups[q] for q in sampled]).mean())
 lo,hi=np.quantile(means,[.025,.975]);r=y.mean();score_gain=1e5*delta.mean()/(r*(1-r))
 print(year,'n',len(y),'pitchers',len(unique),'brier_gain',delta.mean(),'row_se',se,'z',delta.mean()/se,'cluster_ci',lo,hi,'score_gain',score_gain,'pass',lo>0)
 return lo,score_gain
results=[audit(2023),audit(2024)];print('worst_score_gain',min(q[1] for q in results),'all_ci_positive',all(q[0]>0 for q in results))
