"""Cross-fit a regularized residual stack across early/mid/late 2024 periods."""
from pathlib import Path
import glob,numpy as np
R=Path(__file__).resolve().parent;E=R/'old/experiments';anchor=np.load(R/'expanded_blend_2024.npz')['p'].astype(float);y=np.load(R/'expanded_blend_2024.npz')['y'].astype(float);month=np.load(E/'v3_subtypes_2024.npz')['month']
paths=list(E.glob('model_v[234]_*.npz'))+[E/'adaptive_gate_2024.npz']+list(R.glob('direct_brier_*.npz'))+list(R.glob('future_teacher_*.npz'))+list(R.glob('trackman_*_2024.npz'))+[R/'structured_gate_2024.npz',R/'expanded_crossseason_2024.npz',R/'expanded_crossseason_pair_2024.npz',R/'expanded_recency_groups_2024.npz']
names=[];cols=[]
for f in paths:
 if not f.exists():continue
 z=np.load(f);key='p' if 'p' in z else None
 if key and z[key].shape==anchor.shape:
  p=z[key].astype(float)
  if np.all(np.isfinite(p)) and np.std(p-anchor)>1e-6:names.append(f.stem);cols.append(p-anchor)
# Add seed posterior members.
z=np.load(R/'seed_pool_2024.npz')
for j,seed in enumerate(z['seeds']):names.append('seed_'+str(seed));cols.append(z['predictions'][j]-anchor)
D=np.column_stack(cols);periods=[month<=5,(month>=6)&(month<=7),month>=8]
def sc(mask,p):q=y[mask];return 1e5*(1-np.mean((q-np.clip(p[mask],1e-5,1-1e-5))**2)/(q.mean()*(1-q.mean())))
print('features',len(names),'anchor',[sc(k,anchor) for k in periods])
results=[]
for alpha in (1e2,3e2,1e3,3e3,1e4,3e4,1e5,3e5,1e6,3e6,1e7):
 improvements=[];coefs=[];cross=anchor.copy()
 for j,va in enumerate(periods):
  tr=np.logical_not(va);mu=D[tr].mean(0);sd=D[tr].std(0)+1e-5;X=(D-mu)/sd;A=np.c_[np.ones(tr.sum()),X[tr]];reg=np.eye(A.shape[1])*alpha;reg[0,0]=0;c=np.linalg.solve(A.T@A+reg,A.T@(y[tr]-anchor[tr]));p=anchor+np.c_[np.ones(len(y)),X]@c;cross[va]=p[va];improvements.append(sc(va,p)-sc(va,anchor));coefs.append(c)
 results.append((min(improvements),np.mean(improvements),alpha,improvements));print(alpha,[round(q,3) for q in improvements],'cross',round(sc(np.ones(len(y),bool),cross),3),flush=True)
print('best',max(results))

# Fit all rows with the selected regularization and report only descriptive in-sample ceiling.
alpha=max(results)[2];mu=D.mean(0);sd=D.std(0)+1e-5;X=(D-mu)/sd;A=np.c_[np.ones(len(y)),X];reg=np.eye(A.shape[1])*alpha;reg[0,0]=0;c=np.linalg.solve(A.T@A+reg,A.T@(y-anchor));p=np.clip(anchor+A@c,1e-5,1-1e-5)
print('full score',sc(np.ones(len(y),bool),p),'periods',[sc(k,p) for k in periods])
for j in np.argsort(np.abs(c[1:]))[::-1][:15]:print(round(c[j+1],6),names[j])
np.savez_compressed('period_crossfit_stack_2024.npz',y=y,p=p,names=np.array(names),coef=c,mean=mu,std=sd,alpha=alpha)

# Prune noisy channels by full-fit coefficient magnitude, then cross-fit again.
order=np.argsort(np.abs(c[1:]))[::-1]
print('pruned crossfit')
for nfeat in (3,5,8,12,16,24,32):
 idx=order[:nfeat];DD=D[:,idx];best_sub=None
 for aa in (1e3,3e3,1e4,3e4,1e5,3e5,1e6):
  cross=anchor.copy();imp=[]
  for va in periods:
   tr=~va;mm=DD[tr].mean(0);ss=DD[tr].std(0)+1e-5;XX=(DD-mm)/ss;AA=np.c_[np.ones(tr.sum()),XX[tr]];rr=np.eye(AA.shape[1])*aa;rr[0,0]=0;cc=np.linalg.solve(AA.T@AA+rr,AA.T@(y[tr]-anchor[tr]));pp=anchor+np.c_[np.ones(len(y)),XX]@cc;cross[va]=pp[va];imp.append(sc(va,pp)-sc(va,anchor))
  row=(min(imp),sc(np.ones(len(y),bool),cross),aa,imp)
  if best_sub is None or row>best_sub:best_sub=row
 print(nfeat,best_sub,[names[j] for j in idx])
