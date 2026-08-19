"""Forward-month ablation around the strongest expanded prediction."""
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent; E=R/'old/experiments'
z=np.load(R/'expanded_blend_2024.npz'); y=z['y'].astype(float); anchor=z['p'].astype(float)
month=np.load(E/'v3_subtypes_2024.npz')['month'].astype(int)
files=[E/'adaptive_gate_2024.npz',E/'model_v2_d7.npz',E/'model_v2_d8.npz',E/'model_v3_2024.npz',
 E/'model_v3_decay_25.npz',E/'model_v3_decay_30.npz',E/'model_v3_decay_40.npz',E/'model_v3_depth_6.npz',
 E/'model_v3_seed_260811.npz',E/'model_v3_seed_3407.npz',E/'model_v3_seed_42.npz',E/'model_v4_d7.npz']

def sc(mask,p):
 q=y[mask]; pp=np.clip(p[mask],1e-5,1-1e-5);return 1e5*(1-np.mean((q-pp)**2)/(q.mean()*(1-q.mean())))

masks={'early':month<=5,'mid':(month>=6)&(month<=7),'late':month>=8,'all':np.ones(len(y),bool)}
print('anchor',{k:sc(v,anchor) for k,v in masks.items()})
rows=[]
for f in files:
 p=np.load(f)['p'].astype(float)
 for w in (-.25,-.1,.1,.2,.35,.5):
  blend=anchor+w*(p-anchor)
  vals={k:sc(v,blend)-sc(v,anchor) for k,v in masks.items()}
  rows.append((min(vals['mid'],vals['late']),vals['all'],vals,f.stem,w))
print('best stable single blends')
for row in sorted(rows,reverse=True)[:20]:print(row)

# Greedy additions must improve both mid and late; the late period is never used for weight fitting,
# only as a robustness veto. This prevents a large full-year score from hiding temporal failure.
cur=anchor.copy(); chosen=[]
for step in range(5):
 candidates=[]
 for f in files:
  if f.stem in [c[0] for c in chosen]:continue
  p=np.load(f)['p'].astype(float)
  for w in (-.15,-.05,.05,.1,.2,.3):
   b=cur+w*(p-cur);dm=sc(masks['mid'],b)-sc(masks['mid'],cur);dl=sc(masks['late'],b)-sc(masks['late'],cur)
   candidates.append((min(dm,dl),dm+dl,f.stem,w,b))
 best=max(candidates,key=lambda q:(q[0],q[1]));
 if best[0]<=0:break
 chosen.append((best[2],best[3],best[0]));cur=best[4]
 print('step',step+1,chosen[-1],{k:sc(v,cur) for k,v in masks.items()})
np.savez_compressed('safe_expansion_2024.npz',y=y,p=cur,month=month)
