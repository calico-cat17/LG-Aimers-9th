"""Reconstruct and audit the site-refitted five-channel model plus affine calibration."""
import numpy as np
from evaluate_crossfit_psych_profiles import skill

A=np.load('evaluation/anchors/adaptive_gate.npz');P=np.load('evaluation/anchors/psych_latent.npz')
S=np.load('evaluation/split_prior_stack.npz');F=np.load('evaluation/psych_regime_film.npz')
M=np.load('evaluation/dynamic_stable_stack.npz');H=np.load('evaluation/anchors/hierarchical_stack.npz')
w=np.array([.6782,.5098,-.3057,-.5914,-.3688]);out={}
for y in (23,24):
 a=A[f'p{y}'];p=a+w[0]*(P[f'p{y}']-a)+w[1]*(S[f'p{y}']-P[f'p{y}'])+w[2]*(F[f'p{y}']-a)+w[3]*(M[f'p{y}']-a)+w[4]*(H[f'p{y}']-a)
 q=np.clip(p+.75*(.03*(p-.49)-.004),1e-6,1-1e-6);print(y,'refit',skill(A[f'y{y}'],p),'calibrated',skill(A[f'y{y}'],q),flush=True)
 out.update({f'y{y}':A[f'y{y}'],f'p{y}':q,f'base{y}':p,f'pitcher{y}':A[f'pitcher{y}']})
np.savez_compressed('evaluation/channel_calibrated.npz',**out)
