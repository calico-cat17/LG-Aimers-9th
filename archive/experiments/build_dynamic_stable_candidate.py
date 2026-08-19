"""Compose stable context channels with the row-local dual dynamic prior."""
import numpy as np

dctx=np.load("evaluation/context_adjusted_psych.npz")
dplat=np.load("evaluation/context_platoon_pitchmix.npz")
dfilm=np.load("evaluation/psych_regime_film.npz")
s=np.load("evaluation/anchors/adaptive_gate.npz")
d=np.load("evaluation/dual_dynamic_prior.npz")
weights=(.75,1.,1.25,0.)
def compose(y):
    p=s[f"p{y}"].copy()
    for weight,z in zip(weights[:3],(dctx,dplat,dfilm)):
        p+=weight*(z[f"p{y}"]-z[f"base{y}"])
    p+=weights[3]*d[f"corr{y}"]
    return np.clip(p,1e-6,1-1e-6)
p23=compose(23);p24=compose(24)
np.savez_compressed(
 "evaluation/dynamic_stable_stack.npz",
 y23=s["y23"],p23=p23,base23=s["p23"],pitcher23=s["pitcher23"],
 y24=s["y24"],p24=p24,base24=s["p24"],pitcher24=s["pitcher24"],
 weights=np.asarray(weights),
)
