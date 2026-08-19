"""Re-evaluate state-space predictions as a diversity channel, not standalone."""
import numpy as np
import pandas as pd
from hypotheses import hstate
from evaluate_crossfit_psych_profiles import load_predictions,skill
from evaluate_psych_residual_on_adaptive import fit_adaptive

raw=pd.read_csv('data/train.csv',low_memory=False)
parts={y:load_predictions(raw,y) for y in (2022,2023,2024)}
r22,y22,g22,o22=parts[2022];r23,y23,g23,o23=parts[2023];r24,y24,g24,o24=parts[2024]
gate=fit_adaptive(g22,y22,o22,2650023)
base={2023:np.clip(o23+gate.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']}
rows={2023:r23,2024:r24};ys={2023:y23,2024:y24};hp={}
for y in (2023,2024):
 yy,hp[y]=hstate(y,irm_lambda=0,epochs=30);assert np.array_equal(yy.astype(np.float32),ys[y].astype(np.float32));print('trained',y,skill(ys[y],hp[y]),flush=True)
grid=np.linspace(0,1,41);trial=[]
for w in grid:
 scores={y:skill(ys[y],np.clip((1-w)*base[y]+w*hp[y],1e-6,1-1e-6)) for y in (2023,2024)}
 gains={y:scores[y]-skill(ys[y],base[y]) for y in scores};trial.append((min(gains.values()),sum(gains.values())/2,w,scores,gains))
best=max(trial,key=lambda z:(z[0],z[1]));print('BEST',best,flush=True)
w=best[2];np.savez_compressed('evaluation/diversity_hstate.npz',y23=y23,p23=(1-w)*base[2023]+w*hp[2023],base23=base[2023],pitcher23=r23.pitcher_id.to_numpy(),y24=y24,p24=(1-w)*base[2024]+w*hp[2024],base24=base[2024],pitcher24=r24.pitcher_id.to_numpy(),hstate23=hp[2023],hstate24=hp[2024],weight=w)
