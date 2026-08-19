"""Train the production 9-regime FiLM residual using forward base predictions."""
import copy,random
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,TensorDataset
from evaluate_crossfit_psych_profiles import load_predictions
from evaluate_psych_residual_on_adaptive import fit_adaptive
from src.context_adjusted_psych import attach_context_adjusted_psych,build_profiles
from src.psych_film_expert import PsychRegimeFiLMExpert,production_arrays

torch.set_num_threads(6);seed=260813;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
raw=pd.read_csv('data/train.csv',low_memory=False);parts={y:load_predictions(raw,y) for y in (2022,2023,2024)}
r22,y22,g22,b22=parts[2022];r23,y23,g23,b23=parts[2023];r24,y24,g24,b24=parts[2024];gate23=fit_adaptive(g22,y22,b22,2020023)
rows={2022:r22,2023:r23,2024:r24};ys={2022:y22,2023:y23,2024:y24};base={2022:b22,2023:np.clip(b23+gate23.predict(g23),1e-6,1-1e-6),2024:np.load('old/experiments/adaptive_gate_2024.npz')['p']}
data={}
for year in rows:
    psych=attach_context_adjusted_psych(rows[year],raw.loc[raw.season.lt(year)],400.,100.)
    data[year]=production_arrays(rows[year],base[year],psych)
arr=[np.concatenate([data[y][j] for y in rows]) for j in range(5)];target=np.concatenate([ys[y]-base[y] for y in rows]).astype(np.float32);bx,px,qx,rel,prior=arr
means=[bx.mean(0),px.mean(0),qx.mean(0)];stds=[bx.std(0)+1e-4,px.std(0)+1e-4,qx.std(0)+1e-4];bx=np.nan_to_num((bx-means[0])/stds[0]);px=np.nan_to_num((px-means[1])/stds[1]);qx=np.nan_to_num((qx-means[2])/stds[2])
ds=TensorDataset(*map(torch.from_numpy,(bx,px,qx,rel,prior,target)));loader=DataLoader(ds,batch_size=4096,shuffle=True,num_workers=0);model=PsychRegimeFiLMExpert(bx.shape[1],px.shape[1],qx.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=5e-3);bestloss=1e9;best=None
for epoch in range(8):
    model.train();total=0
    for b,p,q,r,pr,y in loader:
        opt.zero_grad();corr,delta,gate,regime,_,members=model(b,p,q,r,pr);loss=((corr-y)**2).mean()+.10*((gate*(pr<.15))**2).mean()+.03*(delta**2).mean()+.02*((regime.mean(0)-1/9)**2).mean()+.15*((members-y[:,None])**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();total+=loss.item()*len(y)
    avg=total/len(ds);print('final epoch',epoch+1,'loss',avg,flush=True)
    if avg<bestloss:bestloss=avg;best=copy.deepcopy(model.state_dict())
model.load_state_dict(best);torch.save(model.state_dict(),'model_hierarchical_stack/psych_regime.pt');np.savez_compressed('model_hierarchical_stack/psych_regime_assets.npz',base_mean=means[0],base_std=stds[0],psych_mean=means[1],psych_std=stds[1],pressure_mean=means[2],pressure_std=stds[2],scale=np.float32(.75))
build_profiles(raw,400.,100.).to_pickle('model_hierarchical_stack/psych_regime_profile.pkl');print('saved production regime assets')
