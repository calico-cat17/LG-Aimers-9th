"""Train a compact TabM probability model with a strict 2024 holdout."""
from __future__ import annotations

import copy, json, os, time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tabm import TabM

from src.deep_preprocessing import TabularPreprocessor


def skill(y, p):
    b = np.mean((y-p)**2); r = np.mean(y)
    return b, max(0., 100000*(1-b/(r*(1-r))))


@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); out=[]
    for xn, xc in loader:
        logits=model(xn.to(device), xc.to(device)).squeeze(-1)
        out.append(torch.sigmoid(logits).mean(1).cpu().numpy())
    return np.concatenate(out)


def make_model(prep, n_num):
    return TabM.make(n_num_features=n_num, cat_cardinalities=prep.cardinalities,
                     d_out=1, n_blocks=3, d_block=256, dropout=0.12, k=8,
                     arch_type='tabm-mini', start_scaling_init='normal')


def main():
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw=pd.read_csv('data/train.csv', low_memory=False)
    mask=raw.season.to_numpy()==2024; split=np.flatnonzero(mask)[0]
    prep=TabularPreprocessor().fit(raw.iloc[:split])
    xn,xc=prep.transform(raw); y=raw.control_success.to_numpy(np.float32)
    tn=torch.from_numpy(xn); tc=torch.from_numpy(xc); ty=torch.from_numpy(y)
    train=TensorDataset(tn[:split],tc[:split],ty[:split]); valid=TensorDataset(tn[split:],tc[split:])
    tl=DataLoader(train,batch_size=4096,shuffle=True,num_workers=0)
    vl=DataLoader(valid,batch_size=16384,shuffle=False,num_workers=0)
    model=make_model(prep,xn.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1.5e-3,weight_decay=1e-4)
    best=(1.,None,0); epochs_run=0
    for epoch in range(12):
        model.train(); start=time.time()
        for a,b,t in tl:
            a,b,t=a.to(device),b.to(device),t.to(device)
            opt.zero_grad(set_to_none=True); z=model(a,b).squeeze(-1)
            # Train every packed member; squared loss aligns directly with Brier.
            loss=((torch.sigmoid(z)-t[:,None])**2).mean()+.05*F.binary_cross_entropy_with_logits(z,t[:,None].expand_as(z))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
        p=predict(model,vl,device); br,sc=skill(y[split:],p)
        print(epoch,br,sc,'sec',time.time()-start,flush=True); epochs_run=epoch+1
        if br < best[0]-2e-6: best=(br,copy.deepcopy(model.state_dict()),0)
        else:
            best=(best[0],best[1],best[2]+1)
            if best[2]>=3: break
    model.load_state_dict(best[1]); p=predict(model,vl,device)
    os.makedirs('model_tabm',exist_ok=True)
    joblib.dump(prep,'model_tabm/preprocessor.joblib',compress=3)
    torch.save({'state_dict':best[1],'n_num':xn.shape[1],'cardinalities':prep.cardinalities},'model_tabm/tabm.pt')
    np.savez_compressed('model_tabm/validation.npz',y=y[split:],p=p)
    json.dump({'epochs':epochs_run,'best_brier':best[0],'skill':skill(y[split:],p)[1]},open('model_tabm/meta.json','w'),indent=2)

if __name__=='__main__': main()
