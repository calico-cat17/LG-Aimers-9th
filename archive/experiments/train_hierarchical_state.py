"""Forward-evaluate the hierarchical state-space model on 2022, 2023 and 2024."""
from __future__ import annotations
import time
import numpy as np, pandas as pd, torch
from torch import nn
from src.hierarchical_state import (HierarchicalState, SITUATIONS, build_situations,
                                    build_state_inputs)
from src.season_delta_features import build_snapshots

torch.manual_seed(0); torch.set_num_threads(6)
tr = pd.read_csv('data/train.csv', low_memory=False)
CONTEXT = ['balls_before','strikes_before','outs_before','inning','num_runners_on','li',
           'score_diff_pitcher_team','home_win_expectancy','asof_pitcher_middle_rate',
           'asof_pitcher_reverse_rate','asof_pitcher_ball_rate','asof_batter_success_rate',
           'asof_batter_middle_rate','asof_pitcher_fastball_rate','asof_pitcher_breaking_rate']

def season_state(frame, snap, prior):
    keys = pd.MultiIndex.from_arrays([frame.pitcher_id.astype(str), frame.season.astype(int)])
    s = snap.set_index(['pitcher_id','season'])
    prev_n = s.snapshot_n.reindex(keys).fillna(0).to_numpy(float)
    prev_c = s['snapshot_asof_pitcher_success_rate_count'].reindex(keys).fillna(0).to_numpy(float)
    n = pd.to_numeric(frame.asof_pitcher_n, errors='coerce').fillna(0).to_numpy(float)
    r = pd.to_numeric(frame.asof_pitcher_success_rate, errors='coerce').fillna(prior).to_numpy(float)
    sn = np.maximum(n - prev_n, 0); sc = np.maximum(n * r - prev_c, 0)
    return np.divide(sc, sn, out=np.full(len(frame), prior), where=sn > 0), sn

def run(target, epochs=30, slope_sd=.15, batter_sd=.05, lr=3e-3, emb_lr=.05, batch=16384):
    hist = tr[tr.season < target]
    snap = build_snapshots(hist); prior = float(hist.control_success.mean())
    frame = tr[tr.season <= target].reset_index(drop=True)
    rate, sn = season_state(frame, snap, prior)
    state = build_state_inputs(frame, rate, sn, prior)
    sit = build_situations(frame)
    ctx = frame[CONTEXT].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(np.float32)
    ctx = (ctx - ctx.mean(0)) / (ctx.std(0) + 1e-6)
    pid = pd.factorize(frame.pitcher_id.astype(str))[0]
    bid = pd.factorize(frame.batter_id.astype(str))[0]
    y = frame.control_success.to_numpy(np.float32); season = frame.season.to_numpy()
    te = season == target; fit = np.flatnonzero(~te); tei = np.flatnonzero(te)
    # recency weights, matching the tree pipeline so the comparison is about structure
    w = np.power(.55, target - season[fit]).astype(np.float32)
    dev = 'cpu'
    T = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=dev)
    S, Z, C, P, B, Y, W = T(state), T(sit), T(ctx), T(pid, torch.long), T(bid, torch.long), T(y), T(w)
    model = HierarchicalState(pid.max()+1, bid.max()+1, ctx.shape[1]).to(dev)
    with torch.no_grad():
        p0 = float(np.average(y[fit], weights=w)); model.season_level.fill_(np.log(p0/(1-p0)))
    emb = list(model.slopes.parameters()) + list(model.batter.parameters())
    ids = {id(q) for q in emb}
    rest = [q for q in model.parameters() if id(q) not in ids]
    # Embeddings see each pitcher only a few times per epoch, so they need their
    # own, much larger step size or they never leave initialisation.
    opt = torch.optim.Adam([{'params': rest, 'lr': lr},
                            {'params': emb, 'lr': emb_lr}])
    lossf = nn.BCEWithLogitsLoss(reduction='none')
    t0 = time.time()
    for ep in range(epochs):
        perm = np.random.permutation(len(fit))
        for st in range(0, len(perm), batch):
            idx = fit[perm[st:st+batch]]; ii = T(idx, torch.long)
            wb = W[perm[st:st+batch]]
            logit = model(S[ii], Z[ii], C[ii], P[ii], B[ii])
            loss = (lossf(logit, Y[ii]) * wb).mean() + model.penalty(P[ii], B[ii], slope_sd, batter_sd)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        ii = T(tei, torch.long)
        p = torch.sigmoid(model(S[ii], Z[ii], C[ii], P[ii], B[ii])).numpy().astype(float)
    yt = y[te].astype(float); V = yt.mean()*(1-yt.mean())
    bias = p.mean()-yt.mean(); Vp = p.var(); Cv = ((p-p.mean())*(yt-yt.mean())).mean(); beta = Cv/Vp
    sc = 1e5*(1-((p-yt)**2).mean()/V)
    print(f'{target}: score={sc:9.1f}  refine={1e5*Cv*Cv/(Vp*V):8.1f} slope={1e5*Vp*(1-beta)**2/V:7.1f} '
          f'intercept={1e5*bias*bias/V:7.1f} beta={beta:.3f} bias={bias:+.4f}  [{time.time()-t0:.0f}s]', flush=True)
    sl = model.slopes.weight.detach().numpy()
    print('   pitcher slope sd by situation: ' +
          '  '.join(f'{n_}={sl[:,k].std():.3f}' for k, n_ in enumerate(SITUATIONS)), flush=True)
    np.save(f'/tmp/hstate_{target}.npy', p)
    return sc

print('=== hierarchical state-space (logit scale, pooled pitcher slopes) ===', flush=True)
for t in (2022, 2023, 2024):
    run(t)
    for sd in (.05, .30):
        run(t, slope_sd=sd)
