"""Candidate hypotheses for autosearch.

Each entry is a function `f(target_season) -> (y_true, prediction)` trained only
on seasons before the target. Add a new hypothesis by writing one function and
registering it at the bottom; autosearch handles evaluation, logging and pruning.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

TRAIN = None
CONTEXT = ['balls_before', 'strikes_before', 'outs_before', 'inning', 'num_runners_on', 'li',
           'score_diff_pitcher_team', 'home_win_expectancy', 'asof_pitcher_middle_rate',
           'asof_pitcher_reverse_rate', 'asof_pitcher_ball_rate', 'asof_batter_success_rate',
           'asof_batter_middle_rate', 'asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate']


def train_frame() -> pd.DataFrame:
    global TRAIN
    if TRAIN is None:
        TRAIN = pd.read_csv('data/train.csv', low_memory=False)
    return TRAIN


# One target's v3 frame is ~4 GB, so exactly one is kept resident. Rebuilding
# costs ~45 s and is far cheaper than the OOM kill that maxsize>1 caused.
@functools.lru_cache(maxsize=1)
def tree_inputs(target: int, trackman: str):
    """v3 features and hierarchical base, with every summary fitted before `target`."""
    from src.preprocessing_v2 import build_v3_features
    from src.season_delta_features import build_snapshots
    from src.season_history_v3 import build_entity_snapshots
    tr = train_frame()
    hist = tr[tr.season < target]
    prior = float(hist.control_success.mean())
    ps = build_snapshots(hist)
    bs = build_entity_snapshots(hist, 'batter_id', 'asof_batter_n',
                                ['asof_batter_success_rate', 'asof_batter_middle_rate'], 'control_success')
    ms = build_entity_snapshots(hist, 'pitcher_id', 'asof_pitcher_pitchmix_n',
                                ['asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate',
                                 'asof_pitcher_offspeed_rate'])
    frame = tr[tr.season <= target].reset_index(drop=True)
    x, base = build_v3_features(frame, prior, ps, bs, ms, trackman)
    return frame, x, base, ps, prior


def residual_tree(target: int, trackman='model/trackman_prior_features.csv',
                  iterations=220, depth=8, lr=.035, decay=.55, seed=260803):
    import gc
    from catboost import CatBoostRegressor, Pool
    from src.preprocessing_v2 import CAT_V2
    frame, x, base, _, _ = tree_inputs(target, trackman)
    y = frame.control_success.to_numpy(float); season = frame.season.to_numpy()
    te = season == target; fit = ~te
    m = CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=lr, loss_function='RMSE',
                          l2_leaf_reg=12, random_strength=.35, bootstrap_type='Bernoulli',
                          subsample=.85, one_hot_max_size=16, random_seed=seed, thread_count=6,
                          allow_writing_files=False, verbose=0)
    pool = Pool(x[fit], (y - base)[fit], weight=np.power(decay, target - season[fit]), cat_features=CAT_V2)
    m.fit(pool)
    del pool; gc.collect()
    out = y[te], np.clip(base[te] + m.predict(x[te]), 1e-5, 1 - 1e-5)
    del m; gc.collect()
    return out


def hstate(target: int, irm_lambda=0.0, epochs=30, slope_sd=.30, batter_sd=.05):
    """Hierarchical state-space model; `irm_lambda` adds the IRM gradient penalty.

    Each earlier season is one IRM environment. The penalty is the squared
    gradient of that environment's risk with respect to a dummy scale on the
    logit, which is large whenever a season would prefer a different output
    scaling - exactly the year-to-year rearrangement seen in 2023.
    """
    import torch
    from torch import nn
    from src.hierarchical_state import HierarchicalState, build_situations, build_state_inputs
    from src.season_delta_features import build_snapshots
    torch.manual_seed(0); torch.set_num_threads(6)
    tr = train_frame()
    hist = tr[tr.season < target]
    prior = float(hist.control_success.mean()); snap = build_snapshots(hist).set_index(['pitcher_id', 'season'])
    frame = tr[tr.season <= target].reset_index(drop=True)
    keys = pd.MultiIndex.from_arrays([frame.pitcher_id.astype(str), frame.season.astype(int)])
    pn = snap.snapshot_n.reindex(keys).fillna(0).to_numpy(float)
    pc = snap['snapshot_asof_pitcher_success_rate_count'].reindex(keys).fillna(0).to_numpy(float)
    n = pd.to_numeric(frame.asof_pitcher_n, errors='coerce').fillna(0).to_numpy(float)
    r = pd.to_numeric(frame.asof_pitcher_success_rate, errors='coerce').fillna(prior).to_numpy(float)
    sn = np.maximum(n - pn, 0); sc = np.maximum(n * r - pc, 0)
    rate = np.divide(sc, sn, out=np.full(len(frame), prior), where=sn > 0)

    S = build_state_inputs(frame, rate, sn, prior); Z = build_situations(frame)
    C = frame[CONTEXT].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(np.float32)
    C = (C - C.mean(0)) / (C.std(0) + 1e-6)
    pid = pd.factorize(frame.pitcher_id.astype(str))[0]
    bid = pd.factorize(frame.batter_id.astype(str))[0]
    y = frame.control_success.to_numpy(np.float32); season = frame.season.to_numpy()
    te = season == target; fit = np.flatnonzero(~te); tei = np.flatnonzero(te)
    T = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d)
    St, Zt, Ct, Pt, Bt, Yt = T(S), T(Z), T(C), T(pid, torch.long), T(bid, torch.long), T(y)
    W = T(np.power(.55, target - season[fit]).astype(np.float32))
    env = season[fit]

    model = HierarchicalState(pid.max() + 1, bid.max() + 1, C.shape[1])
    with torch.no_grad():
        p0 = float(np.average(y[fit], weights=np.power(.55, target - season[fit])))
        model.season_level.fill_(np.log(p0 / (1 - p0)))
    emb = list(model.slopes.parameters()) + list(model.batter.parameters())
    ids = {id(q) for q in emb}
    opt = torch.optim.Adam([{'params': [q for q in model.parameters() if id(q) not in ids], 'lr': 3e-3},
                            {'params': emb, 'lr': .05}])
    lf = torch.nn.BCEWithLogitsLoss(reduction='none')
    for _ in range(epochs):
        perm = np.random.permutation(len(fit))
        for st in range(0, len(perm), 16384):
            j = perm[st:st + 16384]; ii = T(fit[j], torch.long)
            logit = model(St[ii], Zt[ii], Ct[ii], Pt[ii], Bt[ii])
            per_row = lf(logit, Yt[ii]) * W[j]
            loss = per_row.mean() + model.penalty(Pt[ii], Bt[ii], slope_sd, batter_sd)
            if irm_lambda > 0:
                scale = torch.ones(1, requires_grad=True)
                penalty = 0.0
                for e in np.unique(env[j]):
                    mask = torch.as_tensor(env[j] == e)
                    if mask.sum() < 64:
                        continue
                    risk = (lf(logit[mask] * scale, Yt[ii][mask]) * W[j][mask.numpy()]).mean()
                    g = torch.autograd.grad(risk, [scale], create_graph=True)[0]
                    penalty = penalty + (g ** 2).sum()
                loss = loss + irm_lambda * penalty
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        ii = T(tei, torch.long)
        p = torch.sigmoid(model(St[ii], Zt[ii], Ct[ii], Pt[ii], Bt[ii])).numpy().astype(float)
    return y[te].astype(float), p


def shrunk_tree(target: int, factor: float):
    """The incumbent tree with its spread pulled in toward its own mean."""
    y, p = residual_tree(target)
    return y, p.mean() + factor * (p - p.mean())


REGISTRY = {
    'baseline_tree': (residual_tree, 'incumbent residual CatBoost, unchanged'),
    'trackman_v2': (functools.partial(residual_tree, trackman='artifacts/trackman_prior_features_v2.csv'),
                    'same tree over the 765-pitcher assignment mapping'),
    'hstate': (hstate, 'hierarchical state-space, no IRM penalty'),
    'irm_hstate_1': (functools.partial(hstate, irm_lambda=1.0), 'hierarchical state + IRM penalty, lambda=1'),
    'irm_hstate_10': (functools.partial(hstate, irm_lambda=10.0), 'hierarchical state + IRM penalty, lambda=10'),
    'irm_hstate_100': (functools.partial(hstate, irm_lambda=100.0), 'hierarchical state + IRM penalty, lambda=100'),
    'tree_shrunk_90': (functools.partial(shrunk_tree, factor=.90), 'incumbent tree, spread x0.90'),
    'tree_shrunk_75': (functools.partial(shrunk_tree, factor=.75), 'incumbent tree, spread x0.75'),
    'tree_lr02': (functools.partial(residual_tree, iterations=600, lr=.02), 'incumbent tree, lr 0.02 / 600 iters'),
}
