"""Optimal one-to-one pitcher_id <-> pitcher_trackman_id assignment.

The existing mapping scores pitchers on their appearance schedule alone and keeps
a pair only when each is the other's best match. Greedy mutual-best throws away
every pitcher whose nearest rival is close, which is why coverage stalls at 63%.

Two changes: a signature that also carries how a pitcher is *used* (batter hand,
count state, inning), and a global assignment instead of greedy matching. An
assignment is one-to-one by construction, so a pair no longer has to win by a
margin - it only has to be the best arrangement overall.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

HAND = {'Left': 1, 'Right': 2}

def blocks(df, pid, hand_col, hand_map=None):
    d = df.copy()
    d['_h'] = d[hand_col].map(hand_map) if hand_map else d[hand_col]
    d['sched'] = d.season.astype(str)+'_'+d.game_month.astype(str)+'_'+d.game_dayofweek.astype(str)
    d['usage'] = d.balls_before.astype(str)+d.strikes_before.astype(str)+'_'+d['_bh'].astype(str)
    d['inn'] = d.inning.clip(1, 10).astype(str)
    out = []
    for col in ('sched', 'usage', 'inn'):
        m = d.groupby([pid, col]).size().unstack(fill_value=0)
        out.append(m)
    return out, d.groupby(pid)['_h'].first()

def main():
    tr = pd.read_csv('data/train.csv', usecols=['season','game_month','game_dayofweek','inning',
        'balls_before','strikes_before','pitcher_id','pitcher_hand','batter_hand'])
    tm = pd.read_csv('data/trackman_history.csv', usecols=['season','game_month','game_dayofweek',
        'inning','balls_before','strikes_before','pitcher_trackman_id','pitcher_hand','batter_hand'])
    tr['_bh'] = tr.batter_hand.astype(int)
    tm['_bh'] = tm.batter_hand.map(HAND).fillna(0).astype(int)
    A, ha = blocks(tr, 'pitcher_id', 'pitcher_hand')
    B, hb = blocks(tm, 'pitcher_trackman_id', 'pitcher_hand', HAND)

    sim = np.zeros((len(A[0]), len(B[0])))
    weights = (0.6, 0.3, 0.1)          # schedule dominates, usage disambiguates
    for w, a, b in zip(weights, A, B):
        cols = a.columns.union(b.columns)
        na = normalize(a.reindex(columns=cols, fill_value=0).to_numpy(float))
        nb = normalize(b.reindex(columns=cols, fill_value=0).to_numpy(float))
        sim += w * (na @ nb.T)
    ha_v = ha.reindex(A[0].index).to_numpy()
    hb_v = hb.reindex(B[0].index).to_numpy()
    sim[ha_v[:, None] != hb_v[None, :]] = -1.0

    rows, cols = linear_sum_assignment(-sim)
    score = sim[rows, cols]
    order = np.sort(sim, axis=1)
    margin = order[:, -1] - order[:, -2]
    mapping = pd.DataFrame({
        'pitcher_id': A[0].index[rows].astype(str),
        'pitcher_trackman_id': B[0].index[cols],
        'mapping_similarity': score,
        'mapping_margin': margin[rows],
    }).sort_values('mapping_similarity', ascending=False)
    mapping.to_csv('artifacts/pitcher_trackman_mapping_v2.csv', index=False)

    n_tr = tr.groupby('pitcher_id').size()
    old = pd.read_csv('artifacts/pitcher_trackman_mapping.csv', dtype={'pitcher_id': str})
    merged = mapping.merge(old[['pitcher_id','pitcher_trackman_id']], on='pitcher_id',
                           how='left', suffixes=('', '_old'))
    same = merged.pitcher_trackman_id.eq(merged.pitcher_trackman_id_old)
    overlap = merged.pitcher_trackman_id_old.notna()
    print(f'assigned {len(mapping)} pitchers')
    print(f'agreement with the existing high-confidence mapping: '
          f'{same[overlap].mean():.3%} of {overlap.sum()} shared pitchers')
    for floor in (0.0, .30, .50, .70, .80, .90):
        keep = mapping[mapping.mapping_similarity >= floor]
        cov = n_tr.reindex(keep.pitcher_id.astype(type(n_tr.index[0]))).sum() / n_tr.sum()
        agree = same[overlap & (merged.mapping_similarity >= floor)].mean()
        print(f'  similarity >= {floor:.2f}: {len(keep):4d} pitchers, '
              f'{cov:6.1%} of train rows, agreement {agree:.1%}')

if __name__ == '__main__':
    main()
