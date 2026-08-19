"""Prior-season Trackman summaries over the expanded assignment mapping.

Everything is a cumulative sum over seasons, so it is built with one groupby and
a shift rather than a loop over pitchers.
"""
from __future__ import annotations
import numpy as np, pandas as pd

METRICS = ['rel_speed','spin_rate','induced_vert_break','horz_break','extension',
           'rel_height','rel_side','zone_speed']
GROUPS = ['fastball','breaking','offspeed','other']
SEASONS = list(range(2019, 2026))

def cumulative(agg, pitchers):
    """Totals over every season strictly before the index season."""
    full = agg.reindex(pd.MultiIndex.from_product([pitchers, SEASONS],
                                                  names=['pitcher_id','season']), fill_value=0.0)
    csum = full.groupby(level=0).cumsum()
    prior = csum.groupby(level=0).shift(1).fillna(0.0)     # seasons < target
    last = full.groupby(level=0).shift(1).fillna(0.0)      # season == target-1
    earlier = prior.groupby(level=0).shift(1).fillna(0.0)  # seasons < target-1
    return prior, last, earlier

def main(floor=0.80, out='artifacts/trackman_prior_features_v2.csv'):
    mapping = pd.read_csv('artifacts/pitcher_trackman_mapping_v2.csv', dtype={'pitcher_id': str})
    mapping = mapping[mapping.mapping_similarity >= floor]
    tm = pd.read_csv('data/trackman_history.csv',
                     usecols=['pitcher_trackman_id','season','pitch_type_group',*METRICS])
    tm = tm.merge(mapping[['pitcher_id','pitcher_trackman_id']], on='pitcher_trackman_id', how='inner')

    parts = {}
    for c in METRICS:
        v = tm[c]
        parts[f'{c}__s'] = v
        parts[f'{c}__q'] = v ** 2
        parts[f'{c}__n'] = v.notna().astype(float)
    for p in GROUPS:
        m = tm.pitch_type_group.eq(p)
        parts[f'grp_{p}__n'] = m.astype(float)
        for c in ['rel_speed','spin_rate','induced_vert_break','horz_break']:
            parts[f'{p}_{c}__s'] = tm[c].where(m)
            parts[f'{p}_{c}__n'] = tm[c].where(m).notna().astype(float)
    wide = pd.DataFrame(parts)
    wide['pitcher_id'] = tm.pitcher_id.to_numpy(); wide['season'] = tm.season.to_numpy()
    agg = wide.groupby(['pitcher_id','season']).sum()
    pitchers = sorted(mapping.pitcher_id.unique())
    prior, last, earlier = cumulative(agg, pitchers)

    idx = prior.index
    out_df = pd.DataFrame(index=idx)
    div = lambda a, b: np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)
    out_df['tm_n'] = prior[[f'{METRICS[0]}__n']].to_numpy().ravel()
    out_df['tm_last_n'] = last[[f'{METRICS[0]}__n']].to_numpy().ravel()
    for c in METRICS:
        for tag, src in (('', prior), ('last_', last)):
            n = src[f'{c}__n'].to_numpy(); s = src[f'{c}__s'].to_numpy(); q = src[f'{c}__q'].to_numpy()
            mean = div(s, n)
            out_df[f'tm_{tag}{c}_mean'] = mean
            out_df[f'tm_{tag}{c}_std'] = np.sqrt(np.clip(div(q, n) - mean ** 2, 0, None))
        em = div(earlier[f'{c}__s'].to_numpy(), earlier[f'{c}__n'].to_numpy())
        out_df[f'tm_{c}_trend'] = out_df[f'tm_last_{c}_mean'].to_numpy() - em
    for tag, src in (('', prior), ('last_', last)):
        tot = sum(src[f'grp_{p}__n'].to_numpy() for p in GROUPS)
        for p in GROUPS:
            out_df[f'tm_{tag}{p}_rate'] = div(src[f'grp_{p}__n'].to_numpy(), tot)
    for p in GROUPS:
        for c in ['rel_speed','spin_rate','induced_vert_break','horz_break']:
            for tag, src in (('', prior), ('last_', last)):
                out_df[f'tm_{tag}{p}_{c}_mean'] = div(src[f'{p}_{c}__s'].to_numpy(),
                                                      src[f'{p}_{c}__n'].to_numpy())
    out_df = out_df.reset_index()
    out_df['tm_mapping_similarity'] = out_df.pitcher_id.map(
        mapping.set_index('pitcher_id').mapping_similarity)
    ref = pd.read_csv('model/trackman_prior_features.csv', nrows=1)
    out_df = out_df.reindex(columns=ref.columns)
    # keep tm_n == 0 rows so the only difference from the old table is the mapping
    out_df.to_csv(out, index=False)
    print('saved', out, out_df.shape, 'pitchers', out_df.pitcher_id.nunique())

if __name__ == '__main__':
    main()
