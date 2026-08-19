"""Leakage-safe prior-season target histories for entities in labelled training data."""
from __future__ import annotations
import numpy as np
import pandas as pd

ENTITIES=('pitcher_id','batter_id','pitcher_team_id','batter_team_id')

def build_prior_season_tables(history: pd.DataFrame) -> dict[str,pd.DataFrame]:
    tables={}
    for entity in ENTITIES:
        g=(history.groupby([entity,'season'],sort=False).control_success
           .agg(['sum','count']).reset_index().sort_values([entity,'season']))
        g['rate']=g['sum']/g['count'];g['prev2_rate']=g.groupby(entity).rate.shift(1)
        g['prev2_n']=g.groupby(entity)['count'].shift(1);g['season']=g['season'].astype(int)+1
        tables[entity]=g.rename(columns={'sum':'prev1_sum','count':'prev1_n','rate':'prev1_rate'})
    return tables

def attach_prior_seasons(x:pd.DataFrame,raw:pd.DataFrame,tables:dict[str,pd.DataFrame],prior:float)->pd.DataFrame:
    out=x.copy();rr=raw.reset_index(drop=True)
    for entity,tab in tables.items():
        prefix={'pitcher_id':'pitcher','batter_id':'batter','pitcher_team_id':'pitcher_team','batter_team_id':'batter_team'}[entity]
        left=pd.DataFrame({entity:rr[entity].astype(str),'season':rr.season.astype(int),'_i':np.arange(len(rr))})
        right=tab.copy();right[entity]=right[entity].astype(str);m=left.merge(right,on=[entity,'season'],how='left',sort=False).sort_values('_i')
        n=m.prev1_n.fillna(0).to_numpy(float);r1=m.prev1_rate.fillna(prior).to_numpy(float);n2=m.prev2_n.fillna(0).to_numpy(float);r2=m.prev2_rate.fillna(prior).to_numpy(float)
        strength=80. if 'team' not in prefix else 400.
        s1=(r1*n+prior*strength)/(n+strength);s2=(r2*n2+prior*strength)/(n2+strength)
        out[f'{prefix}_prevseason_n']=n;out[f'{prefix}_prevseason_success']=s1;out[f'{prefix}_prev2season_success']=s2;out[f'{prefix}_season_trend']=s1-s2;out[f'{prefix}_prevseason_reliability']=n/(n+strength)
    out['prevseason_matchup_gap']=out.pitcher_prevseason_success-out.batter_prevseason_success
    out['team_prevseason_gap']=out.pitcher_team_prevseason_success-out.batter_team_prevseason_success
    return out
