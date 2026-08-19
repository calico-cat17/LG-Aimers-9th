"""Build compact prior-season Trackman tables for mapping confidence ablation."""
from pathlib import Path
import numpy as np,pandas as pd

R=Path(__file__).resolve().parent;metrics=['rel_speed','spin_rate','induced_vert_break','horz_break','extension','rel_height','rel_side','zone_speed']
tm=pd.read_csv(R/'data/trackman_history.csv',usecols=['pitcher_trackman_id','season','pitch_type_group']+metrics)
mp=pd.read_csv(R/'artifacts/mapping_audit_coarse.csv')

# Annual sufficient statistics, later accumulated strictly before target season.
g=tm.groupby(['pitcher_trackman_id','season'],observed=True)
annual=g[metrics].agg(['count','sum'])
for c in metrics:annual[(c,'sumsq')]=g[c].apply(lambda x:np.square(x.dropna().to_numpy(float)).sum())
annual.columns=['__'.join(c) for c in annual.columns];annual=annual.reset_index()
pitch=tm.groupby(['pitcher_trackman_id','season','pitch_type_group'],observed=True).size().unstack(fill_value=0)

specs=[]
for threshold,margin,tag in [(.8,.03,'cov80'),(.7,.05,'cov89'),(.5,.05,'cov96')]:
 specs.append((tag,mp.loc[mp.mutual & (mp.similarity>=threshold)&(mp.margin>=margin)].copy()))
mix=pd.read_csv(R/'artifacts/mapping_audit_pitchmix.csv');strict=mix.loc[mix.mutual&(mix.similarity>=.9)&(mix.margin>=.03)];consensus=mix.loc[mix.mutual&mix.agree&(mix.similarity>=.3)];specs.append(('consensus',pd.concat([strict,consensus]).drop_duplicates('pitcher_id')))
for tag,mapping in specs:
 rows=[]
 for rec in mapping.itertuples(index=False):
  a=annual.loc[annual.pitcher_trackman_id.eq(rec.pitcher_trackman_id)].sort_values('season');q=pitch.loc[rec.pitcher_trackman_id] if rec.pitcher_trackman_id in pitch.index.levels[0] else pd.DataFrame()
  for target in range(2019,2026):
   h=a.loc[a.season<target];row={'pitcher_id':str(rec.pitcher_id),'season':target,'tm_mapping_similarity':rec.similarity,'tm_n':int(h['rel_speed__count'].sum()) if len(h) else 0}
   for c in metrics:
    n=h[f'{c}__count'].sum();sm=h[f'{c}__sum'].sum();ss=h[f'{c}__sumsq'].sum();row[f'tm_{c}_mean']=sm/n if n else np.nan;row[f'tm_{c}_std']=np.sqrt(max((ss-sm*sm/n)/(n-1),0)) if n>1 else np.nan
   if len(q): counts=q.loc[q.index<target].sum(axis=0);den=counts.sum()
   else:counts=pd.Series(dtype=float);den=0
   for typ in ['fastball','breaking','offspeed','other']:row[f'tm_{typ}_rate']=counts.get(typ,0)/den if den else np.nan
   rows.append(row)
 out=pd.DataFrame(rows);path=R/f'artifacts/trackman_prior_{tag}.csv';out.to_csv(path,index=False);print(tag,len(mapping),out.shape,path,flush=True)
