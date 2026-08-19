"""Build prior-only pitch-choice probabilities from mapped Trackman history."""
import numpy as np
import pandas as pd

GROUPS=['fastball','breaking','offspeed','other']
tm=pd.read_csv('data/trackman_history.csv',usecols=['pitcher_trackman_id','season','balls_before','strikes_before','batter_hand','pitch_type_group'])
mapping=pd.read_csv('artifacts/pitcher_trackman_mapping.csv',dtype={'pitcher_id':str})
tm=tm.merge(mapping[['pitcher_id','pitcher_trackman_id','mapping_similarity']],on='pitcher_trackman_id',how='inner')
tm['pitcher_id']=tm.pitcher_id.astype(str);tm['batter_hand']=tm.batter_hand.astype(str)
rows=[]
for target in (2022,2023,2024,2025):
 h=tm.loc[tm.season.lt(target)].copy()
 overall=h.groupby(['pitcher_id','pitch_type_group']).size().unstack(fill_value=0).reindex(columns=GROUPS,fill_value=0)
 total=overall.sum(1);prior=(overall+50*np.array([.5,.3,.15,.05]))/(total.to_numpy()[:,None]+50)
 context=h.groupby(['pitcher_id','balls_before','strikes_before','batter_hand','pitch_type_group']).size().unstack(fill_value=0).reindex(columns=GROUPS,fill_value=0)
 idx=context.index;parent=prior.reindex(idx.get_level_values('pitcher_id')).to_numpy();n=context.sum(1).to_numpy();prob=(context.to_numpy()+100*parent)/(n[:,None]+100)
 out=idx.to_frame(index=False);out['season']=target;out['latent_pitch_n']=n
 out['latent_mapping_similarity']=mapping.set_index('pitcher_id').mapping_similarity.reindex(out.pitcher_id).to_numpy()
 for j,g in enumerate(GROUPS):out[f'latent_{g}_prob']=prob[:,j]
 rows.append(out)
pd.concat(rows,ignore_index=True).to_csv('artifacts/latent_pitch_context.csv',index=False)
print('saved',sum(map(len,rows)))
