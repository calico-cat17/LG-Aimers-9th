import pandas as pd
p='artifacts/trackman_prior_features.csv';x=pd.read_csv(p)
keys=['pitcher_id','season','tm_mapping_similarity','tm_n']
metrics=['rel_speed','spin_rate','induced_vert_break','horz_break','extension','rel_height','rel_side','zone_speed']
cols=keys+[f'tm_{m}_{s}' for m in metrics for s in ('mean','std')]+[f'tm_{q}_rate' for q in ('fastball','breaking','offspeed','other')]+['tm_last_n']+[f'tm_last_{m}_std' for m in metrics]+[f'tm_{m}_trend' for m in metrics]+[f'tm_last_{q}_rate' for q in ('fastball','breaking','offspeed','other')]
x[cols].to_csv('artifacts/trackman_selected_features.csv',index=False)
print(len(cols),x[cols].shape)
