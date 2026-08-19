"""Base Trackman table plus compact pitch-family physical arsenal."""
import pandas as pd
full=pd.read_csv('artifacts/trackman_prior_features.csv',dtype={'pitcher_id':str});base=pd.read_csv('model/trackman_prior_features.csv',nrows=1).columns.tolist();extra=[]
for typ in ['fastball','breaking','offspeed']:
 for metric in ['rel_speed','spin_rate','induced_vert_break','horz_break']:extra.append(f'tm_{typ}_{metric}_mean')
full[base+extra].to_csv('artifacts/trackman_prior_arsenal_core.csv',index=False)
print(len(base+extra),extra)
