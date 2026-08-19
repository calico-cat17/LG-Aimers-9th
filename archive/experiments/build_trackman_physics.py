"""Compact physics-informed Trackman representation and arsenal prototypes."""
import numpy as np,pandas as pd
p=pd.read_csv('model/trackman_prior_features.csv',dtype={'pitcher_id':str})
p['tm_speed_loss']=p.tm_rel_speed_mean-p.tm_zone_speed_mean
p['tm_move_magnitude']=np.hypot(p.tm_induced_vert_break_mean,p.tm_horz_break_mean)
p['tm_move_angle']=np.arctan2(p.tm_induced_vert_break_mean,p.tm_horz_break_mean)
p['tm_spin_per_speed']=p.tm_spin_rate_mean/p.tm_rel_speed_mean.clip(lower=1)
p['tm_release_variance']=p[['tm_extension_std','tm_rel_height_std','tm_rel_side_std']].pow(2).sum(axis=1)
p['tm_movement_variance']=p[['tm_induced_vert_break_std','tm_horz_break_std']].pow(2).sum(axis=1)
p['tm_velocity_retention']=p.tm_zone_speed_mean/p.tm_rel_speed_mean.clip(lower=1)
mix=p[['tm_fastball_rate','tm_breaking_rate','tm_offspeed_rate','tm_other_rate']].clip(lower=1e-6)
p['tm_mix_entropy']=-(mix*np.log(mix)).sum(axis=1)

cols=['tm_rel_speed_mean','tm_spin_rate_mean','tm_induced_vert_break_mean','tm_horz_break_mean','tm_extension_mean','tm_rel_height_mean','tm_rel_side_mean','tm_fastball_rate','tm_breaking_rate','tm_offspeed_rate']
v=p.loc[p.tm_n>0,cols].copy();med=v.median();v=v.fillna(med);mu=v.mean();sd=v.std().replace(0,1);z=((v-mu)/sd).clip(-4,4).to_numpy(float)
rng=np.random.default_rng(260811);k=12;cent=z[rng.choice(len(z),k,replace=False)]
for _ in range(40):
 dist=((z[:,None,:]-cent[None,:,:])**2).sum(2);lab=dist.argmin(1);new=np.array([z[lab==j].mean(0) if np.any(lab==j) else cent[j] for j in range(k)])
 if np.max(np.abs(new-cent))<1e-5:break
 cent=new
for j in range(k):p[f'tm_cluster_{j}']=0
p.loc[p.tm_n>0,[f'tm_cluster_{j}' for j in range(k)]]=np.eye(k,dtype=np.int8)[lab]
p.to_csv('artifacts/trackman_prior_physics.csv',index=False)
print(p.shape,'cluster counts',np.bincount(lab,minlength=k).tolist())
